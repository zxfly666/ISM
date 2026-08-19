"""Aggregate Level-1 training, mechanism probes, and generation metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ism_diffusion.scale_evaluation import (
    correlation_band_errors,
    open_ensemble_metrics,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/level1_rapid"))
    parser.add_argument(
        "--parent-data", type=Path, default=Path("data/level1/parents_l1024.npz")
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=8675309)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bootstrap_mean(values: np.ndarray, draws: int, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if not len(values):
        return {"n": 0, "mean": float("nan"), "ci95": [float("nan")] * 2}
    chunk = min(draws, 500)
    means = []
    remaining = draws
    while remaining:
        take = min(chunk, remaining)
        indices = rng.integers(len(values), size=(take, len(values)))
        means.append(values[indices].mean(axis=1))
        remaining -= take
    distribution = np.concatenate(means)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
        "ci95": [
            float(np.quantile(distribution, 0.025)),
            float(np.quantile(distribution, 0.975)),
        ],
    }


def selected_map(rows: list[dict], model: str, condition_key: str, condition: str) -> dict:
    output = {}
    for row in rows:
        if row["model"] == model and row[condition_key] == condition:
            key = (row["sample_id"], row["t"], row.get("physical_stride", ""))
            output[key] = float(row.get("nll", row.get("kl")))
    return output


def paired_difference(left: dict, right: dict) -> np.ndarray:
    keys = sorted(set(left) & set(right))
    return np.asarray([left[key] - right[key] for key in keys], dtype=np.float64)


def training_results(root: Path) -> tuple[dict, bool]:
    output = {}
    hashes = []
    for name in ("T0", "T3", "Pphase", "Punit"):
        history = json.loads((root / name / "history.json").read_text())
        performance = json.loads((root / name / "performance.json").read_text())
        checkpoint = torch.load(
            root / name / "last.pt", map_location="cpu", weights_only=False
        )
        hashes.append(checkpoint["initialization_hash"])
        output[name] = {
            "step": int(checkpoint["step"]),
            "initialization_hash": checkpoint["initialization_hash"],
            "final": history[-1],
            "best_anchor_nelbo": float(
                min(row["validation"]["anchor_w48_s1"]["nelbo"] for row in history)
            ),
            "performance": performance,
            "history": history,
        }
    return output, len(set(hashes)) == 1


def coordinate_results(rows: list[dict], draws: int, seed: int) -> tuple[dict, bool]:
    output = {}
    all_pass = True
    comparisons = (
        ("T3_Unit", "T3", "CorrectCoord", "T3", "UnitCoord"),
        ("T3_WrongScale", "T3", "CorrectCoord", "T3", "WrongScale"),
        ("Punit_Unit", "T3", "CorrectCoord", "Punit", "UnitCoord"),
    )
    for stride in (3, 6):
        stride_rows = [row for row in rows if int(row["physical_stride"]) == stride]
        output[str(stride)] = {}
        for index, (label, lm, lc, rm, rc) in enumerate(comparisons):
            left = selected_map(stride_rows, lm, "condition", lc)
            right = selected_map(stride_rows, rm, "condition", rc)
            delta = paired_difference(left, right)
            result = bootstrap_mean(delta, draws, seed + 100 * stride + index)
            per_t = {}
            for t_value in sorted({row["t"] for row in stride_rows}):
                t_rows = [row for row in stride_rows if row["t"] == t_value]
                t_delta = paired_difference(
                    selected_map(t_rows, lm, "condition", lc),
                    selected_map(t_rows, rm, "condition", rc),
                )
                per_t[t_value] = float(t_delta.mean())
            result["per_t_mean"] = per_t
            result["negative_t_bins"] = int(sum(value < 0 for value in per_t.values()))
            result["pass_direction"] = bool(
                result["mean"] < 0
                and result["negative_t_bins"] >= max(1, len(per_t) - 1)
            )
            output[str(stride)][label] = result
            all_pass &= result["pass_direction"]
    return output, bool(all_pass)


def markov_results(rows: list[dict], draws: int, seed: int) -> tuple[dict, bool, bool, float]:
    output = {}
    pad_drift = max(
        abs(float(row["logit_drift"]))
        for row in rows
        if row["context"] == "Large-PAD"
    )
    pollution_present = False
    improvement_flags = []
    for context_index, context in enumerate(("Large-MASK", "Large-Visible")):
        output[context] = {}
        deltas = {}
        for model_index, model in enumerate(("T0", "T3")):
            context_values = selected_map(rows, model, "context", context)
            small_values = selected_map(rows, model, "context", "Small")
            delta = paired_difference(context_values, small_values)
            deltas[model] = {
                key: context_values[key] - small_values[key]
                for key in set(context_values) & set(small_values)
            }
            result = bootstrap_mean(
                delta, draws, seed + 100 * context_index + model_index
            )
            output[context][f"{model}_delta_kl"] = result
            if model == "T0" and result["ci95"][0] > 0:
                pollution_present = True
        cross = paired_difference(deltas["T3"], deltas["T0"])
        comparison = bootstrap_mean(
            cross, draws, seed + 1000 + context_index
        )
        comparison["improved"] = bool(comparison["mean"] < 0)
        output[context]["T3_minus_T0_delta"] = comparison
        improvement_flags.append(comparison["improved"])
    pollution_improved = pollution_present and any(improvement_flags)
    return output, pollution_present, pollution_improved, float(pad_drift)


def generation_results(root: Path) -> tuple[dict, bool, bool]:
    metrics = json.loads((root / "generation" / "metrics.json").read_text())
    reference_error = metrics["reference"]["target_control_correlation_error"]
    models = metrics["models"]
    t0_expanded = models["T0"]["connected_correlation_error"]["expanded"]["nrmse"]
    t3_expanded = models["T3"]["connected_correlation_error"]["expanded"]["nrmse"]
    noise_expanded = reference_error["expanded"]["nrmse"]
    improvement = float(t0_expanded - t3_expanded)

    target = metrics["reference"]["target"]
    control = metrics["reference"]["control"]
    scalar_checks = {}
    for key in ("abs_magnetization_mean", "low_frequency_power_mean"):
        t0_error = abs(models["T0"]["metrics"][key] - target[key])
        t3_error = abs(models["T3"]["metrics"][key] - target[key])
        envelope = abs(control[key] - target[key])
        scalar_checks[key] = {
            "T0_error": float(t0_error),
            "T3_error": float(t3_error),
            "mc_envelope": float(envelope),
            "no_opposite_degradation": bool(t3_error <= t0_error + envelope),
        }
    generation_signal = bool(
        improvement > noise_expanded
        and all(value["no_opposite_degradation"] for value in scalar_checks.values())
    )

    with np.load(root / "generation" / "samples_and_correlations.npz") as data:
        target_spins = data["mc_target"]
        rankings = {}
        for steps, prefix in ((64, "model"), (128, "sensitivity")):
            errors = {}
            _, _, _, target_c = open_ensemble_metrics(target_spins, max_radius=32)
            for name in ("T0", "T3"):
                _, _, _, connected = open_ensemble_metrics(
                    data[f"{prefix}_{name}"], max_radius=32
                )
                errors[name] = correlation_band_errors(connected, target_c)["expanded"][
                    "nrmse"
                ]
            rankings[str(steps)] = errors
    stable = (rankings["64"]["T3"] < rankings["64"]["T0"]) == (
        rankings["128"]["T3"] < rankings["128"]["T0"]
    )
    output = {
        "metrics": metrics,
        "expanded_nrmse": {
            "T0": float(t0_expanded),
            "T3": float(t3_expanded),
            "MC_control": float(noise_expanded),
            "T0_minus_T3": improvement,
        },
        "scalar_checks": scalar_checks,
        "sampler_rankings": rankings,
        "sampler_ranking_stable": bool(stable),
        "generation_signal": generation_signal,
    }
    return output, generation_signal, bool(stable)


def parent_diagnostics(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
    return metadata["diagnostics"]


def make_plots(root: Path, training: dict, markov: dict, coordinate: dict) -> None:
    figures = root / "final" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, result in training.items():
        steps = [row["step"] for row in result["history"]]
        axes[0].plot(
            steps,
            [row["validation"]["anchor_w48_s1"]["nelbo"] for row in result["history"]],
            label=name,
        )
        axes[1].plot(
            steps,
            [row["validation"]["held_w24_s3"]["nelbo"] for row in result["history"]],
            label=name,
        )
    axes[0].set_title("ID: W=48, stride=1")
    axes[1].set_title("Held-out: W=24, stride=3")
    for axis in axes:
        axis.set_xlabel("training step")
        axis.set_ylabel("validation NLL")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "training_curves.png", dpi=180)
    figure.savefig(figures / "training_curves.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    labels, values, errors = [], [], []
    for context in ("Large-MASK", "Large-Visible"):
        for model in ("T0", "T3"):
            result = markov[context][f"{model}_delta_kl"]
            labels.append(f"{model}\n{context}")
            values.append(result["mean"])
            errors.append(
                [result["mean"] - result["ci95"][0], result["ci95"][1] - result["mean"]]
            )
    axis.bar(np.arange(len(labels)), values, color=["#777777", "#1764ab"] * 2)
    axis.errorbar(
        np.arange(len(labels)), values, yerr=np.asarray(errors).T, fmt="none", color="black"
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_ylabel(r"$\Delta_{MB}$ (KL)")
    axis.set_title("Exact Markov-blanket contamination")
    figure.tight_layout()
    figure.savefig(figures / "markov_probe.png", dpi=180)
    figure.savefig(figures / "markov_probe.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, stride in zip(axes, ("3", "6")):
        labels = list(coordinate[stride])
        values = [coordinate[stride][label]["mean"] for label in labels]
        lower = [
            coordinate[stride][label]["mean"] - coordinate[stride][label]["ci95"][0]
            for label in labels
        ]
        upper = [
            coordinate[stride][label]["ci95"][1] - coordinate[stride][label]["mean"]
            for label in labels
        ]
        axis.bar(np.arange(len(labels)), values, color="#1764ab")
        axis.errorbar(np.arange(len(labels)), values, yerr=[lower, upper], fmt="none", color="black")
        axis.axhline(0, color="black", linewidth=1)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
        axis.set_title(f"physical stride s={stride}")
        axis.set_ylabel("paired NLL: CorrectCoord - baseline")
    figure.tight_layout()
    figure.savefig(figures / "coordinate_probe.png", dpi=180)
    figure.savefig(figures / "coordinate_probe.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = args.root
    final = root / "final"
    final.mkdir(parents=True, exist_ok=True)
    training, same_initialization = training_results(root)
    coordinate_rows = read_csv(root / "probes" / "coordinates" / "per_sample.csv")
    markov_rows = read_csv(root / "probes" / "markov" / "per_sample.csv")
    coordinate, coordinate_signal = coordinate_results(
        coordinate_rows, args.bootstrap, args.seed
    )
    markov, pollution_present, pollution_improved, pad_drift = markov_results(
        markov_rows, args.bootstrap, args.seed + 10_000
    )
    generation, generation_signal, sampler_stable = generation_results(root)
    diagnostics = parent_diagnostics(args.parent_data)

    t0_final = training["T0"]["final"]
    t3_final = training["T3"]["final"]
    nll_delta = float(
        t3_final["validation"]["anchor_w48_s1"]["nelbo"]
        - t0_final["validation"]["anchor_w48_s1"]["nelbo"]
    )
    short = generation["metrics"]["models"]
    short_t0 = short["T0"]["connected_correlation_error"]["short"]["nrmse"]
    short_t3 = short["T3"]["connected_correlation_error"]["short"]["nrmse"]
    short_noise = generation["metrics"]["reference"]["target_control_correlation_error"]["short"]["nrmse"]
    id_guard = bool(nll_delta <= 0.01 and short_t3 <= short_t0 + short_noise)

    mc_valid = bool(
        diagnostics["energy_split_rhat"] < 1.1
        and diagnostics["abs_magnetization_split_rhat"] < 1.1
    )
    steps_valid = all(result["step"] == 8000 for result in training.values())
    invalid_reasons = []
    if not mc_valid:
        invalid_reasons.append("MC chain diagnostics failed")
    if not same_initialization:
        invalid_reasons.append("model initialization hashes differ")
    if pad_drift > 1e-3:
        invalid_reasons.append("PAD invariance failed")
    if not sampler_stable:
        invalid_reasons.append("64/128-step sampler ranking reversed")
    if not steps_valid:
        invalid_reasons.append("one or more models did not reach 8000 steps")

    pollution_signal = pollution_present and pollution_improved
    if invalid_reasons:
        label = "INVALID PILOT"
    elif id_guard and coordinate_signal and pollution_signal and generation_signal:
        label = "STRONG GO"
    elif id_guard and coordinate_signal and (pollution_signal or generation_signal):
        label = "GO"
    elif generation_signal and not coordinate_signal:
        label = "CONDITIONAL GO"
    else:
        label = "NO-GO"

    summary = {
        "decision": label,
        "invalid_reasons": invalid_reasons,
        "validity": {
            "mc_valid": mc_valid,
            "same_initialization": same_initialization,
            "pad_max_abs_logit_drift": pad_drift,
            "sampler_ranking_stable": sampler_stable,
            "all_models_8000_steps": steps_valid,
        },
        "signals": {
            "id_guard": id_guard,
            "coordinate_signal": coordinate_signal,
            "pollution_present": pollution_present,
            "pollution_improved": pollution_improved,
            "pollution_signal": pollution_signal,
            "generation_signal": generation_signal,
        },
        "id": {
            "T3_minus_T0_anchor_nll": nll_delta,
            "short_nrmse": {"T0": short_t0, "T3": short_t3, "MC_control": short_noise},
        },
        "mc_diagnostics": diagnostics,
        "training": {
            name: {key: value for key, value in result.items() if key != "history"}
            for name, result in training.items()
        },
        "coordinate_probe": coordinate,
        "markov_probe": markov,
        "generation": generation,
    }
    write_json(final / "level1_summary.json", summary)
    make_plots(root, training, markov, coordinate)

    report = f"""# 第一级 Scale-Aware Ising 快速判断

## 结论

自动化、预注册规则给出的标签为：**{label}**。

该标签仅用于决定是否进入多 seed 与更大尺度实验，不构成论文结论。

## 有效性

- MC 有效：{mc_valid}
- 四组初始化一致：{same_initialization}
- PAD 最大绝对 logit 漂移：{pad_drift:.3e}
- 64/128-step 排名稳定：{sampler_stable}
- 四组均达到 8,000 steps：{steps_valid}

## 三类信号

- ID guard：{id_guard}（T3-T0 anchor NLL = {nll_delta:+.6f}）
- Coordinate signal：{coordinate_signal}
- Pollution signal：{pollution_signal}（T0 中可检测污染：{pollution_present}；T3 改善：{pollution_improved}）
- Generation signal：{generation_signal}

## 解释边界

这是单 seed pilot。即使标签为 GO，也只能说明“训练时随机化真实坐标尺度”的研究假设值得继续验证；必须经过多 seed、W=128/更大 context 与更强基线后，才能写成稳定有效的方法结论。
"""
    (final / "LEVEL1_DECISION_ZH.md").write_text(report, encoding="utf-8")
    print(json.dumps({"decision": label, "signals": summary["signals"]}, indent=2))


if __name__ == "__main__":
    main()
