"""Pre-registered Stage-2A decision, adaptation advice, and paper-style figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ism_diffusion.scale_evaluation import write_json


PUBLICATION_STYLE = {
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 13,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 1.8,
    "legend.frameon": False,
    "svg.fonttype": "none",
}
COLORS = {
    "T0": "#272727",
    "T3": "#B64342",
    "Punit": "#767676",
    "LG-Punit": "#42949E",
    "LG-T3": "#0F4D92",
    "Dense-T3+": "#E9A6A1",
    "Dense-Punit+": "#CFCECE",
}


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must be NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=named_path, required=True)
    parser.add_argument("--coordinate-summary", type=Path, required=True)
    parser.add_argument("--coordinate-data", type=Path, required=True)
    parser.add_argument("--markov-root", type=Path, required=True)
    parser.add_argument("--generation-metrics", type=Path, required=True)
    parser.add_argument("--generation-data", type=Path, required=True)
    parser.add_argument("--sampler-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def bootstrap_mean_ci(
    values: np.ndarray, draws: int, rng: np.random.Generator
) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 250):
        count = min(250, draws - start)
        indices = rng.integers(len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "samples": int(len(values)),
    }


def latest_validation(payload: dict, name: str) -> float:
    records = [
        row
        for row in payload.get("history", [])
        if name in row.get("validation", {})
    ]
    if not records:
        raise ValueError(f"checkpoint has no validation geometry {name}")
    return float(records[-1]["validation"][name]["nelbo"])


def load_markov(args: argparse.Namespace) -> tuple[list[dict], Path]:
    summaries = []
    csv_paths = []
    for directory in sorted(args.markov_root.glob("w*")):
        summary_path = directory / "markov" / "summary.json"
        csv_path = directory / "markov" / "per_sample.csv"
        if not summary_path.exists() or not csv_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["width"] = int(summary["large_width"])
        summaries.append(summary)
        csv_paths.append((summary["width"], csv_path))
    if not summaries:
        raise ValueError("no complete Markov widths found")
    return summaries, max(csv_paths, key=lambda item: item[0])[1]


def markov_width_rows(summaries: list[dict]) -> list[dict]:
    output = []
    for summary in summaries:
        table = {
            (row["model"], float(row["t"]), row["context"]): row
            for row in summary["groups"]
        }
        models = sorted({key[0] for key in table})
        times = sorted({key[1] for key in table})
        for model in models:
            for context in ("Large-MASK", "Large-Visible", "Large-PAD"):
                deltas = []
                for t_value in times:
                    small = table[(model, t_value, "Small")]["kl_mean"]
                    deltas.append(table[(model, t_value, context)]["kl_mean"] - small)
                output.append(
                    {
                        "width": int(summary["width"]),
                        "model": model,
                        "context": context,
                        "delta_kl": float(np.mean(deltas)),
                    }
                )
    return output


def paired_markov_ci(
    csv_path: Path,
    model: str,
    context: str,
    draws: int,
    rng: np.random.Generator,
) -> dict:
    values: dict[tuple[int, float, str], float] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["model"] != model:
                continue
            values[(int(row["sample_id"]), float(row["t"]), row["context"])] = float(
                row["kl"]
            )
    samples = sorted({key[0] for key in values})
    times = sorted({key[1] for key in values})
    deltas = np.asarray(
        [
            np.mean(
                [
                    values[(sample, t_value, context)]
                    - values[(sample, t_value, "Small")]
                    for t_value in times
                ]
            )
            for sample in samples
        ]
    )
    return bootstrap_mean_ci(deltas, draws, rng)


def plot_training(output: Path, checkpoints: dict[str, dict]) -> None:
    plt.rcParams.update(PUBLICATION_STYLE)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    geometries = (("anchor_w48_s1", "ID: W=48, s=1"), ("held_w24_s3", "OOD: W=24, s=3"))
    for axis, (geometry, title) in zip(axes, geometries):
        for name, payload in checkpoints.items():
            rows = [row for row in payload.get("history", []) if geometry in row.get("validation", {})]
            if not rows:
                continue
            axis.plot(
                [row["step"] for row in rows],
                [row["validation"][geometry]["nelbo"] for row in rows],
                color=COLORS.get(name, "#767676"),
                linewidth=2.2,
                label=name,
            )
        axis.set_title(title)
        axis.set_xlabel("training update")
        axis.set_ylabel("masked-site NLL")
        axis.grid(alpha=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=max(len(labels), 1))
    figure.tight_layout(rect=(0, 0, 1, 0.9), pad=1.5)
    figure.savefig(output / "01_training_curves.png", dpi=300)
    figure.savefig(output / "01_training_curves.pdf")
    plt.close(figure)


def plot_coordinate(output: Path, data: dict) -> None:
    plt.rcParams.update(PUBLICATION_STYLE)
    names = [str(value) for value in data["model_names"]]
    strides = data["physical_strides"]
    scales = data["coordinate_scales"]
    nll = data["nll"]
    figure, axes = plt.subplots(1, len(strides), figsize=(6 * len(strides), 4.3), squeeze=False)
    for stride_index, physical_stride in enumerate(strides):
        axis = axes[0, stride_index]
        for model_index, name in enumerate(names):
            values = nll[model_index, stride_index].mean(axis=(0, 2))
            axis.plot(scales, values, marker="o", linewidth=2.1, color=COLORS.get(name, "#767676"), label=name)
        axis.axvline(float(physical_stride), color="#B64342", linestyle=":", linewidth=2, label="true scale")
        axis.set_xscale("log")
        axis.set_title(f"physical stride = {int(physical_stride)}")
        axis.set_xlabel("coordinate scale")
        axis.set_ylabel("paired NLL")
        axis.grid(alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6))
    figure.tight_layout(rect=(0, 0, 1, 0.88), pad=1.5)
    figure.savefig(output / "02_coordinate_response.png", dpi=300)
    figure.savefig(output / "02_coordinate_response.pdf")
    plt.close(figure)


def plot_markov(output: Path, rows: list[dict]) -> None:
    plt.rcParams.update(PUBLICATION_STYLE)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, context, title in zip(
        axes,
        ("Large-MASK", "Large-Visible"),
        ("Distant MASK pollution", "Distant visible-spin pollution"),
    ):
        selected = [row for row in rows if row["context"] == context]
        for model in sorted({row["model"] for row in selected}):
            model_rows = sorted(
                [row for row in selected if row["model"] == model],
                key=lambda row: row["width"],
            )
            axis.plot(
                [row["width"] for row in model_rows],
                [row["delta_kl"] for row in model_rows],
                marker="o",
                linewidth=2.1,
                color=COLORS.get(model, "#767676"),
                label=model,
            )
        axis.axhline(0, color="#272727", linestyle="--", linewidth=1.3)
        axis.set_title(title)
        axis.set_xlabel("context width W")
        axis.set_ylabel(r"$\Delta_{MB}$ (KL)")
        axis.grid(alpha=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=max(len(labels), 1))
    figure.tight_layout(rect=(0, 0, 1, 0.89), pad=1.5)
    figure.savefig(output / "03_markov_scaling.png", dpi=300)
    figure.savefig(output / "03_markov_scaling.pdf")
    plt.close(figure)


def plot_generation(output: Path, metrics: dict, data: dict) -> None:
    plt.rcParams.update(PUBLICATION_STYLE)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    target = metrics["reference"]["target"]
    target_connected = data["connected_mc_target"]
    radii = data["radii_mc_target"]
    axes[0].plot(radii[1:], target_connected[1:], color="#272727", linewidth=2.5, label="MC target")
    model_names = list(metrics["models"])
    for name in model_names:
        safe = name.lower().replace(" ", "_")
        axes[0].plot(
            data[f"radii_{safe}"][1:],
            data[f"connected_{safe}"][1:],
            linewidth=2.1,
            color=COLORS.get(name, "#767676"),
            label=name,
        )
    axes[0].set_xlabel("distance r")
    axes[0].set_ylabel("connected G(r)")
    axes[0].set_title("Generated two-point correlation")
    axes[0].grid(alpha=0.18)

    x = np.arange(len(model_names))
    short = [metrics["models"][name]["connected_correlation_error"]["short"]["nrmse"] for name in model_names]
    expanded = [metrics["models"][name]["connected_correlation_error"]["expanded"]["nrmse"] for name in model_names]
    width = 0.36
    axes[1].bar(x - width / 2, short, width, label="short r=1..8", color="#3775BA", edgecolor="black", linewidth=1.0)
    axes[1].bar(x + width / 2, expanded, width, label="expanded r=17..32", color="#E9A6A1", edgecolor="black", linewidth=1.0, hatch="//")
    axes[1].set_xticks(x, model_names, rotation=25, ha="right")
    axes[1].set_ylabel("G(r) NRMSE")
    axes[1].set_title("Short/long-range fidelity")
    axes[1].grid(axis="y", alpha=0.18)
    axes[1].legend()
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6))
    figure.tight_layout(rect=(0, 0, 1, 0.88), pad=1.5)
    figure.savefig(output / "04_generation_physics.png", dpi=300)
    figure.savefig(output / "04_generation_physics.pdf")
    plt.close(figure)


def plot_gate(output: Path, checkpoints: dict[str, dict]) -> None:
    plt.rcParams.update(PUBLICATION_STYLE)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for name in ("LG-Punit", "LG-T3"):
        payload = checkpoints.get(name)
        if payload is None:
            continue
        rows = [row for row in payload.get("history", []) if row.get("diagnostics")]
        if not rows:
            continue
        last = rows[-1]
        for axis, geometry in zip(axes, ("anchor_w48_s1", "held_w24_s3")):
            diagnostic = last["diagnostics"].get(geometry, {}).get("per_t", [])
            axis.plot(
                [row["t"] for row in diagnostic],
                [row["gate_mean"] for row in diagnostic],
                marker="o",
                linewidth=2.1,
                color=COLORS[name],
                label=name,
            )
    axes[0].set_title("ID gate")
    axes[1].set_title("Held-out stride gate")
    for axis in axes:
        axis.set_xlabel("mask fraction t")
        axis.set_ylabel("mean global gate")
        axis.grid(alpha=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=max(len(labels), 1))
    figure.tight_layout(rect=(0, 0, 1, 0.88), pad=1.5)
    figure.savefig(output / "05_gate_mechanism.png", dpi=300)
    figure.savefig(output / "05_gate_mechanism.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    checkpoints = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in args.checkpoint
    }
    coordinate_summary = json.loads(args.coordinate_summary.read_text(encoding="utf-8"))
    coordinate_npz = np.load(args.coordinate_data, allow_pickle=False)
    coordinate_data = {key: coordinate_npz[key] for key in coordinate_npz.files}
    markov_summaries, largest_markov_csv = load_markov(args)
    markov_rows = markov_width_rows(markov_summaries)
    generation = json.loads(args.generation_metrics.read_text(encoding="utf-8"))
    generation_npz = np.load(args.generation_data, allow_pickle=False)
    generation_data = {key: generation_npz[key] for key in generation_npz.files}
    sampler = json.loads(args.sampler_selection.read_text(encoding="utf-8"))

    id_t0 = latest_validation(checkpoints["T0"], "anchor_w48_s1")
    id_lg = latest_validation(checkpoints["LG-T3"], "anchor_w48_s1")
    id_delta = id_lg - id_t0

    names = [str(value) for value in coordinate_data["model_names"]]
    model_index = {name: index for index, name in enumerate(names)}
    strides = coordinate_data["physical_strides"]
    scales = coordinate_data["coordinate_scales"]
    nll = coordinate_data["nll"]
    coordinate_deltas = []
    coordinate_scale_passes = []
    for stride_index, stride in enumerate(strides):
        true_index = int(np.argmin(np.abs(scales - stride)))
        unit_index = int(np.argmin(np.abs(scales - 1.0)))
        lg_values = nll[model_index["LG-T3"], stride_index]
        unit_values = lg_values[:, unit_index].mean(axis=0)
        true_values = lg_values[:, true_index].mean(axis=0)
        coordinate_deltas.append(true_values - unit_values)
        mean_curve = lg_values.mean(axis=(0, 2))
        best_scale = float(scales[int(np.argmin(mean_curve))])
        coordinate_scale_passes.append(abs(math.log(best_scale / float(stride))) <= math.log(1.5))
    coordinate_ci = bootstrap_mean_ci(
        np.concatenate(coordinate_deltas), args.bootstrap_draws, rng
    )
    punit_deltas = []
    for stride_index, stride in enumerate(strides):
        true_index = int(np.argmin(np.abs(scales - stride)))
        unit_index = int(np.argmin(np.abs(scales - 1.0)))
        lg = nll[model_index["LG-T3"], stride_index, :, true_index].mean(axis=0)
        punit = nll[model_index["LG-Punit"], stride_index, :, unit_index].mean(axis=0)
        punit_deltas.append(lg - punit)
    coordinate_vs_punit = bootstrap_mean_ci(
        np.concatenate(punit_deltas), args.bootstrap_draws, rng
    )

    markov_names = [name for name in ("T0", "T3", "Dense-T3+", "Dense-Punit+", "LG-Punit", "LG-T3") if name in checkpoints]
    markov_mask = {
        name: paired_markov_ci(
            largest_markov_csv, name, "Large-MASK", args.bootstrap_draws, rng
        )
        for name in markov_names
    }
    markov_visible = {
        name: paired_markov_ci(
            largest_markov_csv, name, "Large-Visible", args.bootstrap_draws, rng
        )
        for name in markov_names
    }
    generation_lg = generation["models"]["LG-T3"]
    short_nrmse = float(generation_lg["connected_correlation_error"]["short"]["nrmse"])
    expanded_nrmse = float(generation_lg["connected_correlation_error"]["expanded"]["nrmse"])
    energy_error = abs(
        float(generation_lg["metrics"]["energy_mean"])
        - float(generation["reference"]["target"]["energy_mean"])
    )

    dense_t3_control = "Dense-T3+" if "Dense-T3+" in markov_mask else "T3"
    checks = {
        "sampler_frozen": sampler.get("status") == "FROZEN",
        "id_noninferiority": id_delta <= 0.003,
        "coordinate_minimum_near_true": bool(all(coordinate_scale_passes)),
        "coordinate_correct_beats_unit": coordinate_ci["mean"] <= -0.002,
        "coordinate_independent_contribution": coordinate_vs_punit["mean"] <= -0.002,
        "markov_mask_reduced_50pct": markov_mask["LG-T3"]["mean"]
        <= 0.5 * markov_mask[dense_t3_control]["mean"],
        "markov_visible_reduced_50pct": markov_visible["LG-T3"]["mean"]
        <= 0.5 * markov_visible[dense_t3_control]["mean"],
        "short_nrmse": short_nrmse <= 0.040,
        "energy_error": energy_error <= 0.032,
        "expanded_nrmse": expanded_nrmse <= 0.080,
    }
    strong = all(checks.values()) and markov_mask["LG-T3"]["mean"] <= markov_mask["T0"]["mean"] and markov_visible["LG-T3"]["mean"] <= markov_visible["T0"]["mean"] and expanded_nrmse <= 0.055
    go = all(checks.values())
    local_fixed = checks["short_nrmse"] and checks["energy_error"] and checks["markov_mask_reduced_50pct"]
    long_kept = checks["expanded_nrmse"]
    if strong:
        label = "STRONG_GO"
    elif go:
        label = "GO"
    elif local_fixed or long_kept:
        label = "CONDITIONAL_GO"
    else:
        label = "NO_GO"

    modifications = []
    if not checks["sampler_frozen"]:
        modifications.append("Do not start Stage 2B generation; repair sampler closure first.")
    if not checks["id_noninferiority"] or not checks["short_nrmse"] or not checks["energy_error"]:
        modifications.append("Strengthen local protection; test LG-HardMarkov diagnostically and reduce initial/global gate capacity.")
    if not checks["markov_mask_reduced_50pct"] or not checks["markov_visible_reduced_50pct"]:
        modifications.append("Add a visibility-conditioned gate penalty before RandomGap training; distant-token contamination remains causal.")
    if not checks["coordinate_minimum_near_true"] or not checks["coordinate_correct_beats_unit"]:
        modifications.append("Do not expand gap range; audit coordinate units/RoPE frequency and matched sampling first.")
    if not checks["coordinate_independent_contribution"]:
        modifications.append("Elevate Randomized-PE and data-only controls in Stage 2B because coordinate benefit is not practically separated.")
    if not checks["expanded_nrmse"]:
        modifications.append("Preserve or widen global expert and retune only on validation; local repair has removed too much long-range signal.")
    if go:
        modifications.append("Proceed with the preregistered UniformStride/RandomGap x Unit/Matched 2x2 and three model seeds.")

    decision = {
        "label": label,
        "checks": checks,
        "metrics": {
            "id_nll_t0": id_t0,
            "id_nll_lg_t3": id_lg,
            "id_delta": id_delta,
            "coordinate_correct_minus_unit": coordinate_ci,
            "lg_t3_minus_lg_punit": coordinate_vs_punit,
            "markov_mask": markov_mask,
            "markov_visible": markov_visible,
            "parameter_matched_dense_control": dense_t3_control,
            "short_nrmse": short_nrmse,
            "expanded_nrmse": expanded_nrmse,
            "energy_abs_error": energy_error,
        },
        "frozen_sampler": sampler.get("frozen_sampler"),
        "stage2b_modifications": modifications,
    }
    write_json(args.output_dir / "stage2a_decision.json", decision)
    plot_training(args.output_dir, checkpoints)
    plot_coordinate(args.output_dir, coordinate_data)
    plot_markov(args.output_dir, markov_rows)
    plot_generation(args.output_dir, generation, generation_data)
    plot_gate(args.output_dir, checkpoints)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
