"""Analyze the single-seed Stage-2B causal resolution screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ism_diffusion.scale_evaluation import write_json


MODEL_ORDER = ["LG-T3", "LG-Punit", "Gap-Unit", "U-RandPE", "Gap-Matched"]
MODEL_COLORS = {
    "LG-T3": "#18549a",
    "LG-Punit": "#3c949e",
    "Gap-Unit": "#7a7a7a",
    "U-RandPE": "#dca6a0",
    "Gap-Matched": "#b9403b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-data", type=Path, required=True)
    parser.add_argument("--probe-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260819)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--practical-threshold", type=float, default=0.002)
    return parser.parse_args()


def paired_bootstrap(
    values: np.ndarray,
    replicates: int,
    seed: int,
) -> dict:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    chunk = 500
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        draws[start:stop] = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "se": float(values.std(ddof=1) / np.sqrt(len(values))),
        "ci95": [float(value) for value in np.quantile(draws, (0.025, 0.975))],
        "samples": int(len(values)),
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
        }
    )


def save_figure(figure: plt.Figure, output: Path, stem: str) -> None:
    figure.savefig(output / f"{stem}.png", dpi=300)
    figure.savefig(output / f"{stem}.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.probe_summary.read_text(encoding="utf-8"))
    with np.load(args.probe_data, allow_pickle=False) as data:
        model_names = [str(value) for value in data["model_names"]]
        geometry_names = [str(value) for value in data["geometry_names"]]
        conditions = [str(value) for value in data["coordinate_conditions"]]
        t_grid = np.asarray(data["t_grid"], dtype=np.float64)
        nll = np.asarray(data["nll"], dtype=np.float64)

    missing = sorted(set(MODEL_ORDER) - set(model_names))
    if missing:
        raise ValueError(f"missing required models: {missing}")
    if "held_mix_3_6_w64" not in geometry_names:
        raise ValueError("held_mix_3_6_w64 geometry is required")
    for condition in ("correct", "unit", "shuffled"):
        if condition not in conditions:
            raise ValueError(f"missing coordinate condition: {condition}")

    model_index = {name: model_names.index(name) for name in model_names}
    geometry_index = {name: geometry_names.index(name) for name in geometry_names}
    condition_index = {name: conditions.index(name) for name in conditions}
    averaged = nll.mean(axis=2)  # model, geometry, coordinate condition, sample
    held = geometry_index["held_mix_3_6_w64"]
    correct = condition_index["correct"]
    unit = condition_index["unit"]
    shuffled = condition_index["shuffled"]

    contrast_specs = [
        ("matched_minus_data_only", "Gap-Matched", correct, "Gap-Unit", correct),
        ("matched_minus_randpe", "Gap-Matched", correct, "U-RandPE", correct),
        ("matched_minus_uniform_t3", "Gap-Matched", correct, "LG-T3", correct),
        ("correct_minus_unit", "Gap-Matched", correct, "Gap-Matched", unit),
        (
            "correct_minus_shuffled",
            "Gap-Matched",
            correct,
            "Gap-Matched",
            shuffled,
        ),
    ]
    contrasts = {}
    for index, (name, left_model, left_condition, right_model, right_condition) in enumerate(
        contrast_specs
    ):
        values = (
            averaged[model_index[left_model], held, left_condition]
            - averaged[model_index[right_model], held, right_condition]
        )
        contrasts[name] = paired_bootstrap(
            values,
            args.bootstrap_replicates,
            args.bootstrap_seed + 7919 * index,
        )

    threshold = float(args.practical_threshold)

    def passes(name: str) -> bool:
        result = contrasts[name]
        return result["mean"] <= -threshold and result["ci95"][1] < 0.0

    checks = {
        "matched_beats_data_only": passes("matched_minus_data_only"),
        "matched_beats_randpe": passes("matched_minus_randpe"),
        "correct_beats_unit": passes("correct_minus_unit"),
        "correct_beats_shuffled": passes("correct_minus_shuffled"),
    }
    if all(checks.values()):
        label = "GO_FULL_2B"
        recommendation = (
            "Run the preregistered 3-seed, 15000-step hierarchical confirmation."
        )
    elif checks["matched_beats_data_only"] and checks["correct_beats_unit"]:
        label = "PARTIAL_CAUSAL_SUPPORT"
        recommendation = (
            "Do not scale to 12 runs yet; refine the randomized-PE mismatch control "
            "and repeat one independent training seed."
        )
    else:
        label = "STOP_FULL_2B"
        recommendation = (
            "Do not claim physical-coordinate causality or launch the 12-run study; "
            "retain Local-Global as a mask-robust architecture result only."
        )

    initialization_hashes = {
        name: summary["models"][name].get("initialization_hash")
        for name in ("Gap-Unit", "U-RandPE", "Gap-Matched")
    }
    initialization_paired = len(set(initialization_hashes.values())) == 1
    checks["paired_initialization"] = initialization_paired
    if not initialization_paired:
        label = "INVALID"
        recommendation = "Initialization hashes differ; repair pairing before inference."

    configure_style()
    geometry_labels = {
        "seen_mix_w64": "seen mix\nW64",
        "held_mix_3_6_w64": "held {3,6}\nW64",
        "fixed3_w64": "gap 3\nW64",
        "fixed6_w64": "gap 6\nW64",
        "fixed10_w48": "gap 10\nW48",
    }

    figure, axis = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    x = np.arange(len(geometry_names))
    for model in MODEL_ORDER:
        values = averaged[model_index[model], :, correct]
        means = values.mean(axis=-1)
        se = values.std(axis=-1, ddof=1) / np.sqrt(values.shape[-1])
        axis.errorbar(
            x,
            means,
            yerr=se,
            marker="o",
            linewidth=1.8,
            markersize=4,
            capsize=2,
            label=model,
            color=MODEL_COLORS[model],
        )
    axis.set_xticks(x, [geometry_labels.get(name, name) for name in geometry_names])
    axis.set_ylabel("masked-site NLL (correct coordinates)")
    axis.set_title("RandomGap causal screen: held-out geometry")
    axis.grid(alpha=0.2)
    axis.legend(ncol=3, frameon=False)
    save_figure(figure, args.output_dir, "01_geometry_nll")

    figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    labels = [spec[0] for spec in contrast_specs]
    means = np.asarray([contrasts[name]["mean"] for name in labels])
    lower = means - np.asarray([contrasts[name]["ci95"][0] for name in labels])
    upper = np.asarray([contrasts[name]["ci95"][1] for name in labels]) - means
    y = np.arange(len(labels))
    axis.errorbar(means, y, xerr=np.vstack((lower, upper)), fmt="o", capsize=3)
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
    axis.axvline(-threshold, color="#b9403b", linestyle=":", linewidth=1.5)
    axis.set_yticks(y, [name.replace("_", " ") for name in labels])
    axis.set_xlabel("paired NLL difference (negative favors matched physical model)")
    axis.set_title("Paired causal contrasts on held-out {3,6} RandomGap")
    axis.grid(axis="x", alpha=0.2)
    save_figure(figure, args.output_dir, "02_paired_contrasts")

    figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    width = 0.24
    condition_x = np.arange(len(conditions))
    for model_offset, model in enumerate(("Gap-Unit", "U-RandPE", "Gap-Matched")):
        values = averaged[model_index[model], held]
        means = values.mean(axis=-1)
        se = values.std(axis=-1, ddof=1) / np.sqrt(values.shape[-1])
        axis.bar(
            condition_x + (model_offset - 1) * width,
            means,
            width,
            yerr=se,
            capsize=2,
            label=model,
            color=MODEL_COLORS[model],
        )
    axis.set_xticks(condition_x, conditions)
    axis.set_ylabel("masked-site NLL")
    axis.set_title("Coordinate counterfactuals on held-out {3,6} RandomGap")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    save_figure(figure, args.output_dir, "03_coordinate_counterfactual")

    figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    for model in MODEL_ORDER:
        values = nll[model_index[model], held, :, correct]
        means = values.mean(axis=-1)
        se = values.std(axis=-1, ddof=1) / np.sqrt(values.shape[-1])
        axis.errorbar(
            t_grid,
            means,
            yerr=se,
            marker="o",
            capsize=2,
            linewidth=1.8,
            label=model,
            color=MODEL_COLORS[model],
        )
    axis.set_xlabel("mask fraction t")
    axis.set_ylabel("masked-site NLL")
    axis.set_title("Noise-level robustness on held-out {3,6} RandomGap")
    axis.grid(alpha=0.2)
    axis.legend(ncol=3, frameon=False)
    save_figure(figure, args.output_dir, "04_nll_by_t")

    decision = {
        "label": label,
        "scope": "single_training_seed_causal_screen",
        "practical_threshold": threshold,
        "checks": checks,
        "contrasts": contrasts,
        "initialization_hashes": initialization_hashes,
        "recommendation": recommendation,
        "probe_settings": summary["settings"],
    }
    write_json(args.output_dir / "stage2b_causal_decision.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
