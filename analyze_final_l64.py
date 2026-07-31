"""Create the final, publication-style analysis for the L=64 pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#0F4D92"
BLUE_LIGHT = "#3775BA"
RED = "#B64342"
GREY = "#767676"
BLACK = "#272727"
TEAL = "#42949E"
PALE_BLUE = "#D8E6F3"
PALE_GREY = "#CFCECE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--sampler-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.5,
            "axes.linewidth": 1.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


def save_figure(figure: plt.Figure, directory: Path, stem: str) -> None:
    figure.tight_layout(pad=1.5)
    figure.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def energy_density(spins: np.ndarray) -> np.ndarray:
    return -(
        spins * np.roll(spins, -1, axis=-1)
        + spins * np.roll(spins, -1, axis=-2)
    ).mean(axis=(-2, -1))


def spectral_features(
    spins: np.ndarray, batch_size: int = 256
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spins = np.asarray(spins, dtype=np.int8)
    rows, length, _ = spins.shape
    coordinate = np.minimum(np.arange(length), length - np.arange(length))
    radial_bins = np.rint(
        np.sqrt(coordinate[:, None] ** 2 + coordinate[None, :] ** 2)
    ).astype(np.int32)
    selectors = [radial_bins == radius for radius in range(length // 2 + 1)]
    radial_correlation = np.empty((rows, length // 2 + 1), dtype=np.float32)
    radial_structure = np.empty_like(radial_correlation)
    s_zero = np.empty(rows, dtype=np.float64)
    s_min = np.empty(rows, dtype=np.float64)
    volume = length * length
    for start in range(0, rows, batch_size):
        stop = min(start + batch_size, rows)
        batch = spins[start:stop].astype(np.float64)
        spectrum = np.fft.fft2(batch, axes=(-2, -1))
        structure = np.abs(spectrum) ** 2 / volume
        correlation = np.fft.ifft2(
            spectrum * spectrum.conj(), axes=(-2, -1)
        ).real / volume
        for radius, selected in enumerate(selectors):
            radial_correlation[start:stop, radius] = correlation[:, selected].mean(axis=1)
            radial_structure[start:stop, radius] = structure[:, selected].mean(axis=1)
        s_zero[start:stop] = structure[:, 0, 0]
        s_min[start:stop] = 0.5 * (structure[:, 1, 0] + structure[:, 0, 1])
    return radial_correlation, radial_structure, s_zero, s_min


def sufficient_arrays(
    spins: np.ndarray, s_zero: np.ndarray, s_min: np.ndarray
) -> dict[str, np.ndarray]:
    magnetization = spins.mean(axis=(-2, -1), dtype=np.float64)
    return {
        "energy": energy_density(spins).astype(np.float64),
        "magnetization": magnetization,
        "abs_magnetization": np.abs(magnetization),
        "m2": magnetization**2,
        "m4": magnetization**4,
        "s_zero": np.asarray(s_zero, dtype=np.float64),
        "s_min": np.asarray(s_min, dtype=np.float64),
    }


def metrics_from_means(means: dict[str, np.ndarray | float], length: int) -> dict[str, np.ndarray | float]:
    m2 = np.asarray(means["m2"])
    m4 = np.asarray(means["m4"])
    binder = 1.0 - m4 / np.maximum(3.0 * m2 * m2, 1e-30)
    ratio = np.maximum(
        np.asarray(means["s_zero"]) / np.maximum(np.asarray(means["s_min"]), 1e-30) - 1.0,
        0.0,
    )
    xi_over_l = np.sqrt(ratio) / (2.0 * np.sin(np.pi / length) * length)
    return {
        "energy_mean": means["energy"],
        "magnetization_mean": means["magnetization"],
        "abs_magnetization_mean": means["abs_magnetization"],
        "binder_u4": binder,
        "xi_over_l": xi_over_l,
    }


def point_metrics(arrays: dict[str, np.ndarray], length: int) -> dict[str, float]:
    means = {key: float(value.mean()) for key, value in arrays.items()}
    return {key: float(value) for key, value in metrics_from_means(means, length).items()}


def reference_bootstrap(
    arrays: dict[str, np.ndarray],
    curves: np.ndarray,
    structures: np.ndarray,
    chain_ids: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    chains = np.unique(chain_ids)
    chain_means = {
        key: np.asarray([values[chain_ids == chain].mean() for chain in chains])
        for key, values in arrays.items()
    }
    chain_curves = np.stack([curves[chain_ids == chain].mean(axis=0) for chain in chains])
    chain_structures = np.stack(
        [structures[chain_ids == chain].mean(axis=0) for chain in chains]
    )
    selection = rng.integers(0, len(chains), size=(repeats, len(chains)))
    boot_means = {
        key: values[selection].mean(axis=1) for key, values in chain_means.items()
    }
    return boot_means, chain_curves[selection].mean(axis=1), chain_structures[selection].mean(axis=1)


def model_bootstrap(
    arrays_by_seed: list[dict[str, np.ndarray]],
    curves_by_seed: list[np.ndarray],
    structures_by_seed: list[np.ndarray],
    repeats: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    keys = tuple(arrays_by_seed[0])
    boot_means = {key: np.empty(repeats, dtype=np.float64) for key in keys}
    boot_curves = np.empty((repeats, curves_by_seed[0].shape[1]), dtype=np.float64)
    boot_structures = np.empty_like(boot_curves)
    seed_count = len(arrays_by_seed)
    for repeat in range(repeats):
        selected_seeds = rng.integers(0, seed_count, size=seed_count)
        accumulator = {key: 0.0 for key in keys}
        curve_accumulator = np.zeros(curves_by_seed[0].shape[1], dtype=np.float64)
        structure_accumulator = np.zeros_like(curve_accumulator)
        for seed_index in selected_seeds:
            rows = len(arrays_by_seed[seed_index][keys[0]])
            selected_rows = rng.integers(0, rows, size=rows)
            for key in keys:
                accumulator[key] += float(
                    arrays_by_seed[seed_index][key][selected_rows].mean()
                )
            curve_accumulator += curves_by_seed[seed_index][selected_rows].mean(axis=0)
            structure_accumulator += structures_by_seed[seed_index][selected_rows].mean(axis=0)
        for key in keys:
            boot_means[key][repeat] = accumulator[key] / seed_count
        boot_curves[repeat] = curve_accumulator / seed_count
        boot_structures[repeat] = structure_accumulator / seed_count
    return boot_means, boot_curves, boot_structures


def confidence_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, (0.025, 0.975))
    return float(low), float(high)


def fit_eta(curve: np.ndarray, r_min: int = 2, r_max: int = 16) -> dict[str, float]:
    radius = np.arange(r_min, r_max + 1, dtype=np.float64)
    values = np.asarray(curve)[r_min : r_max + 1]
    slope, intercept = np.polyfit(np.log(radius), np.log(values), 1)
    predicted = intercept + slope * np.log(radius)
    residual = float(np.sum((np.log(values) - predicted) ** 2))
    total = float(np.sum((np.log(values) - np.log(values).mean()) ** 2))
    return {
        "eta": float(-slope),
        "amplitude": float(np.exp(intercept)),
        "r_squared": float(1.0 - residual / total),
    }


def load_sampler_metrics(directory: Path) -> list[dict]:
    rows = []
    for steps in (24, 48, 96, 128):
        payload = json.loads((directory / f"validation_steps{steps}.json").read_text())
        rows.append({"steps": steps, "model": payload["model"], "reference": payload["reference"]})
    return rows


def main() -> None:
    args = parse_args()
    figures = args.output_dir / "figures"
    tables = args.output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with np.load(args.reference, allow_pickle=False) as payload:
        reference = np.asarray(payload["test"], dtype=np.int8)
        reference_chain_ids = np.asarray(payload["test_chain_id"], dtype=np.int16)
        reference_metadata = json.loads(str(payload["metadata"].item()))
    model_by_seed = []
    model_seed_names = []
    for path in args.model:
        with np.load(path, allow_pickle=False) as payload:
            model_by_seed.append(np.asarray(payload["model"], dtype=np.int8))
        model_seed_names.append(path.stem.replace("model_", ""))
    model = np.concatenate(model_by_seed, axis=0)
    np.savez_compressed(
        args.output_dir / "model_samples_4608.npz",
        model=model,
        seed_id=np.concatenate(
            [np.full(len(rows), index, dtype=np.int8) for index, rows in enumerate(model_by_seed)]
        ),
    )
    length = int(reference.shape[-1])

    reference_corr, reference_sf, reference_s0, reference_sk = spectral_features(reference)
    model_features = [spectral_features(rows) for rows in model_by_seed]
    model_corr_by_seed = [item[0] for item in model_features]
    model_sf_by_seed = [item[1] for item in model_features]
    reference_arrays = sufficient_arrays(reference, reference_s0, reference_sk)
    model_arrays_by_seed = [
        sufficient_arrays(rows, features[2], features[3])
        for rows, features in zip(model_by_seed, model_features, strict=True)
    ]
    model_arrays = {
        key: np.concatenate([rows[key] for rows in model_arrays_by_seed])
        for key in model_arrays_by_seed[0]
    }

    reference_boot_means, reference_boot_corr, reference_boot_sf = reference_bootstrap(
        reference_arrays,
        reference_corr,
        reference_sf,
        reference_chain_ids,
        args.bootstrap,
        rng,
    )
    model_boot_means, model_boot_corr, model_boot_sf = model_bootstrap(
        model_arrays_by_seed,
        model_corr_by_seed,
        model_sf_by_seed,
        args.bootstrap,
        rng,
    )
    reference_metrics = point_metrics(reference_arrays, length)
    model_metrics = point_metrics(model_arrays, length)
    reference_boot_metrics = metrics_from_means(reference_boot_means, length)
    model_boot_metrics = metrics_from_means(model_boot_means, length)

    reference_raw = reference_corr.mean(axis=0)
    model_raw = np.concatenate(model_corr_by_seed).mean(axis=0)
    reference_connected = reference_raw - reference_metrics["magnetization_mean"] ** 2
    model_connected = model_raw - model_metrics["magnetization_mean"] ** 2
    reference_structure = reference_sf.mean(axis=0)
    model_structure = np.concatenate(model_sf_by_seed).mean(axis=0)
    radius = np.arange(length // 2 + 1)
    k = 2.0 * np.pi * radius / length
    selected_r = np.arange(1, 17)
    correlation_nrmse = float(
        np.sqrt(np.mean((model_raw[selected_r] - reference_raw[selected_r]) ** 2))
        / np.sqrt(np.mean(reference_raw[selected_r] ** 2))
    )
    correlation_ratio = model_raw / reference_raw
    scaling = {
        "model": fit_eta(model_raw),
        "reference": fit_eta(reference_raw),
    }

    model_seed_metrics = [point_metrics(rows, length) for rows in model_arrays_by_seed]
    mode_positive = float(np.mean(model_arrays["magnetization"] > 0))
    mode_negative = float(np.mean(model_arrays["magnetization"] < 0))
    metric_keys = (
        "energy_mean",
        "abs_magnetization_mean",
        "binder_u4",
        "xi_over_l",
    )
    metric_rows = []
    for ensemble, point, boot in (
        ("MC reference", reference_metrics, reference_boot_metrics),
        ("Model pooled", model_metrics, model_boot_metrics),
    ):
        for key in metric_keys:
            low, high = confidence_interval(np.asarray(boot[key]))
            metric_rows.append(
                {
                    "ensemble": ensemble,
                    "metric": key,
                    "estimate": point[key],
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    with (tables / "final_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_rows[0].keys())
        writer.writeheader()
        writer.writerows(metric_rows)

    seed_rows = []
    for seed_name, metrics in zip(model_seed_names, model_seed_metrics, strict=True):
        seed_rows.append({"seed": seed_name} | metrics)
    with (tables / "seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=seed_rows[0].keys())
        writer.writeheader()
        writer.writerows(seed_rows)

    correlation_rows = [
        {
            "r": int(r),
            "model_raw": float(model_raw[r]),
            "reference_raw": float(reference_raw[r]),
            "model_connected": float(model_connected[r]),
            "reference_connected": float(reference_connected[r]),
            "model_to_reference": float(correlation_ratio[r]),
        }
        for r in radius
    ]
    with (tables / "correlation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=correlation_rows[0].keys())
        writer.writeheader()
        writer.writerows(correlation_rows)

    sampler_rows = load_sampler_metrics(args.sampler_dir)
    flat_sampler_rows = []
    for row in sampler_rows:
        flat_sampler_rows.append(
            {"steps": row["steps"]}
            | {f"model_{key}": row["model"][key] for key in metric_keys}
            | {f"reference_{key}": row["reference"][key] for key in metric_keys}
        )
    with (tables / "sampler_convergence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_sampler_rows[0].keys())
        writer.writeheader()
        writer.writerows(flat_sampler_rows)

    checks = {
        "spin_flip_symmetry": abs(model_metrics["magnetization_mean"]) < 0.10
        and 0.35 <= mode_positive <= 0.65,
        "energy": abs(model_metrics["energy_mean"] - reference_metrics["energy_mean"]) < 0.02,
        "abs_magnetization": abs(
            model_metrics["abs_magnetization_mean"]
            - reference_metrics["abs_magnetization_mean"]
        )
        < 0.05,
        "binder_u4": abs(model_metrics["binder_u4"] - reference_metrics["binder_u4"]) < 0.04,
        "xi_over_l": abs(model_metrics["xi_over_l"] - reference_metrics["xi_over_l"]) < 0.10,
        "correlation_nrmse_r1_16": correlation_nrmse < 0.08,
    }
    summary = {
        "configuration": {
            "lattice_size": length,
            "model_samples": int(len(model)),
            "model_sampling_seeds": model_seed_names,
            "reference_samples": int(len(reference)),
            "reference_chains": int(len(np.unique(reference_chain_ids))),
            "sampler": "ancestral",
            "reverse_steps": 128,
            "sampling_logit_temperature": 1.0,
        },
        "model": model_metrics,
        "reference": reference_metrics,
        "absolute_error": {
            key: abs(model_metrics[key] - reference_metrics[key]) for key in metric_keys
        },
        "mode_balance": {
            "positive_fraction": mode_positive,
            "negative_fraction": mode_negative,
        },
        "correlation": {
            "nrmse_r1_16": correlation_nrmse,
            "ratio_r8": float(correlation_ratio[8]),
            "ratio_r16": float(correlation_ratio[16]),
            "ratio_r32": float(correlation_ratio[32]),
            "scaling_fit_r2_16": scaling,
        },
        "checks": checks,
        "all_pilot_checks_passed": bool(all(checks.values())),
        "reference_metadata": {
            "convergence": reference_metadata["convergence"],
            "seed": reference_metadata["seed"],
        },
    }
    (args.output_dir / "final_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    apply_style()
    history = json.loads(args.history.read_text())
    steps = np.asarray([row["step"] for row in history])

    figure, axes = plt.subplots(1, 3, figsize=(15.3, 4.4))
    axes[0].plot(steps, [row["validation_nelbo"] for row in history], color=BLUE, lw=2.4)
    axes[0].axhline(np.log(2), color=GREY, ls="--", lw=1.4, label=r"$\log 2$")
    axes[0].set(xlabel="training step", ylabel="validation NELBO", title="A  Validation objective")
    axes[0].legend(fontsize=9)
    for t_value, color in zip((0.01, 0.4, 0.8, 0.95, 0.99, 1.0), (BLUE, BLUE_LIGHT, TEAL, RED, "#9A4D8E", BLACK)):
        axes[1].plot(
            steps,
            [row["validation_per_t"][str(t_value)] for row in history],
            lw=2.0,
            color=color,
            label=f"t={t_value:g}",
        )
    axes[1].set(xlabel="training step", ylabel="masked CE", title="B  Noise-level calibration")
    axes[1].legend(fontsize=8, ncol=2)
    axes[2].plot(steps, [row["learning_rate"] for row in history], color=BLACK, lw=2.2, label="learning rate")
    gradient_axis = axes[2].twinx()
    gradient_axis.plot(steps, [row["gradient_norm"] for row in history], color=BLUE, lw=1.5, alpha=0.75, label="gradient norm")
    axes[2].set(xlabel="training step", ylabel="learning rate", title="C  Optimization state")
    gradient_axis.set_ylabel("gradient norm", color=BLUE)
    axes[2].tick_params(axis="y", colors=BLUE)
    save_figure(figure, figures, "01_training_curves")

    figure, axes = plt.subplots(1, 4, figsize=(16.5, 3.9))
    sampler_labels = {
        "energy_mean": "energy",
        "abs_magnetization_mean": r"$\langle|m|\rangle$",
        "binder_u4": r"Binder $U_4$",
        "xi_over_l": r"$\xi_2/L$",
    }
    sampler_steps = [row["steps"] for row in sampler_rows]
    for axis, key in zip(axes, metric_keys, strict=True):
        axis.plot(sampler_steps, [row["model"][key] for row in sampler_rows], "o-", color=BLUE, lw=2.3, label="model")
        axis.plot(sampler_steps, [row["reference"][key] for row in sampler_rows], "--", color=BLACK, lw=1.8, label="validation MC")
        axis.axvline(128, color=RED, ls=":", lw=1.4)
        axis.set(xlabel="reverse steps", ylabel=sampler_labels[key], xticks=sampler_steps)
    axes[0].legend(fontsize=8)
    figure.suptitle("Sampler convergence on validation ensemble", y=1.02, fontsize=13)
    save_figure(figure, figures, "02_sampler_convergence")

    display_rng = np.random.default_rng(7301)
    figure, axes = plt.subplots(2, 6, figsize=(14.5, 5.2))
    for row, (samples, label) in enumerate(((model, "Model"), (reference, "MC reference"))):
        selected = display_rng.choice(len(samples), size=6, replace=False)
        for column, index in enumerate(selected):
            axes[row, column].imshow(samples[index], cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
            magnetization = float(samples[index].mean())
            axes[row, column].set_title(f"{label}\nm={magnetization:+.2f}", fontsize=9)
            axes[row, column].axis("off")
    save_figure(figure, figures, "03_random_samples")

    targets = (-0.72, -0.50, -0.20, 0.20, 0.50, 0.72)
    figure, axes = plt.subplots(2, len(targets), figsize=(14.5, 5.2))
    for row, (samples, label, magnetization) in enumerate(
        (
            (model, "Model", model_arrays["magnetization"]),
            (reference, "MC reference", reference_arrays["magnetization"]),
        )
    ):
        for column, target in enumerate(targets):
            index = int(np.argmin(np.abs(magnetization - target)))
            axes[row, column].imshow(samples[index], cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
            axes[row, column].set_title(f"{label}\nm={magnetization[index]:+.2f}", fontsize=9)
            axes[row, column].axis("off")
    save_figure(figure, figures, "04_magnetization_matched_samples")

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.1))
    histogram_specs = (
        ("energy", "energy density", 48),
        ("magnetization", "signed magnetization", 50),
        ("abs_magnetization", "absolute magnetization", 45),
    )
    for axis, (key, label, bins) in zip(axes, histogram_specs, strict=True):
        axis.hist(reference_arrays[key], bins=bins, density=True, histtype="step", color=BLACK, lw=2.2, label="MC reference")
        axis.hist(model_arrays[key], bins=bins, density=True, color=BLUE, alpha=0.32, edgecolor=BLUE, lw=0.8, label="model")
        axis.set(xlabel=label, ylabel="density")
    axes[0].legend(fontsize=9)
    figure.suptitle("Final ensemble distributions", y=1.02, fontsize=13)
    save_figure(figure, figures, "05_ensemble_distributions")

    figure, axes = plt.subplots(2, 2, figsize=(11.8, 8.6))
    r = radius[1:]
    ref_corr_low, ref_corr_high = np.quantile(reference_boot_corr[:, 1:], (0.025, 0.975), axis=0)
    mod_corr_low, mod_corr_high = np.quantile(model_boot_corr[:, 1:], (0.025, 0.975), axis=0)
    axes[0, 0].fill_between(r, ref_corr_low, ref_corr_high, color=PALE_GREY, alpha=0.7)
    axes[0, 0].fill_between(r, mod_corr_low, mod_corr_high, color=PALE_BLUE, alpha=0.75)
    axes[0, 0].loglog(r, reference_raw[1:], color=BLACK, lw=2.4, label="MC reference")
    axes[0, 0].loglog(r, model_raw[1:], color=BLUE, lw=2.4, label="model")
    theory = reference_raw[4] * (r / 4.0) ** (-0.25)
    axes[0, 0].loglog(r, theory, "--", color=TEAL, lw=1.7, label=r"$r^{-1/4}$ guide")
    axes[0, 0].axvspan(2, 16, color=PALE_GREY, alpha=0.13)
    axes[0, 0].set(xlabel="distance r", ylabel="raw C(r)", title="A  Two-point correlation")
    axes[0, 0].legend(fontsize=8)
    paired_ratio = model_boot_corr[:, 1:] / reference_boot_corr[:, 1:]
    ratio_low, ratio_high = np.quantile(paired_ratio, (0.025, 0.975), axis=0)
    axes[0, 1].fill_between(r, ratio_low, ratio_high, color=PALE_BLUE, alpha=0.8)
    axes[0, 1].plot(r, correlation_ratio[1:], color=BLUE, lw=2.4)
    axes[0, 1].axhline(1.0, color=BLACK, ls="--", lw=1.5)
    axes[0, 1].axvline(16, color=GREY, ls=":", lw=1.3)
    axes[0, 1].set(xlabel="distance r", ylabel=r"$C_{model}/C_{MC}$", title="B  Correlation fidelity", xlim=(1, 32))
    valid_k = radius > 0
    ref_sf_low, ref_sf_high = np.quantile(reference_boot_sf[:, valid_k], (0.025, 0.975), axis=0)
    mod_sf_low, mod_sf_high = np.quantile(model_boot_sf[:, valid_k], (0.025, 0.975), axis=0)
    axes[1, 0].fill_between(k[valid_k], ref_sf_low, ref_sf_high, color=PALE_GREY, alpha=0.7)
    axes[1, 0].fill_between(k[valid_k], mod_sf_low, mod_sf_high, color=PALE_BLUE, alpha=0.75)
    axes[1, 0].loglog(k[valid_k], reference_structure[valid_k], color=BLACK, lw=2.4, label="MC reference")
    axes[1, 0].loglog(k[valid_k], model_structure[valid_k], color=BLUE, lw=2.4, label="model")
    axes[1, 0].set(xlabel=r"wave number $|k|$", ylabel=r"$S(k)$", title="C  Structure factor")
    axes[1, 0].legend(fontsize=8)
    structure_ratio = model_structure[valid_k] / reference_structure[valid_k]
    sf_ratio_boot = model_boot_sf[:, valid_k] / reference_boot_sf[:, valid_k]
    sf_ratio_low, sf_ratio_high = np.quantile(sf_ratio_boot, (0.025, 0.975), axis=0)
    axes[1, 1].fill_between(k[valid_k], sf_ratio_low, sf_ratio_high, color=PALE_BLUE, alpha=0.8)
    axes[1, 1].semilogx(k[valid_k], structure_ratio, color=BLUE, lw=2.4)
    axes[1, 1].axhline(1.0, color=BLACK, ls="--", lw=1.5)
    axes[1, 1].set(xlabel=r"wave number $|k|$", ylabel=r"$S_{model}/S_{MC}$", title="D  Spectral fidelity")
    save_figure(figure, figures, "06_correlation_and_structure")

    figure, axes = plt.subplots(1, 4, figsize=(15.5, 3.9))
    display_labels = {
        "energy_mean": "energy",
        "abs_magnetization_mean": r"$\langle|m|\rangle$",
        "binder_u4": r"Binder $U_4$",
        "xi_over_l": r"$\xi_2/L$",
    }
    for axis, key in zip(axes, metric_keys, strict=True):
        points = (reference_metrics[key], model_metrics[key])
        intervals = (
            confidence_interval(np.asarray(reference_boot_metrics[key])),
            confidence_interval(np.asarray(model_boot_metrics[key])),
        )
        errors = np.asarray(
            [
                [points[i] - intervals[i][0] for i in range(2)],
                [intervals[i][1] - points[i] for i in range(2)],
            ]
        )
        axis.errorbar((0, 1), points, yerr=errors, fmt="o", color=BLACK, ecolor=GREY, capsize=4, ms=7)
        axis.scatter((1,), (points[1],), color=BLUE, s=55, zorder=3)
        axis.set(xticks=(0, 1), xticklabels=("MC", "model"), ylabel=display_labels[key])
        margin = max(abs(points[1] - points[0]) * 2.2, max(errors.flatten()) * 1.8, 1e-3)
        center = 0.5 * (points[0] + points[1])
        axis.set_ylim(center - margin, center + margin)
    figure.suptitle("Dimensionless and thermodynamic observables (95% CI)", y=1.02, fontsize=13)
    save_figure(figure, figures, "07_scalar_observables")

    figure, axes = plt.subplots(1, 4, figsize=(15.5, 3.9))
    for axis, key in zip(axes, metric_keys, strict=True):
        for index, metrics in enumerate(model_seed_metrics):
            axis.scatter(index, metrics[key], color=BLUE_LIGHT, s=55, zorder=3)
        axis.axhline(reference_metrics[key], color=BLACK, ls="--", lw=1.8, label="MC reference")
        axis.scatter(3.3, model_metrics[key], color=BLUE, marker="D", s=55, label="pooled model")
        axis.set(xticks=(0, 1, 2, 3.3), xticklabels=(*model_seed_names, "pooled"), ylabel=display_labels[key])
        axis.tick_params(axis="x", rotation=25)
    axes[0].legend(fontsize=8)
    figure.suptitle("Sampling-seed stability", y=1.02, fontsize=13)
    save_figure(figure, figures, "08_seed_stability")

    pass_text = "PASS" if summary["all_pilot_checks_passed"] else "PARTIAL"
    report = f"""# L=64 离散扩散最终实验报告

## 结论

总体判定：**{pass_text}**。

- 模型样本：{len(model):,} 张，3 个冻结 sampling seed；MC reference：{len(reference):,} 张、8 条独立链。
- sampler：ancestral，128 reverse steps，sampling logit temperature=1.0。
- 能量：model {model_metrics['energy_mean']:.6f}，MC {reference_metrics['energy_mean']:.6f}。
- `<|m|>`：model {model_metrics['abs_magnetization_mean']:.6f}，MC {reference_metrics['abs_magnetization_mean']:.6f}。
- Binder U4：model {model_metrics['binder_u4']:.6f}，MC {reference_metrics['binder_u4']:.6f}。
- xi/L：model {model_metrics['xi_over_l']:.6f}，MC {reference_metrics['xi_over_l']:.6f}。
- `G(r)` normalized RMSE（r=1..16）：{correlation_nrmse:.4%}。
- `G_model/G_MC`：r=8/16/32 为 {correlation_ratio[8]:.4f}/{correlation_ratio[16]:.4f}/{correlation_ratio[32]:.4f}。
- scaling eta（r=2..16）：model {scaling['model']['eta']:.4f}，MC {scaling['reference']['eta']:.4f}。
- 正/负磁化比例：{mode_positive:.3f}/{mode_negative:.3f}。

## 自动判据

""" + "\n".join(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`" for name, passed in checks.items()
    ) + """

## 图表

1. `figures/01_training_curves.png`：训练与分噪声层验证。
2. `figures/02_sampler_convergence.png`：24/48/96/128步敏感性。
3. `figures/03_random_samples.png`：固定随机索引样本网格。
4. `figures/04_magnetization_matched_samples.png`：同磁化宏观态比较。
5. `figures/05_ensemble_distributions.png`：能量与磁化分布。
6. `figures/06_correlation_and_structure.png`：G(r)、S(k)及相对误差。
7. `figures/07_scalar_observables.png`：关键标量与95% CI。
8. `figures/08_seed_stability.png`：三个sampling seed稳定性。
"""
    (args.output_dir / "FINAL_REPORT_ZH.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
