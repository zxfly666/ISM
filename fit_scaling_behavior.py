"""Fit and plot finite-size scaling of L=64 Ising two-point correlations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "MC": "#272727",
    "step 15k": "#B64342",
    "step 30k": "#0F4D92",
    "theory": "#42949E",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step15", type=Path, required=True)
    parser.add_argument("--step30", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def sample_radial_raw(samples: np.ndarray) -> np.ndarray:
    spins = np.asarray(samples, dtype=np.float64)
    if spins.ndim != 3 or spins.shape[-1] != spins.shape[-2]:
        raise ValueError("samples must have shape [N, L, L]")
    length = spins.shape[-1]
    spectrum = np.fft.fft2(spins, axes=(-2, -1))
    correlation = np.fft.ifft2(
        spectrum * spectrum.conj(), axes=(-2, -1)
    ).real / (length * length)
    coordinate = np.minimum(np.arange(length), length - np.arange(length))
    bins = np.rint(
        np.sqrt(coordinate[:, None] ** 2 + coordinate[None, :] ** 2)
    ).astype(np.int32)
    return np.stack(
        [
            correlation[:, bins == radius].mean(axis=1)
            for radius in range(length // 2 + 1)
        ],
        axis=1,
    )


def fit_eta(
    curve: np.ndarray,
    r_min: int,
    r_max: int,
    length: int,
    chord_distance: bool = False,
) -> dict[str, float]:
    radius = np.arange(r_min, r_max + 1, dtype=np.float64)
    selected = np.asarray(curve, dtype=np.float64)[r_min : r_max + 1]
    if chord_distance:
        distance = length / np.pi * np.sin(np.pi * radius / length)
    else:
        distance = radius
    if np.any(selected <= 0):
        raise ValueError("correlation must be positive in the fit window")
    x = np.log(distance)
    y = np.log(selected)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "eta": float(-slope),
        "amplitude": float(np.exp(intercept)),
        "r_squared": float(1.0 - residual / total) if total > 0 else 1.0,
    }


def bootstrap_curves(
    per_sample: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    rows = len(per_sample)
    indices = rng.integers(0, rows, size=(repeats, rows))
    return per_sample[indices].mean(axis=1)


def bootstrap_fit(
    curves: np.ndarray,
    r_min: int,
    r_max: int,
    length: int,
    chord_distance: bool,
) -> dict[str, float]:
    values = np.asarray(
        [
            fit_eta(curve, r_min, r_max, length, chord_distance)["eta"]
            for curve in curves
        ]
    )
    return {
        "eta_bootstrap_mean": float(values.mean()),
        "eta_bootstrap_se": float(values.std(ddof=1)),
        "eta_ci95_low": float(np.quantile(values, 0.025)),
        "eta_ci95_high": float(np.quantile(values, 0.975)),
    }


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.linewidth": 1.6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.step15, allow_pickle=False) as payload:
        model_15 = np.asarray(payload["model"])
    with np.load(args.step30, allow_pickle=False) as payload:
        model_30 = np.asarray(payload["model"])
        reference = np.asarray(payload["reference"])
    ensembles = {
        "MC": reference,
        "step 15k": model_15,
        "step 30k": model_30,
    }
    length = int(reference.shape[-1])
    rng = np.random.default_rng(args.seed)
    per_sample = {
        name: sample_radial_raw(samples) for name, samples in ensembles.items()
    }
    curves = {name: values.mean(axis=0) for name, values in per_sample.items()}
    boot = {
        name: bootstrap_curves(values, args.bootstrap, rng)
        for name, values in per_sample.items()
    }

    windows = ((2, 8), (2, 12), (2, 16), (4, 16), (4, 24))
    fit_report: dict[str, dict] = {}
    for name, curve in curves.items():
        fit_report[name] = {}
        for r_min, r_max in windows:
            key = f"r{r_min}_{r_max}"
            ordinary = fit_eta(curve, r_min, r_max, length, False)
            ordinary.update(
                bootstrap_fit(boot[name], r_min, r_max, length, False)
            )
            chord = fit_eta(curve, r_min, r_max, length, True)
            chord.update(
                bootstrap_fit(boot[name], r_min, r_max, length, True)
            )
            fit_report[name][key] = {
                "radial_power_law": ordinary,
                "chord_distance_power_law": chord,
            }

    ratio_report: dict[str, dict[str, float]] = {}
    for name in ("step 15k", "step 30k"):
        ratio = curves[name] / curves["MC"]
        ratio_report[name] = {
            str(radius): float(ratio[radius])
            for radius in (1, 2, 4, 8, 16, 24, 32)
        }

    report = {
        "lattice_size": length,
        "samples_per_ensemble": {
            name: int(len(values)) for name, values in ensembles.items()
        },
        "estimator": (
            "Raw periodic C(r)=<s_i s_{i+r}>, radial bins rounded to nearest "
            "integer. Ordinary least squares is performed in log-log space."
        ),
        "warning": (
            "The 128-sample MC subset is a provisional finite-size target. "
            "Final uncertainties require the independent >=10k reference ensemble."
        ),
        "fits": fit_report,
        "model_to_mc_ratio": ratio_report,
    }
    (args.output_dir / "scaling_fit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    apply_style()
    figure, axes = plt.subplots(1, 3, figsize=(15.4, 4.5))
    radius = np.arange(1, length // 2 + 1)

    axis = axes[0]
    for name in ("MC", "step 15k", "step 30k"):
        low, high = np.quantile(boot[name][:, 1:], (0.025, 0.975), axis=0)
        axis.fill_between(radius, low, high, color=COLORS[name], alpha=0.12)
        axis.loglog(
            radius,
            curves[name][1:],
            color=COLORS[name],
            linewidth=2.3,
            label=name,
        )
    theory = curves["MC"][4] * (radius / 4.0) ** (-0.25)
    axis.loglog(
        radius,
        theory,
        "--",
        color=COLORS["theory"],
        linewidth=1.8,
        label=r"$r^{-1/4}$ guide",
    )
    axis.axvspan(2, 16, color="#CFCECE", alpha=0.16, zorder=0)
    axis.set(xlabel=r"distance $r$", ylabel=r"raw $C(r)$", title="A  Correlation scaling")
    axis.legend(fontsize=9)

    axis = axes[1]
    for name in ("step 15k", "step 30k"):
        ratio = curves[name][1:] / curves["MC"][1:]
        ratio_boot = boot[name][:, 1:] / boot["MC"][:, 1:]
        low, high = np.quantile(ratio_boot, (0.025, 0.975), axis=0)
        axis.fill_between(radius, low, high, color=COLORS[name], alpha=0.12)
        axis.plot(radius, ratio, color=COLORS[name], linewidth=2.3, label=name)
    axis.axhline(1.0, color=COLORS["MC"], linestyle="--", linewidth=1.5)
    axis.axvline(16, color="#767676", linestyle=":", linewidth=1.3)
    axis.set(
        xlabel=r"distance $r$",
        ylabel=r"$C_{model}(r)/C_{MC}(r)$",
        title="B  Finite-size fidelity",
        xlim=(1, 32),
        ylim=(0.82, 1.03),
    )
    axis.legend(fontsize=9)

    axis = axes[2]
    r_max_values = np.asarray((8, 12, 16, 20, 24))
    for name in ("MC", "step 15k", "step 30k"):
        eta = [
            fit_eta(curves[name], 2, int(r_max), length, False)["eta"]
            for r_max in r_max_values
        ]
        axis.plot(
            r_max_values,
            eta,
            marker="o",
            markersize=5,
            color=COLORS[name],
            linewidth=2.2,
            label=name,
        )
    axis.axhline(0.25, color=COLORS["theory"], linestyle="--", linewidth=1.8)
    axis.text(23.6, 0.252, r"theory $\eta=1/4$", ha="right", va="bottom", fontsize=9)
    axis.set(
        xlabel=r"fit-window endpoint $r_{max}$",
        ylabel=r"effective $\hat{\eta}$",
        title=r"C  Window sensitivity ($r_{min}=2$)",
        xticks=r_max_values,
        ylim=(0.18, 0.265),
    )

    figure.tight_layout(pad=1.6)
    figure.savefig(
        args.output_dir / "scaling_behavior.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        args.output_dir / "scaling_behavior.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)

    summary = {
        name: fit_report[name]["r2_16"]["radial_power_law"]
        for name in fit_report
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
