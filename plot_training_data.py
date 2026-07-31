"""Visual and numerical audit for chain-split Ising training data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ism_diffusion.diagnostics import (
    autocorrelation_fft,
    integrated_autocorrelation_time,
    local_gibbs_calibration,
    split_rhat,
)
from ism_diffusion.ising import BETA_CRITICAL, energy_density, magnetization
from ism_diffusion.metrics import (
    ensemble_summary,
    radial_average_periodic,
    structure_factor,
)


BLUE = "#0072B2"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREY = "#666666"
CHAIN_COLOURS = (BLUE, GREEN, RED, PURPLE, ORANGE, SKY, "#332288", "#88CCEE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-lag", type=int, default=48)
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def save_figure(figure: plt.Figure, base: Path) -> None:
    figure.tight_layout()
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def load_dataset(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "train",
            "val",
            "test",
            "train_chain_id",
            "val_chain_id",
            "test_chain_id",
            "metadata",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"dataset is missing keys: {sorted(missing)}")
        arrays = {key: np.asarray(payload[key]) for key in required - {"metadata"}}
        metadata = json.loads(str(payload["metadata"].item()))
    return arrays, metadata


def plot_sample_grid(
    train: np.ndarray, chain_ids: np.ndarray, output_dir: Path
) -> None:
    unique_chains = np.unique(chain_ids)
    shown_chains = unique_chains[: min(4, len(unique_chains))]
    figure, axes = plt.subplots(
        4,
        len(shown_chains),
        figsize=(2.4 * len(shown_chains), 8.4),
        squeeze=False,
    )
    for column, chain_id in enumerate(shown_chains):
        indices = np.flatnonzero(chain_ids == chain_id)
        selected = indices[
            np.rint(np.linspace(0, len(indices) - 1, 4)).astype(np.int64)
        ]
        for row, sample_index in enumerate(selected):
            axis = axes[row, column]
            axis.imshow(
                train[sample_index],
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                interpolation="nearest",
            )
            axis.set_title(
                f"chain {int(chain_id)}, saved {int(sample_index - indices[0])}"
            )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
    figure.suptitle(
        "Critical Ising training samples (Wolff, periodic boundary)",
        y=1.005,
        fontsize=13,
    )
    save_figure(figure, output_dir / "training_samples")


def chain_series(
    samples: np.ndarray, chain_ids: np.ndarray
) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    ids = [int(value) for value in np.unique(chain_ids)]
    energy_chains = []
    abs_m_chains = []
    for chain_id in ids:
        selected = samples[chain_ids == chain_id]
        energy_chains.append(energy_density(selected))
        abs_m_chains.append(np.abs(magnetization(selected)))
    return ids, energy_chains, abs_m_chains


def plot_chain_diagnostics(
    samples: np.ndarray,
    chain_ids: np.ndarray,
    max_lag: int,
    output_dir: Path,
) -> tuple[list[dict], dict[str, float]]:
    ids, energy_chains, abs_m_chains = chain_series(samples, chain_ids)
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.2))
    for index, (chain_id, energy, abs_m) in enumerate(
        zip(ids, energy_chains, abs_m_chains, strict=True)
    ):
        colour = CHAIN_COLOURS[index % len(CHAIN_COLOURS)]
        axes[0, 0].plot(
            energy,
            color=colour,
            alpha=0.72,
            linewidth=0.75,
            label=f"chain {chain_id}",
        )
        axes[0, 1].plot(abs_m, color=colour, alpha=0.72, linewidth=0.75)
        energy_acf = autocorrelation_fft(energy)
        mag_acf = autocorrelation_fft(abs_m)
        last_energy = min(max_lag, len(energy_acf) - 1)
        last_mag = min(max_lag, len(mag_acf) - 1)
        axes[1, 0].plot(
            np.arange(last_energy + 1),
            energy_acf[: last_energy + 1],
            color=colour,
            alpha=0.8,
            linewidth=1.1,
        )
        axes[1, 1].plot(
            np.arange(last_mag + 1),
            mag_acf[: last_mag + 1],
            color=colour,
            alpha=0.8,
            linewidth=1.1,
        )

    axes[0, 0].axhline(
        -math.sqrt(2.0),
        color=GREY,
        linestyle="--",
        linewidth=1.2,
        label=r"$-\sqrt{2}$ (infinite-$L$)",
    )
    axes[0, 0].set_title("Energy trace after burn-in")
    axes[0, 0].set_xlabel("saved configuration")
    axes[0, 0].set_ylabel("energy per spin")
    axes[0, 0].legend(ncol=3, frameon=False)
    axes[0, 1].set_title(r"$|m|$ trace after burn-in")
    axes[0, 1].set_xlabel("saved configuration")
    axes[0, 1].set_ylabel(r"$|m|$")
    axes[1, 0].set_title("Energy autocorrelation")
    axes[1, 0].set_xlabel("lag (saved configurations)")
    axes[1, 0].set_ylabel("ACF")
    axes[1, 1].set_title(r"$|m|$ autocorrelation")
    axes[1, 1].set_xlabel("lag (saved configurations)")
    axes[1, 1].set_ylabel("ACF")
    for axis in axes[1]:
        axis.axhline(0.0, color="#BBBBBB", linewidth=0.8)
    save_figure(figure, output_dir / "chain_diagnostics")

    rows = []
    for chain_id, energy, abs_m in zip(
        ids, energy_chains, abs_m_chains, strict=True
    ):
        energy_diag = integrated_autocorrelation_time(energy)
        mag_diag = integrated_autocorrelation_time(abs_m)
        rows.append(
            {
                "chain_id": chain_id,
                "samples": int(len(energy)),
                "energy_mean": float(np.mean(energy)),
                "abs_magnetization_mean": float(np.mean(abs_m)),
                "energy_tau_int": energy_diag["tau_int"],
                "energy_ess": energy_diag["ess"],
                "abs_magnetization_tau_int": mag_diag["tau_int"],
                "abs_magnetization_ess": mag_diag["ess"],
            }
        )
    convergence = {
        "energy_split_rhat": split_rhat(energy_chains),
        "abs_magnetization_split_rhat": split_rhat(abs_m_chains),
        "minimum_chain_ess": float(
            min(
                min(row["energy_ess"], row["abs_magnetization_ess"])
                for row in rows
            )
        ),
    }
    return rows, convergence


def raw_correlation_map(samples: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(structure_factor(samples)).real


def plot_physics_sanity(
    samples: np.ndarray,
    beta: float,
    output_dir: Path,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    energy = energy_density(samples)
    magnetization_values = magnetization(samples)
    calibration = local_gibbs_calibration(samples, beta)
    correlation = raw_correlation_map(samples)
    radii, radial = radial_average_periodic(correlation)

    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    axes[0, 0].hist(
        energy,
        bins=34,
        density=True,
        color=BLUE,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.35,
    )
    axes[0, 0].axvline(
        -math.sqrt(2.0),
        color=GREY,
        linestyle="--",
        linewidth=1.4,
        label=r"$-\sqrt{2}$ (infinite-$L$)",
    )
    axes[0, 0].axvline(
        float(np.mean(energy)),
        color=RED,
        linewidth=1.4,
        label=f"sample mean {np.mean(energy):.4f}",
    )
    axes[0, 0].set_title("Energy distribution")
    axes[0, 0].set_xlabel("energy per spin")
    axes[0, 0].set_ylabel("density")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].hist(
        magnetization_values,
        bins=41,
        density=True,
        color=GREEN,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.35,
    )
    axes[0, 1].axvline(0.0, color=GREY, linestyle="--", linewidth=1.0)
    axes[0, 1].set_title("Signed magnetization (spin-flip symmetry)")
    axes[0, 1].set_xlabel("m")
    axes[0, 1].set_ylabel("density")

    valid = (radii >= 1) & np.isfinite(radial) & (radial > 0)
    axes[1, 0].loglog(
        radii[valid],
        radial[valid],
        "o-",
        color=BLUE,
        markersize=3.5,
        linewidth=1.2,
        label="sampled raw G(r)",
    )
    valid_r = radii[valid]
    if len(valid_r):
        anchor_index = min(1, len(valid_r) - 1)
        anchor_r = valid_r[anchor_index]
        anchor_g = radial[valid][anchor_index]
        theory = anchor_g * (valid_r / anchor_r) ** (-0.25)
        axes[1, 0].loglog(
            valid_r,
            theory,
            "--",
            color=GREY,
            linewidth=1.2,
            label=r"critical guide $\propto r^{-1/4}$",
        )
    axes[1, 0].set_title("Periodic two-point correlation")
    axes[1, 0].set_xlabel("periodic distance r")
    axes[1, 0].set_ylabel("G(r)")
    axes[1, 0].legend(frameon=False)

    neighbours = np.asarray([row["neighbour_sum"] for row in calibration])
    empirical = np.asarray([row["empirical_p_plus"] for row in calibration])
    exact = np.asarray([row["exact_p_plus"] for row in calibration])
    error = np.asarray([row["binomial_se"] for row in calibration])
    axes[1, 1].errorbar(
        neighbours,
        empirical,
        yerr=2.0 * error,
        fmt="o",
        color=BLUE,
        capsize=3,
        label=r"empirical $\pm 2$ binomial SE",
    )
    axes[1, 1].plot(
        neighbours,
        exact,
        "s--",
        color=RED,
        linewidth=1.2,
        label="exact Ising conditional",
    )
    axes[1, 1].set_ylim(-0.03, 1.03)
    axes[1, 1].set_xticks(neighbours)
    axes[1, 1].set_title("Local Gibbs calibration")
    axes[1, 1].set_xlabel("sum of four neighbours")
    axes[1, 1].set_ylabel(r"$P(s_i=+1\mid\mathrm{neighbours})$")
    axes[1, 1].legend(frameon=False)
    save_figure(figure, output_dir / "physics_sanity")

    identities = {
        "g_zero_error": float(abs(correlation[0, 0] - 1.0)),
        "energy_correlation_error": float(
            abs(
                np.mean(energy)
                + correlation[1, 0]
                + correlation[0, 1]
            )
        ),
        "structure_sum_rule_error": float(
            abs(np.sum(structure_factor(samples)) / samples.shape[-1] ** 2 - 1.0)
        ),
        "max_local_gibbs_error": float(
            max(row["absolute_error"] for row in calibration)
        ),
    }
    return calibration, identities


def plot_split_overview(
    arrays: dict[str, np.ndarray], output_dir: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    for split, colour in zip(
        ("train", "val", "test"), (BLUE, GREEN, RED), strict=True
    ):
        samples = arrays[split]
        axes[0].hist(
            energy_density(samples),
            bins=32,
            density=True,
            histtype="step",
            linewidth=1.5,
            color=colour,
            label=split,
        )
        axes[1].hist(
            np.abs(magnetization(samples)),
            bins=32,
            density=True,
            histtype="step",
            linewidth=1.5,
            color=colour,
            label=split,
        )
    axes[0].set_title("Split consistency: energy")
    axes[0].set_xlabel("energy per spin")
    axes[0].set_ylabel("density")
    axes[1].set_title(r"Split consistency: $|m|$")
    axes[1].set_xlabel(r"$|m|$")
    axes[1].set_ylabel("density")
    for axis in axes:
        axis.legend(frameon=False)
    save_figure(figure, output_dir / "split_overview")


def main() -> None:
    args = parse_args()
    apply_style()
    arrays, metadata = load_dataset(args.data)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    chain_sets = {
        split: set(int(value) for value in np.unique(arrays[f"{split}_chain_id"]))
        for split in ("train", "val", "test")
    }
    overlaps = {
        "train_val": sorted(chain_sets["train"] & chain_sets["val"]),
        "train_test": sorted(chain_sets["train"] & chain_sets["test"]),
        "val_test": sorted(chain_sets["val"] & chain_sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"chain leakage detected: {overlaps}")

    plot_sample_grid(
        arrays["train"], arrays["train_chain_id"], output_dir
    )
    chain_rows, convergence = plot_chain_diagnostics(
        arrays["train"],
        arrays["train_chain_id"],
        max_lag=args.max_lag,
        output_dir=output_dir,
    )
    beta = float(metadata.get("beta", BETA_CRITICAL))
    calibration, identities = plot_physics_sanity(
        arrays["train"], beta=beta, output_dir=output_dir
    )
    plot_split_overview(arrays, output_dir)

    summaries = {
        split: ensemble_summary(arrays[split])
        for split in ("train", "val", "test")
    }
    train_summary = summaries["train"]
    checks = {
        "chain_splits_disjoint": not any(overlaps.values()),
        "binary_spins_only": bool(
            all(np.isin(arrays[split], (-1, 1)).all() for split in ("train", "val", "test"))
        ),
        "g_zero_identity": identities["g_zero_error"] < 1e-10,
        "energy_correlation_identity": identities["energy_correlation_error"] < 1e-10,
        "structure_sum_rule": identities["structure_sum_rule_error"] < 1e-10,
        "critical_energy_sanity": abs(train_summary["energy_mean"] + math.sqrt(2.0)) <= 0.08,
        "spin_flip_symmetry": abs(train_summary["magnetization_mean"]) <= 0.10,
        "binder_sanity": abs(train_summary["binder_u4"] - 0.6106901) <= 0.10,
        "xi_over_l_sanity": abs(train_summary["xi_over_l"] - 0.9050488) <= 0.20,
        "split_rhat": max(
            convergence["energy_split_rhat"],
            convergence["abs_magnetization_split_rhat"],
        )
        <= 1.05,
        "minimum_chain_ess": convergence["minimum_chain_ess"] >= 30.0,
        "local_gibbs_calibration": identities["max_local_gibbs_error"] <= 0.05,
    }
    report = {
        "data_path": str(args.data.resolve()),
        "metadata": metadata,
        "chain_sets": {key: sorted(value) for key, value in chain_sets.items()},
        "chain_overlaps": overlaps,
        "split_summaries": summaries,
        "chain_diagnostics": chain_rows,
        "convergence": convergence,
        "local_gibbs_calibration": calibration,
        "exact_identity_errors": identities,
        "checks": checks,
        "all_checks_passed": bool(all(checks.values())),
    }
    (output_dir / "data_diagnostics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    check_lines = "\n".join(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in checks.items()
    )
    markdown = f"""# L={metadata['lattice_size']} 训练数据采样审计

- 数据：`{args.data.resolve()}`
- 临界逆温：{beta:.8f}
- 训练/验证/测试链：{len(chain_sets['train'])}/{len(chain_sets['val'])}/{len(chain_sets['test'])}
- 每个 split 样本数：{len(arrays['train'])}/{len(arrays['val'])}/{len(arrays['test'])}
- 训练集 `<e>`：{train_summary['energy_mean']:.6f}
- 训练集 `<|m|>`：{train_summary['abs_magnetization_mean']:.6f}
- 训练集 Binder U4：{train_summary['binder_u4']:.6f}
- 训练集 xi/L：{train_summary['xi_over_l']:.6f}
- energy / |m| split-R-hat：{convergence['energy_split_rhat']:.4f} / {convergence['abs_magnetization_split_rhat']:.4f}
- 最小单链 ESS：{convergence['minimum_chain_ess']:.1f}

## 自动检查

{check_lines}

这些检查借鉴了同学项目的链级审计和局域物理校准，但这里针对完整的周期 L×L 格点计算，没有沿用其 open-window crop 假设。
"""
    (output_dir / "DATA_REPORT_ZH.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
