"""Diagnose equilibration and critical physics of full periodic L=256 samples."""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from ism_diffusion.diagnostics import (
    autocorrelation_fft,
    integrated_autocorrelation_time,
    split_rhat,
)
from ism_diffusion.ising import BETA_CRITICAL, energy_density, magnetization
from ism_diffusion.ising_numba import simulate_wolff_chain, warm_up_numba
from ism_diffusion.metrics import (
    ensemble_summary,
    radial_average_periodic,
    structure_factor,
)


BLUE = "#0F4D92"
BLUE_SECONDARY = "#3775BA"
GREEN = "#009E73"
ORANGE = "#E69F00"
RED = "#B64342"
PURPLE = "#9A4D8E"
GREY = "#767676"
CONDITION_COLOURS = (RED, ORANGE, GREEN, BLUE)
SPIN_CMAP = ListedColormap(["#3B4CC0", "#B40426"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lattice-size", type=int, default=256)
    parser.add_argument("--burn-in-sweeps", default="1,10,50,500")
    parser.add_argument("--chains-per-condition", type=int, default=4)
    parser.add_argument("--samples-per-chain", type=int, default=128)
    parser.add_argument("--sweeps-between", type=float, default=5.0)
    parser.add_argument("--pilot-cluster-steps", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def save_figure(figure: plt.Figure, base: Path) -> None:
    figure.tight_layout(pad=1.5)
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _chain_worker(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    samples, metadata = simulate_wolff_chain(
        length=int(task["lattice_size"]),
        beta=float(BETA_CRITICAL),
        burn_in_sweeps=float(task["burn_in_sweeps"]),
        sweeps_between_samples=float(task["sweeps_between"]),
        n_samples=int(task["samples_per_chain"]),
        seed=int(task["seed"]),
        adaptation_sweeps=3.0,
        pilot_cluster_steps=int(task["pilot_cluster_steps"]),
        initial_state=str(task["initial_state"]),
    )
    return {
        **task,
        "samples": samples,
        "sampler_metadata": metadata,
        "elapsed_seconds": time.perf_counter() - started,
    }


def fit_eta(radii: np.ndarray, correlation: np.ndarray) -> dict[str, float]:
    valid = (
        (radii >= 4)
        & (radii <= 32)
        & np.isfinite(correlation)
        & (correlation > 0)
    )
    x = np.log(radii[valid])
    y = np.log(correlation[valid])
    if len(x) < 3:
        return {"eta_hat": float("nan"), "eta_r_squared": float("nan")}
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "eta_hat": float(-slope),
        "eta_r_squared": float(1.0 - residual / total) if total else float("nan"),
    }


def summarize_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = np.concatenate([row["samples"] for row in rows])
    energies = [energy_density(row["samples"]) for row in rows]
    signed_magnetizations = [magnetization(row["samples"]) for row in rows]
    absolute_magnetizations = [np.abs(values) for values in signed_magnetizations]
    raw_correlation = np.fft.ifft2(structure_factor(samples)).real
    radii, radial_correlation = radial_average_periodic(raw_correlation)
    eta = fit_eta(radii, radial_correlation)

    chain_rows = []
    for row, energy, signed_m, abs_m in zip(
        rows,
        energies,
        signed_magnetizations,
        absolute_magnetizations,
        strict=True,
    ):
        energy_diag = integrated_autocorrelation_time(energy)
        magnetization_diag = integrated_autocorrelation_time(abs_m)
        chain_rows.append(
            {
                "chain_index": int(row["chain_index"]),
                "initial_state": row["initial_state"],
                "seed": int(row["seed"]),
                "energy_mean": float(np.mean(energy)),
                "magnetization_mean": float(np.mean(signed_m)),
                "abs_magnetization_mean": float(np.mean(abs_m)),
                "energy_ess": energy_diag["ess"],
                "abs_magnetization_ess": magnetization_diag["ess"],
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "sampler_metadata": row["sampler_metadata"],
            }
        )

    convergence = {
        "energy_split_rhat": split_rhat(energies),
        "abs_magnetization_split_rhat": split_rhat(absolute_magnetizations),
        "minimum_chain_ess": float(
            min(
                min(row["energy_ess"], row["abs_magnetization_ess"])
                for row in chain_rows
            )
        ),
    }
    return {
        "samples": samples,
        "summary": ensemble_summary(samples),
        "chain_rows": chain_rows,
        "convergence": convergence,
        "radii": radii,
        "radial_correlation": radial_correlation,
        **eta,
    }


def exact_l4_validation(seed: int) -> dict[str, Any]:
    length = 4
    n_sites = length * length
    integers = np.arange(1 << n_sites, dtype=np.uint32)
    bits = (
        (integers[:, None] >> np.arange(n_sites, dtype=np.uint32)) & 1
    ).astype(np.int8)
    exact_samples = (2 * bits - 1).reshape(-1, length, length)
    exact_energy = energy_density(exact_samples)
    exact_magnetization = magnetization(exact_samples)
    log_weight = -float(BETA_CRITICAL) * exact_energy * n_sites
    weight = np.exp(log_weight - np.max(log_weight))
    weight /= np.sum(weight)
    exact_m2 = float(np.sum(weight * exact_magnetization**2))
    exact_m4 = float(np.sum(weight * exact_magnetization**4))
    exact = {
        "energy_mean": float(np.sum(weight * exact_energy)),
        "abs_magnetization_mean": float(
            np.sum(weight * np.abs(exact_magnetization))
        ),
        "binder_u4": float(1.0 - exact_m4 / (3.0 * exact_m2**2)),
    }

    mc_samples, _ = simulate_wolff_chain(
        length=4,
        beta=float(BETA_CRITICAL),
        burn_in_sweeps=100.0,
        sweeps_between_samples=2.0,
        n_samples=50_000,
        seed=seed,
        adaptation_sweeps=3.0,
        pilot_cluster_steps=64,
        initial_state="random",
    )
    mc = ensemble_summary(mc_samples)
    errors = {
        key: abs(float(mc[key]) - float(exact[key]))
        for key in exact
    }
    return {"exact": exact, "mc": mc, "absolute_error": errors}


def plot_burnin_sensitivity(
    burn_values: list[float],
    conditions: dict[float, dict[str, Any]],
    output_dir: Path,
) -> None:
    metrics = [
        ("energy_mean", "Energy per spin", -math.sqrt(2.0)),
        ("abs_magnetization_mean", r"$\langle |m|\rangle$", None),
        ("binder_u4", r"Binder $U_4$", 0.6106901),
        ("xi_over_l", r"$\xi_2/L$", 0.9050488),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(14.0, 8.0))
    x = np.asarray(burn_values, dtype=np.float64)
    for axis, (key, title, reference) in zip(
        axes.flat[:4], metrics, strict=True
    ):
        values = np.asarray(
            [conditions[burn]["summary"][key] for burn in burn_values]
        )
        for index, burn in enumerate(burn_values):
            chain_values = [
                row[key]
                for row in conditions[burn]["chain_rows"]
                if key in row
            ]
            if chain_values:
                axis.scatter(
                    np.full(len(chain_values), burn),
                    chain_values,
                    s=22,
                    color=CONDITION_COLOURS[index],
                    alpha=0.45,
                )
        axis.plot(x, values, "o-", color=BLUE, linewidth=1.8)
        if reference is not None:
            axis.axhline(
                reference,
                color=GREY,
                linestyle="--",
                linewidth=1.2,
                label="critical reference",
            )
            axis.legend()
        axis.set_xscale("log")
        axis.set_title(title)
        axis.set_xlabel("declared burn-in sweeps")

    eta = [conditions[burn]["eta_hat"] for burn in burn_values]
    axes[1, 1].plot(x, eta, "o-", color=BLUE, linewidth=1.8)
    axes[1, 1].axhline(0.25, color=GREY, linestyle="--", linewidth=1.2)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title(r"Correlation exponent $\hat{\eta}$, $4\leq r\leq32$")
    axes[1, 1].set_xlabel("declared burn-in sweeps")

    minimum_ess = [
        conditions[burn]["convergence"]["minimum_chain_ess"]
        for burn in burn_values
    ]
    energy_rhat = [
        conditions[burn]["convergence"]["energy_split_rhat"]
        for burn in burn_values
    ]
    mag_rhat = [
        conditions[burn]["convergence"]["abs_magnetization_split_rhat"]
        for burn in burn_values
    ]
    axis = axes[1, 2]
    axis.plot(x, energy_rhat, "o-", color=RED, label="energy R-hat")
    axis.plot(x, mag_rhat, "s-", color=ORANGE, label="|m| R-hat")
    axis.axhline(1.05, color=GREY, linestyle="--", linewidth=1.0)
    axis.set_xscale("log")
    axis.set_title("Independent-chain convergence")
    axis.set_xlabel("declared burn-in sweeps")
    axis.set_ylabel("split R-hat")
    axis.legend(loc="upper right")
    ess_axis = axis.twinx()
    ess_axis.plot(x, minimum_ess, "^-", color=GREEN, label="minimum ESS")
    ess_axis.set_ylabel("minimum chain ESS")
    save_figure(figure, output_dir / "burnin_sensitivity")


def plot_correlations(
    burn_values: list[float],
    conditions: dict[float, dict[str, Any]],
    output_dir: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for index, burn in enumerate(burn_values):
        condition = conditions[burn]
        radii = condition["radii"]
        correlation = condition["radial_correlation"]
        valid = (radii >= 1) & (radii <= 64) & (correlation > 0)
        axis.loglog(
            radii[valid],
            correlation[valid],
            "o-",
            markersize=2.8,
            linewidth=1.2,
            color=CONDITION_COLOURS[index],
            label=f"burn-in {burn:g}",
        )
    anchor = conditions[burn_values[-1]]
    r = anchor["radii"]
    g = anchor["radial_correlation"]
    valid = (r >= 4) & (r <= 64) & (g > 0)
    selected_r = r[valid]
    if len(selected_r):
        anchor_index = int(np.argmin(np.abs(selected_r - 4)))
        guide = g[valid][anchor_index] * (
            selected_r / selected_r[anchor_index]
        ) ** (-0.25)
        axis.loglog(
            selected_r,
            guide,
            "--",
            color=GREY,
            linewidth=1.5,
            label=r"guide $r^{-1/4}$",
        )
    axis.set_xlabel("periodic distance r")
    axis.set_ylabel("raw two-point correlation G(r)")
    axis.set_title(r"Full periodic $L=256$: burn-in sensitivity of $G(r)$")
    axis.legend(ncol=2)
    save_figure(figure, output_dir / "correlation_burnin")


def plot_magnetization_and_samples(
    burn_values: list[float],
    conditions: dict[float, dict[str, Any]],
    output_dir: Path,
) -> None:
    figure = plt.figure(figsize=(14.0, 7.7))
    grid = figure.add_gridspec(2, 3, height_ratios=(0.72, 1.28))
    histogram_axis = figure.add_subplot(grid[0, :])
    for index, burn in enumerate(burn_values):
        values = magnetization(conditions[burn]["samples"])
        histogram_axis.hist(
            values,
            bins=41,
            density=True,
            histtype="step",
            linewidth=1.6,
            color=CONDITION_COLOURS[index],
            label=f"burn-in {burn:g}",
        )
    histogram_axis.axvline(0.0, color=GREY, linestyle="--", linewidth=1.0)
    histogram_axis.set_xlabel("signed magnetization m")
    histogram_axis.set_ylabel("density")
    histogram_axis.set_title("Magnetization distribution across independent chains")
    histogram_axis.legend(ncol=len(burn_values))

    final_samples = conditions[burn_values[-1]]["samples"]
    final_m = magnetization(final_samples)
    final_energy = energy_density(final_samples)
    abs_m = np.abs(final_m)
    median_abs_m = float(np.median(abs_m))
    selected = [
        int(np.argmin(abs_m)),
        int(np.argmin(np.abs(abs_m - median_abs_m))),
        int(np.argmax(abs_m)),
    ]
    labels = ("smallest |m|", "typical |m|", "largest |m|")
    for column, (sample_index, label) in enumerate(
        zip(selected, labels, strict=True)
    ):
        axis = figure.add_subplot(grid[1, column])
        axis.imshow(
            final_samples[sample_index],
            cmap=SPIN_CMAP,
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        axis.set_title(
            f"{label}\n"
            f"e={final_energy[sample_index]:.4f}, "
            f"m={final_m[sample_index]:.4f}"
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    save_figure(figure, output_dir / "magnetization_and_samples")
    np.savez_compressed(
        output_dir / "selected_l256_samples.npz",
        samples=final_samples[selected],
        energy=final_energy[selected],
        magnetization=final_m[selected],
        labels=np.asarray(labels),
    )


def plot_final_chain_traces(
    final_rows: list[dict[str, Any]], output_dir: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.6, 7.4))
    colours = (BLUE, GREEN, RED, PURPLE)
    for index, row in enumerate(final_rows):
        samples = row["samples"]
        energy = energy_density(samples)
        signed_m = magnetization(samples)
        energy_acf = autocorrelation_fft(energy)
        mag_acf = autocorrelation_fft(np.abs(signed_m))
        label = f"{row['initial_state']}, seed {row['seed']}"
        axes[0, 0].plot(energy, color=colours[index], alpha=0.8, label=label)
        axes[0, 1].plot(signed_m, color=colours[index], alpha=0.8)
        last = min(32, len(energy_acf) - 1)
        axes[1, 0].plot(
            np.arange(last + 1),
            energy_acf[: last + 1],
            color=colours[index],
        )
        axes[1, 1].plot(
            np.arange(last + 1),
            mag_acf[: last + 1],
            color=colours[index],
        )
    axes[0, 0].axhline(
        -math.sqrt(2.0), color=GREY, linestyle="--", linewidth=1.0
    )
    axes[0, 0].set_title("Energy traces, burn-in 500")
    axes[0, 0].set_ylabel("energy per spin")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("Signed magnetization traces, burn-in 500")
    axes[0, 1].set_ylabel("m")
    axes[1, 0].set_title("Energy ACF")
    axes[1, 1].set_title("|m| ACF")
    for axis in axes[1]:
        axis.axhline(0.0, color="#BBBBBB", linewidth=0.8)
        axis.set_xlabel("lag")
        axis.set_ylabel("ACF")
    for axis in axes[0]:
        axis.set_xlabel("saved configuration")
    save_figure(figure, output_dir / "final_chain_traces")


def main() -> None:
    args = parse_args()
    apply_style()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    burn_values = [
        float(value.strip())
        for value in args.burn_in_sweeps.split(",")
        if value.strip()
    ]
    if not burn_values:
        raise ValueError("at least one burn-in value is required")
    if args.chains_per_condition < 4:
        raise ValueError("use at least four chains per condition")

    initial_states = ("random", "random", "plus", "minus")
    seed_sequence = np.random.SeedSequence(args.seed)
    children = seed_sequence.spawn(len(burn_values) * args.chains_per_condition)
    seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in children
    ]
    tasks = []
    seed_index = 0
    for burn in burn_values:
        for chain_index in range(args.chains_per_condition):
            tasks.append(
                {
                    "lattice_size": args.lattice_size,
                    "burn_in_sweeps": burn,
                    "sweeps_between": args.sweeps_between,
                    "samples_per_chain": args.samples_per_chain,
                    "pilot_cluster_steps": args.pilot_cluster_steps,
                    "chain_index": chain_index,
                    "initial_state": initial_states[
                        chain_index % len(initial_states)
                    ],
                    "seed": seeds[seed_index],
                }
            )
            seed_index += 1

    warm_up_numba()
    completed: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(tasks))
    ) as executor:
        futures = [executor.submit(_chain_worker, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            completed.append(row)
            print(
                f"burn={row['burn_in_sweeps']:g}, "
                f"chain={row['chain_index']}, init={row['initial_state']}, "
                f"elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )

    grouped: dict[float, list[dict[str, Any]]] = {
        burn: [] for burn in burn_values
    }
    for row in completed:
        grouped[float(row["burn_in_sweeps"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["chain_index"]))

    conditions = {
        burn: summarize_condition(grouped[burn]) for burn in burn_values
    }
    exact_validation = exact_l4_validation(args.seed + 99_999)

    plot_burnin_sensitivity(burn_values, conditions, output_dir)
    plot_correlations(burn_values, conditions, output_dir)
    plot_magnetization_and_samples(burn_values, conditions, output_dir)
    plot_final_chain_traces(grouped[burn_values[-1]], output_dir)

    serializable_conditions = {}
    for burn in burn_values:
        condition = conditions[burn]
        serializable_conditions[str(burn)] = {
            "summary": condition["summary"],
            "chain_rows": condition["chain_rows"],
            "convergence": condition["convergence"],
            "eta_hat": condition["eta_hat"],
            "eta_r_squared": condition["eta_r_squared"],
            "radii": condition["radii"].tolist(),
            "radial_correlation": condition["radial_correlation"].tolist(),
        }

    final = conditions[burn_values[-1]]
    final_summary = final["summary"]
    earlier = conditions[burn_values[-2]] if len(burn_values) > 1 else final
    stability = {
        "energy_change_last_two": abs(
            final_summary["energy_mean"] - earlier["summary"]["energy_mean"]
        ),
        "abs_m_change_last_two": abs(
            final_summary["abs_magnetization_mean"]
            - earlier["summary"]["abs_magnetization_mean"]
        ),
        "binder_change_last_two": abs(
            final_summary["binder_u4"] - earlier["summary"]["binder_u4"]
        ),
        "xi_over_l_change_last_two": abs(
            final_summary["xi_over_l"] - earlier["summary"]["xi_over_l"]
        ),
    }
    checks = {
        "l4_transition_kernel_matches_exact": max(
            exact_validation["absolute_error"].values()
        )
        < 0.01,
        "energy_near_critical_limit": abs(
            final_summary["energy_mean"] + math.sqrt(2.0)
        )
        < 0.015,
        "signed_magnetization_symmetric": abs(
            final_summary["magnetization_mean"]
        )
        < 0.10,
        "binder_near_torus_reference": abs(
            final_summary["binder_u4"] - 0.6106901
        )
        < 0.06,
        "xi_over_l_near_torus_reference": abs(
            final_summary["xi_over_l"] - 0.9050488
        )
        < 0.15,
        "eta_near_one_quarter": abs(final["eta_hat"] - 0.25) < 0.08,
        "independent_chain_rhat": max(
            final["convergence"]["energy_split_rhat"],
            final["convergence"]["abs_magnetization_split_rhat"],
        )
        < 1.05,
        "minimum_chain_ess": final["convergence"]["minimum_chain_ess"] >= 30,
        "last_two_burnins_stable": (
            stability["energy_change_last_two"] < 0.01
            and stability["abs_m_change_last_two"] < 0.05
            and stability["binder_change_last_two"] < 0.04
            and stability["xi_over_l_change_last_two"] < 0.12
        ),
    }
    report = {
        "configuration": vars(args) | {"output_dir": str(args.output_dir)},
        "critical_beta": float(BETA_CRITICAL),
        "wolff_add_probability": float(
            1.0 - math.exp(-2.0 * float(BETA_CRITICAL))
        ),
        "conditions": serializable_conditions,
        "exact_l4_validation": exact_validation,
        "last_two_burnin_stability": stability,
        "checks": checks,
        "all_checks_passed": bool(all(checks.values())),
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "diagnostics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
