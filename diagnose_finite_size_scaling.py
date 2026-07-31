"""Measure critical finite-size magnetization scaling without cropping.

This diagnostic addresses a common visual misconception: at the two-dimensional
Ising critical point the signed ensemble mean is zero, but a typical finite
periodic configuration has nonzero absolute magnetization.  The latter decays
only as L^{-beta/nu} = L^{-1/8}.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ism_diffusion.diagnostics import split_rhat
from ism_diffusion.ising import BETA_CRITICAL, energy_density, magnetization
from ism_diffusion.ising_numba import simulate_wolff_chain, warm_up_numba
from ism_diffusion.metrics import ensemble_summary


BLUE = "#16569A"
RED = "#BA4642"
GREY = "#777777"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/finite_size_scaling"),
    )
    parser.add_argument("--sizes", type=str, default="16,32,64,128,256")
    parser.add_argument("--chains-per-size", type=int, default=4)
    parser.add_argument("--samples-per-chain", type=int, default=128)
    parser.add_argument("--burn-in-sweeps", type=float, default=50.0)
    parser.add_argument("--sweeps-between", type=float, default=5.0)
    parser.add_argument("--pilot-cluster-steps", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def run_chain(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    samples, metadata = simulate_wolff_chain(
        length=int(task["size"]),
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
        "metadata": metadata,
        "elapsed_seconds": time.perf_counter() - started,
    }


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )


def main() -> None:
    args = parse_args()
    sizes = [
        int(value.strip()) for value in args.sizes.split(",") if value.strip()
    ]
    if len(sizes) < 3 or any(size < 4 for size in sizes):
        raise ValueError("provide at least three lattice sizes, all >= 4")
    if args.chains_per_size < 4:
        raise ValueError("use at least four independent chains per size")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial_states = ("plus", "minus", "plus", "minus")
    seed_sequence = np.random.SeedSequence(args.seed)
    children = seed_sequence.spawn(len(sizes) * args.chains_per_size)
    seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in children
    ]
    tasks: list[dict[str, Any]] = []
    seed_index = 0
    for size in sizes:
        for chain_index in range(args.chains_per_size):
            tasks.append(
                {
                    "size": size,
                    "chain_index": chain_index,
                    "initial_state": initial_states[
                        chain_index % len(initial_states)
                    ],
                    "seed": seeds[seed_index],
                    "burn_in_sweeps": args.burn_in_sweeps,
                    "sweeps_between": args.sweeps_between,
                    "samples_per_chain": args.samples_per_chain,
                    "pilot_cluster_steps": args.pilot_cluster_steps,
                }
            )
            seed_index += 1

    warm_up_numba()
    started = time.perf_counter()
    completed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(tasks))
    ) as executor:
        futures = [executor.submit(run_chain, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            completed.append(row)
            print(
                f"L={row['size']}, chain={row['chain_index']}, "
                f"init={row['initial_state']}, "
                f"elapsed={row['elapsed_seconds']:.2f}s",
                flush=True,
            )

    rows: list[dict[str, Any]] = []
    for size in sizes:
        selected = sorted(
            (row for row in completed if int(row["size"]) == size),
            key=lambda row: int(row["chain_index"]),
        )
        samples = np.concatenate([row["samples"] for row in selected])
        summary = ensemble_summary(samples)
        chain_abs_m = [
            np.abs(magnetization(row["samples"])) for row in selected
        ]
        chain_energy = [
            energy_density(row["samples"]) for row in selected
        ]
        chain_means = np.asarray(
            [float(values.mean()) for values in chain_abs_m]
        )
        all_abs_m = np.concatenate(chain_abs_m)
        abs_m_se = float(
            chain_means.std(ddof=1) / math.sqrt(len(chain_means))
        )
        rows.append(
            {
                "lattice_size": size,
                **summary,
                "abs_magnetization_chain_se": abs_m_se,
                "majority_spin_fraction_mean": float(
                    0.5 * (1.0 + summary["abs_magnetization_mean"])
                ),
                "abs_magnetization_quantiles": {
                    str(probability): float(value)
                    for probability, value in zip(
                        (0.05, 0.25, 0.5, 0.75, 0.95),
                        np.quantile(
                            all_abs_m, (0.05, 0.25, 0.5, 0.75, 0.95)
                        ),
                        strict=True,
                    )
                },
                "fraction_abs_m_below_0_10": float(
                    np.mean(all_abs_m < 0.10)
                ),
                "fraction_abs_m_below_0_20": float(
                    np.mean(all_abs_m < 0.20)
                ),
                "energy_split_rhat": split_rhat(chain_energy),
                "abs_magnetization_split_rhat": split_rhat(chain_abs_m),
                "chains": [
                    {
                        "chain_index": int(row["chain_index"]),
                        "initial_state": str(row["initial_state"]),
                        "seed": int(row["seed"]),
                        "abs_magnetization_mean": float(
                            chain_abs_m[index].mean()
                        ),
                        "energy_mean": float(chain_energy[index].mean()),
                        "elapsed_seconds": float(row["elapsed_seconds"]),
                        "sampler_metadata": row["metadata"],
                    }
                    for index, row in enumerate(selected)
                ],
            }
        )

    lattice_sizes = np.asarray(sizes, dtype=np.float64)
    abs_m = np.asarray(
        [row["abs_magnetization_mean"] for row in rows],
        dtype=np.float64,
    )
    abs_m_se = np.asarray(
        [row["abs_magnetization_chain_se"] for row in rows],
        dtype=np.float64,
    )
    fitted_slope, fitted_intercept = np.polyfit(
        np.log(lattice_sizes), np.log(abs_m), 1
    )
    predicted = fitted_intercept + fitted_slope * np.log(lattice_sizes)
    residual = float(np.sum((np.log(abs_m) - predicted) ** 2))
    total = float(
        np.sum((np.log(abs_m) - np.mean(np.log(abs_m))) ** 2)
    )
    r_squared = float(1.0 - residual / total)

    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].errorbar(
        lattice_sizes,
        abs_m,
        yerr=abs_m_se,
        fmt="o-",
        color=BLUE,
        capsize=3,
        linewidth=1.8,
        label="Wolff MC",
    )
    guide = abs_m[-1] * (lattice_sizes / lattice_sizes[-1]) ** (-0.125)
    axes[0].plot(
        lattice_sizes,
        guide,
        "--",
        color=GREY,
        linewidth=1.5,
        label=r"critical guide $L^{-1/8}$",
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("lattice size L")
    axes[0].set_ylabel(r"$\langle |m| \rangle$")
    axes[0].set_title(
        rf"Finite-size scaling: fitted slope {fitted_slope:.3f}"
    )
    axes[0].legend()

    majority = 0.5 * (1.0 + abs_m)
    axes[1].errorbar(
        lattice_sizes,
        majority,
        yerr=0.5 * abs_m_se,
        fmt="o-",
        color=RED,
        capsize=3,
        linewidth=1.8,
    )
    axes[1].axhline(
        0.5,
        color=GREY,
        linestyle="--",
        linewidth=1.2,
        label="perfect 50/50 balance",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylim(0.49, max(0.9, float(majority.max()) + 0.02))
    axes[1].set_xlabel("lattice size L")
    axes[1].set_ylabel("mean majority-spin fraction")
    axes[1].set_title("A typical finite critical image is not 50/50")
    axes[1].legend()
    figure.tight_layout(pad=1.5)
    figure.savefig(
        args.output_dir / "finite_size_magnetization.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        args.output_dir / "finite_size_magnetization.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)

    report = {
        "configuration": vars(args) | {"output_dir": str(args.output_dir)},
        "critical_beta": float(BETA_CRITICAL),
        "expected_exponent_beta_over_nu": 0.125,
        "fitted_log_log_slope": float(fitted_slope),
        "fitted_exponent_beta_over_nu": float(-fitted_slope),
        "fit_r_squared": r_squared,
        "rows": rows,
        "all_rhat_below_1_05": bool(
            all(
                max(
                    row["energy_split_rhat"],
                    row["abs_magnetization_split_rhat"],
                )
                < 1.05
                for row in rows
            )
        ),
        "total_elapsed_seconds": float(time.perf_counter() - started),
    }
    (args.output_dir / "finite_size_scaling.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
