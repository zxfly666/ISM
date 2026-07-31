"""Generate and render one complete periodic critical Ising lattice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from ism_diffusion.ising import BETA_CRITICAL, energy_density, magnetization
from ism_diffusion.ising_numba import simulate_wolff_chain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lattice-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--burn-in-sweeps", type=float, default=500.0)
    parser.add_argument("--sweeps-between", type=float, default=5.0)
    parser.add_argument("--pilot-cluster-steps", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples, diagnostics = simulate_wolff_chain(
        length=args.lattice_size,
        beta=float(BETA_CRITICAL),
        burn_in_sweeps=args.burn_in_sweeps,
        sweeps_between_samples=args.sweeps_between,
        n_samples=1,
        seed=args.seed,
        adaptation_sweeps=3.0,
        pilot_cluster_steps=args.pilot_cluster_steps,
    )
    sample = samples[0]
    energy = float(energy_density(sample[None])[0])
    signed_magnetization = float(magnetization(sample[None])[0])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "lattice_size": args.lattice_size,
        "beta": float(BETA_CRITICAL),
        "seed": args.seed,
        "energy_density": energy,
        "magnetization": signed_magnetization,
        "sampler": diagnostics,
    }
    np.savez_compressed(
        args.output_dir / "l256_preview_sample.npz",
        spins=sample,
        metadata=np.asarray(json.dumps(metadata)),
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 13,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )
    figure, axis = plt.subplots(figsize=(8.2, 8.6))
    axis.imshow(
        sample,
        cmap=ListedColormap(["#3B4CC0", "#B40426"]),
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    axis.set_title(
        rf"Critical Ising, full periodic $L={args.lattice_size}$"
        + "\n"
        + rf"$e={energy:.5f}$, $m={signed_magnetization:.5f}$",
        pad=12,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout(pad=1.0)
    figure.savefig(
        args.output_dir / "l256_preview.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    (args.output_dir / "l256_preview_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
