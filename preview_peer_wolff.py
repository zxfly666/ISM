"""Render one full L=256 sample using the peer Hackathon-3 Wolff code."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--peer-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lattice-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2019611124)
    parser.add_argument("--initial-state", default="random")
    parser.add_argument("--adaptation-sweeps", type=float, default=5.0)
    parser.add_argument("--pilot-cluster-steps", type=int, default=256)
    parser.add_argument("--pilot-rounds", type=int, default=5)
    parser.add_argument("--burnin-sweeps", type=float, default=50.0)
    parser.add_argument("--sweeps-between", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.peer_root / "src"
    if not (source_dir / "critical_ising" / "wolff.py").exists():
        raise FileNotFoundError(f"peer Wolff source not found under {source_dir}")
    sys.path.insert(0, str(source_dir))

    # Imports deliberately come from the peer repository, not ism_diffusion.
    from critical_ising.observables import (  # noqa: PLC0415
        energy_per_spin,
        magnetization_per_spin,
    )
    from critical_ising.wolff import (  # noqa: PLC0415
        BETA_CRITICAL,
        simulate_wolff_chain,
    )

    # The peer public API requires n_samples >= 2.  We retain and render only
    # the first state; the second exists only transiently inside this call.
    samples, sampler_metadata = simulate_wolff_chain(
        length=args.lattice_size,
        beta=float(BETA_CRITICAL),
        burnin_sweeps=args.burnin_sweeps,
        sweeps_between_samples=args.sweeps_between,
        n_samples=2,
        seed=args.seed,
        adaptation_sweeps=args.adaptation_sweeps,
        pilot_cluster_steps=args.pilot_cluster_steps,
        pilot_rounds=args.pilot_rounds,
        initial_state=args.initial_state,
    )
    sample = samples[0].copy()
    energy = float(energy_per_spin(sample[None])[0])
    signed_magnetization = float(magnetization_per_spin(sample[None])[0])
    positive_fraction = float(np.mean(sample == 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": "external/Hackathon-3-src/src/critical_ising/wolff.py",
        "source_implementation": "peer Hackathon-3, unmodified",
        "lattice_size": args.lattice_size,
        "periodic": True,
        "cropped": False,
        "beta": float(BETA_CRITICAL),
        "seed": args.seed,
        "initial_state": args.initial_state,
        "energy_density": energy,
        "magnetization": signed_magnetization,
        "positive_spin_fraction": positive_fraction,
        "exported_images": 1,
        "peer_api_internal_samples": 2,
        "sampler": sampler_metadata,
    }
    np.savez_compressed(
        args.output_dir / "peer_wolff_l256_sample.npz",
        spins=sample,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )
    figure, axis = plt.subplots(figsize=(8.0, 8.5))
    axis.imshow(
        sample,
        cmap=ListedColormap(["#3B4CC0", "#B40426"]),
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    axis.set_title(
        rf"Peer Hackathon-3 Wolff, full periodic $L={args.lattice_size}$"
        + "\n"
        + rf"$e={energy:.5f}$, $m={signed_magnetization:.5f}$, "
        + rf"$f_+={positive_fraction:.3f}$",
        pad=12,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout(pad=1.0)
    figure.savefig(
        args.output_dir / "peer_wolff_l256.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    (args.output_dir / "peer_wolff_l256_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
