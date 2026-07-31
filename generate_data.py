"""Generate chain-split critical Ising data with Wolff Monte Carlo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ism_diffusion.ising import (
    BETA_CRITICAL,
    energy_density,
    generate_independent_chains,
    magnetization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lattice-size", type=int, default=32)
    parser.add_argument("--beta", type=float, default=float(BETA_CRITICAL))
    parser.add_argument("--chains", type=int, default=16)
    parser.add_argument("--train-chains", type=int, default=12)
    parser.add_argument("--val-chains", type=int, default=2)
    parser.add_argument("--test-chains", type=int, default=2)
    parser.add_argument("--samples-per-chain", type=int, default=1250)
    parser.add_argument("--burn-in-sweeps", type=int, default=500)
    parser.add_argument("--sweeps-between", type=int, default=5)
    parser.add_argument("--adaptation-sweeps", type=int, default=3)
    parser.add_argument("--pilot-cluster-steps", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--backend",
        choices=("auto", "python", "numba"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_total = args.train_chains + args.val_chains + args.test_chains
    if split_total != args.chains:
        raise ValueError(
            "train_chains + val_chains + test_chains must equal chains"
        )
    seed_sequence = np.random.SeedSequence(args.seed)
    chain_seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(args.chains)
    ]
    print(
        f"Generating {args.chains} chains at L={args.lattice_size}, "
        f"beta={args.beta:.8f} ..."
    )
    samples, chain_ids, chain_diagnostics = generate_independent_chains(
        lattice_size=args.lattice_size,
        chain_seeds=chain_seeds,
        samples_per_chain=args.samples_per_chain,
        burn_in_sweeps=args.burn_in_sweeps,
        sweeps_between=args.sweeps_between,
        beta=args.beta,
        adaptation_sweeps=args.adaptation_sweeps,
        pilot_cluster_steps=args.pilot_cluster_steps,
        workers=args.workers,
        backend=args.backend,
        return_chain_metadata=True,
    )

    train_end = args.train_chains
    val_end = train_end + args.val_chains
    train_selector = chain_ids < train_end
    val_selector = (chain_ids >= train_end) & (chain_ids < val_end)
    test_selector = chain_ids >= val_end
    splits = {
        "train": samples[train_selector],
        "val": samples[val_selector],
        "test": samples[test_selector],
        "train_chain_id": chain_ids[train_selector],
        "val_chain_id": chain_ids[val_selector],
        "test_chain_id": chain_ids[test_selector],
    }
    metadata = {
        "lattice_size": args.lattice_size,
        "beta": args.beta,
        "critical_beta_exact": float(BETA_CRITICAL),
        "chains": args.chains,
        "train_chains": args.train_chains,
        "val_chains": args.val_chains,
        "test_chains": args.test_chains,
        "samples_per_chain": args.samples_per_chain,
        "burn_in_sweeps": args.burn_in_sweeps,
        "sweeps_between": args.sweeps_between,
        "adaptation_sweeps": args.adaptation_sweeps,
        "pilot_cluster_steps": args.pilot_cluster_steps,
        "workers": args.workers,
        "sampler_backend": chain_diagnostics[0]["backend"],
        "production_interval": (
            "A post-burn-in pilot fixes the number of Wolff cluster updates "
            "between all retained configurations."
        ),
        "seed": args.seed,
        "chain_seeds": chain_seeds,
        "chain_diagnostics": chain_diagnostics,
        "note": "ESS is measured after sampling; sweep spacing is not assumed independent.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **splits,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    for name in ("train", "val", "test"):
        energy = energy_density(splits[name])
        mag = magnetization(splits[name])
        print(
            f"{name:5s}: {len(splits[name]):6d} samples, "
            f"<e>={energy.mean(): .5f}, <|m|>={np.abs(mag).mean(): .5f}"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
