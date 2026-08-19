"""Generate bit-packed parent fields for the rapid scale experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ism_diffusion.diagnostics import integrated_autocorrelation_time, split_rhat
from ism_diffusion.ising import (
    BETA_CRITICAL,
    energy_density,
    generate_independent_chains,
    magnetization,
)
from ism_diffusion.scale_data import pack_spins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lattice-size", type=int, required=True)
    parser.add_argument("--train-chains", type=int, default=6)
    parser.add_argument("--val-chains", type=int, default=2)
    parser.add_argument("--target-chains", type=int, default=2)
    parser.add_argument("--control-chains", type=int, default=2)
    parser.add_argument("--samples-per-chain", type=int, default=128)
    parser.add_argument("--burn-in-sweeps", type=int, default=20)
    parser.add_argument("--sweeps-between", type=int, default=2)
    parser.add_argument("--adaptation-sweeps", type=int, default=3)
    parser.add_argument("--pilot-cluster-steps", type=int, default=128)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args()


def chain_diagnostics(
    samples: np.ndarray, chain_ids: np.ndarray, chains: int
) -> dict:
    energies = energy_density(samples)
    abs_magnetization = np.abs(magnetization(samples))
    energy_chains = [energies[chain_ids == chain] for chain in range(chains)]
    magnetization_chains = [
        abs_magnetization[chain_ids == chain] for chain in range(chains)
    ]
    return {
        "energy": [
            integrated_autocorrelation_time(values) for values in energy_chains
        ],
        "abs_magnetization": [
            integrated_autocorrelation_time(values)
            for values in magnetization_chains
        ],
        "energy_split_rhat": split_rhat(energy_chains),
        "abs_magnetization_split_rhat": split_rhat(magnetization_chains),
        "energy_mean": float(energies.mean()),
        "abs_magnetization_mean": float(abs_magnetization.mean()),
    }


def main() -> None:
    args = parse_args()
    split_counts = {
        "train": args.train_chains,
        "val": args.val_chains,
        "test_target": args.target_chains,
        "test_control": args.control_chains,
    }
    chains = sum(split_counts.values())
    seed_sequence = np.random.SeedSequence(args.seed)
    seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(chains)
    ]
    initial_states = [
        "random" if chain % 2 == 0 else "plus" for chain in range(chains)
    ]
    samples, chain_ids, chain_metadata = generate_independent_chains(
        lattice_size=args.lattice_size,
        chain_seeds=seeds,
        samples_per_chain=args.samples_per_chain,
        burn_in_sweeps=args.burn_in_sweeps,
        sweeps_between=args.sweeps_between,
        beta=float(BETA_CRITICAL),
        adaptation_sweeps=args.adaptation_sweeps,
        pilot_cluster_steps=args.pilot_cluster_steps,
        workers=min(args.workers, chains),
        backend="numba",
        return_chain_metadata=True,
        initial_states=initial_states,
    )
    diagnostics = chain_diagnostics(samples, chain_ids, chains)

    arrays: dict[str, np.ndarray] = {}
    offset = 0
    split_manifest: dict[str, list[int]] = {}
    for split, count in split_counts.items():
        selected_chains = list(range(offset, offset + count))
        selected = np.isin(chain_ids, selected_chains)
        arrays[f"{split}_packed"] = pack_spins(samples[selected])
        arrays[f"{split}_chain_id"] = chain_ids[selected] - offset
        split_manifest[split] = selected_chains
        offset += count

    metadata = {
        "lattice_size": args.lattice_size,
        "beta": float(BETA_CRITICAL),
        "samples_per_chain": args.samples_per_chain,
        "burn_in_sweeps": args.burn_in_sweeps,
        "sweeps_between": args.sweeps_between,
        "adaptation_sweeps": args.adaptation_sweeps,
        "pilot_cluster_steps": args.pilot_cluster_steps,
        "seed": args.seed,
        "chain_seeds": seeds,
        "initial_states": initial_states,
        "split_counts": split_counts,
        "split_manifest": split_manifest,
        "chain_metadata": chain_metadata,
        "diagnostics": diagnostics,
        "storage": "np.packbits along the final lattice axis, little endian",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **arrays,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    summary = {
        "output": str(args.output),
        "size_bytes": args.output.stat().st_size,
        "lattice_size": args.lattice_size,
        "chains": chains,
        "samples": int(len(samples)),
        "diagnostics": diagnostics,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
