"""Generate an independent high-statistics Ising physics reference ensemble."""

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
from ism_diffusion.metrics import ensemble_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lattice-size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=float(BETA_CRITICAL))
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--samples-per-chain", type=int, default=1250)
    parser.add_argument("--burn-in-sweeps", type=int, default=500)
    parser.add_argument("--sweeps-between", type=int, default=5)
    parser.add_argument("--adaptation-sweeps", type=int, default=3)
    parser.add_argument("--pilot-cluster-steps", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backend", choices=("auto", "python", "numba"), default="numba")
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_sequence = np.random.SeedSequence(args.seed)
    chain_seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(args.chains)
    ]
    samples, chain_ids, chain_metadata = generate_independent_chains(
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
    energy_chains = [energy_density(samples[chain_ids == index]) for index in range(args.chains)]
    abs_m_chains = [
        np.abs(magnetization(samples[chain_ids == index]))
        for index in range(args.chains)
    ]
    chain_diagnostics = []
    for index in range(args.chains):
        energy_diag = integrated_autocorrelation_time(energy_chains[index])
        mag_diag = integrated_autocorrelation_time(abs_m_chains[index])
        chain_diagnostics.append(
            {
                "chain_id": index,
                "energy_tau_int": energy_diag["tau_int"],
                "energy_ess": energy_diag["ess"],
                "abs_magnetization_tau_int": mag_diag["tau_int"],
                "abs_magnetization_ess": mag_diag["ess"],
            }
        )
    convergence = {
        "energy_split_rhat": split_rhat(energy_chains),
        "abs_magnetization_split_rhat": split_rhat(abs_m_chains),
        "minimum_chain_ess": min(
            min(row["energy_ess"], row["abs_magnetization_ess"])
            for row in chain_diagnostics
        ),
    }
    metadata = {
        "purpose": "independent high-statistics physics reference; never used for training or selection",
        "lattice_size": args.lattice_size,
        "beta": args.beta,
        "chains": args.chains,
        "samples_per_chain": args.samples_per_chain,
        "burn_in_sweeps": args.burn_in_sweeps,
        "sweeps_between": args.sweeps_between,
        "adaptation_sweeps": args.adaptation_sweeps,
        "pilot_cluster_steps": args.pilot_cluster_steps,
        "seed": args.seed,
        "chain_seeds": chain_seeds,
        "sampler_backend": chain_metadata[0]["backend"],
        "chain_metadata": chain_metadata,
        "chain_diagnostics": chain_diagnostics,
        "convergence": convergence,
        "ensemble_summary": ensemble_summary(samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        test=samples.astype(np.int8),
        test_chain_id=chain_ids.astype(np.int16),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    print(json.dumps(metadata["ensemble_summary"] | convergence, indent=2))
    print(f"Saved {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
