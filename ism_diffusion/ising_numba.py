"""Numba-accelerated fixed-interval Wolff sampling.

The production-interval design is adapted from the reviewed Hackathon-3
``critical_ising.wolff`` implementation: all state-dependent adaptation is
completed before retained samples, then a fixed cluster-update count is used
throughout production.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from numba import njit
except ModuleNotFoundError as error:  # pragma: no cover - depends on environment
    raise ModuleNotFoundError(
        "The numba sampler backend requires `pip install numba`."
    ) from error


@njit(cache=True, nogil=True)
def _wolff_cluster_flip(
    spins: np.ndarray,
    beta: float,
    marks: np.ndarray,
    stack: np.ndarray,
    cluster_sites: np.ndarray,
    marker: int,
) -> int:
    length = spins.shape[0]
    n_sites = length * length
    p_add = 1.0 - math.exp(-2.0 * beta)

    seed = np.random.randint(0, n_sites)
    seed_row = seed // length
    seed_col = seed - seed_row * length
    target_spin = spins[seed_row, seed_col]

    stack_size = 1
    cluster_size = 1
    stack[0] = seed
    cluster_sites[0] = seed
    marks[seed] = marker

    while stack_size > 0:
        stack_size -= 1
        site = stack[stack_size]
        row = site // length
        col = site - row * length
        neighbours = (
            ((row - 1) % length) * length + col,
            ((row + 1) % length) * length + col,
            row * length + ((col - 1) % length),
            row * length + ((col + 1) % length),
        )
        for neighbour in neighbours:
            if marks[neighbour] == marker:
                continue
            neighbour_row = neighbour // length
            neighbour_col = neighbour - neighbour_row * length
            if spins[neighbour_row, neighbour_col] != target_spin:
                continue
            if np.random.random() >= p_add:
                continue
            marks[neighbour] = marker
            stack[stack_size] = neighbour
            stack_size += 1
            cluster_sites[cluster_size] = neighbour
            cluster_size += 1

    for index in range(cluster_size):
        site = cluster_sites[index]
        row = site // length
        col = site - row * length
        spins[row, col] = -spins[row, col]
    return cluster_size


@njit(cache=True, nogil=True)
def _simulate_wolff_chain(
    length: int,
    beta: float,
    burn_in_sweeps: float,
    sweeps_between_samples: float,
    n_samples: int,
    seed: int,
    adaptation_sweeps: float,
    pilot_cluster_steps: int,
    initial_state_code: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    n_sites = length * length
    spins = np.empty((length, length), dtype=np.int8)
    for row in range(length):
        for column in range(length):
            if initial_state_code == 1:
                spins[row, column] = 1
            elif initial_state_code == -1:
                spins[row, column] = -1
            else:
                spins[row, column] = 1 if np.random.random() < 0.5 else -1

    marks = np.zeros(n_sites, dtype=np.int32)
    stack = np.empty(n_sites, dtype=np.int32)
    cluster_sites = np.empty(n_sites, dtype=np.int32)
    marker = 1

    # State-dependent stopping is allowed only in this discarded adaptation.
    adaptation_target = max(1, int(math.ceil(adaptation_sweeps * n_sites)))
    adaptation_updated = 0
    adaptation_steps = 0
    while adaptation_updated < adaptation_target:
        adaptation_updated += _wolff_cluster_flip(
            spins, beta, marks, stack, cluster_sites, marker
        )
        marker += 1
        adaptation_steps += 1

    initial_pilot_sum = 0.0
    for _ in range(pilot_cluster_steps):
        initial_pilot_sum += _wolff_cluster_flip(
            spins, beta, marks, stack, cluster_sites, marker
        )
        marker += 1
    initial_pilot_mean = initial_pilot_sum / pilot_cluster_steps
    initial_steps_per_sweep = n_sites / initial_pilot_mean

    burn_in_cluster_steps = max(
        1, int(math.ceil(burn_in_sweeps * initial_steps_per_sweep))
    )
    burn_in_cluster_sum = 0.0
    for _ in range(burn_in_cluster_steps):
        burn_in_cluster_sum += _wolff_cluster_flip(
            spins, beta, marks, stack, cluster_sites, marker
        )
        marker += 1

    production_pilot_sum = 0.0
    for _ in range(pilot_cluster_steps):
        production_pilot_sum += _wolff_cluster_flip(
            spins, beta, marks, stack, cluster_sites, marker
        )
        marker += 1
    production_pilot_mean = production_pilot_sum / pilot_cluster_steps
    production_steps_per_sweep = n_sites / production_pilot_mean
    fixed_cluster_steps_per_gap = max(
        1,
        int(
            math.ceil(
                sweeps_between_samples * production_steps_per_sweep
            )
        ),
    )

    samples = np.empty((n_samples, length, length), dtype=np.int8)
    realized_gap_sweeps = np.empty(n_samples, dtype=np.float64)
    production_cluster_count = 0
    production_cluster_sum = 0.0
    production_cluster_max = 0
    for sample_index in range(n_samples):
        gap_updated = 0
        for _ in range(fixed_cluster_steps_per_gap):
            cluster_size = _wolff_cluster_flip(
                spins, beta, marks, stack, cluster_sites, marker
            )
            marker += 1
            gap_updated += cluster_size
            production_cluster_count += 1
            production_cluster_sum += cluster_size
            if cluster_size > production_cluster_max:
                production_cluster_max = cluster_size
        realized_gap_sweeps[sample_index] = gap_updated / n_sites
        samples[sample_index] = spins

    raw_metadata = np.array(
        [
            float(adaptation_updated),
            float(adaptation_steps),
            float(initial_pilot_mean),
            float(burn_in_cluster_steps),
            float(burn_in_cluster_sum),
            float(production_pilot_mean),
            float(fixed_cluster_steps_per_gap),
            float(production_cluster_count),
            float(production_cluster_sum),
            float(production_cluster_max),
        ],
        dtype=np.float64,
    )
    return samples, realized_gap_sweeps, raw_metadata


def simulate_wolff_chain(
    length: int,
    beta: float,
    burn_in_sweeps: float,
    sweeps_between_samples: float,
    n_samples: int,
    seed: int,
    adaptation_sweeps: float = 3.0,
    pilot_cluster_steps: int = 128,
    initial_state: str = "random",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a complete periodic lattice chain and production diagnostics."""

    if length < 4:
        raise ValueError("length must be at least 4")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if burn_in_sweeps <= 0.0 or sweeps_between_samples <= 0.0:
        raise ValueError("sweep counts must be positive")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if adaptation_sweeps <= 0.0 or pilot_cluster_steps < 16:
        raise ValueError(
            "adaptation_sweeps must be positive and pilot_cluster_steps >= 16"
        )
    initial_state_codes = {"minus": -1, "random": 0, "plus": 1}
    if initial_state not in initial_state_codes:
        raise ValueError("initial_state must be minus, random, or plus")

    samples, realized_gap_sweeps, raw = _simulate_wolff_chain(
        int(length),
        float(beta),
        float(burn_in_sweeps),
        float(sweeps_between_samples),
        int(n_samples),
        int(seed),
        float(adaptation_sweeps),
        int(pilot_cluster_steps),
        int(initial_state_codes[initial_state]),
    )
    production_count = int(raw[7])
    metadata: dict[str, Any] = {
        "backend": "numba",
        "initial_state": initial_state,
        "adaptation_sweeps": float(adaptation_sweeps),
        "adaptation_updated_spins": int(raw[0]),
        "adaptation_cluster_steps": int(raw[1]),
        "pilot_cluster_steps": int(pilot_cluster_steps),
        "initial_pilot_mean_cluster_size": float(raw[2]),
        "burn_in_cluster_steps": int(raw[3]),
        "burn_in_mean_cluster_size": float(raw[4] / raw[3]),
        "production_pilot_mean_cluster_size": float(raw[5]),
        "fixed_cluster_steps_per_gap": int(raw[6]),
        "mean_realized_gap_sweeps": float(np.mean(realized_gap_sweeps)),
        "production_cluster_updates": production_count,
        "production_mean_cluster_size": float(raw[8] / production_count),
        "production_max_cluster_size": int(raw[9]),
    }
    return samples, metadata


def warm_up_numba() -> None:
    """Compile and cache the kernels before multiprocessing starts."""

    simulate_wolff_chain(
        length=4,
        beta=0.2,
        burn_in_sweeps=1.0,
        sweeps_between_samples=1.0,
        n_samples=2,
        seed=1,
        adaptation_sweeps=1.0,
        pilot_cluster_steps=16,
    )
