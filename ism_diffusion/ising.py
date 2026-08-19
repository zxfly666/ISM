"""Wolff-cluster Monte Carlo for the periodic 2D ferromagnetic Ising model."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


BETA_CRITICAL = 0.5 * np.log(1.0 + np.sqrt(2.0))


@dataclass
class WolffSampler:
    """Sample square-lattice Ising configurations with periodic boundaries.

    Retained configurations are separated by a fixed number of cluster updates.
    The update count is calibrated before production from an unretained pilot.
    This avoids observing the Markov chain at state-dependent stopping times.
    """

    lattice_size: int
    beta: float = float(BETA_CRITICAL)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.lattice_size < 2:
            raise ValueError("lattice_size must be at least 2")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        self.rng = np.random.default_rng(self.seed)
        self.spins = self.rng.choice(
            np.array([-1, 1], dtype=np.int8),
            size=(self.lattice_size, self.lattice_size),
        )
        self.p_add = float(1.0 - np.exp(-2.0 * self.beta))
        self._marks = np.zeros_like(self.spins, dtype=np.int32)
        self._mark = 0

    def wolff_step(self) -> int:
        """Build and flip one cluster; return the number of flipped spins."""

        L = self.lattice_size
        sx = int(self.rng.integers(L))
        sy = int(self.rng.integers(L))
        target = self.spins[sx, sy]

        self._mark += 1
        if self._mark == np.iinfo(np.int32).max:
            self._marks.fill(0)
            self._mark = 1
        mark = self._mark

        stack = [(sx, sy)]
        cluster = [(sx, sy)]
        self._marks[sx, sy] = mark

        while stack:
            x, y = stack.pop()
            neighbours = (
                ((x + 1) % L, y),
                ((x - 1) % L, y),
                (x, (y + 1) % L),
                (x, (y - 1) % L),
            )
            for nx, ny in neighbours:
                if self._marks[nx, ny] == mark:
                    continue
                if self.spins[nx, ny] != target:
                    continue
                if self.rng.random() >= self.p_add:
                    continue
                self._marks[nx, ny] = mark
                stack.append((nx, ny))
                cluster.append((nx, ny))

        xs, ys = zip(*cluster)
        self.spins[np.asarray(xs), np.asarray(ys)] *= -1
        return len(cluster)

    def sweep(self) -> int:
        """Perform an adaptive sweep for unretained adaptation only.

        The number of cluster updates depends on the visited states, so this
        method must not be used to decide when a production sample is saved.
        """

        target_flips = self.lattice_size**2
        flipped = 0
        clusters = 0
        while flipped < target_flips:
            flipped += self.wolff_step()
            clusters += 1
        return clusters

    def run_cluster_steps(self, steps: int) -> np.ndarray:
        """Run a fixed number of Wolff updates and return their cluster sizes."""

        if steps < 1:
            raise ValueError("steps must be positive")
        sizes = np.empty(int(steps), dtype=np.int32)
        for index in range(int(steps)):
            sizes[index] = self.wolff_step()
        return sizes

    def calibrate_steps_per_sweep(
        self, pilot_cluster_steps: int = 128
    ) -> tuple[float, float]:
        """Estimate fixed cluster updates per lattice-equivalent sweep."""

        if pilot_cluster_steps < 16:
            raise ValueError("pilot_cluster_steps must be at least 16")
        sizes = self.run_cluster_steps(pilot_cluster_steps)
        mean_cluster_size = float(np.mean(sizes))
        steps_per_sweep = self.lattice_size**2 / mean_cluster_size
        return float(steps_per_sweep), mean_cluster_size

    def thermalize(self, sweeps: int) -> None:
        """Thermalize with a fixed update count chosen by an unretained pilot."""

        if sweeps < 1:
            raise ValueError("sweeps must be positive")
        steps_per_sweep, _ = self.calibrate_steps_per_sweep()
        self.run_cluster_steps(max(1, math.ceil(sweeps * steps_per_sweep)))

    def sample(self, count: int, sweeps_between: int = 5) -> np.ndarray:
        """Return samples using a fixed production interval.

        A short pilot is consumed before the first retained configuration.
        """

        if count < 1:
            raise ValueError("count must be positive")
        if sweeps_between < 1:
            raise ValueError("sweeps_between must be positive")
        steps_per_sweep, _ = self.calibrate_steps_per_sweep()
        gap_cluster_steps = max(
            1, math.ceil(sweeps_between * steps_per_sweep)
        )
        out = np.empty(
            (count, self.lattice_size, self.lattice_size), dtype=np.int8
        )
        for i in range(count):
            self.run_cluster_steps(gap_cluster_steps)
            out[i] = self.spins
        return out

    def sample_equilibrium(
        self,
        count: int,
        burn_in_sweeps: int,
        sweeps_between: int,
        adaptation_sweeps: int = 3,
        pilot_cluster_steps: int = 128,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        """Thermalize, freeze the production interval, and retain samples."""

        if count < 1:
            raise ValueError("count must be positive")
        if burn_in_sweeps < 1 or sweeps_between < 1:
            raise ValueError("sweep counts must be positive")
        if adaptation_sweeps < 1:
            raise ValueError("adaptation_sweeps must be positive")
        if pilot_cluster_steps < 16:
            raise ValueError("pilot_cluster_steps must be at least 16")

        # The state-dependent stopping rule is confined to an unretained
        # adaptation phase. Every subsequent stage has a count fixed before it
        # starts, so production observations do not occur at stopping times.
        adaptation_target = adaptation_sweeps * self.lattice_size**2
        adaptation_updated = 0
        adaptation_cluster_steps = 0
        while adaptation_updated < adaptation_target:
            adaptation_updated += self.wolff_step()
            adaptation_cluster_steps += 1

        initial_steps_per_sweep, initial_cluster_mean = (
            self.calibrate_steps_per_sweep(pilot_cluster_steps)
        )
        burn_in_cluster_steps = max(
            1, math.ceil(burn_in_sweeps * initial_steps_per_sweep)
        )
        burn_in_sizes = self.run_cluster_steps(burn_in_cluster_steps)

        production_steps_per_sweep, production_pilot_cluster_mean = (
            self.calibrate_steps_per_sweep(pilot_cluster_steps)
        )
        gap_cluster_steps = max(
            1, math.ceil(sweeps_between * production_steps_per_sweep)
        )
        samples = np.empty(
            (count, self.lattice_size, self.lattice_size), dtype=np.int8
        )
        realized_gap_sweeps = np.empty(count, dtype=np.float64)
        production_cluster_sizes: list[np.ndarray] = []
        for sample_index in range(count):
            sizes = self.run_cluster_steps(gap_cluster_steps)
            production_cluster_sizes.append(sizes)
            realized_gap_sweeps[sample_index] = (
                float(np.sum(sizes)) / self.lattice_size**2
            )
            samples[sample_index] = self.spins

        production_sizes = np.concatenate(production_cluster_sizes)
        metadata: dict[str, float | int] = {
            "adaptation_sweeps": int(adaptation_sweeps),
            "adaptation_updated_spins": int(adaptation_updated),
            "adaptation_cluster_steps": int(adaptation_cluster_steps),
            "pilot_cluster_steps": int(pilot_cluster_steps),
            "initial_pilot_mean_cluster_size": initial_cluster_mean,
            "burn_in_cluster_steps": int(burn_in_cluster_steps),
            "burn_in_mean_cluster_size": float(np.mean(burn_in_sizes)),
            "production_pilot_mean_cluster_size": production_pilot_cluster_mean,
            "fixed_cluster_steps_per_gap": int(gap_cluster_steps),
            "mean_realized_gap_sweeps": float(np.mean(realized_gap_sweeps)),
            "production_cluster_updates": int(len(production_sizes)),
            "production_mean_cluster_size": float(np.mean(production_sizes)),
            "production_max_cluster_size": int(np.max(production_sizes)),
        }
        return samples, metadata


def _generate_chain(
    lattice_size: int,
    seed: int,
    samples_per_chain: int,
    burn_in_sweeps: int,
    sweeps_between: int,
    beta: float,
    adaptation_sweeps: int,
    pilot_cluster_steps: int,
    backend: str,
    initial_state: str = "random",
) -> tuple[np.ndarray, dict[str, Any]]:
    if backend == "numba":
        from .ising_numba import simulate_wolff_chain

        samples, metadata = simulate_wolff_chain(
            length=lattice_size,
            beta=beta,
            burn_in_sweeps=burn_in_sweeps,
            sweeps_between_samples=sweeps_between,
            n_samples=samples_per_chain,
            seed=int(seed),
            adaptation_sweeps=adaptation_sweeps,
            pilot_cluster_steps=pilot_cluster_steps,
            initial_state=initial_state,
        )
        metadata["seed"] = int(seed)
        return samples, metadata

    sampler = WolffSampler(
        lattice_size=lattice_size, beta=beta, seed=int(seed)
    )
    if initial_state == "plus":
        sampler.spins.fill(1)
    elif initial_state == "minus":
        sampler.spins.fill(-1)
    elif initial_state != "random":
        raise ValueError("initial_state must be random, plus, or minus")
    samples, metadata = sampler.sample_equilibrium(
        count=samples_per_chain,
        burn_in_sweeps=burn_in_sweeps,
        sweeps_between=sweeps_between,
        adaptation_sweeps=adaptation_sweeps,
        pilot_cluster_steps=pilot_cluster_steps,
    )
    metadata["backend"] = "python"
    metadata["seed"] = int(seed)
    metadata["initial_state"] = initial_state
    return samples, metadata


def generate_independent_chains(
    lattice_size: int,
    chain_seeds: Iterable[int],
    samples_per_chain: int,
    burn_in_sweeps: int,
    sweeps_between: int,
    beta: float = float(BETA_CRITICAL),
    adaptation_sweeps: int = 3,
    pilot_cluster_steps: int = 128,
    workers: int = 1,
    backend: str = "auto",
    return_chain_metadata: bool = False,
    initial_states: Iterable[str] | None = None,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]
):
    """Generate samples and their chain ids.

    Splitting by returned chain id, before any augmentation or cropping, avoids
    leaking correlated configurations between train and validation/test sets.
    Each production chain uses a fixed number of Wolff cluster updates between
    saved configurations.
    """

    seeds = [int(seed) for seed in chain_seeds]
    if not seeds:
        raise ValueError("chain_seeds must be non-empty")
    if workers < 1:
        raise ValueError("workers must be positive")
    if backend not in {"auto", "python", "numba"}:
        raise ValueError("backend must be auto, python, or numba")
    states = (
        ["random"] * len(seeds)
        if initial_states is None
        else [str(state) for state in initial_states]
    )
    if len(states) != len(seeds):
        raise ValueError("initial_states must match chain_seeds")
    if any(state not in {"random", "plus", "minus"} for state in states):
        raise ValueError("initial_states must be random, plus, or minus")
    resolved_backend = backend
    if resolved_backend == "auto":
        try:
            import numba  # noqa: F401
        except ModuleNotFoundError:
            resolved_backend = "python"
        else:
            resolved_backend = "numba"
    if resolved_backend == "numba":
        from .ising_numba import warm_up_numba

        warm_up_numba()
    arguments = [
        (
            lattice_size,
            seed,
            samples_per_chain,
            burn_in_sweeps,
            sweeps_between,
            beta,
            adaptation_sweeps,
            pilot_cluster_steps,
            resolved_backend,
            state,
        )
        for seed, state in zip(seeds, states, strict=True)
    ]
    if workers == 1:
        results = [_generate_chain(*argument) for argument in arguments]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(seeds))) as executor:
            results = list(executor.map(_generate_chain_from_tuple, arguments))

    chunks: list[np.ndarray] = []
    chain_ids: list[np.ndarray] = []
    chain_metadata: list[dict[str, Any]] = []
    for chain_id, (samples, metadata) in enumerate(results):
        chunks.append(samples)
        chain_ids.append(
            np.full(samples_per_chain, chain_id, dtype=np.int16)
        )
        chain_metadata.append({"chain_id": chain_id, **metadata})
    combined = np.concatenate(chunks)
    combined_ids = np.concatenate(chain_ids)
    if return_chain_metadata:
        return combined, combined_ids, chain_metadata
    return combined, combined_ids


def _generate_chain_from_tuple(
    arguments: tuple[int, int, int, int, int, float, int, int, str, str]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pickle-friendly adapter for ``ProcessPoolExecutor.map``."""

    return _generate_chain(*arguments)


def energy_density(spins: np.ndarray) -> np.ndarray:
    """Energy per site for J=1, counting every bond once."""

    spins = np.asarray(spins)
    horizontal = spins * np.roll(spins, shift=-1, axis=-1)
    vertical = spins * np.roll(spins, shift=-1, axis=-2)
    return -(horizontal + vertical).mean(axis=(-2, -1))


def magnetization(spins: np.ndarray) -> np.ndarray:
    return np.asarray(spins).mean(axis=(-2, -1))
