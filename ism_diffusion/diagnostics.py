"""Monte Carlo chain diagnostics used by data and model evaluation."""

from __future__ import annotations

import math

import numpy as np


def autocorrelation_fft(series: np.ndarray) -> np.ndarray:
    """Return the normalized autocorrelation of a one-dimensional series."""

    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("series must be one-dimensional with length >= 2")
    centered = values - values.mean()
    variance = float(np.dot(centered, centered))
    if variance <= 0.0:
        return np.ones(1, dtype=np.float64)
    fft_length = 1 << (2 * len(values) - 1).bit_length()
    transformed = np.fft.rfft(centered, n=fft_length)
    autocovariance = np.fft.irfft(
        transformed * transformed.conj(), n=fft_length
    )[: len(values)]
    autocovariance /= np.arange(len(values), 0, -1, dtype=np.float64)
    return autocovariance / autocovariance[0]


def integrated_autocorrelation_time(series: np.ndarray) -> dict[str, float]:
    """Estimate integrated autocorrelation time with a positive-pair window."""

    values = np.asarray(series, dtype=np.float64)
    acf = autocorrelation_fft(values)
    if len(acf) == 1:
        return {
            "tau_int": 0.5,
            "statistical_inefficiency": 1.0,
            "ess": float(len(values)),
            "window_lag": 0.0,
        }

    positive_pairs: list[float] = []
    previous_pair = float("inf")
    lag = 1
    while lag < len(acf):
        pair_sum = float(acf[lag])
        if lag + 1 < len(acf):
            pair_sum += float(acf[lag + 1])
        if pair_sum <= 0.0:
            break
        pair_sum = min(pair_sum, previous_pair)
        previous_pair = pair_sum
        positive_pairs.append(pair_sum)
        lag += 2

    tau = max(0.5 + float(np.sum(positive_pairs)), 0.5)
    inefficiency = 2.0 * tau
    return {
        "tau_int": float(tau),
        "statistical_inefficiency": float(inefficiency),
        "ess": float(min(len(values) / inefficiency, len(values))),
        "window_lag": float(max(lag - 1, 0)),
    }


def split_rhat(chains: list[np.ndarray]) -> float:
    """Return the classical split-R-hat for equally defined scalar chains."""

    if len(chains) < 2:
        return float("nan")
    minimum_length = min(len(np.asarray(chain)) for chain in chains)
    half = minimum_length // 2
    if half < 4:
        return float("nan")
    split = []
    for chain in chains:
        values = np.asarray(chain, dtype=np.float64)[: 2 * half]
        split.extend((values[:half], values[half:]))
    matrix = np.stack(split, axis=0)
    within = float(np.mean(np.var(matrix, axis=1, ddof=1)))
    if within <= 0.0:
        return 1.0
    between = float(half * np.var(np.mean(matrix, axis=1), ddof=1))
    variance_hat = ((half - 1.0) / half) * within + between / half
    return float(math.sqrt(max(variance_hat / within, 0.0)))


def local_gibbs_calibration(
    samples: np.ndarray, beta: float
) -> list[dict[str, float]]:
    """Compare empirical local conditionals with the exact Ising law."""

    spins = np.asarray(samples, dtype=np.int8)
    if spins.ndim != 3 or spins.shape[-1] != spins.shape[-2]:
        raise ValueError("samples must have shape [samples, L, L]")
    neighbour_sum = (
        np.roll(spins, 1, axis=-2)
        + np.roll(spins, -1, axis=-2)
        + np.roll(spins, 1, axis=-1)
        + np.roll(spins, -1, axis=-1)
    )
    rows: list[dict[str, float]] = []
    for value in (-4, -2, 0, 2, 4):
        selected = neighbour_sum == value
        count = int(np.count_nonzero(selected))
        positive = int(np.count_nonzero(spins[selected] == 1))
        empirical = positive / count if count else float("nan")
        exact = 1.0 / (1.0 + math.exp(-2.0 * beta * value))
        standard_error = (
            math.sqrt(max(empirical * (1.0 - empirical), 0.0) / count)
            if count and np.isfinite(empirical)
            else float("nan")
        )
        rows.append(
            {
                "neighbour_sum": float(value),
                "count": float(count),
                "empirical_p_plus": float(empirical),
                "exact_p_plus": float(exact),
                "absolute_error": float(abs(empirical - exact)),
                "binomial_se": float(standard_error),
            }
        )
    return rows
