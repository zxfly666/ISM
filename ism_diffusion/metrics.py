"""Physics-facing ensemble metrics for generated Ising configurations."""

from __future__ import annotations

import numpy as np

from .ising import energy_density, magnetization


def correlation_map(spins: np.ndarray) -> np.ndarray:
    """Connected periodic two-point function averaged over samples and origins."""

    spins = np.asarray(spins, dtype=np.float64)
    size = spins.shape[-1] * spins.shape[-2]
    spectrum = np.fft.fft2(spins, axes=(-2, -1))
    autocorrelation = np.fft.ifft2(
        spectrum * spectrum.conj(), axes=(-2, -1)
    ).real / size
    ensemble_mean = float(spins.mean())
    return autocorrelation.mean(axis=0) - ensemble_mean**2


def structure_factor(spins: np.ndarray) -> np.ndarray:
    spins = np.asarray(spins, dtype=np.float64)
    size = spins.shape[-1] * spins.shape[-2]
    spectrum = np.fft.fft2(spins, axes=(-2, -1))
    return (np.abs(spectrum) ** 2 / size).mean(axis=0)


def radial_average_periodic(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially average a square periodic field around index (0, 0)."""

    L = field.shape[0]
    coord = np.minimum(np.arange(L), L - np.arange(L))
    radius = np.sqrt(coord[:, None] ** 2 + coord[None, :] ** 2)
    bins = np.rint(radius).astype(np.int32)
    max_bin = L // 2
    values = np.zeros(max_bin + 1, dtype=np.float64)
    counts = np.zeros(max_bin + 1, dtype=np.int64)
    for r in range(max_bin + 1):
        selected = bins == r
        values[r] = field[selected].mean() if selected.any() else np.nan
        counts[r] = int(selected.sum())
    return np.arange(max_bin + 1), values


def ensemble_summary(spins: np.ndarray) -> dict[str, float]:
    spins = np.asarray(spins, dtype=np.int8)
    if spins.ndim != 3 or spins.shape[-1] != spins.shape[-2]:
        raise ValueError("spins must have shape [samples, L, L]")
    L = spins.shape[-1]
    energy = energy_density(spins)
    mag = magnetization(spins)
    m2 = float(np.mean(mag**2))
    m4 = float(np.mean(mag**4))
    binder = float(1.0 - m4 / (3.0 * m2 * m2)) if m2 else float("nan")

    sf = structure_factor(spins)
    s0 = float(sf[0, 0])
    sk_min = float(0.5 * (sf[1, 0] + sf[0, 1]))
    ratio = max(s0 / max(sk_min, 1e-12) - 1.0, 0.0)
    xi = float(np.sqrt(ratio) / (2.0 * np.sin(np.pi / L)))

    return {
        "samples": int(len(spins)),
        "lattice_size": int(L),
        "energy_mean": float(energy.mean()),
        "energy_std": float(energy.std(ddof=1)) if len(energy) > 1 else 0.0,
        "magnetization_mean": float(mag.mean()),
        "abs_magnetization_mean": float(np.abs(mag).mean()),
        "magnetization_std": float(mag.std(ddof=1)) if len(mag) > 1 else 0.0,
        "binder_u4": binder,
        "xi_over_l": xi / L,
    }


def compare_ensembles(model_spins: np.ndarray, reference_spins: np.ndarray) -> dict:
    model = ensemble_summary(model_spins)
    reference = ensemble_summary(reference_spins)
    scalar_keys = (
        "energy_mean",
        "abs_magnetization_mean",
        "binder_u4",
        "xi_over_l",
    )
    absolute_error = {
        key: abs(float(model[key]) - float(reference[key])) for key in scalar_keys
    }
    return {
        "model": model,
        "reference": reference,
        "absolute_error": absolute_error,
    }
