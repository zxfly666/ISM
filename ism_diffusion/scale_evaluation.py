"""Evaluation helpers for the Level-1 scale-aware Ising experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .scale_model import CoordinateDenseDenoiser, CoordinateDenoiserConfig


def load_coordinate_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[CoordinateDenseDenoiser, dict]:
    """Load the EMA denoiser and its complete training payload."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = CoordinateDenseDenoiser(
        CoordinateDenoiserConfig(**config["model"])
    )
    state = payload.get("ema", payload["model"])
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, payload


def load_scale_model(checkpoint_path: Path, device: torch.device):
    """Load either a Level-1 dense model or a Stage-2 local--global model."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("model_type") == "local_global" or "d_local" in payload[
        "config"
    ].get("model", {}):
        from .stage2_model import (
            LocalGlobalDenoiserConfig,
            LocalGlobalScaleDenoiser,
        )

        model = LocalGlobalScaleDenoiser(
            LocalGlobalDenoiserConfig(**payload["config"]["model"])
        )
    else:
        model = CoordinateDenseDenoiser(
            CoordinateDenoiserConfig(**payload["config"]["model"])
        )
    state = payload.get("ema", payload["model"])
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, payload


def centered_coordinate_grid(
    batch_size: int,
    width: int,
    stride: float,
    device: torch.device,
) -> torch.Tensor:
    """Create a translation-centered external-coordinate grid."""

    axis = torch.arange(width, device=device, dtype=torch.float32)
    axis -= float(width // 2)
    axis *= float(stride)
    rows, columns = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack((rows, columns), dim=-1)
    return grid[None].expand(batch_size, -1, -1, -1).clone()


def crop_periodic_windows(
    parents: np.ndarray,
    parent_indices: np.ndarray,
    origins_x: np.ndarray,
    origins_y: np.ndarray,
    width: int,
    spin_stride: int,
    centered: bool = False,
) -> np.ndarray:
    """Extract matched windows from periodic parent configurations."""

    lattice_size = int(parents.shape[-1])
    if centered:
        offsets = np.arange(width, dtype=np.int64) - width // 2
    else:
        offsets = np.arange(width, dtype=np.int64)
    offsets *= int(spin_stride)
    windows = np.empty((len(parent_indices), width, width), dtype=np.int8)
    for index, parent_index in enumerate(parent_indices):
        x = (int(origins_x[index]) + offsets) % lattice_size
        y = (int(origins_y[index]) + offsets) % lattice_size
        windows[index] = parents[int(parent_index)][np.ix_(x, y)]
    return windows


def spin_tokens(spins: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy((np.asarray(spins) > 0).astype(np.int64)).to(device)


def open_energy_density(spins: np.ndarray) -> np.ndarray:
    """Nearest-neighbour energy density without wrapping the crop boundary."""

    values = np.asarray(spins, dtype=np.float64)
    horizontal = values[:, :, :-1] * values[:, :, 1:]
    vertical = values[:, :-1, :] * values[:, 1:, :]
    bonds = horizontal.sum(axis=(1, 2)) + vertical.sum(axis=(1, 2))
    count = horizontal.shape[1] * horizontal.shape[2]
    count += vertical.shape[1] * vertical.shape[2]
    return -bonds / float(count)


def open_radial_correlation(
    spins: np.ndarray, max_radius: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw and ensemble-connected open-window radial two-point functions.

    Zero-padding removes the artificial wraparound introduced by a periodic FFT.
    Each displacement is normalized by its exact overlap count before radial
    averaging.
    """

    values = np.asarray(spins, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != values.shape[-2]:
        raise ValueError("spins must have shape [samples,L,L]")
    lattice_size = int(values.shape[-1])
    if max_radius is None:
        max_radius = lattice_size // 2
    max_radius = min(int(max_radius), lattice_size - 1)

    padded = 2 * lattice_size
    spectrum = np.fft.fft2(values, s=(padded, padded), axes=(-2, -1))
    sums = np.fft.ifft2(
        spectrum * spectrum.conj(), axes=(-2, -1)
    ).real.mean(axis=0)
    sums = np.fft.fftshift(sums)
    center = padded // 2
    offsets = np.arange(-max_radius, max_radius + 1, dtype=np.int64)
    displacement = sums[
        center - max_radius : center + max_radius + 1,
        center - max_radius : center + max_radius + 1,
    ]
    overlap_x = lattice_size - np.abs(offsets)
    overlap = overlap_x[:, None] * overlap_x[None, :]
    displacement = displacement / overlap

    dx, dy = np.meshgrid(offsets, offsets, indexing="ij")
    radius_bins = np.rint(np.sqrt(dx * dx + dy * dy)).astype(np.int64)
    radii = np.arange(max_radius + 1, dtype=np.int64)
    raw = np.empty(max_radius + 1, dtype=np.float64)
    for radius in radii:
        selected = radius_bins == radius
        raw[radius] = float(displacement[selected].mean())
    connected = raw - float(values.mean()) ** 2
    return radii, raw, connected


def low_frequency_power(spins: np.ndarray, cutoff_modes: float = 2.0) -> np.ndarray:
    """Hann-windowed low-frequency power fraction for each open crop."""

    values = np.asarray(spins, dtype=np.float64)
    lattice_size = int(values.shape[-1])
    window_1d = np.hanning(lattice_size)
    window = window_1d[:, None] * window_1d[None, :]
    centered = values - values.mean(axis=(1, 2), keepdims=True)
    spectrum = np.fft.fft2(centered * window[None], axes=(-2, -1))
    power = np.abs(spectrum) ** 2
    frequency = np.fft.fftfreq(lattice_size) * lattice_size
    fx, fy = np.meshgrid(frequency, frequency, indexing="ij")
    low = (fx * fx + fy * fy <= float(cutoff_modes) ** 2) & ~(
        (fx == 0) & (fy == 0)
    )
    return power[:, low].sum(axis=1) / np.maximum(
        power.sum(axis=(1, 2)), 1e-12
    )


def open_ensemble_metrics(
    spins: np.ndarray, max_radius: int | None = None
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(spins, dtype=np.int8)
    magnetization = values.mean(axis=(1, 2), dtype=np.float64)
    energy = open_energy_density(values)
    low_power = low_frequency_power(values)
    radii, raw, connected = open_radial_correlation(values, max_radius=max_radius)
    summary = {
        "samples": int(len(values)),
        "width": int(values.shape[-1]),
        "energy_mean": float(energy.mean()),
        "energy_std": float(energy.std(ddof=1)) if len(values) > 1 else 0.0,
        "magnetization_mean": float(magnetization.mean()),
        "abs_magnetization_mean": float(np.abs(magnetization).mean()),
        "magnetization_std": (
            float(magnetization.std(ddof=1)) if len(values) > 1 else 0.0
        ),
        "low_frequency_power_mean": float(low_power.mean()),
        "low_frequency_power_std": (
            float(low_power.std(ddof=1)) if len(values) > 1 else 0.0
        ),
    }
    return summary, radii, raw, connected


def correlation_band_errors(
    model: np.ndarray, reference: np.ndarray
) -> dict[str, dict[str, float]]:
    bands = {"short": (1, 8), "medium": (9, 16), "expanded": (17, 32)}
    output: dict[str, dict[str, float]] = {}
    for name, (start, stop) in bands.items():
        stop = min(stop, len(model) - 1, len(reference) - 1)
        if start > stop:
            continue
        delta = model[start : stop + 1] - reference[start : stop + 1]
        scale = np.sqrt(np.mean(reference[start : stop + 1] ** 2))
        output[name] = {
            "start": start,
            "stop": stop,
            "mae": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta * delta))),
            "nrmse": float(np.sqrt(np.mean(delta * delta)) / max(scale, 1e-12)),
        }
    return output


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
