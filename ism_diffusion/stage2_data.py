"""Physical-geometry samplers for Stage 2A and Stage 2B."""

from __future__ import annotations

import numpy as np
import torch

from .scale_data import augment_geometry, sample_windows


STAGE2_VARIANTS = {
    "LG-Punit",
    "LG-T3",
    "LG-U-Unit",
    "LG-U-Matched",
    "LG-Gap-Unit",
    "LG-Gap-Matched",
    "LG-U-RandPE",
}


def stage2_geometry(
    variant: str,
    widths: list[int],
    spacings: list[int],
    rng: np.random.Generator,
) -> tuple[int, str, int, str]:
    """Return ``width, geometry_mode, spacing, coordinate_mode``."""

    if variant not in STAGE2_VARIANTS:
        raise ValueError(f"unknown Stage-2 variant: {variant}")
    width = int(widths[int(rng.integers(len(widths)))])
    spacing = int(spacings[int(rng.integers(len(spacings)))])
    if variant == "LG-U-RandPE":
        mode = "random_pe"
    elif "Gap" in variant:
        mode = "random_gap"
    else:
        mode = "uniform"
    if variant.endswith(("T3", "Matched")):
        coordinates = "matched"
    elif variant.endswith("RandPE"):
        coordinates = "randomized"
    else:
        coordinates = "unit"
    return width, mode, spacing, coordinates


def _random_offsets(
    width: int,
    gaps: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if width < 1:
        raise ValueError("width must be positive")
    if width == 1:
        return np.zeros(1, dtype=np.int64)
    increments = rng.choice(gaps, size=width - 1, replace=True)
    return np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(increments)))


def sample_random_gap_windows(
    parents: np.ndarray,
    width: int,
    gaps: list[int] | np.ndarray,
    coordinate_mode: str,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
    augment: bool = True,
    fixed_gap: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a separable nonuniform grid from real integer parent sites.

    Each sample receives independently drawn x/y integer increments.  Matched
    coordinates contain those exact physical offsets; unit coordinates contain
    only the rank indices and therefore form the causal coordinate control.
    """

    parents = np.asarray(parents)
    if parents.ndim != 3 or parents.shape[-1] != parents.shape[-2]:
        raise ValueError("parents must have shape [samples,L,L]")
    if coordinate_mode not in {"matched", "unit"}:
        raise ValueError("coordinate_mode must be matched or unit")
    gap_values = np.asarray(gaps, dtype=np.int64)
    if gap_values.ndim != 1 or len(gap_values) == 0 or np.any(gap_values < 1):
        raise ValueError("gaps must be a nonempty list of positive integers")
    if fixed_gap is not None:
        if int(fixed_gap) < 1:
            raise ValueError("fixed_gap must be positive")
        gap_values = np.asarray([int(fixed_gap)], dtype=np.int64)
    lattice_size = int(parents.shape[-1])
    if (width - 1) * int(gap_values.max()) >= lattice_size // 2:
        raise ValueError("maximum random-gap span must stay below half parent size")

    parent_indices = rng.integers(len(parents), size=batch_size)
    origins_x = rng.integers(lattice_size, size=batch_size)
    origins_y = rng.integers(lattice_size, size=batch_size)
    batch = np.empty((batch_size, width, width), dtype=np.int64)
    coordinate_batch = np.empty((batch_size, width, width, 2), dtype=np.float32)
    rank = np.arange(width, dtype=np.float32)
    rank_x, rank_y = np.meshgrid(rank, rank, indexing="ij")

    for row in range(batch_size):
        offset_x = _random_offsets(width, gap_values, rng)
        offset_y = _random_offsets(width, gap_values, rng)
        x = (origins_x[row] + offset_x) % lattice_size
        y = (origins_y[row] + offset_y) % lattice_size
        batch[row] = (parents[parent_indices[row]][np.ix_(x, y)] > 0).astype(
            np.int64
        )
        if coordinate_mode == "matched":
            grid_x, grid_y = np.meshgrid(offset_x, offset_y, indexing="ij")
            coordinate_batch[row, ..., 0] = grid_x
            coordinate_batch[row, ..., 1] = grid_y
        else:
            coordinate_batch[row, ..., 0] = rank_x
            coordinate_batch[row, ..., 1] = rank_y

    tokens = torch.from_numpy(batch).to(device=device, non_blocking=True)
    coordinates = torch.from_numpy(coordinate_batch).to(
        device=device, non_blocking=True
    )
    valid_mask = torch.ones_like(tokens, dtype=torch.bool)
    if augment:
        tokens, coordinates, valid_mask = augment_geometry(
            tokens, coordinates, valid_mask, rng
        )
    return tokens, coordinates, valid_mask


def sample_random_pe_control_windows(
    parents: np.ndarray,
    width: int,
    gaps: list[int] | np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
    augment: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random-position control with coordinates independent of spin geometry.

    Spin tokens come from ordinary contiguous windows, while coordinates use
    the same random-gap distribution as :func:`sample_random_gap_windows`.
    With the same RNG seed, this function and the matched RandomGap sampler
    consume the same parent/origin/gap random stream.  The control therefore
    isolates randomized positional regularization from physically matched
    sampling.
    """

    parents = np.asarray(parents)
    if parents.ndim != 3 or parents.shape[-1] != parents.shape[-2]:
        raise ValueError("parents must have shape [samples,L,L]")
    gap_values = np.asarray(gaps, dtype=np.int64)
    if gap_values.ndim != 1 or len(gap_values) == 0 or np.any(gap_values < 1):
        raise ValueError("gaps must be a nonempty list of positive integers")
    lattice_size = int(parents.shape[-1])
    if (width - 1) * int(gap_values.max()) >= lattice_size // 2:
        raise ValueError("maximum randomized-PE span must stay below half parent size")

    parent_indices = rng.integers(len(parents), size=batch_size)
    origins_x = rng.integers(lattice_size, size=batch_size)
    origins_y = rng.integers(lattice_size, size=batch_size)
    batch = np.empty((batch_size, width, width), dtype=np.int64)
    coordinate_batch = np.empty((batch_size, width, width, 2), dtype=np.float32)
    rank = np.arange(width, dtype=np.int64)

    for row in range(batch_size):
        offset_x = _random_offsets(width, gap_values, rng)
        offset_y = _random_offsets(width, gap_values, rng)
        # Tokens deliberately ignore the randomized positional gaps.
        x = (origins_x[row] + rank) % lattice_size
        y = (origins_y[row] + rank) % lattice_size
        batch[row] = (parents[parent_indices[row]][np.ix_(x, y)] > 0).astype(
            np.int64
        )
        grid_x, grid_y = np.meshgrid(offset_x, offset_y, indexing="ij")
        coordinate_batch[row, ..., 0] = grid_x
        coordinate_batch[row, ..., 1] = grid_y

    tokens = torch.from_numpy(batch).to(device=device, non_blocking=True)
    coordinates = torch.from_numpy(coordinate_batch).to(
        device=device, non_blocking=True
    )
    valid_mask = torch.ones_like(tokens, dtype=torch.bool)
    if augment:
        tokens, coordinates, valid_mask = augment_geometry(
            tokens, coordinates, valid_mask, rng
        )
    return tokens, coordinates, valid_mask


def sample_stage2_batch(
    parents: np.ndarray,
    variant: str,
    widths: list[int],
    spacings: list[int],
    batch_sizes: dict[int, int],
    rng: np.random.Generator,
    device: torch.device,
    augment: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, int | str],
]:
    width, geometry_mode, spacing, coordinate_mode = stage2_geometry(
        variant, widths, spacings, rng
    )
    batch_size = int(batch_sizes[width])
    if geometry_mode == "uniform":
        coordinate_stride = spacing if coordinate_mode == "matched" else 1
        batch = sample_windows(
            parents,
            width=width,
            spin_stride=spacing,
            coordinate_stride=coordinate_stride,
            batch_size=batch_size,
            rng=rng,
            device=device,
            augment=augment,
        )
    elif geometry_mode == "random_gap":
        batch = sample_random_gap_windows(
            parents,
            width=width,
            gaps=spacings,
            coordinate_mode=coordinate_mode,
            batch_size=batch_size,
            rng=rng,
            device=device,
            augment=augment,
        )
    else:
        batch = sample_random_pe_control_windows(
            parents,
            width=width,
            gaps=spacings,
            batch_size=batch_size,
            rng=rng,
            device=device,
            augment=augment,
        )
    metadata: dict[str, int | str] = {
        "width": width,
        "geometry_mode": geometry_mode,
        "spacing": spacing,
        "coordinate_mode": coordinate_mode,
    }
    return (*batch, metadata)
