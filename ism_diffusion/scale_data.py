"""Packed parent fields and matched coordinate-window sampling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ParentSplit:
    spins: np.ndarray
    chain_ids: np.ndarray
    lattice_size: int
    metadata: dict


def pack_spins(spins: np.ndarray) -> np.ndarray:
    binary = np.asarray(spins) > 0
    return np.packbits(binary, axis=-1, bitorder="little")


def unpack_spins(packed: np.ndarray, lattice_size: int) -> np.ndarray:
    binary = np.unpackbits(
        np.asarray(packed), axis=-1, count=int(lattice_size), bitorder="little"
    )
    return (2 * binary.astype(np.int8) - 1).astype(np.int8, copy=False)


def load_parent_split(path: Path, split: str) -> ParentSplit:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        lattice_size = int(metadata["lattice_size"])
        packed = data[f"{split}_packed"]
        chain_ids = data[f"{split}_chain_id"].astype(np.int16)
    spins = unpack_spins(packed, lattice_size)
    return ParentSplit(spins, chain_ids, lattice_size, metadata)


def coordinate_grid(
    batch_size: int,
    width: int,
    coordinate_stride: int,
    device: torch.device,
) -> torch.Tensor:
    axis = torch.arange(width, device=device, dtype=torch.float32)
    axis = axis * float(coordinate_stride)
    rows, columns = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack((rows, columns), dim=-1)
    return grid[None].expand(batch_size, -1, -1, -1).clone()


def sample_windows(
    parents: np.ndarray,
    width: int,
    spin_stride: int,
    coordinate_stride: int,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
    augment: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a homogeneous geometry batch from periodic parent fields."""

    if parents.ndim != 3 or parents.shape[-1] != parents.shape[-2]:
        raise ValueError("parents must have shape [samples,L,L]")
    lattice_size = parents.shape[-1]
    if (width - 1) * spin_stride >= lattice_size // 2:
        raise ValueError("window span must stay below half the parent size")
    parent_indices = rng.integers(len(parents), size=batch_size)
    origins_x = rng.integers(lattice_size, size=batch_size)
    origins_y = rng.integers(lattice_size, size=batch_size)
    offsets = spin_stride * np.arange(width, dtype=np.int64)
    batch = np.empty((batch_size, width, width), dtype=np.int64)
    for row in range(batch_size):
        x = (origins_x[row] + offsets) % lattice_size
        y = (origins_y[row] + offsets) % lattice_size
        batch[row] = (parents[parent_indices[row]][np.ix_(x, y)] > 0).astype(
            np.int64
        )
    tokens = torch.from_numpy(batch).to(device=device, non_blocking=True)
    coordinates = coordinate_grid(
        batch_size, width, coordinate_stride, device=device
    )
    valid_mask = torch.ones_like(tokens, dtype=torch.bool)
    if augment:
        tokens, coordinates, valid_mask = augment_geometry(
            tokens, coordinates, valid_mask, rng
        )
    return tokens, coordinates, valid_mask


def augment_geometry(
    tokens: torch.Tensor,
    coordinates: torch.Tensor,
    valid_mask: torch.Tensor,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply matched D4 and exact global spin-flip augmentation."""

    out_tokens = torch.empty_like(tokens)
    out_coordinates = torch.empty_like(coordinates)
    out_valid = torch.empty_like(valid_mask)
    transforms = rng.integers(8, size=len(tokens))
    flips = rng.random(len(tokens)) < 0.5
    for index, transform in enumerate(transforms):
        token = tokens[index]
        coordinate = coordinates[index]
        valid = valid_mask[index]
        rotation = int(transform % 4)
        if rotation:
            token = torch.rot90(token, rotation, dims=(-2, -1))
            coordinate = torch.rot90(coordinate, rotation, dims=(0, 1))
            valid = torch.rot90(valid, rotation, dims=(-2, -1))
        if transform >= 4:
            token = torch.flip(token, dims=(-1,))
            coordinate = torch.flip(coordinate, dims=(1,))
            valid = torch.flip(valid, dims=(-1,))
        if flips[index]:
            token = 1 - token
        out_tokens[index] = token
        out_coordinates[index] = coordinate
        out_valid[index] = valid
    return out_tokens, out_coordinates, out_valid


def variant_geometry(
    variant: str,
    widths: list[int],
    strides: list[int],
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    """Return width, spin stride, coordinate stride for a Level-1 group."""

    if variant == "T0":
        return max(widths), 1, 1
    width = int(widths[int(rng.integers(len(widths)))])
    stride = int(strides[int(rng.integers(len(strides)))])
    if variant == "T3":
        return width, stride, stride
    if variant == "Pphase":
        return width, 1, stride
    if variant == "Punit":
        return width, stride, 1
    raise ValueError(f"unknown Level-1 variant: {variant}")
