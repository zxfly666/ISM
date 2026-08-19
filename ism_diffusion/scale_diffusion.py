"""Coordinate-aware absorbing diffusion used by the rapid scale pilot."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass
class CoordinateDiffusionLoss:
    loss: torch.Tensor
    masked_ce: torch.Tensor
    mask_fraction: torch.Tensor
    mean_t: torch.Tensor


class CoordinateAbsorbingDiffusion:
    def __init__(
        self,
        mask_id: int = 2,
        pad_id: int = 3,
        t_min: float = 0.01,
        t_max: float = 1.0,
        full_mask_probability: float = 0.02,
    ):
        if not (0.0 < t_min < t_max <= 1.0):
            raise ValueError("require 0 < t_min < t_max <= 1")
        if not (0.0 <= full_mask_probability < 1.0):
            raise ValueError("full_mask_probability must be in [0,1)")
        self.mask_id = int(mask_id)
        self.pad_id = int(pad_id)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.full_mask_probability = float(full_mask_probability)

    def draw_t(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        t = torch.rand(batch_size, device=device, generator=generator)
        t = self.t_min + (self.t_max - self.t_min) * t
        if self.full_mask_probability:
            endpoint = torch.rand(
                batch_size, device=device, generator=generator
            ) < self.full_mask_probability
            t = torch.where(endpoint, torch.ones_like(t), t)
        return t

    def corrupt(
        self,
        clean: torch.Tensor,
        t: torch.Tensor,
        valid_mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if clean.ndim != 3 or valid_mask.shape != clean.shape:
            raise ValueError("clean and valid_mask must have shape [B,H,W]")
        random_mask = torch.rand(
            clean.shape, device=clean.device, generator=generator
        ) < t[:, None, None]
        masked = random_mask & valid_mask
        noisy = clean.masked_fill(masked, self.mask_id)
        noisy = noisy.masked_fill(~valid_mask, self.pad_id)
        return noisy, masked

    def training_loss(
        self,
        model,
        clean: torch.Tensor,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> CoordinateDiffusionLoss:
        t = self.draw_t(clean.shape[0], clean.device, generator)
        noisy, masked = self.corrupt(clean, t, valid_mask, generator)
        logits = model(noisy, t, coordinates, valid_mask)
        ce = F.cross_entropy(logits.float(), clean, reduction="none")
        weights = masked.to(ce.dtype) / t[:, None, None]
        valid_count = valid_mask.sum().clamp_min(1)
        masked_count = masked.sum().clamp_min(1)
        return CoordinateDiffusionLoss(
            loss=(ce * weights).sum() / valid_count,
            masked_ce=(ce * masked).sum() / masked_count,
            mask_fraction=masked.sum().float() / valid_count,
            mean_t=t.mean(),
        )

    @torch.no_grad()
    def validation_nelbo(
        self,
        model,
        clean: torch.Tensor,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
        t_grid: Iterable[float],
        seed: int,
        batch_size: int,
    ) -> tuple[float, dict[float, float]]:
        device = clean.device
        per_t: dict[float, float] = {}
        for grid_index, t_value in enumerate(t_grid):
            generator = torch.Generator(device=device).manual_seed(
                int(seed + 1009 * grid_index)
            )
            total = 0.0
            count = 0
            for start in range(0, len(clean), batch_size):
                rows = clean[start : start + batch_size]
                row_coordinates = coordinates[start : start + batch_size]
                row_valid = valid_mask[start : start + batch_size]
                t = torch.full(
                    (len(rows),),
                    float(t_value),
                    device=device,
                    dtype=torch.float32,
                )
                noisy, masked = self.corrupt(rows, t, row_valid, generator)
                logits = model(noisy, t, row_coordinates, row_valid)
                ce = F.cross_entropy(logits.float(), rows, reduction="none")
                total += float(ce[masked].sum().item())
                count += int(masked.sum().item())
            per_t[float(t_value)] = total / max(count, 1)
        return sum(per_t.values()) / max(len(per_t), 1), per_t

    @torch.no_grad()
    def sample(
        self,
        model,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
        steps: int = 64,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if steps < 1 or temperature <= 0.0:
            raise ValueError("steps and temperature must be positive")
        device = next(model.parameters()).device
        coordinates = coordinates.to(device)
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)
        shape = valid_mask.shape
        tokens = torch.full(shape, self.mask_id, dtype=torch.long, device=device)
        tokens.masked_fill_(~valid_mask, self.pad_id)
        batch = shape[0]
        times = [
            math.cos(0.5 * math.pi * step / steps) ** 2
            for step in range(steps + 1)
        ]
        for step in range(steps):
            current_mask = tokens.eq(self.mask_id) & valid_mask
            if not bool(current_mask.any()):
                break
            t = times[step]
            s = times[step + 1]
            model_t = torch.full(
                (batch,), max(t, self.t_min), device=device, dtype=torch.float32
            )
            logits = model(tokens, model_t, coordinates, valid_mask)
            probabilities = F.softmax(logits.float() / temperature, dim=1)
            sampled = torch.multinomial(
                probabilities.permute(0, 2, 3, 1).reshape(-1, 2),
                num_samples=1,
                generator=generator,
            ).reshape(shape)
            reveal_probability = min(max(1.0 - s / max(t, 1e-12), 0.0), 1.0)
            reveal = (
                torch.rand(shape, device=device, generator=generator)
                < reveal_probability
            ) & current_mask
            tokens = torch.where(reveal, sampled, tokens)
        remaining = tokens.eq(self.mask_id) & valid_mask
        if bool(remaining.any()):
            model_t = remaining.sum(dim=(1, 2)).float()
            model_t /= valid_mask.sum(dim=(1, 2)).clamp_min(1)
            model_t.clamp_min_(self.t_min)
            logits = model(tokens, model_t, coordinates, valid_mask)
            sampled = torch.multinomial(
                F.softmax(logits.float() / temperature, dim=1)
                .permute(0, 2, 3, 1)
                .reshape(-1, 2),
                num_samples=1,
                generator=generator,
            ).reshape(shape)
            tokens = torch.where(remaining, sampled, tokens)
        return tokens
