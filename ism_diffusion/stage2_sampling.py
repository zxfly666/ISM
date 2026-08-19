"""Frozen sampler candidates for Stage 0 and subsequent Stage-2 evaluation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .scale_diffusion import CoordinateAbsorbingDiffusion


def _categorical_sample(
    probabilities: torch.Tensor,
    shape: torch.Size,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.multinomial(
        probabilities.permute(0, 2, 3, 1).reshape(-1, 2),
        num_samples=1,
        generator=generator,
    ).reshape(shape)


@torch.no_grad()
def confidence_decode(
    model,
    coordinates: torch.Tensor,
    valid_mask: torch.Tensor,
    steps: int = 64,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
    mask_id: int = 2,
    pad_id: int = 3,
    t_min: float = 0.01,
    return_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[dict]]:
    """MaskGIT-style decoding that reveals highest-confidence masked sites."""

    if steps < 1 or temperature <= 0:
        raise ValueError("steps and temperature must be positive")
    device = next(model.parameters()).device
    coordinates = coordinates.to(device)
    valid_mask = valid_mask.to(device=device, dtype=torch.bool)
    shape = valid_mask.shape
    tokens = torch.full(shape, mask_id, dtype=torch.long, device=device)
    tokens.masked_fill_(~valid_mask, pad_id)
    counts = valid_mask.sum(dim=(1, 2))
    trace: list[dict] = []

    for step in range(steps):
        current_mask = tokens.eq(mask_id) & valid_mask
        if not bool(current_mask.any()):
            break
        t_value = math.cos(0.5 * math.pi * step / steps) ** 2
        next_fraction = math.cos(0.5 * math.pi * (step + 1) / steps) ** 2
        model_t = torch.full(
            (shape[0],), max(t_value, t_min), dtype=torch.float32, device=device
        )
        logits = model(tokens, model_t, coordinates, valid_mask)
        probabilities = F.softmax(logits.float() / temperature, dim=1)
        sampled = _categorical_sample(probabilities, shape, generator)
        confidence = probabilities.gather(1, sampled[:, None]).squeeze(1)
        candidate = torch.where(current_mask, sampled, tokens)

        next_tokens = candidate.clone()
        target_mask_counts = torch.round(counts.float() * next_fraction).long()
        for row in range(shape[0]):
            masked_flat = torch.nonzero(
                current_mask[row].flatten(), as_tuple=False
            ).squeeze(1)
            target = min(int(target_mask_counts[row]), int(masked_flat.numel()))
            if target <= 0:
                continue
            masked_confidence = confidence[row].flatten()[masked_flat]
            keep_mask_indices = masked_flat[
                torch.topk(masked_confidence, k=target, largest=False).indices
            ]
            next_tokens[row].flatten()[keep_mask_indices] = mask_id
        tokens = next_tokens.masked_fill(~valid_mask, pad_id)
        if return_trace:
            trace.append(
                {
                    "step": step + 1,
                    "model_t": float(model_t[0]),
                    "masked_fraction": float(
                        (tokens.eq(mask_id) & valid_mask).sum().item()
                        / counts.sum().item()
                    ),
                }
            )

    remaining = tokens.eq(mask_id) & valid_mask
    if bool(remaining.any()):
        model_t = remaining.sum(dim=(1, 2)).float() / counts.clamp_min(1)
        logits = model(
            tokens, model_t.clamp_min(t_min), coordinates, valid_mask
        )
        sampled = _categorical_sample(
            F.softmax(logits.float() / temperature, dim=1), shape, generator
        )
        tokens = torch.where(remaining, sampled, tokens)
    return (tokens, trace) if return_trace else tokens


@torch.no_grad()
def checkerboard_refine(
    model,
    tokens: torch.Tensor,
    coordinates: torch.Tensor,
    valid_mask: torch.Tensor,
    sweeps: int,
    temperature: float,
    generator: torch.Generator | None,
    mask_id: int = 2,
    pad_id: int = 3,
    t_min: float = 0.01,
) -> torch.Tensor:
    """Blocked model pseudo-Gibbs correction on the two checkerboard colours."""

    if sweeps < 0:
        raise ValueError("sweeps must be nonnegative")
    device = next(model.parameters()).device
    tokens = tokens.to(device).clone()
    coordinates = coordinates.to(device)
    valid_mask = valid_mask.to(device=device, dtype=torch.bool)
    height, width = tokens.shape[-2:]
    rows, columns = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    colour = (rows + columns) % 2
    valid_counts = valid_mask.sum(dim=(1, 2)).clamp_min(1)
    for _ in range(sweeps):
        for selected_colour in (0, 1):
            selected = valid_mask & colour.eq(selected_colour)[None]
            noisy = tokens.masked_fill(selected, mask_id).masked_fill(
                ~valid_mask, pad_id
            )
            model_t = selected.sum(dim=(1, 2)).float() / valid_counts
            logits = model(
                noisy, model_t.clamp_min(t_min), coordinates, valid_mask
            )
            sampled = _categorical_sample(
                F.softmax(logits.float() / temperature, dim=1),
                tokens.shape,
                generator,
            )
            tokens = torch.where(selected, sampled, tokens)
    return tokens


@torch.no_grad()
def sample_with_method(
    model,
    coordinates: torch.Tensor,
    valid_mask: torch.Tensor,
    method: str,
    steps: int,
    temperature: float,
    generator: torch.Generator | None,
    refinement_sweeps: int = 2,
) -> torch.Tensor:
    """Dispatch a pre-registered Stage-0 sampler candidate."""

    normalized = method.lower().replace("-", "_")
    if normalized in {"reveal", "irreversible_reveal", "s0"}:
        diffusion = CoordinateAbsorbingDiffusion()
        return diffusion.sample(
            model,
            coordinates,
            valid_mask,
            steps=steps,
            temperature=temperature,
            generator=generator,
        )
    if normalized in {"confidence", "confidence_remask", "s1", "s2"}:
        tokens = confidence_decode(
            model,
            coordinates,
            valid_mask,
            steps=steps,
            temperature=temperature,
            generator=generator,
        )
        if normalized in {"s2"}:
            tokens = checkerboard_refine(
                model,
                tokens,
                coordinates,
                valid_mask,
                sweeps=refinement_sweeps,
                temperature=temperature,
                generator=generator,
            )
        return tokens
    raise ValueError(f"unknown sampling method: {method}")
