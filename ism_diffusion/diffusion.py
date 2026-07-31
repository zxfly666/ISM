"""Absorbing-state discrete diffusion objective and iterative sampler."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass
class DiffusionLoss:
    loss: torch.Tensor
    masked_ce: torch.Tensor
    mask_fraction: torch.Tensor
    mean_t: torch.Tensor


class AbsorbingDiffusion:
    """Masked diffusion over binary Ising spins.

    Clean tokens are 0/1 and ``mask_id=2`` is the absorbing state. For a
    uniformly sampled continuous noise level t, each site is masked with
    probability t. The MDLM estimator is

        mean_over_sites[ 1(masked) * CE / t ].

    An optional atom at t=1 trains the exact all-mask endpoint needed for
    unconditional generation. It is best viewed as an endpoint regularizer in
    addition to the continuous-time NELBO.
    """

    def __init__(
        self,
        mask_id: int = 2,
        t_min: float = 0.01,
        t_max: float = 1.0,
        full_mask_probability: float = 0.05,
    ):
        if not (0.0 < t_min < t_max <= 1.0):
            raise ValueError("require 0 < t_min < t_max <= 1")
        if not (0.0 <= full_mask_probability < 1.0):
            raise ValueError("full_mask_probability must be in [0, 1)")
        self.mask_id = int(mask_id)
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
            endpoint = (
                torch.rand(batch_size, device=device, generator=generator)
                < self.full_mask_probability
            )
            t = torch.where(endpoint, torch.ones_like(t), t)
        return t

    def corrupt(
        self,
        clean: torch.Tensor,
        t: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if clean.ndim != 3:
            raise ValueError("clean tokens must have shape [batch, height, width]")
        if t.shape != (clean.shape[0],):
            raise ValueError("t must have shape [batch]")
        masked = torch.rand(
            clean.shape, device=clean.device, generator=generator
        ) < t[:, None, None]
        noisy = clean.masked_fill(masked, self.mask_id)
        return noisy, masked

    def training_loss(
        self,
        model,
        clean: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> DiffusionLoss:
        t = self.draw_t(clean.shape[0], clean.device, generator)
        noisy, masked = self.corrupt(clean, t, generator)
        logits = model(noisy, t)
        ce = F.cross_entropy(logits.float(), clean, reduction="none")
        weights = masked.to(ce.dtype) / t[:, None, None]
        loss = (ce * weights).sum() / clean.numel()
        masked_count = masked.sum().clamp_min(1)
        return DiffusionLoss(
            loss=loss,
            masked_ce=(ce * masked).sum() / masked_count,
            mask_fraction=masked.float().mean(),
            mean_t=t.mean(),
        )

    @torch.no_grad()
    def validation_nelbo(
        self,
        model,
        clean: torch.Tensor,
        t_grid: Iterable[float],
        seed: int = 0,
        batch_size: int = 32,
    ) -> tuple[float, dict[float, float]]:
        """Deterministic fixed-grid estimate in nats per spin."""

        per_t: dict[float, float] = {}
        device = clean.device
        for grid_index, t_value in enumerate(t_grid):
            t_value = float(t_value)
            generator = torch.Generator(device=device).manual_seed(
                int(seed + 1009 * grid_index)
            )
            total = 0.0
            count = 0
            for start in range(0, len(clean), batch_size):
                rows = clean[start : start + batch_size]
                t = torch.full(
                    (len(rows),), t_value, device=device, dtype=torch.float32
                )
                noisy, masked = self.corrupt(rows, t, generator)
                if not bool(masked.any()):
                    continue
                logits = model(noisy, t)
                ce = F.cross_entropy(logits.float(), rows, reduction="none")
                total += float(ce[masked].sum().item())
                count += int(masked.sum().item())
            per_t[t_value] = total / max(count, 1)
        return sum(per_t.values()) / max(len(per_t), 1), per_t

    @torch.no_grad()
    def sample(
        self,
        model,
        shape: tuple[int, int, int],
        steps: int = 32,
        temperature: float = 1.0,
        selection_noise: float = 0.05,
        method: str = "ancestral",
        corrector_steps: int = 4,
        corrector_mask_ratio: float = 0.1,
        generator: torch.Generator | None = None,
        known_tokens: torch.Tensor | None = None,
        known_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from all mask or complete an inpainting mask.

        ``ancestral`` follows the exact reverse reveal probabilities implied by
        the absorbing forward process. ``ancestral_corrector`` then performs
        blocked conditional resampling, allowing early decisions to be revised.
        ``confidence_corrector`` adds early stochastic backtracking to the
        MaskGIT-style decoder; ``confidence`` retains that decoder as a baseline.
        """

        if steps < 1:
            raise ValueError("steps must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if method not in {
            "ancestral",
            "ancestral_corrector",
            "confidence",
            "confidence_corrector",
        }:
            raise ValueError(
                "method must be ancestral, ancestral_corrector, "
                "confidence, or confidence_corrector"
            )
        if corrector_steps < 0:
            raise ValueError("corrector_steps must be non-negative")
        if not (0.0 <= corrector_mask_ratio <= 1.0):
            raise ValueError("corrector_mask_ratio must be in [0, 1]")
        if method in {"ancestral", "ancestral_corrector"}:
            return self._sample_ancestral(
                model=model,
                shape=shape,
                steps=steps,
                temperature=temperature,
                corrector_steps=(
                    corrector_steps if method == "ancestral_corrector" else 0
                ),
                corrector_mask_ratio=corrector_mask_ratio,
                generator=generator,
                known_tokens=known_tokens,
                known_mask=known_mask,
            )

        if selection_noise < 0:
            raise ValueError("selection_noise must be non-negative")
        batch, height, width = shape
        device = next(model.parameters()).device
        tokens = torch.full(
            shape, self.mask_id, dtype=torch.long, device=device
        )
        if known_tokens is not None or known_mask is not None:
            if known_tokens is None or known_mask is None:
                raise ValueError("known_tokens and known_mask must be provided together")
            known_tokens = known_tokens.to(device=device, dtype=torch.long)
            known_mask = known_mask.to(device=device, dtype=torch.bool)
            if known_tokens.shape != shape or known_mask.shape != shape:
                raise ValueError("known tensors must match sample shape")
            tokens = torch.where(known_mask, known_tokens, tokens)
        else:
            known_mask = torch.zeros(shape, dtype=torch.bool, device=device)

        initial_unknown = (~known_mask).sum(dim=(1, 2))
        for step in range(steps):
            # Revisit a small random subset of early commitments. The following
            # confidence update decides which revised proposals are safe to
            # recommit, while later iterations anneal back to ordinary decoding.
            if (
                method == "confidence_corrector"
                and 0 < step <= corrector_steps
                and corrector_mask_ratio > 0
            ):
                editable_visible = tokens.ne(self.mask_id) & ~known_mask
                anneal = 1.0 - float(step) / steps
                remask = (
                    torch.rand(shape, device=device, generator=generator)
                    < corrector_mask_ratio * anneal
                ) & editable_visible
                tokens = tokens.masked_fill(remask, self.mask_id)

            current_mask = tokens.eq(self.mask_id)
            if not bool(current_mask.any()):
                break
            mask_fraction = current_mask.float().mean(dim=(1, 2))
            logits = model(tokens, mask_fraction.clamp_min(self.t_min))
            probabilities = F.softmax(logits.float() / temperature, dim=1)
            flat_prob = probabilities.permute(0, 2, 3, 1).reshape(-1, 2)
            sampled = torch.multinomial(
                flat_prob, num_samples=1, generator=generator
            ).reshape(batch, height, width)
            confidence = probabilities.gather(1, sampled[:, None]).squeeze(1)
            # At the all-mask endpoint every site is translation-equivalent, so
            # deterministic top-k would resolve confidence ties by flat index and
            # amplify tiny class biases. Gumbel perturbations make early commits
            # spatially random; their scale vanishes as the mask fraction falls.
            if selection_noise:
                uniform = torch.rand(
                    confidence.shape, device=device, generator=generator
                ).clamp_(1e-6, 1.0 - 1e-6)
                gumbel = -torch.log(-torch.log(uniform))
                confidence = torch.log(confidence.clamp_min(1e-12))
                confidence = confidence + (
                    selection_noise
                    * mask_fraction[:, None, None]
                    * gumbel
                )
            confidence = confidence.masked_fill(~current_mask, -torch.inf)

            progress = float(step + 1) / steps
            remaining_ratio = math.cos(progress * math.pi / 2.0) ** 2
            target_remaining = torch.floor(
                initial_unknown.float() * remaining_ratio
            ).to(torch.long)

            for row in range(batch):
                current = int(current_mask[row].sum().item())
                fill = current - int(target_remaining[row].item())
                if fill <= 0:
                    continue
                candidate = confidence[row].flatten()
                selected = candidate.topk(min(fill, current)).indices
                flat_tokens = tokens[row].flatten()
                flat_sampled = sampled[row].flatten()
                flat_tokens[selected] = flat_sampled[selected]

        if bool(tokens.eq(self.mask_id).any()):
            t = tokens.eq(self.mask_id).float().mean(dim=(1, 2)).clamp_min(self.t_min)
            logits = model(tokens, t)
            final = torch.multinomial(
                F.softmax(logits.float() / temperature, dim=1)
                .permute(0, 2, 3, 1)
                .reshape(-1, 2),
                num_samples=1,
                generator=generator,
            ).reshape(shape)
            tokens = torch.where(tokens.eq(self.mask_id), final, tokens)
        return tokens

    @torch.no_grad()
    def _sample_ancestral(
        self,
        model,
        shape: tuple[int, int, int],
        steps: int,
        temperature: float,
        corrector_steps: int,
        corrector_mask_ratio: float,
        generator: torch.Generator | None,
        known_tokens: torch.Tensor | None,
        known_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, height, width = shape
        device = next(model.parameters()).device
        tokens = torch.full(shape, self.mask_id, dtype=torch.long, device=device)
        if known_tokens is not None or known_mask is not None:
            if known_tokens is None or known_mask is None:
                raise ValueError("known_tokens and known_mask must be provided together")
            known_tokens = known_tokens.to(device=device, dtype=torch.long)
            known_mask = known_mask.to(device=device, dtype=torch.bool)
            if known_tokens.shape != shape or known_mask.shape != shape:
                raise ValueError("known tensors must match sample shape")
            tokens = torch.where(known_mask, known_tokens, tokens)
        else:
            known_mask = torch.zeros(shape, dtype=torch.bool, device=device)

        # The forward process masks a clean token with probability t. Therefore,
        # conditioned on a token being masked at t, its exact probability of
        # becoming visible at the next level s < t is 1 - s/t.
        times = [
            math.cos(0.5 * math.pi * step / steps) ** 2
            for step in range(steps + 1)
        ]
        for step in range(steps):
            current_mask = tokens.eq(self.mask_id)
            if not bool(current_mask.any()):
                break
            t = times[step]
            s = times[step + 1]
            # The network was trained with the forward-process time parameter,
            # not the realized finite-lattice mask fraction.  Using the exact
            # schedule value keeps ancestral sampling aligned with its derived
            # reverse kernel; the two only coincide in expectation.
            model_t = torch.full(
                (batch,),
                max(t, self.t_min),
                device=device,
                dtype=torch.float32,
            )
            logits = model(tokens, model_t)
            probabilities = F.softmax(logits.float() / temperature, dim=1)
            sampled = torch.multinomial(
                probabilities.permute(0, 2, 3, 1).reshape(-1, 2),
                num_samples=1,
                generator=generator,
            ).reshape(batch, height, width)
            reveal_probability = min(max(1.0 - s / max(t, 1e-12), 0.0), 1.0)
            reveal = (
                torch.rand(shape, device=device, generator=generator)
                < reveal_probability
            ) & current_mask
            tokens = torch.where(reveal, sampled, tokens)

        # Guard against floating-point endpoint effects.
        remaining = tokens.eq(self.mask_id)
        if bool(remaining.any()):
            model_t = remaining.float().mean(dim=(1, 2)).clamp_min(self.t_min)
            logits = model(tokens, model_t)
            sampled = torch.multinomial(
                F.softmax(logits.float() / temperature, dim=1)
                .permute(0, 2, 3, 1)
                .reshape(-1, 2),
                num_samples=1,
                generator=generator,
            ).reshape(shape)
            tokens = torch.where(remaining, sampled, tokens)

        # Blocked denoise steps act as a lightweight corrector. Unlike
        # irreversible confidence commits, these steps may revise any generated
        # token while preserving user-supplied observations.
        editable = ~known_mask
        for _ in range(corrector_steps):
            remask = (
                torch.rand(shape, device=device, generator=generator)
                < corrector_mask_ratio
            ) & editable
            if not bool(remask.any()):
                continue
            corrupted = tokens.masked_fill(remask, self.mask_id)
            model_t = remask.float().mean(dim=(1, 2)).clamp_min(self.t_min)
            logits = model(corrupted, model_t)
            sampled = torch.multinomial(
                F.softmax(logits.float() / temperature, dim=1)
                .permute(0, 2, 3, 1)
                .reshape(-1, 2),
                num_samples=1,
                generator=generator,
            ).reshape(shape)
            tokens = torch.where(remask, sampled, tokens)
        return tokens
