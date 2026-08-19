"""Local--global coordinate denoiser for the Stage-2 scale experiment.

The local path is physically radius limited.  The global path retains dense
coordinate-aware attention.  Their hidden states never mix; only a gated
global logit residual is added to the local logits.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import timestep_embedding
from .scale_model import (
    AdaptiveLayerNorm1D,
    CoordinateDenseBlock,
    CoordinateDenoiserConfig,
    apply_2d_rope,
)


@dataclass
class LocalGlobalDenoiserConfig:
    d_local: int = 64
    local_heads: int = 4
    local_blocks: int = 4
    d_global: int = 128
    global_heads: int = 4
    global_blocks: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    vocab_size: int = 4
    output_classes: int = 2
    rope_base: float = 10_000.0
    physical_radius: float = 1.0
    gate_hidden: int = 128
    gate_initial: float = 0.1
    hard_markov_gate: bool = False

    def validate(self) -> None:
        for width, heads, name in (
            (self.d_local, self.local_heads, "local"),
            (self.d_global, self.global_heads, "global"),
        ):
            if width % heads:
                raise ValueError(f"{name} width must be divisible by heads")
            if (width // heads) % 4:
                raise ValueError(f"{name} head dimension must be divisible by four")
        if self.local_blocks < 1 or self.global_blocks < 1:
            raise ValueError("local_blocks and global_blocks must be positive")
        if self.physical_radius < 0:
            raise ValueError("physical_radius must be nonnegative")
        if not 0.0 < self.gate_initial < 1.0:
            raise ValueError("gate_initial must lie strictly between zero and one")
        if self.vocab_size != 4 or self.output_classes != 2:
            raise ValueError("expected spin-, spin+, MASK, PAD and two outputs")

    def to_dict(self) -> dict:
        return asdict(self)


class LocalPhysicalAttention(nn.Module):
    """Dense kernel with a hard mask defined in physical coordinate units."""

    def __init__(self, config: LocalGlobalDenoiserConfig):
        super().__init__()
        self.width = int(config.d_local)
        self.heads = int(config.local_heads)
        self.head_dim = self.width // self.heads
        self.dropout = float(config.dropout)
        self.rope_base = float(config.rope_base)
        self.radius = float(config.physical_radius)
        self.qkv = nn.Linear(self.width, 3 * self.width, bias=False)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.proj = nn.Linear(self.width, self.width, bias=False)

    def _attention_bias(
        self,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        x = coordinates[..., 0]
        y = coordinates[..., 1]
        distance = (x[:, :, None] - x[:, None, :]).abs()
        distance += (y[:, :, None] - y[:, None, :]).abs()
        allowed = (distance <= self.radius + 1e-6) & valid_mask[:, None, :]
        # Invalid queries are discarded after attention, but SDPA must still
        # see at least one finite key to avoid producing NaNs for those rows.
        eye = torch.eye(
            coordinates.shape[1], device=coordinates.device, dtype=torch.bool
        )
        allowed |= (~valid_mask[:, :, None]) & eye[None]
        bias = torch.zeros(
            (coordinates.shape[0], 1, coordinates.shape[1], coordinates.shape[1]),
            device=coordinates.device,
            dtype=dtype,
        )
        return bias.masked_fill(~allowed[:, None], -torch.inf)

    def forward(
        self,
        x: torch.Tensor,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_2d_rope(q, coordinates, self.rope_base)
        k = apply_2d_rope(k, coordinates, self.rope_base)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self._attention_bias(coordinates, valid_mask, q.dtype),
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, self.width)
        attended = self.proj(attended)
        return attended * valid_mask[:, :, None].to(attended.dtype)


class LocalPhysicalBlock(nn.Module):
    def __init__(self, config: LocalGlobalDenoiserConfig):
        super().__init__()
        width = int(config.d_local)
        self.attn_norm = AdaptiveLayerNorm1D(width)
        self.mlp_norm = AdaptiveLayerNorm1D(width)
        self.attention = LocalPhysicalAttention(config)
        hidden = int(width * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, width),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.dropout(
            self.attention(self.attn_norm(x, time), coordinates, valid_mask)
        )
        x = x + self.dropout(self.mlp(self.mlp_norm(x, time)))
        return x * valid_mask[:, :, None].to(x.dtype)


class LocalPointwiseBlock(nn.Module):
    """Capacity without spatial propagation, preserving total local radius."""

    def __init__(self, config: LocalGlobalDenoiserConfig):
        super().__init__()
        width = int(config.d_local)
        self.mlp_norm = AdaptiveLayerNorm1D(width)
        hidden = int(width * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, width),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        del coordinates
        x = x + self.dropout(self.mlp(self.mlp_norm(x, time)))
        return x * valid_mask[:, :, None].to(x.dtype)


class _CoordinateBranch(nn.Module):
    def __init__(
        self,
        width: int,
        blocks: nn.ModuleList,
        vocab_size: int,
    ):
        super().__init__()
        self.width = int(width)
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.time_mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.SiLU(),
            nn.Linear(4 * width, width),
        )
        self.blocks = blocks
        self.output_norm = nn.LayerNorm(width)

    def forward(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = tokens.shape[0]
        flat_valid = valid_mask.flatten(1)
        flat_coordinates = coordinates.reshape(batch, -1, 2)
        hidden = self.token_embedding(tokens).reshape(batch, -1, self.width)
        time = self.time_mlp(timestep_embedding(t, self.width))
        hidden = (hidden + time[:, None]) * flat_valid[:, :, None].to(hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, time, flat_coordinates, flat_valid)
        return F.silu(self.output_norm(hidden)), time


class LocalGlobalScaleDenoiser(nn.Module):
    """Local base predictor plus a gated dense global logit residual."""

    def __init__(self, config: LocalGlobalDenoiserConfig):
        super().__init__()
        config.validate()
        self.config = config

        local_blocks = nn.ModuleList(
            [LocalPhysicalBlock(config)]
            + [LocalPointwiseBlock(config) for _ in range(config.local_blocks - 1)]
        )
        global_config = CoordinateDenoiserConfig(
            d_model=config.d_global,
            n_heads=config.global_heads,
            n_blocks=config.global_blocks,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            vocab_size=config.vocab_size,
            output_classes=config.output_classes,
            rope_base=config.rope_base,
        )
        global_blocks = nn.ModuleList(
            [CoordinateDenseBlock(global_config) for _ in range(config.global_blocks)]
        )
        self.local = _CoordinateBranch(
            config.d_local, local_blocks, config.vocab_size
        )
        self.global_branch = _CoordinateBranch(
            config.d_global, global_blocks, config.vocab_size
        )
        self.local_output = nn.Linear(config.d_local, config.output_classes)
        self.global_residual = nn.Linear(config.d_global, config.output_classes)
        nn.init.zeros_(self.local_output.weight)
        nn.init.zeros_(self.local_output.bias)
        nn.init.zeros_(self.global_residual.weight)
        nn.init.zeros_(self.global_residual.bias)

        gate_input = config.d_local + 2 * config.d_global
        self.gate = nn.Sequential(
            nn.Linear(gate_input, config.gate_hidden),
            nn.SiLU(),
            nn.Linear(config.gate_hidden, 1),
        )
        final_gate = self.gate[-1]
        nn.init.zeros_(final_gate.weight)
        initial_logit = math.log(config.gate_initial / (1.0 - config.gate_initial))
        nn.init.constant_(final_gate.bias, initial_logit)

    @staticmethod
    def _validate_inputs(
        tokens: torch.Tensor,
        t: torch.Tensor,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch,height,width]")
        batch, height, width = tokens.shape
        if t.shape != (batch,):
            raise ValueError("t must have shape [batch]")
        if coordinates.shape != (batch, height, width, 2):
            raise ValueError("coordinates must have shape [batch,H,W,2]")
        if valid_mask is None:
            valid_mask = torch.ones_like(tokens, dtype=torch.bool)
        if valid_mask.shape != tokens.shape:
            raise ValueError("valid_mask must match tokens")
        if not bool(valid_mask.any(dim=(1, 2)).all()):
            raise ValueError("each sample must contain at least one valid token")
        return valid_mask

    def _apply_hard_markov_gate(
        self,
        gate: torch.Tensor,
        tokens: torch.Tensor,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch = tokens.shape[0]
        coords = coordinates.reshape(batch, -1, 2)
        dx = (coords[:, :, None, 0] - coords[:, None, :, 0]).abs()
        dy = (coords[:, :, None, 1] - coords[:, None, :, 1]).abs()
        nearest = ((dx + dy - 1.0).abs() <= 1e-6) & valid_mask.flatten(1)[:, None]
        visible = tokens.flatten(1).lt(2)
        neighbour_count = nearest.sum(dim=-1)
        visible_count = (nearest & visible[:, None]).sum(dim=-1)
        exact_local = (neighbour_count == 4) & (visible_count == 4)
        return gate.masked_fill(exact_local[:, :, None], 0.0)

    def forward(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ):
        valid_mask = self._validate_inputs(tokens, t, coordinates, valid_mask)
        batch, height, width = tokens.shape
        local_hidden, _ = self.local(tokens, t, coordinates, valid_mask)
        global_hidden, global_time = self.global_branch(
            tokens, t, coordinates, valid_mask
        )
        gate_time = global_time[:, None].expand(-1, global_hidden.shape[1], -1)
        gate = torch.sigmoid(
            self.gate(torch.cat((local_hidden, global_hidden, gate_time), dim=-1))
        )
        if self.config.hard_markov_gate:
            gate = self._apply_hard_markov_gate(
                gate, tokens, coordinates, valid_mask
            )
        local_logits = self.local_output(local_hidden)
        global_logits = self.global_residual(global_hidden)
        flat_logits = local_logits + gate * global_logits
        flat_valid = valid_mask.flatten(1)
        flat_logits = flat_logits * flat_valid[:, :, None].to(flat_logits.dtype)
        logits = flat_logits.reshape(batch, height, width, -1).permute(0, 3, 1, 2)
        if not return_diagnostics:
            return logits
        diagnostics = {
            "gate": gate.reshape(batch, height, width),
            "local_logits": local_logits.reshape(batch, height, width, -1).permute(
                0, 3, 1, 2
            ),
            "global_residual": global_logits.reshape(
                batch, height, width, -1
            ).permute(0, 3, 1, 2),
        }
        return logits, diagnostics
