"""Coordinate-aware dense denoiser for the Level-1 scale experiment."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import timestep_embedding


@dataclass
class CoordinateDenoiserConfig:
    d_model: int = 128
    n_heads: int = 4
    n_blocks: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    vocab_size: int = 4
    output_classes: int = 2
    rope_base: float = 10_000.0

    def validate(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        head_dim = self.d_model // self.n_heads
        if head_dim % 4:
            raise ValueError("head_dim must be divisible by four for 2D RoPE")
        if self.n_blocks < 1:
            raise ValueError("n_blocks must be positive")
        if self.vocab_size != 4 or self.output_classes != 2:
            raise ValueError("expected spin-, spin+, MASK, PAD and two outputs")

    def to_dict(self) -> dict:
        return asdict(self)


class AdaptiveLayerNorm1D(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.affine = nn.Linear(width, 2 * width)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        scale, shift = self.affine(time).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale[:, None, :]) + shift[:, None, :]


def _apply_axis_rope(
    values: torch.Tensor,
    positions: torch.Tensor,
    base: float,
) -> torch.Tensor:
    """Apply rotary encoding to ``[B,H,N,D_axis]`` values."""

    axis_dim = values.shape[-1]
    if axis_dim % 2:
        raise ValueError("RoPE axis dimension must be even")
    frequencies = torch.exp(
        -math.log(float(base))
        * torch.arange(0, axis_dim, 2, device=values.device, dtype=torch.float32)
        / axis_dim
    )
    angles = positions.float()[:, None, :, None] * frequencies[None, None, None, :]
    cosine = torch.cos(angles).to(values.dtype)
    sine = torch.sin(angles).to(values.dtype)
    even = values[..., 0::2]
    odd = values[..., 1::2]
    rotated = torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    )
    return rotated.flatten(-2)


def apply_2d_rope(
    values: torch.Tensor,
    coordinates: torch.Tensor,
    base: float,
) -> torch.Tensor:
    """Apply independent x/y RoPE using external physical coordinates."""

    half = values.shape[-1] // 2
    x_part = _apply_axis_rope(values[..., :half], coordinates[..., 0], base)
    y_part = _apply_axis_rope(values[..., half:], coordinates[..., 1], base)
    return torch.cat((x_part, y_part), dim=-1)


class CoordinateDenseAttention(nn.Module):
    def __init__(self, config: CoordinateDenoiserConfig):
        super().__init__()
        self.width = config.d_model
        self.heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = float(config.dropout)
        self.rope_base = float(config.rope_base)
        self.qkv = nn.Linear(self.width, 3 * self.width, bias=False)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.proj = nn.Linear(self.width, self.width, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        qkv = self.qkv(x).view(
            batch, length, 3, self.heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_2d_rope(q, coordinates, self.rope_base)
        k = apply_2d_rope(k, coordinates, self.rope_base)

        key_bias = torch.zeros(
            (batch, 1, 1, length), device=x.device, dtype=q.dtype
        )
        key_bias.masked_fill_(~valid_mask[:, None, None, :], -torch.inf)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=key_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, self.width)
        attended = self.proj(attended)
        return attended * valid_mask[:, :, None].to(attended.dtype)


class CoordinateDenseBlock(nn.Module):
    def __init__(self, config: CoordinateDenoiserConfig):
        super().__init__()
        width = config.d_model
        self.attn_norm = AdaptiveLayerNorm1D(width)
        self.mlp_norm = AdaptiveLayerNorm1D(width)
        self.attention = CoordinateDenseAttention(config)
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


class CoordinateDenseDenoiser(nn.Module):
    """Predict clean Ising spins on a coordinate-indexed valid set."""

    def __init__(self, config: CoordinateDenoiserConfig):
        super().__init__()
        config.validate()
        self.config = config
        width = config.d_model
        self.token_embedding = nn.Embedding(config.vocab_size, width)
        self.time_mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.SiLU(),
            nn.Linear(4 * width, width),
        )
        self.blocks = nn.ModuleList(
            [CoordinateDenseBlock(config) for _ in range(config.n_blocks)]
        )
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, config.output_classes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch, height, width]")
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

        flat_valid = valid_mask.reshape(batch, height * width)
        flat_coordinates = coordinates.reshape(batch, height * width, 2)
        x = self.token_embedding(tokens).reshape(batch, height * width, -1)
        time = self.time_mlp(timestep_embedding(t, self.config.d_model))
        x = (x + time[:, None, :]) * flat_valid[:, :, None].to(x.dtype)
        for block in self.blocks:
            x = block(x, time, flat_coordinates, flat_valid)
        logits = self.output(F.silu(self.output_norm(x)))
        logits = logits.reshape(batch, height, width, 2).permute(0, 3, 1, 2)
        return logits.masked_fill(~valid_mask[:, None, :, :], 0.0)
