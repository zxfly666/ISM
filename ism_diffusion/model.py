"""Size-flexible axial denoiser for masked 2D Ising configurations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DenoiserConfig:
    d_model: int = 256
    n_heads: int = 8
    n_blocks: int = 10
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    vocab_size: int = 3
    output_classes: int = 2

    def validate(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_blocks < 1:
            raise ValueError("n_blocks must be positive")
        if self.vocab_size != 3 or self.output_classes != 2:
            raise ValueError("Ising diffusion expects 3 input tokens and 2 outputs")

    def to_dict(self) -> dict:
        return asdict(self)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Continuous sinusoidal embedding for t in [0, 1]."""

    half = dim // 2
    if half == 0:
        return t[:, None]
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    phase = (1000.0 * t.float())[:, None] * frequencies[None, :]
    emb = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, width: int, time_width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.affine = nn.Linear(time_width, 2 * width)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        scale, shift = self.affine(time).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale[:, None, None, :]) + shift[
            :, None, None, :
        ]


class PeriodicAxialAttention(nn.Module):
    """Attention along rows or columns with a toroidal power-law distance prior.

    The learned bias is ``-slope * log(1 + cyclic_distance)``. It is translation
    equivariant on a periodic lattice, has no learned maximum position, and
    therefore remains defined when inference lattices are larger than training
    lattices.
    """

    def __init__(self, width: int, heads: int, axis: str, dropout: float):
        super().__init__()
        if axis not in {"row", "column"}:
            raise ValueError("axis must be 'row' or 'column'")
        self.width = width
        self.heads = heads
        self.axis = axis
        self.head_dim = width // heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        initial_slopes = torch.logspace(-1.0, 0.3, heads)
        self.raw_slopes = nn.Parameter(torch.log(torch.expm1(initial_slopes)))

    def _relative_bias(
        self, length: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        pos = torch.arange(length, device=device)
        direct = (pos[:, None] - pos[None, :]).abs()
        distance = torch.minimum(direct, length - direct).to(torch.float32)
        slopes = F.softplus(self.raw_slopes.float())[:, None, None]
        bias = -slopes * torch.log1p(distance)[None, :, :]
        return bias.to(dtype=dtype)[None, :, :, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        if self.axis == "row":
            sequence = x.reshape(batch * height, width, channels)
            length = width
        else:
            sequence = x.transpose(1, 2).reshape(batch * width, height, channels)
            length = height

        qkv = self.qkv(sequence).view(
            sequence.shape[0], length, 3, self.heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self._relative_bias(length, x.device, q.dtype),
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(
            sequence.shape[0], length, channels
        )
        attended = self.proj(attended)

        if self.axis == "row":
            return attended.reshape(batch, height, width, channels)
        return attended.reshape(batch, width, height, channels).transpose(1, 2)


class AxialBlock(nn.Module):
    def __init__(self, config: DenoiserConfig):
        super().__init__()
        width = config.d_model
        self.row_norm = AdaptiveLayerNorm(width, width)
        self.col_norm = AdaptiveLayerNorm(width, width)
        self.mlp_norm = AdaptiveLayerNorm(width, width)
        self.row_attention = PeriodicAxialAttention(
            width, config.n_heads, "row", config.dropout
        )
        self.col_attention = PeriodicAxialAttention(
            width, config.n_heads, "column", config.dropout
        )
        hidden = int(width * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, width),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.row_attention(self.row_norm(x, time)))
        x = x + self.dropout(self.col_attention(self.col_norm(x, time)))
        x = x + self.dropout(self.mlp(self.mlp_norm(x, time)))
        return x


class AxialDenoiser(nn.Module):
    """Predict clean {-1,+1} tokens at each spatial position."""

    def __init__(self, config: DenoiserConfig):
        super().__init__()
        config.validate()
        self.config = config
        width = config.d_model
        self.token_embedding = nn.Embedding(config.vocab_size, width)
        self.local_stem = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.SiLU(),
            nn.Linear(4 * width, width),
        )
        self.blocks = nn.ModuleList(
            [AxialBlock(config) for _ in range(config.n_blocks)]
        )
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, config.output_classes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch, height, width]")
        if t.shape != (tokens.shape[0],):
            raise ValueError("t must have shape [batch]")
        x = self.token_embedding(tokens)
        local = self.local_stem(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = x + local
        time = self.time_mlp(timestep_embedding(t, self.config.d_model))
        x = x + time[:, None, None, :]
        for block in self.blocks:
            x = block(x, time)
        return self.output(F.silu(self.output_norm(x))).permute(0, 3, 1, 2)
