"""J key-source intervention, preserving original parameter names/order/shapes.

The historical scale_model.py stays byte-for-byte unchanged. Both paths use the
original dense Q/K/V tensor shapes; observed keys are not compressed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ism_diffusion.model import timestep_embedding
from ism_diffusion.scale_model import CoordinateDenseDenoiser, CoordinateDenoiserConfig, apply_2d_rope


def allowed_keys(tokens, valid, mode):
    if tokens.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError("Boolean validity mask must match tokens")
    if mode not in ("dense", "observed_only"):
        raise ValueError(mode)
    if not bool(valid.flatten(1).any(1).all()):
        raise ValueError("Each sample needs a valid token")
    if mode == "dense":
        return valid
    observed = valid & ((tokens == 0) | (tokens == 1))
    has_observed = observed.flatten(1).any(1).reshape((-1,) + (1,)*(tokens.ndim-1))
    return torch.where(has_observed, observed, valid)


def attention(attn, x, coordinates, valid, keys):
    batch, length, _ = x.shape
    q, k, v = attn.qkv(x).view(batch, length, 3, attn.heads, attn.head_dim).unbind(2)
    q = attn.q_norm(q).transpose(1, 2)
    k = attn.k_norm(k).transpose(1, 2)
    v = v.transpose(1, 2)
    q = apply_2d_rope(q, coordinates, attn.rope_base)
    k = apply_2d_rope(k, coordinates, attn.rope_base)
    bias = torch.zeros((batch, 1, 1, length), device=x.device, dtype=q.dtype)
    bias.masked_fill_(~keys[:, None, None, :], -torch.inf)
    result = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
        dropout_p=attn.dropout if attn.training else 0.)
    result = result.transpose(1, 2).reshape(batch, length, attn.width)
    return attn.proj(result) * valid[:, :, None].to(result.dtype)


class InterventionDenoiser(CoordinateDenseDenoiser):
    """Same registered modules as the original; intervention is only key bias."""
    def __init__(self, config: CoordinateDenoiserConfig, mode: str):
        if mode not in ("dense", "observed_only"):
            raise ValueError(mode)
        super().__init__(config)
        self.attention_mode = mode  # plain string, no new parameters/buffers

    def forward(self, tokens, t, coordinates, valid_mask=None):
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [B,H,W] or [B,1,N]")
        b, h, w = tokens.shape
        if t.shape != (b,) or coordinates.shape != (b, h, w, 2):
            raise ValueError("Input time/coordinate shape mismatch")
        valid_mask = torch.ones_like(tokens, dtype=torch.bool) if valid_mask is None else valid_mask
        keys = allowed_keys(tokens, valid_mask, self.attention_mode).reshape(b, h*w)
        valid = valid_mask.reshape(b, h*w)
        coords = coordinates.reshape(b, h*w, 2)
        x = self.token_embedding(tokens).reshape(b, h*w, -1)
        time = self.time_mlp(timestep_embedding(t, self.config.d_model))
        x = (x + time[:, None, :]) * valid[:, :, None].to(x.dtype)
        for block in self.blocks:
            x = x + block.dropout(attention(block.attention, block.attn_norm(x, time), coords, valid, keys))
            x = x + block.dropout(block.mlp(block.mlp_norm(x, time)))
            x = x * valid[:, :, None].to(x.dtype)
        logits = self.output(F.silu(self.output_norm(x)))
        logits = logits.reshape(b, h, w, 2).permute(0, 3, 1, 2)
        return logits.masked_fill(~valid_mask[:, None, :, :], 0.)
