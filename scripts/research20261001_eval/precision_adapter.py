"""Evaluation-only arithmetic revisions; never alter saved model parameters.

Candidate order is fixed before validation. Formal conditional predictions retain
the unmodified FP32 model. Generation uses one common accepted mode for every arm.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch
import torch.nn.functional as F

CANDIDATES = ("bf16_fp32_head", "bf16_fp32_conditioning_head",
              "bf16_fp32_conditioning_rope_head")


def _fp32(x):
    if torch.is_tensor(x):
        return x.float() if x.is_floating_point() else x
    if isinstance(x, tuple):
        return tuple(_fp32(v) for v in x)
    if isinstance(x, list):
        return [_fp32(v) for v in x]
    if isinstance(x, dict):
        return {k: _fp32(v) for k, v in x.items()}
    return x


def _attention(self, x, coordinates, valid_mask):
    from ism_diffusion.scale_model import apply_2d_rope
    batch, length, _ = x.shape
    qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim)
    q, k, v = qkv.unbind(dim=2)
    v = v.transpose(1, 2)
    with torch.autocast(device_type=x.device.type, enabled=False):
        q = self.q_norm(q.float()).transpose(1, 2)
        k = self.k_norm(k.float()).transpose(1, 2)
        q = apply_2d_rope(q, coordinates, self.rope_base)
        k = apply_2d_rope(k, coordinates, self.rope_base)
    q, k = q.to(v.dtype), k.to(v.dtype)
    bias = torch.zeros((batch, 1, 1, length), device=x.device, dtype=q.dtype)
    bias.masked_fill_(~valid_mask[:, None, None, :], -torch.inf)
    attended = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                              dropout_p=0.)
    attended = attended.transpose(1, 2).reshape(batch, length, self.width)
    attended = self.proj(attended)
    return attended * valid_mask[:, :, None].to(attended.dtype)


@contextmanager
def precision_mode(model, mode):
    """Scoped instance-forward adapters; state_dict and original files unchanged."""
    if mode not in (*CANDIDATES, "fp32", "bf16_original"):
        raise ValueError(mode)
    changes = []

    def full(module):
        original = module.forward

        def forward(this, *args, **kwargs):
            with torch.autocast(device_type=next(this.parameters()).device.type,
                                enabled=False):
                return original(*_fp32(args), **_fp32(kwargs))

        changes.append((module, original))
        module.forward = MethodType(forward, module)

    try:
        if mode in CANDIDATES:
            full(model.output)
        if mode in CANDIDATES[1:]:
            full(model.time_mlp)
            for block in model.blocks:
                full(block.attn_norm.affine)
                full(block.mlp_norm.affine)
        if mode == CANDIDATES[2]:
            for block in model.blocks:
                changes.append((block.attention, block.attention.forward))
                block.attention.forward = MethodType(_attention, block.attention)
        yield model
    finally:
        for module, forward in reversed(changes):
            module.forward = forward
