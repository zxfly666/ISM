"""FP16 inference candidate after the three BF16 revisions failed validation."""
from contextlib import contextmanager
from types import MethodType

import torch

from precision_adapter import precision_mode


@contextmanager
def fp16_mode(model):
    original = model.forward
    with precision_mode(model, "bf16_fp32_head"):
        def forward(self, *args, **kwargs):
            device = next(self.parameters()).device.type
            with torch.autocast(device_type=device, dtype=torch.float16,
                                enabled=device == "cuda"):
                return original(*args, **kwargs)
        model.forward = MethodType(forward, model)
        try:
            yield model
        finally:
            model.forward = original
