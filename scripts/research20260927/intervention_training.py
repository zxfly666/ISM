"""J initialization/continuation, losses, fixed learning rates and recovery.

This is a library, not a launcher. Importing it never starts a CUDA context,
allocates a model or creates output files. The caller owns the absolute clock.
"""
from __future__ import annotations

import copy
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ism_diffusion import geometry_study as gs
from ism_diffusion.scale_model import CoordinateDenoiserConfig
import intervention_common as c
import intervention_data as d
from intervention_model import InterventionDenoiser


def new_model(seed, mode, device):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    m = InterventionDenoiser(CoordinateDenoiserConfig(**c.MODEL), mode).to(device)
    if sum(p.numel() for p in m.parameters()) != c.CONFIG["model"]["parameters"]:
        raise RuntimeError("Unexpected parameter count")
    return m


def optimizer(m):
    return torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(.9, .95), weight_decay=.05,
                            fused=next(m.parameters()).device.type == "cuda")


def restore_rng(p, device):
    torch.set_rng_state(p["torch_rng"].cpu())
    np.random.set_state(p["numpy_rng"])
    random.setstate(p["python_rng"])
    if torch.device(device).type == "cuda":
        if len(p["cuda_rng"]) != torch.cuda.device_count():
            raise RuntimeError("CUDA RNG device count mismatch")
        torch.cuda.set_rng_state_all([x.cpu() for x in p["cuda_rng"]])


def check_optimizer(o, expected_step):
    p = o.state_dict()
    if not p["state"]:
        raise RuntimeError("Optimizer moments missing")
    for row in p["state"].values():
        if int(row["step"]) != expected_step:
            raise RuntimeError("Unexpected optimizer update count")
        for key in ("exp_avg", "exp_avg_sq"):
            if key not in row or not bool(torch.isfinite(row[key]).all()):
                raise RuntimeError("Missing/nonfinite AdamW moments")


def initialize(ident, device, base_path=None):
    m = new_model(ident["seed"], ident["attention_mode"], device)
    e = copy.deepcopy(m).eval().requires_grad_(False)
    o = optimizer(m)
    meta = dict(step=0, global_step=ident["base_step"], elapsed_seconds=0.,
                paired_data_digest="", actual_input_digest="", identity=ident,
                fresh_initialization=ident["cohort"] == "fresh")
    if ident["cohort"] == "continuation":
        if base_path is None:
            raise ValueError("Continuation requires the verified complete I-F state")
        expected = c.CONFIG["base_final_sha256"][ident["pair_index"]]
        if c.sha(base_path) != expected:
            raise RuntimeError("Old final checkpoint SHA mismatch")
        p = torch.load(base_path, map_location="cpu", weights_only=False)
        if (p["study"], p["seed"], p["arm"], p["step"], p["protocol_hash"]) != (
                "geometry_identification_20260926", ident["seed"], "I-F", 12000, c.CONFIG["base_protocol_hash"]):
            raise RuntimeError("Foreign or incomplete I-F base")
        m.load_state_dict(p["model"], strict=True)
        e.load_state_dict(p["ema"], strict=True)
        o.load_state_dict(p["optimizer"])
        check_optimizer(o, 12000)
        restore_rng(p, device)
        meta.update(base_final_sha256=expected, base_initial_hash=p["metadata"]["initial_hash"],
                    base_raw_hash=gs.model_hash(m), base_ema_hash=gs.model_hash(e))
    elif ident["cohort"] != "fresh" or base_path is not None:
        raise ValueError("Fresh cohort cannot inherit a checkpoint")
    meta["initial_raw_hash"] = gs.model_hash(m)
    meta["initial_ema_hash"] = gs.model_hash(e)
    return m, e, o, meta


def learning_rate(cohort, step):
    spec = c.COHORTS[cohort]["lr"]
    if not 1 <= step <= spec["total_updates"]:
        raise ValueError((cohort, step))
    if step <= spec["warmup_updates"]:
        return spec["start"]+(spec["peak"]-spec["start"])*step/spec["warmup_updates"]
    fraction = (step-spec["warmup_updates"])/(spec["total_updates"]-spec["warmup_updates"])
    return spec["end"]+.5*(spec["peak"]-spec["end"])*(1+math.cos(math.pi*fraction))


def objective(logits, b, device):
    if b["sparse"]:
        q = torch.as_tensor(b["queries"], device=device)
        y = torch.as_tensor(b["labels"], device=device)
        valid = torch.as_tensor(b["query_valid"], device=device)
        if int(valid.sum()) != 512:
            raise ValueError("Sparse update must supervise exactly 512 hidden slots")
        lp = logits.float().flatten(2).transpose(1, 2).log_softmax(-1).gather(1, q[:, :, None].expand(-1, -1, 2))
        return -lp.gather(2, y[:, :, None]).squeeze(-1)[valid].sum()/512
    y = torch.as_tensor(b["clean"], device=device, dtype=torch.long)
    mask = torch.as_tensor(b["mask"], device=device)
    t = torch.as_tensor(b["t"], device=device)
    if y.numel() != c.TOKENS:
        raise ValueError("Token budget mismatch")
    nll = F.cross_entropy(logits.float(), y, reduction="none")
    return (nll*mask/t[:, None, None]).sum()/c.TOKENS


def update(m, e, o, b, cohort, step, amp=True):
    device = next(m.parameters()).device
    m.train(); o.zero_grad(set_to_none=True)
    x = torch.as_tensor(b["noisy"], device=device, dtype=torch.long)
    xy = torch.as_tensor(b["coords"], device=device)
    t = torch.as_tensor(b["t"], device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
        logits = m(x, t, xy)
    loss = objective(logits, b, device)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Nonfinite objective")
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
    if not bool(torch.isfinite(grad)):
        raise RuntimeError("Nonfinite gradient")
    rate = learning_rate(cohort, step)
    for group in o.param_groups:
        group["lr"] = rate
    o.step(); gs.ema_update(e, m)
    return dict(loss=float(loss.detach()), grad_norm=float(grad), lr=rate,
                tokens=c.TOKENS, width=b["width"], kind=b["kind"], sparse=bool(b["sparse"]),
                supervised_slots=int(b["query_valid"].sum()) if b["sparse"] else int(b["mask"].sum()),
                paired_data_hash=b["paired_data_hash"], actual_input_hash=b["actual_input_hash"])


def checkpoint(path, m, e, o, meta, protocol_hash, *, immutable=True):
    path = Path(path)
    if immutable and path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ident = meta["identity"]
    if meta["global_step"] != ident["base_step"]+meta["step"]:
        raise ValueError("Relative/global update mismatch")
    p = dict(study=c.CONFIG["study"], identity=ident, protocol_hash=protocol_hash,
             config=dict(model=c.MODEL, attention_mode=ident["attention_mode"]),
             model=m.state_dict(), ema=e.state_dict(), optimizer=o.state_dict(), metadata=meta,
             step=meta["step"], global_step=meta["global_step"],
             torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
             cuda_rng=torch.cuda.get_rng_state_all() if next(m.parameters()).is_cuda else [])
    tmp = path.with_name(path.name+".tmp")
    with tmp.open("xb") as f:
        torch.save(p, f)
    os.replace(tmp, path)


def restore(path, ident, protocol_hash, device):
    p = torch.load(path, map_location="cpu", weights_only=False)
    if (p["study"], p["identity"], p["protocol_hash"]) != (c.CONFIG["study"], ident, protocol_hash):
        raise RuntimeError("Foreign J checkpoint")
    if not 0 < p["step"] <= ident["updates"] or p["global_step"] != ident["base_step"]+p["step"]:
        raise RuntimeError("Wrong saved update count")
    m = new_model(ident["seed"], ident["attention_mode"], device)
    e = copy.deepcopy(m).eval().requires_grad_(False)
    o = optimizer(m)
    m.load_state_dict(p["model"], strict=True)
    e.load_state_dict(p["ema"], strict=True)
    o.load_state_dict(p["optimizer"])
    check_optimizer(o, p["global_step"])
    restore_rng(p, device)
    return m, e, o, p["metadata"]


@torch.inference_mode()
def predict(m, b, *, check, batch_size=2):
    device = next(m.parameters()).device
    m.eval()
    parts = []
    for start in range(0, len(b["t"]), batch_size):
        check()
        sl = slice(start, start+batch_size)
        logits = m(torch.as_tensor(b["noisy"][sl], device=device, dtype=torch.long),
                   torch.as_tensor(b["t"][sl], device=device),
                   torch.as_tensor(b["input_coordinates"][sl], device=device))
        q = torch.as_tensor(b["queries"][sl], device=device, dtype=torch.long)
        parts.append(logits.float().softmax(1)[:, 1].flatten(1).gather(1, q).cpu().numpy())
    p = np.concatenate(parts).astype(np.float64)
    if not np.isfinite(p).all():
        raise RuntimeError("Nonfinite probability")
    result = dict(probability=p, common_data_hash=np.array(d.common_hash(b)),
                  actual_input_hash=np.array(c.array_hash(b["noisy"], b["input_coordinates"], b["t"], b["queries"])),
                  parent=b["parent"], chain=b["chain"], target_type=b["target_type"])
    target_type = str(b["target_type"])
    if target_type != "reference_only":
        y = b["labels"].astype(np.float64)
        pc = np.clip(p, 1e-12, 1-1e-12)
        ce = -(y*np.log(pc)+(1-y)*np.log1p(-pc))
        result.update(ce=ce, brier=y*(1-p)**2+(1-y)*p**2, labels=y)
        if target_type == "soft":
            yc = np.clip(y, 1e-15, 1-1e-15)
            entropy = -(y*np.log(yc)+(1-y)*np.log1p(-yc))
            result.update(kl=ce-entropy, probability_error=p-y)
    return result
