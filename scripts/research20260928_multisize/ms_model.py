"""Unchanged dense model, equal microbatch schedule and recoverable states."""
from __future__ import annotations

import copy
import hashlib
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import ms_common as c
import ms_data as d

sys.path.insert(0, str(c.ROOT))
sys.path.insert(0, str(c.ROOT/"scripts/research20260927"))
from intervention_model import InterventionDenoiser
from ism_diffusion.scale_model import CoordinateDenoiserConfig


def configure():
    torch.set_num_threads(c.CFG["torch_threads"])
    torch.set_num_interop_threads(c.CFG["torch_interop_threads"])
    torch.use_deterministic_algorithms(True)


def model_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def new_model(label):
    torch.manual_seed(c.initial_seed(label))
    model = InterventionDenoiser(CoordinateDenoiserConfig(**c.CFG["model"]), "dense")
    assert sum(p.numel() for p in model.parameters()) == c.CFG["model_parameters"]
    return model


def optimizer(model):
    s = c.CFG["optimizer"]
    return torch.optim.AdamW(model.parameters(), lr=c.CFG["learning_rate"]["peak"],
        betas=tuple(s["betas"]), weight_decay=s["weight_decay"], foreach=False)


def fresh(label):
    model = new_model(label)
    return model, copy.deepcopy(model).eval().requires_grad_(False), optimizer(model)


def tensor_bank(bank):
    types = dict(tokens=torch.long, coordinates=torch.float32, t=torch.float32,
                 query=torch.long, target=torch.float32, valid=torch.bool)
    return {k: torch.as_tensor(bank[k], dtype=kind) for k, kind in types.items() if k in bank}


def prepare_tensors(bank, views):
    source = views[4]["source_row"]
    assert np.array_equal(source, views[6]["source_row"])
    lookup = np.full(len(bank["query"]), -1, dtype=np.int64)
    lookup[source] = np.arange(len(source))
    return dict(base=tensor_bank(bank), views={s: tensor_bank(views[s]) for s in (4, 6)}, lookup=lookup)


def learning_rate(step):
    s = c.CFG["learning_rate"]
    assert 1 <= step <= c.CFG["steps"]
    if step <= s["warmup"]:
        return s["peak"]*step/s["warmup"]
    f = (step-s["warmup"])/(c.CFG["steps"]-s["warmup"])
    return s["end"]+.5*(s["peak"]-s["end"])*(1+math.cos(math.pi*f))


def update(model, ema, opt, tensors, ids, step, side):
    c.deadline()
    started = time.perf_counter()
    assert len(ids) == 48 and side in (4, 6)
    local = tensors["lookup"][ids[32:]]
    assert (local >= 0).all()
    model.train()
    opt.zero_grad(set_to_none=True)
    losses = []
    actual = []
    for tb, selected in ((tensors["base"], ids[:32]), (tensors["views"][side], local)):
        ix = torch.as_tensor(selected, dtype=torch.long)
        logits = model(tb["tokens"][ix], tb["t"][ix], tb["coordinates"][ix])
        lp = logits.flatten(2).transpose(1, 2).log_softmax(-1)[torch.arange(len(ix)), tb["query"][ix]]
        y = tb["target"][ix]
        losses.append(-(y*lp[:, 1]+(1-y)*lp[:, 0]))
        actual.extend([tb[k][ix].numpy() for k in ("tokens","coordinates","t","query","target")])
    per_row = torch.cat(losses)
    loss = per_row.mean()
    assert torch.isfinite(loss)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), c.CFG["optimizer"]["grad_clip"])
    assert torch.isfinite(grad)
    lr = learning_rate(step)
    for group in opt.param_groups:
        group["lr"] = lr
    opt.step()
    with torch.no_grad():
        decay = c.CFG["optimizer"]["ema_decay"]
        for e, p in zip(ema.parameters(), model.parameters()):
            e.mul_(decay).add_(p, alpha=1-decay)
    return dict(step=int(step), loss=float(loss.detach()),
        family_losses=[float(v.mean()) for v in per_row.detach().split(16)],
        grad_norm=float(grad), lr=lr, side=side, actual_input_digest=c.ahash(*actual),
        microbatch_tokens=[32*16,16*side*side], seconds=time.perf_counter()-started)


@torch.inference_mode()
def predict(model, bank, batch_size=32):
    model.eval()
    tb = tensor_bank(bank)
    result = []
    for start in range(0, len(bank["query"]), batch_size):
        c.deadline()
        sl = slice(start, min(start+batch_size, len(bank["query"])))
        valid = None if "valid" not in tb else tb["valid"][sl]
        logits = model(tb["tokens"][sl], tb["t"][sl], tb["coordinates"][sl], valid)
        p = logits.softmax(1)[:, 1].flatten(1)
        result.append(p[torch.arange(len(p)), tb["query"][sl]].numpy())
    value = np.concatenate(result).astype(np.float64)
    assert np.isfinite(value).all()
    return value


def checkpoint(path, state, label, arm, step, protocol_hash, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(str(path))
    model, ema, opt = state
    payload = dict(study=c.CFG["study"], config=c.CFG, seed_label=label, arm=arm,
        step=step, next_step=step+1, protocol_hash=protocol_hash, model=model.state_dict(),
        ema=ema.state_dict(), optimizer=opt.state_dict(), torch_rng=torch.get_rng_state(),
        data_rng="SeedSequence(root,seed,step,physical_examples); no arm",
        raw_hash=model_hash(model), ema_hash=model_hash(ema), time=time.time())
    tmp = path.with_name(path.name+".tmp")
    with tmp.open("xb") as f:
        torch.save(payload, f)
    os.replace(str(tmp), str(path))


def restore(path):
    p = torch.load(path, map_location="cpu", weights_only=False)
    assert p["study"] == c.CFG["study"] and p["config"] == c.CFG
    state = fresh(p["seed_label"])
    model, ema, opt = state
    model.load_state_dict(p["model"])
    ema.load_state_dict(p["ema"])
    opt.load_state_dict(p["optimizer"])
    torch.set_rng_state(p["torch_rng"])
    assert model_hash(model) == p["raw_hash"] and model_hash(ema) == p["ema_hash"]
    assert opt.state and all(int(s["step"]) == p["step"] for s in opt.state.values())
    return state, p


def same_state(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same_state(x, y) for x, y in zip(left, right))
    return left == right


def old_model(seed, expected_sha):
    path = c.PARENT/("training/seed_%d/final.pt" % seed)
    assert c.sha(path) == expected_sha
    p = torch.load(path, map_location="cpu", weights_only=False)
    assert p["seed_label"] == seed and p["step"] == 2048 and p["config"]["model"] == c.CFG["model"]
    model = new_model(seed)
    model.load_state_dict(p["model"])
    assert model_hash(model) == p["raw_hash"]
    return model
