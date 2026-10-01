"""Full-state continuation and FP32 conditional prediction; no launcher at import."""
from __future__ import annotations

import copy
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import endpoint_common as c
import endpoint_data as d
c.import_old()
from ism_diffusion import geometry_study as gs


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=3e-5, betas=(.9, .95), weight_decay=.05,
                            fused=next(model.parameters()).device.type == "cuda")


def rng_restore(payload, device):
    torch.set_rng_state(payload["torch_rng"].cpu())
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    if torch.device(device).type == "cuda":
        if len(payload["cuda_rng"]) != torch.cuda.device_count():
            raise RuntimeError("CUDA RNG count mismatch")
        torch.cuda.set_rng_state_all([s.cpu() for s in payload["cuda_rng"]])


def check_optimizer(opt, expected):
    state = opt.state_dict()["state"]
    assert state
    for row in state.values():
        assert int(row["step"]) == expected
        for key in ("exp_avg", "exp_avg_sq"):
            assert key in row and bool(torch.isfinite(row[key]).all())


def base_hashes():
    receipt = c.read(c.DOCS / "ENDPOINT_MASK_SAMPLER_DISENTANGLEMENT_BUDGET_20261001.json")
    return {row["lineage"]: row["sha256"] for row in receipt["base_final_hashes_from_existing_manifests_not_reloaded_in_this_design_check"]}


def initialize(seed, arm, device="cuda"):
    assert seed in c.SEEDS and arm in c.ARMS
    path = c.base_path(seed)
    expected = base_hashes()[seed]
    assert c.sha(path) == expected
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert (payload["study"], payload["arm"], payload["seed"], payload["step"]) == ("geometry_identification_20260926", "I-F", seed, 12000)
    assert payload["protocol_hash"] == "c9acba5d414257edfcc23545d51974ba7cc1c1eb3cfc10bbee1753fc12482164"
    model = gs.new_model(seed, device=device)
    assert sum(p.numel() for p in model.parameters()) == 1976706
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = optimizer(model)
    model.load_state_dict(payload["model"], strict=True)
    ema.load_state_dict(payload["ema"], strict=True)
    opt.load_state_dict(payload["optimizer"])
    if torch.device(device).type == "cpu":
        for group in opt.param_groups:
            group["fused"] = False
    check_optimizer(opt, 12000)
    rng_restore(payload, device)
    meta = dict(seed=seed, arm=arm, step=0, global_step=12000, base_sha256=expected,
                base_initial_hash=payload["metadata"]["initial_hash"],
                initial_raw_hash=gs.model_hash(model), initial_ema_hash=gs.model_hash(ema),
                physical_digest="", native_digest="", supervised_digest="", elapsed_seconds=0.)
    return model, ema, opt, meta


def learning_rate(step):
    assert 1 <= step <= 8000
    if step <= 256:
        return 3e-5 + (1e-4 - 3e-5)*step/256
    return 1e-5 + .5*(1e-4 - 1e-5)*(1+math.cos(math.pi*(step-256)/(8000-256)))


def objective(logits, batch, device):
    if batch["sparse"]:
        q = torch.as_tensor(batch["queries"], device=device)
        y = torch.as_tensor(batch["labels"], device=device)
        valid = torch.as_tensor(batch["query_valid"], device=device)
        assert int(valid.sum()) == 512
        lp = logits.float().flatten(2).transpose(1, 2).log_softmax(-1).gather(1, q[:, :, None].expand(-1, -1, 2))
        return -lp.gather(2, y[:, :, None]).squeeze(-1)[valid].sum()/512
    clean = torch.as_tensor(batch["clean"], device=device, dtype=torch.long)
    mask = torch.as_tensor(batch["mask"], device=device)
    nll = F.cross_entropy(logits.float(), clean, reduction="none")
    if batch["endpoint"]:
        return ((nll*mask).sum((1, 2))/mask.sum((1, 2))).mean()
    time = torch.as_tensor(batch["t"], device=device)
    return (nll*mask/time[:, None, None]).sum()/c.TOKENS


def update(model, ema, opt, batch, step, amp=True):
    device = next(model.parameters()).device
    model.train(); opt.zero_grad(set_to_none=True)
    x = torch.as_tensor(batch["noisy"], device=device, dtype=torch.long)
    xy = torch.as_tensor(batch["coords"], device=device)
    t = torch.as_tensor(batch["t"], device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
        logits = model(x, t, xy)
    loss = objective(logits, batch, device)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Nonfinite loss")
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not bool(torch.isfinite(grad)):
        raise RuntimeError("Nonfinite gradient")
    lr = learning_rate(step)
    for group in opt.param_groups:
        group["lr"] = lr
    opt.step(); gs.ema_update(ema, model)
    return dict(loss=float(loss.detach()), grad_norm=float(grad), lr=lr, width=batch["width"],
                kind=batch["kind"], repeat=batch["repeat"], sparse=batch["sparse"], endpoint=batch["endpoint"],
                input_tokens=c.TOKENS, supervised_slots=512 if batch["sparse"] else int(batch["mask"].sum()),
                M=batch["mask_counts"].tolist(), model_t=batch["t"].tolist(), q_mask=batch["q_mask"].tolist(),
                physical_hash=batch["physical_hash"], native_hash=batch["native_hash"], supervision_hash=batch["supervision_hash"])


def save_state(path, model, ema, opt, meta, protocol_hash, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(path)
    assert meta["global_step"] == 12000+meta["step"]
    payload = dict(study=c.STUDY, protocol_hash=protocol_hash, seed=meta["seed"], arm=meta["arm"],
                   step=meta["step"], global_step=meta["global_step"], config=dict(model=c.MODEL), metadata=meta,
                   model=model.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(),
                   torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                   cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [])
    temp = path.with_name(path.name+".tmp")
    with temp.open("xb") as f:
        torch.save(payload, f)
    os.replace(temp, path)


def restore(path, seed, arm, protocol_hash, device="cuda"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert (payload["study"], payload["seed"], payload["arm"], payload["protocol_hash"]) == (c.STUDY, seed, arm, protocol_hash)
    assert 0 <= payload["step"] <= 8000 and payload["global_step"] == 12000+payload["step"]
    model = gs.new_model(seed, device=device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = optimizer(model)
    model.load_state_dict(payload["model"], strict=True)
    ema.load_state_dict(payload["ema"], strict=True)
    opt.load_state_dict(payload["optimizer"])
    if torch.device(device).type == "cpu":
        for group in opt.param_groups:
            group["fused"] = False
    check_optimizer(opt, payload["global_step"])
    rng_restore(payload, device)
    return model, ema, opt, payload["metadata"]


def load_ema(path, seed, device="cuda"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["seed"] == seed
    model = gs.new_model(seed, device=device).eval().requires_grad_(False)
    model.load_state_dict(payload["ema"], strict=True)
    return model


def metrics(probability, target):
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1-1e-12)
    y = np.asarray(target, dtype=np.float64)
    ce = -(y*np.log(p)+(1-y)*np.log1p(-p))
    yc = np.clip(y, 1e-15, 1-1e-15)
    entropy = -(y*np.log(yc)+(1-y)*np.log1p(-yc))
    return dict(ce=ce, brier=y*(1-p)**2+(1-y)*p**2, kl=ce-entropy, probability_error=p-y)


@torch.inference_mode()
def predict(model, bank, batch_size=2, amp=False, check=c.check):
    d.validate_bank(bank)
    model.eval()
    device = next(model.parameters()).device
    output = []
    for start in range(0, len(bank["t"]), batch_size):
        check(); sl = slice(start, start+batch_size)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            logits = model(torch.as_tensor(bank["noisy"][sl], device=device, dtype=torch.long),
                           torch.as_tensor(bank["t"][sl], device=device), torch.as_tensor(bank["coords"][sl], device=device))
        q = torch.as_tensor(bank["queries"][sl], device=device)
        output.append(logits.float().softmax(1)[:, 1].flatten(1).gather(1, q).cpu().numpy())
    p = np.concatenate(output).astype(np.float64)
    assert np.isfinite(p).all()
    values = metrics(p, bank["target"])
    if not str(bank["target_type"]).startswith("exact"):
        values.pop("kl")
    return dict(probability=p, target=bank["target"], parent=bank["parent"], chain=bank["chain"],
                common_native_hash=np.array(d.bank_hash(bank)), target_type=bank["target_type"], **values)

