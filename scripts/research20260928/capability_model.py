"""Use the unchanged J model; no repaired attention architecture is added."""
from __future__ import annotations

import copy
import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import capability_common as c

sys.path.insert(0, str(c.ROOT))
sys.path.insert(0, str(c.ROOT / "scripts/research20260927"))
from intervention_model import InterventionDenoiser
from ism_diffusion.scale_model import CoordinateDenoiserConfig


def configure():
    torch.set_num_threads(c.CONFIG["torch_threads"])
    torch.set_num_interop_threads(c.CONFIG["torch_interop_threads"])
    torch.use_deterministic_algorithms(True)


def model_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def new_model(label, mode="dense", diagnostic_head=False):
    seed = c.model_seed(label, "structural" if diagnostic_head else "training")
    torch.manual_seed(seed)
    model = InterventionDenoiser(CoordinateDenoiserConfig(**c.CONFIG["model"]), mode)
    if sum(p.numel() for p in model.parameters()) != c.CONFIG["model_parameters"]:
        raise RuntimeError("Model parameter count changed")
    if diagnostic_head:
        with torch.no_grad():
            model.output.weight.normal_(0, .04)
            model.output.bias.normal_(0, .02)
    return model


def optimizer(model):
    spec = c.CONFIG["optimizer"]
    return torch.optim.AdamW(model.parameters(), lr=c.CONFIG["learning_rate"]["peak"],
                            betas=tuple(spec["betas"]), weight_decay=spec["weight_decay"],
                            foreach=False)


def tensor_bank(bank):
    result = dict(tokens=torch.as_tensor(bank["tokens"], dtype=torch.long),
                  coordinates=torch.as_tensor(bank["coordinates"], dtype=torch.float32),
                  t=torch.as_tensor(bank["t"], dtype=torch.float32),
                  query=torch.as_tensor(bank["query"], dtype=torch.long))
    if "target" in bank:
        result["target"] = torch.as_tensor(bank["target"], dtype=torch.float32)
    if "valid" in bank:
        result["valid"] = torch.as_tensor(bank["valid"], dtype=torch.bool)
    return result


def batch_indices(bank, label, step):
    random = c.stream(label, step, "training_examples")
    return np.concatenate([random.choice(np.flatnonzero((bank["k"] == k) & bank["train"]),
                                         c.CONFIG["batch_per_family"], replace=True)
                           for k in (1, 2, 4)]).astype(np.int64)


def learning_rate(step):
    spec = c.CONFIG["learning_rate"]
    if not 1 <= step <= c.CONFIG["steps"]:
        raise ValueError("Step outside frozen schedule")
    if step <= spec["warmup"]:
        return spec["peak"]*step/spec["warmup"]
    fraction = (step-spec["warmup"])/(c.CONFIG["steps"]-spec["warmup"])
    return spec["end"] + .5*(spec["peak"]-spec["end"])*(1+math.cos(math.pi*fraction))


def update(model, ema, opt, tb, indices, step):
    model.train(); opt.zero_grad(set_to_none=True)
    idx = torch.as_tensor(indices, dtype=torch.long)
    logits = model(tb["tokens"][idx], tb["t"][idx], tb["coordinates"][idx])
    logp = logits.flatten(2).transpose(1, 2).log_softmax(-1)
    lp = logp[torch.arange(len(idx)), tb["query"][idx]]
    y = tb["target"][idx]
    loss = -(y*lp[:, 1] + (1-y)*lp[:, 0]).mean()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Nonfinite soft-query loss")
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), c.CONFIG["optimizer"]["grad_clip"])
    if not bool(torch.isfinite(grad)):
        raise RuntimeError("Nonfinite gradient")
    rate = learning_rate(step)
    for group in opt.param_groups:
        group["lr"] = rate
    opt.step()
    with torch.no_grad():
        decay = c.CONFIG["optimizer"]["ema_decay"]
        for p_ema, p in zip(ema.parameters(), model.parameters()):
            p_ema.mul_(decay).add_(p, alpha=1-decay)
    return dict(step=int(step), loss=float(loss.detach()), grad_norm=float(grad), lr=rate,
                input_hash=c.array_hash(tb["tokens"][idx].numpy(), tb["coordinates"][idx].numpy(),
                                        tb["t"][idx].numpy(), tb["query"][idx].numpy()),
                batch_row_hash=c.array_hash(indices))


@torch.inference_mode()
def predict(model, bank, batch_size=64):
    model.eval()
    tb = tensor_bank(bank)
    parts = []
    for start in range(0, len(bank["query"]), batch_size):
        c.check_deadline()
        stop = min(start+batch_size, len(bank["query"]))
        sl = slice(start, stop)
        valid = tb.get("valid")
        logits = model(tb["tokens"][sl], tb["t"][sl], tb["coordinates"][sl],
                       None if valid is None else valid[sl])
        p = logits.softmax(1)[:, 1].flatten(1)
        parts.append(p[torch.arange(stop-start), tb["query"][sl]].numpy())
    result = np.concatenate(parts).astype(np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite prediction")
    return result


def subset(bank, ids):
    n = len(bank["query"])
    return {key: value[ids].copy() for key, value in bank.items()
            if value.ndim > 0 and len(value) == n}


def transformed_checks(model, bank):
    ids = np.concatenate([np.flatnonzero(bank["k"] == k)[:16] for k in (1, 2, 4)])
    base = subset(bank, ids)
    p = predict(model, base)
    shifted = copy.deepcopy(base)
    shifted["coordinates"] += np.array([7, -11], dtype=np.float32)
    permutation = c.stream(0, 0, "structural_permutation").permutation(16)
    permuted = copy.deepcopy(base)
    permuted["tokens"] = permuted["tokens"][:, :, permutation]
    permuted["coordinates"] = permuted["coordinates"][:, :, permutation]
    permuted["query"] = np.argsort(permutation)[permuted["query"]]
    pad = copy.deepcopy(base)
    extra = np.arange(16, dtype=np.float32).reshape(1, 1, 8, 2)+20
    pad["coordinates"] = np.concatenate([pad["coordinates"], np.repeat(extra, len(ids), 0)], axis=2)
    pad["tokens"] = np.concatenate([pad["tokens"], np.full((len(ids), 1, 8), 3, dtype=np.int64)], axis=2)
    pad["valid"] = np.concatenate([np.ones((len(ids), 1, 16), dtype=bool),
                                    np.zeros((len(ids), 1, 8), dtype=bool)], axis=2)
    mask = copy.deepcopy(pad)
    mask["tokens"][:, :, 16:] = 2
    mask["valid"][:] = True
    flipped = copy.deepcopy(base)
    seen = flipped["tokens"] < 2
    flipped["tokens"][seen] = 1-flipped["tokens"][seen]
    values = dict(translation_max=float(np.max(np.abs(predict(model, shifted)-p))),
                  permutation_max=float(np.max(np.abs(predict(model, permuted)-p))),
                  PAD_max=float(np.max(np.abs(predict(model, pad)-p))),
                  added_MASK_fixed_clock_max=float(np.max(np.abs(predict(model, mask)-p))),
                  spin_flip_max=float(np.max(np.abs(predict(model, flipped)+p-1))),
                  repeated_forward_exact=bool(np.array_equal(predict(model, base), p)))
    tol = c.CONFIG["strict_probability_tolerance"]
    if max(values[key] for key in ("translation_max", "permutation_max", "PAD_max")) > tol:
        raise RuntimeError("Strict invariance check failed: " + repr(values))
    if model.attention_mode == "observed_only" and values["added_MASK_fixed_clock_max"] > tol:
        raise RuntimeError("Observed-only fixed-clock structural check failed")
    values["spin_flip_is_learned_not_architectural"] = True
    values["dense_MASK_invariance_is_not_a_required_identity"] = True
    return values


def structural_suite(bank, out):
    all_results = []
    ids = np.flatnonzero((bank["k"] == 1) | ((bank["k"] == 2) & np.isin(bank["pattern"], [0, 3])))
    small = subset(bank, ids)
    for label in c.CONFIG["seed_labels"]:
        pair = []
        for mode in ("dense", "observed_only"):
            model = new_model(label, mode, True)
            p = predict(model, small)
            identity = model_hash(model)
            result = dict(seed_label=label, initialization_seed=c.model_seed(label, "structural"),
                          mode=mode, model_hash=identity, checks=transformed_checks(model, bank),
                          same_symbol_geometry_spread={})
            for k in (1, 2):
                for pattern in (0, (1 << k)-1):
                    sel = (small["k"] == k) & (small["pattern"] == pattern)
                    name = "k%d_pattern%d" % (k, pattern)
                    result["same_symbol_geometry_spread"][name] = dict(
                        model=float(np.ptp(p[sel])), truth=float(np.ptp(small["target"][sel])),
                        minimum_possible_max_error_if_constant=float(np.ptp(small["target"][sel])/2))
            result["K1_numerical_distance_sensitivity"] = bool(max(
                result["same_symbol_geometry_spread"]["k1_pattern%d" % ptn]["model"] for ptn in (0, 1)) > 1e-6)
            if mode == "observed_only":
                result["known_geometry_degeneracy_reproduced"] = bool(max(
                    row["model"] for row in result["same_symbol_geometry_spread"].values()) <= c.CONFIG["strict_probability_tolerance"])
                if not result["known_geometry_degeneracy_reproduced"]:
                    raise RuntimeError("O degeneracy reasoning and code disagree")
            c.save(out/("structural_%s_%s.npz" % (label, mode)), source_row=ids, probability=p,
                   target=small["target"], model_hash=np.array(identity))
            pair.append(result)
        if pair[0]["model_hash"] != pair[1]["model_hash"]:
            raise RuntimeError("D/O structural weights are not paired")
        all_results.extend(pair)
    c.write(out/"structural_summary.json", dict(status="completed", results=all_results,
        expected_O_expressivity_failure_is_not_software_failure=True, trained_O_models=0))
    return all_results


def checkpoint(path, model, ema, opt, label, step, protocol_hash):
    path = Path(path)
    if path.exists():
        raise FileExistsError(str(path))
    payload = dict(study=c.CONFIG["study"], config=c.CONFIG, seed_label=label,
                   model=model.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(),
                   step=step, next_step=step+1, protocol_hash=protocol_hash,
                   torch_rng=torch.get_rng_state(), data_rng="SeedSequence addressed by seed,step,role",
                   raw_hash=model_hash(model), ema_hash=model_hash(ema))
    temp = path.with_name(path.name+".tmp")
    with temp.open("xb") as f:
        torch.save(payload, f)
    if path.exists():
        raise FileExistsError(str(path))
    os.replace(str(temp), str(path))


def restore(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["study"] != c.CONFIG["study"] or payload["config"] != c.CONFIG:
        raise ValueError("Foreign diagnostic checkpoint")
    model = new_model(payload["seed_label"])
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = optimizer(model)
    model.load_state_dict(payload["model"]); ema.load_state_dict(payload["ema"])
    opt.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    if model_hash(model) != payload["raw_hash"] or model_hash(ema) != payload["ema_hash"]:
        raise RuntimeError("Checkpoint tensor hash mismatch")
    if not opt.state or any(int(row["step"]) != payload["step"] for row in opt.state.values()):
        raise RuntimeError("Missing/incomplete optimizer state")
    return model, ema, opt, payload

