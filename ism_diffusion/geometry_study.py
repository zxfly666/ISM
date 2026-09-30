"""Paired, token-balanced geometry experiment (2026-09-21).

The three arms share clean data, physical geometry, augmentation, time, masks,
initialization and update count. Only the coordinates supplied to RoPE differ.
Random streams are stateless and named, so restart/microbatching cannot advance
another arm's data stream. Existing experiment implementations are not changed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .scale_data import load_parent_split
from .scale_model import CoordinateDenseDenoiser, CoordinateDenoiserConfig


WIDTHS = (16, 24, 32, 48)
ARMS = ("A", "B", "C")
GAPS = (1, 2, 4, 8)
TOKENS_PER_UPDATE = 18432
MODEL = dict(d_model=128, n_heads=4, n_blocks=7, mlp_ratio=4.0,
             dropout=0.0, vocab_size=4, output_classes=2, rope_base=10000.0)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(*values):
    h = hashlib.sha256()
    for value in values:
        a = np.asarray(value)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def rng(seed, index, tag):
    tag_id = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "little")
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(index), tag_id]))


def stream_seed(seed, index, tag):
    return int(rng(seed, index, tag).integers(0, 2**63 - 1))


def schedule(seed, step):
    cycle, offset = divmod(step - 1, 8)
    cells = rng(seed, cycle, "schedule").permutation(8)
    cell = int(cells[offset])
    return WIDTHS[cell % 4], "continuous" if cell < 4 else "gap"


def offsets(random, batch, width, gaps):
    increments = random.choice(np.asarray(gaps), size=(batch, width - 1))
    return np.concatenate([np.zeros((batch, 1), dtype=np.int64), increments.cumsum(1)], axis=1)


def coordinate_arrays(batch, width, kind, gaps, seed, index, augment=False):
    true_rng = rng(seed, index, "physical_geometry")
    fake_rng = rng(seed, index, "independent_geometry")
    used_gaps = (1,) if kind == "continuous" else tuple(gaps)
    axes = [offsets(true_rng, batch, width, used_gaps) for _ in range(2)]
    fake_axes = [offsets(fake_rng, batch, width, (1,) if kind == "continuous" else GAPS)
                 for _ in range(2)]
    rank = np.broadcast_to(np.arange(width), (batch, width))

    def grid(xs, ys):
        coords = np.empty((batch, width, width, 2), dtype=np.float32)
        coords[..., 0] = xs[:, :, None]
        coords[..., 1] = ys[:, None, :]
        return coords

    arrays = {"A": grid(*axes), "B": grid(rank, rank), "C": grid(*fake_axes)}
    # Transform the physical coordinate vectors, not just token order. For a
    # permutation-equivariant model, jointly permuting tokens/coords alone is
    # not a rotation augmentation of the positional attention law.
    if augment:
        transforms = rng(seed, index, "d4").integers(0, 8, size=batch)
        for row, code in enumerate(transforms):
            for coords in arrays.values():
                if code & 1:
                    coords[row] = coords[row][..., ::-1].copy()
                if code & 2:
                    coords[row, ..., 0] *= -1
                if code & 4:
                    coords[row, ..., 1] *= -1
    return arrays, axes


def make_batch(parent, batch, width, kind, gaps, seed, index, augment=False):
    coords, axes = coordinate_arrays(batch, width, kind, gaps, seed, index, augment)
    lattice = parent.lattice_size
    if any(int(a.max()) >= lattice // 2 for a in axes):
        raise ValueError("physical axis span must be below the parent half width")
    prng = rng(seed, index, "parent")
    parents = prng.integers(0, len(parent.spins), size=batch)
    origins = rng(seed, index, "origin").integers(0, lattice, size=(batch, 2))
    xx = (origins[:, 0, None] + axes[0]) % lattice
    yy = (origins[:, 1, None] + axes[1]) % lattice
    spins = parent.spins[parents[:, None, None], xx[:, :, None], yy[:, None, :]]
    clean = (spins > 0).astype(np.int64)
    if augment:
        flip = rng(seed, index, "spin_flip").random(batch) < .5
        clean[flip] = 1 - clean[flip]
    return dict(clean=clean, coords=coords, axes=axes,
                parent=parents, chain=parent.chain_ids[parents], origin=origins)


def corrupt_batch(clean, seed, index, t_value=None):
    # All randomness is generated before microbatch slicing, preserving paired
    # masks when hardware changes the gradient-accumulation layout.
    if t_value is None:
        t = rng(seed, index, "time").uniform(.01, 1., len(clean)).astype(np.float32)
        endpoint = rng(seed, index, "endpoint").random(len(clean)) < .02
        t[endpoint] = 1.
    else:
        t = np.full(len(clean), t_value, dtype=np.float32)
    mask = rng(seed, index, "mask").random(clean.shape) < t[:, None, None]
    return t, mask, np.where(mask, 2, clean)


def new_model(seed, device="cuda"):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    return CoordinateDenseDenoiser(CoordinateDenoiserConfig(**MODEL)).to(device)


def model_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def ema_update(ema, model):
    for dst, src in zip(ema.parameters(), model.parameters()):
        dst.lerp_(src, .001)
    for dst, src in zip(ema.buffers(), model.buffers()):
        dst.copy_(src)


def train_update(model, ema, optimizer, batch, arm, seed, step, total_steps, microbatch):
    t, mask, noisy = corrupt_batch(batch["clean"], seed, step)
    clean = torch.as_tensor(batch["clean"], device="cuda")
    coords = torch.as_tensor(batch["coords"][arm], device="cuda")
    tt = torch.as_tensor(t, device="cuda")
    mm = torch.as_tensor(mask, device="cuda")
    xx = torch.as_tensor(noisy, device="cuda")
    total_tokens = clean.numel()
    optimizer.zero_grad(set_to_none=True)
    losses, ce_sums = [], []
    for start in range(0, len(clean), microbatch):
        sl = slice(start, start + microbatch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xx[sl], tt[sl], coords[sl])
            ce = F.cross_entropy(logits.float(), clean[sl], reduction="none")
            objective = (ce * mm[sl] / tt[sl, None, None]).sum() / total_tokens
        objective.backward()
        losses.append(objective.detach())
        ce_sums.append((ce.detach() * mm[sl]).sum())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    warmup = min(1000, max(1, total_steps // 16))
    lr = 3e-4 * step / warmup if step <= warmup else (
        3e-5 + .5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * (step - warmup) / max(total_steps - warmup, 1))))
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    ema_update(ema, model)
    values = torch.stack([torch.stack(losses).sum(), torch.stack(ce_sums).sum() / mm.sum().clamp_min(1), norm]).float().cpu().tolist()
    if not all(math.isfinite(v) for v in values):
        raise RuntimeError("nonfinite loss/gradient; stop instead of silently continuing")
    metadata = dict(loss=values[0], masked_ce=values[1], grad_norm=values[2], lr=lr,
                    tokens=total_tokens, masked_targets=int(mask.sum()),
                    inverse_t_weight=float((mask / t[:, None, None]).sum()), mean_t=float(t.mean()),
                    paired_hash=array_hash(batch["clean"], t, mask))
    return metadata


@torch.inference_mode()
def evaluate(model, parent, arm, definitions, samples=64, seed=99101, masks=1, microbatch=8):
    records = []
    model.eval()
    for geometry_id, definition in enumerate(definitions):
        width, kind, gaps = definition["width"], definition["kind"], definition["gaps"]
        batch = make_batch(parent, samples, width, kind, gaps, seed, geometry_id, False)
        for ti, tv in enumerate((.2, .5, .8, .95)):
            for realization in range(masks):
                t, mask, noisy = corrupt_batch(batch["clean"], seed + 1000 * geometry_id,
                                               100 * ti + realization, tv)
                for start in range(0, samples, microbatch):
                    sl = slice(start, min(start + microbatch, samples))
                    xx = torch.as_tensor(noisy[sl], device="cuda")
                    cc = torch.as_tensor(batch["coords"][arm][sl], device="cuda")
                    tt = torch.as_tensor(t[sl], device="cuda")
                    yy = torch.as_tensor(batch["clean"][sl], device="cuda")
                    mm = torch.as_tensor(mask[sl], device="cuda")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(xx, tt, cc)
                    ce = F.cross_entropy(logits.float(), yy, reduction="none")
                    prob = logits.float().softmax(1)[:, 1]
                    count = mm.sum((1, 2)).clamp_min(1)
                    ce_row = ((ce * mm).sum((1, 2)) / count).cpu().numpy()
                    brier = (((prob - yy)**2 * mm).sum((1, 2)) / count).cpu().numpy()
                    for j, row in enumerate(range(start, sl.stop)):
                        records.append(dict(geometry=definition["name"], sample=row,
                            parent=int(batch["parent"][row]), chain=int(batch["chain"][row]),
                            t=tv, mask_id=realization, ce=float(ce_row[j]), brier=float(brier[j]),
                            masked_targets=int(mask[row].sum())))
    return records


VAL_DEFS = [dict(name="continuous32", width=32, kind="continuous", gaps=[1]),
            dict(name="randomgap24", width=24, kind="gap", gaps=list(GAPS))]
TEST_DEFS = [dict(name="held_gap57_w32", width=32, kind="gap", gaps=[5, 7]),
             dict(name="held_s10_w48", width=48, kind="gap", gaps=[10]),
             dict(name="continuous48", width=48, kind="continuous", gaps=[1]),
             dict(name="continuous64", width=64, kind="continuous", gaps=[1])]


def balanced_risk(records):
    # Equal chain -> parent -> crop/mask weighting within each task and t.
    groups = {}
    for r in records:
        groups.setdefault((r["geometry"], r["t"], r["chain"], r["parent"]), []).append(r["ce"])
    chains = {}
    for (g, t, c, _), values in groups.items():
        chains.setdefault((g, t, c), []).append(float(np.mean(values)))
    tasks = {}
    for (g, t, _), values in chains.items():
        tasks.setdefault((g, t), []).append(float(np.mean(values)))
    per_task = {f"{g}/t={t}": float(np.mean(v)) for (g, t), v in tasks.items()}
    return dict(mean_ce=float(np.mean(list(per_task.values()))), per_task=per_task)


def train_one(parent, validation, output, arm, seed, config, deadline):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training_complete.json").exists():
        return json.loads((output / "training_complete.json").read_text())
    model = new_model(seed)
    init_hash = model_hash(model)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9, .95),
                                  weight_decay=.05, fused=True)
    steps = config["steps"]
    counters = {f"{kind}/w{w}": dict(updates=0, tokens=0, windows=0, masked=0, inverse_t_weight=0.)
                for kind in ("continuous", "gap") for w in WIDTHS}
    best, previous_elapsed, start_step = float("inf"), 0., 1
    if (output / "last.pt").exists():
        state = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if state["protocol_hash"] != config["protocol_hash"]:
            raise RuntimeError("cannot resume a different frozen protocol")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        counters, best = state["counters"], state["best_validation"]
        previous_elapsed, start_step = state["elapsed_seconds"], state["step"] + 1
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        # Keep rolled-back log tails as evidence, but avoid duplicate steps in
        # resumed histories and paired-hash comparisons.
        for filename in ("train.jsonl", "paired_hashes.jsonl"):
            path = output / filename
            if path.exists():
                lines = path.read_text(encoding="utf-8").splitlines()
                kept, rolled_back = [], []
                for line in lines:
                    try:
                        record_step = json.loads(line)["step"]
                        (kept if record_step < start_step else rolled_back).append(line)
                    except (ValueError, KeyError):
                        rolled_back.append(line)
                if rolled_back:
                    path.with_name(filename + f".rollback_{time.time_ns()}").write_text("\n".join(rolled_back)+"\n", encoding="utf-8")
                    path.write_text("\n".join(kept)+("\n" if kept else ""), encoding="utf-8")
    atomic_json(output / "run_config.json", dict(config, arm=arm, seed=seed, initialization_hash=init_hash))
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    def save(step, filename):
        payload = dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
            config=dict(model=MODEL, variant=arm, seed=seed), step=step, counters=counters,
            best_validation=best, initialization_hash=init_hash, protocol_hash=config["protocol_hash"],
            elapsed_seconds=previous_elapsed + time.monotonic() - started,
            rng_strategy="stateless_named_streams_v1", torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(),
            python_rng=random.getstate())
        path = output / filename
        torch.save(payload, str(path) + ".tmp")
        os.replace(str(path) + ".tmp", path)

    with (output / "train.jsonl").open("a", encoding="utf-8", buffering=1) as log:
        for step in range(start_step, steps + 1):
            if step == start_step or step % 50 == 0:
                if time.time() >= deadline:
                    save(step - 1, "last.pt")
                    return dict(status="budget_stop", step=step - 1, arm=arm, seed=seed)
            width, kind = schedule(seed, step)
            batch_size = TOKENS_PER_UPDATE // width**2
            batch = make_batch(parent, batch_size, width, kind, GAPS, seed, step, True)
            model.train()
            stats = train_update(model, ema, optimizer, batch, arm, seed, step, steps,
                                 config["microbatch"][str(width)])
            counter = counters[f"{kind}/w{width}"]
            counter["updates"] += 1
            counter["tokens"] += stats["tokens"]
            counter["windows"] += batch_size
            counter["masked"] += stats["masked_targets"]
            counter["inverse_t_weight"] += stats["inverse_t_weight"]
            if step <= 100:
                with (output / "paired_hashes.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(step=step, hash=stats["paired_hash"])) + "\n")
            if step % 20 == 0 or step == 1:
                record = dict(step=step, arm=arm, seed=seed, width=width, kind=kind,
                    elapsed=previous_elapsed + time.monotonic() - started, **stats)
                log.write(json.dumps(record) + "\n")
            if step % int(config.get("checkpoint_interval", 1000)) == 0 or step == steps:
                risk = balanced_risk(evaluate(ema, validation, arm, VAL_DEFS))
                record = dict(step=step, arm=arm, seed=seed, validation=risk,
                              elapsed=previous_elapsed + time.monotonic() - started)
                log.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                if risk["mean_ce"] < best:
                    best = risk["mean_ce"]
                    save(step, "best_val.pt")
                save(step, "last.pt")
    save(steps, "final.pt")
    result = dict(status="trained", arm=arm, seed=seed, steps=steps,
                  valid_tokens=steps * TOKENS_PER_UPDATE, counters=counters,
                  elapsed_seconds=previous_elapsed + time.monotonic() - started,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  checkpoint_sha256=file_hash(output / "final.pt"), initialization_hash=init_hash)
    atomic_json(output / "training_complete.json", result)
    del model, ema, optimizer
    torch.cuda.empty_cache()
    return result
