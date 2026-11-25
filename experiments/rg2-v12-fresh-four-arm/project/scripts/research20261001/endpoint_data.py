"""Stateless paired physical data and exact local banks; CPU/numpy only."""
from __future__ import annotations

import json
from dataclasses import dataclass
import numpy as np
import endpoint_common as c


@dataclass
class Parent:
    spins: np.ndarray
    chain_ids: np.ndarray
    lattice_size: int
    metadata: dict


def load_parent(path, split):
    z = c.load(path)
    md = json.loads(str(z["metadata"]))
    binary = np.unpackbits(z[split + "_packed"], axis=-1, count=int(md["lattice_size"]), bitorder="little")
    return Parent((2 * binary.astype(np.int8) - 1), z[split + "_chain_id"].astype(np.int16), int(md["lattice_size"]), md)


def pack(spins):
    return np.packbits(spins > 0, axis=-1, bitorder="little")


def increments(width, kind):
    q = width // 4 - 1
    if kind == "continuous":
        return np.ones(width - 1, dtype=np.int64)
    if kind == "train_gap":
        return np.repeat([1, 2, 4, 8], [q+1, q, q+2, q])
    if kind == "held_gap":
        return np.repeat([3, 6], [3*q+3, q])
    raise ValueError(kind)


def physical_axes(width, kind, batch, root, index, role):
    r = c.rng(root, index, role)
    axes = np.zeros((batch, 2, width), dtype=np.int64)
    inc = increments(width, kind)
    for i in range(batch):
        for dim in range(2):
            axes[i, dim, 1:] = r.permutation(inc).cumsum()
    assert axes.max() < 512
    return axes


def coordinates(axes, d4=None):
    b, _, w = axes.shape
    z = np.empty((b, w, w, 2), dtype=np.float32)
    z[..., 0] = axes[:, 0, :, None]
    z[..., 1] = axes[:, 1, None, :]
    if d4 is not None:
        for i, code in enumerate(d4):
            if int(code) & 1:
                z[i] = z[i][..., ::-1].copy()
            if int(code) & 2:
                z[i, ..., 0] *= -1
            if int(code) & 4:
                z[i, ..., 1] *= -1
    return z


def sample(parent, ids, axes, origin):
    x = (origin[:, 0, None] + axes[:, 0]) % parent.lattice_size
    y = (origin[:, 1, None] + axes[:, 1]) % parent.lattice_size
    spins = parent.spins[ids[:, None, None], x[:, :, None], y[:, None, :]]
    assert np.isin(spins, [-1, 1]).all()
    return (spins > 0).astype(np.int64)


def train_schedule(seed, step):
    if not 1 <= step <= 8000:
        raise ValueError(step)
    cycle, off = divmod(step - 1, 32)
    value = int(c.rng(c.CFG["role_seeds"]["training"], cycle, f"schedule_{seed}").permutation(32)[off])
    cell, repeat = divmod(value, 4)
    width = c.WIDTHS[cell % 4]
    return dict(width=width, kind="continuous" if cell < 4 else "train_gap",
                repeat=repeat, batch=c.TOKENS // width**2, cell=cell)


def training_batch(parent, seed, step, arm):
    if arm not in c.ARMS:
        raise ValueError(arm)
    sc = train_schedule(seed, step)
    w, b = sc["width"], sc["batch"]
    n = w*w
    root = c.CFG["role_seeds"]["training"]
    def rand(role):
        return c.rng(root, step, f"s{seed}_{role}")
    axes = physical_axes(w, sc["kind"], b, root, step, f"s{seed}_axes")
    ids = rand("parent").integers(len(parent.spins), size=b)
    origin = rand("origin").integers(parent.lattice_size, size=(b, 2))
    clean = sample(parent, ids, axes, origin)
    flip = rand("flip").random(b) < .5
    clean[flip] = 1-clean[flip]
    d4 = rand("d4").integers(8, size=b)
    xy = coordinates(axes, d4)
    physical = c.digest(clean, axes, ids, origin, flip, d4)
    sparse = sc["repeat"] == 3
    endpoint = arm == "explicit-late-mask" and sc["repeat"] == 2
    query = np.empty((b, 0), dtype=np.int64)
    valid = np.empty((b, 0), dtype=bool)
    labels = np.empty((b, 0), dtype=np.int64)
    if sparse:
        k = rand("sparse_k").choice([1, 2, 4, 8, 16, 32], size=b)
        count, rem = divmod(512, b)
        counts = np.full(b, count)
        counts[rand("query_allocation").permutation(b)[:rem]] += 1
        noisy = np.full((b, n), 2, dtype=np.int64)
        query = np.zeros((b, 64), dtype=np.int64)
        valid = np.zeros((b, 64), dtype=bool)
        labels = np.zeros((b, 64), dtype=np.int64)
        r = rand("sparse_query")
        for i in range(b):
            order = r.permutation(n)
            q, evidence = order[:counts[i]], order[counts[i]:counts[i]+k[i]]
            query[i, :counts[i]] = q
            valid[i, :counts[i]] = True
            labels[i, :counts[i]] = clean.reshape(b, n)[i, q]
            noisy[i, evidence] = clean.reshape(b, n)[i, evidence]
        noisy = noisy.reshape(b, w, w)
        mask = noisy == 2
        qmask = (1-k/n).astype(np.float32)
        t = qmask.copy()
    elif endpoint:
        allowed = [m for m in [1, 2, 4, 8, 16, 32] if m/n <= .02]
        count = rand("late_M").choice(allowed, size=b)
        mask = np.zeros((b, n), dtype=bool)
        r = rand("late_positions")
        for i, m in enumerate(count):
            mask[i, r.choice(n, size=int(m), replace=False)] = True
        mask = mask.reshape(b, w, w)
        qmask = (count/n).astype(np.float32)
        t = np.maximum(qmask, .002).astype(np.float32)
        noisy = np.where(mask, 2, clean)
    else:
        low = .01 if arm == "original-support" else .002
        t = (low + (1-low)*rand("ordinary_t").random(b)).astype(np.float32)
        t[rand("full_mask").random(b) < .02] = 1
        qmask = t.copy()
        mask = rand("ordinary_mask").random(clean.shape) < t[:, None, None]
        noisy = np.where(mask, 2, clean)
    return dict(**sc, clean=clean, coords=xy, physical_axes=axes, parent=ids, origin=origin,
                spin_flip=flip, d4=d4, noisy=noisy, mask=mask, t=t, q_mask=qmask,
                sparse=sparse, endpoint=endpoint, queries=query, query_valid=valid, labels=labels,
                physical_hash=physical, native_hash=c.digest(noisy, xy, t, query, valid, labels),
                supervision_hash=c.digest(clean, mask, query, valid, labels),
                mask_counts=mask.sum((1, 2)).astype(np.int64))


def select_parents(parent, count):
    groups = [np.flatnonzero(parent.chain_ids == v) for v in np.unique(parent.chain_ids)]
    if count < len(groups):
        selected = np.arange(count)*len(groups)//count
        return np.array([groups[i][len(groups[i])//2] for i in selected])
    assert count % len(groups) == 0
    each = count // len(groups)
    assert all(len(g) >= each for g in groups)
    # Even stride for new 128-per-chain reference; stable endpoint-inclusive for validation.
    return np.concatenate([g[np.arange(each)*len(g)//each] for g in groups])


def checkerboard_positions(width, color):
    x, y = np.indices((width, width))
    mask = (x > 0) & (x < width-1) & (y > 0) & (y < width-1) & ((x+y) % 2 == color)
    return np.flatnonzero(mask)


def oracle_probability(tokens, query):
    b, w, _ = tokens.shape
    x, y = query // w, query % w
    assert np.all((x > 0) & (x < w-1) & (y > 0) & (y < w-1))
    rows = np.arange(b)[:, None]
    neighbors = np.stack([tokens[rows, x+dx, y+dy] for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]], -1)
    assert np.isin(neighbors, [0, 1]).all()
    return 1 / (1 + np.exp(-2*c.BETA*(2*neighbors.astype(np.float64)-1).sum(-1)))


def local_bank(parent, ids, width, mask_count, tag, clock=.002, crop=None):
    ids = np.asarray(ids, dtype=np.int64)
    b = len(ids)
    axes = np.broadcast_to(np.arange(width), (b, 2, width)).copy()
    root = c.CFG["role_seeds"]["banks"]
    if crop is None:
        origin = np.stack([c.rng(root, int(i), f"origin_{tag}_w{width}").integers(parent.lattice_size, size=2) for i in ids])
        clean = sample(parent, ids, axes, origin)
    else:
        clean, origin = crop
    noisy = clean.copy()
    q = np.empty((b, mask_count), dtype=np.int64)
    for i, parent_id in enumerate(ids):
        r = c.rng(root, int(parent_id), f"mask_{tag}_w{width}_m{mask_count}")
        candidates = checkerboard_positions(width, int(r.integers(2)))
        q[i] = r.choice(candidates, size=mask_count, replace=False)
        noisy[i].flat[q[i]] = 2
    target = oracle_probability(noisy, q)
    return dict(clean=clean.astype(np.int8), noisy=noisy.astype(np.int8), coords=coordinates(axes),
                queries=q, labels=clean.reshape(b, -1)[np.arange(b)[:, None], q].astype(np.int8),
                target=target, target_type=np.array("exact_local"),
                t=np.full(b, max(mask_count/width**2, clock), dtype=np.float32),
                parent=ids, chain=parent.chain_ids[ids], origin=origin, width=np.array(width),
                mask_count=np.array(mask_count), clock_floor=np.array(clock), kind=np.array("continuous"))


def stress_bank(parent, width):
    ids = select_parents(parent, 4)
    base = local_bank(parent, ids, width, 1, "stress")
    # A fixed center, 16 neighbor patterns for each of four backgrounds.
    clean = np.repeat(base["clean"], 16, axis=0)
    center = width//2
    rows = len(clean)
    for i in range(rows):
        pattern = i % 16
        for j, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            clean[i, center+dx, center+dy] = (pattern >> j) & 1
    noisy = clean.copy()
    noisy[:, center, center] = 2
    q = np.full((rows, 1), center*width+center, dtype=np.int64)
    axes = np.broadcast_to(np.arange(width), (rows, 2, width)).copy()
    return dict(clean=clean, noisy=noisy, coords=coordinates(axes), queries=q,
                target=oracle_probability(noisy, q), labels=np.zeros((rows, 1), dtype=np.int8),
                target_type=np.array("exact_constructed_no_empirical_label"),
                t=np.full(rows, max(1/width**2, .002), dtype=np.float32),
                parent=np.repeat(ids, 16), chain=np.repeat(parent.chain_ids[ids], 16),
                pattern=np.tile(np.arange(16), 4), width=np.array(width), mask_count=np.array(1))


def retention_bank(parent, ids, kind, visible):
    ids = np.asarray(ids)
    b, w = len(ids), 48
    tag = f"retention_{kind}_{visible}"
    root = c.CFG["role_seeds"]["banks"]
    axes = physical_axes(w, kind, b, root, visible, tag)
    origin = np.stack([c.rng(root, int(i), tag+"_origin").integers(parent.lattice_size, size=2) for i in ids])
    clean = sample(parent, ids, axes, origin)
    noisy = np.full((b, w*w), 2, dtype=np.int8)
    q = np.empty((b, 64), dtype=np.int64)
    evidence = np.empty((b, visible), dtype=np.int64)
    for i, pid in enumerate(ids):
        order = c.rng(root, int(pid), tag+"_mask").permutation(w*w)
        q[i], evidence[i] = order[:64], order[64:64+visible]
        noisy[i, evidence[i]] = clean.reshape(b, -1)[i, evidence[i]]
    return dict(clean=clean.astype(np.int8), noisy=noisy.reshape(b, w, w), coords=coordinates(axes),
                queries=q, evidence=evidence, target=clean.reshape(b, -1)[np.arange(b)[:, None], q],
                labels=clean.reshape(b, -1)[np.arange(b)[:, None], q], target_type=np.array("empirical_label"),
                t=np.full(b, 1-visible/w**2, dtype=np.float32), parent=ids, chain=parent.chain_ids[ids],
                origin=origin, physical_axes=axes, width=np.array(w), visible=np.array(visible), kind=np.array(kind))


def validate_bank(b):
    noisy, q = b["noisy"], b["queries"]
    rows = np.arange(len(q))[:, None]
    assert np.all(noisy.reshape(len(q), -1)[rows, q] == 2)
    assert np.isfinite(b["target"]).all() and np.all((b["target"] >= 0) & (b["target"] <= 1))
    if str(b["target_type"]).startswith("exact"):
        assert np.allclose(oracle_probability(noisy, q), b["target"], atol=1e-15, rtol=0)
        assert np.all((noisy == 2).sum((1, 2)) == q.shape[1])
    else:
        assert np.array_equal(b["target"], b["clean"].reshape(len(q), -1)[rows, q])
    assert all(len(np.unique(x)) == len(x) for x in q)


def bank_hash(b):
    return c.digest(b["noisy"], b["coords"], b["t"], b["queries"], b["target"], b["parent"], b["chain"])


def concatenate(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def physical_stats(spins, max_lag=None, region_mask=None):
    """Open x/y pair sums, matching old definitions, int64 accumulators."""
    s = np.asarray(spins, dtype=np.int8)
    assert s.ndim == 3 and s.shape[1] == s.shape[2] and np.isin(s, [-1, 1]).all()
    b, w, _ = s.shape
    lag = min(w-1, w//2 if max_lag is None else max_lag)
    sums = np.empty((b, lag+1), dtype=np.float64)
    counts = np.empty((b, lag+1), dtype=np.int64)
    region = np.ones((w, w), dtype=bool) if region_mask is None else np.asarray(region_mask, dtype=bool)
    sums[:, 0] = region.sum(); counts[:, 0] = region.sum()
    for r in range(1, lag+1):
        mx, my = region[:-r] & region[r:], region[:, :-r] & region[:, r:]
        sums[:, r] = ((s[:, :-r]*s[:, r:])*mx).sum((1, 2), dtype=np.int64) + ((s[:, :, :-r]*s[:, :, r:])*my).sum((1, 2), dtype=np.int64)
        counts[:, r] = mx.sum() + my.sum()
    m = s[:, region].mean(1, dtype=np.float64)
    ex, ey = region[:-1] & region[1:], region[:, :-1] & region[:, 1:]
    # Preserve historical generation energy PER OPEN BOND, not per lattice site.
    energy = -(((s[:, :-1]*s[:, 1:])*ex).sum((1, 2), dtype=np.int64) + ((s[:, :, :-1]*s[:, :, 1:])*ey).sum((1, 2), dtype=np.int64)) / max(ex.sum()+ey.sum(), 1)
    return dict(pair_sum=sums, pair_count=counts, m=m, m2=m*m, abs_m=np.abs(m), energy=energy)


def majority(spins):
    s = np.asarray(spins)
    b, w, _ = s.shape
    assert w % 3 == 0 and np.isin(s, [-1, 1]).all()
    block_sum = s.reshape(b, w//3, 3, w//3, 3).sum((2, 4))
    assert np.all(block_sum != 0)
    return np.sign(block_sum).astype(np.int8)


def multiscale_stats(spins):
    w = spins.shape[-1]
    result = {"full": physical_stats(spins)}
    if w == 96:
        result["center48"] = physical_stats(spins[:, 24:72, 24:72])
        edge = np.ones((w, w), dtype=bool); edge[8:-8, 8:-8] = False
        result["edge8"] = physical_stats(spins, region_mask=edge)
        result["majority32"] = physical_stats(majority(spins))
        result["decimation32"] = physical_stats(spins[:, ::3, ::3])
        origin = [physical_stats(majority(spins[:, x:x+90, y:y+90])) for x in range(3) for y in range(3)]
        # Preserve each phase before averaging, nested within original image.
        result["majority30_origins"] = {k: np.stack([v[k] for v in origin], axis=1) for k in origin[0]}
    return result
