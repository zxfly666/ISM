"""J physical data, fixed-information views and analytic controls (NumPy only).

Data streams deliberately have no attention-arm argument. D and O receive the
same batch. Formal reference fields never enter training_batch.
"""
from __future__ import annotations

import numpy as np
import intervention_common as c


def increments(width, kind):
    q = width // 4 - 1
    if kind == "continuous":
        return np.ones(width-1, dtype=np.int64)
    if kind == "train_gap":
        return np.repeat([1, 2, 4, 8], [q+1, q, q+2, q])
    if kind == "held_gap":
        return np.repeat([3, 6], [3*q+3, q])
    raise ValueError(kind)


def axes(width, kind, batch, seed, index):
    random = c.rng(seed, index, "physical_axes")
    inc = increments(width, kind)
    out = np.zeros((batch, 2, width), dtype=np.int64)
    for row in out:
        for axis in row:
            axis[1:] = random.permutation(inc).cumsum()
    if out.max() >= 512:
        raise ValueError("Physical span must be below half of parent width")
    return out


def coordinates(a, d4=None):
    b, _, width = a.shape
    out = np.empty((b, width, width, 2), dtype=np.float32)
    out[..., 0] = a[:, 0, :, None]
    out[..., 1] = a[:, 1, None, :]
    if d4 is not None:
        for i, code in enumerate(d4):
            if int(code) & 1:
                out[i] = out[i, ..., ::-1].copy()
            if int(code) & 2:
                out[i, ..., 0] *= -1
            if int(code) & 4:
                out[i, ..., 1] *= -1
    return out


def sampled(parent, ids, a, origins):
    if parent.lattice_size != 1024:
        raise ValueError("L1024 parent required")
    x = (origins[:, 0, None] + a[:, 0]) % 1024
    y = (origins[:, 1, None] + a[:, 1]) % 1024
    spins = parent.spins[ids[:, None, None], x[:, :, None], y[:, None, :]]
    if not np.isin(spins, [-1, 1]).all():
        raise ValueError("Invalid physical spins")
    return (spins > 0).astype(np.int8)


def schedule(data_seed, step, total):
    if not 1 <= step <= total or total not in (4000, 24000):
        raise ValueError((step, total))
    cycle, offset = divmod(step-1, 32)
    cell, repeat = divmod(int(c.rng(data_seed, cycle, "balanced_schedule").permutation(32)[offset]), 4)
    width = c.WIDTHS[cell % 4]
    return dict(width=width, kind="continuous" if cell < 4 else "train_gap",
                sparse=repeat == 3, batch=c.TOKENS // width**2, latent_cell=cell)


def training_batch(parent, identity, step):
    seed = identity["data_seed"]
    sc = schedule(seed, step, identity["updates"])
    b, w = sc["batch"], sc["width"]
    n = w*w
    a = axes(w, sc["kind"], b, seed, step)
    ids = c.rng(seed, step, "parent").integers(len(parent.spins), size=b)
    origins = c.rng(seed, step, "origin").integers(1024, size=(b, 2))
    clean = sampled(parent, ids, a, origins).astype(np.int64)
    flip = c.rng(seed, step, "spin_flip").random(b) < .5
    clean[flip] = 1-clean[flip]
    d4 = c.rng(seed, step, "d4").integers(8, size=b)
    coords = coordinates(a, d4)
    if sc["sparse"]:
        k = c.rng(seed, step, "sparse_k").choice([1, 2, 4, 8, 16, 32], size=b)
        count, rem = divmod(512, b)
        counts = np.full(b, count)
        counts[c.rng(seed, step, "query_allocation").permutation(b)[:rem]] += 1
        noisy = np.full((b, n), 2, dtype=np.int64)
        query = np.zeros((b, 64), dtype=np.int64)
        qvalid = np.zeros((b, 64), dtype=bool)
        labels = np.zeros((b, 64), dtype=np.int64)
        random = c.rng(seed, step, "query_and_observations")
        flat = clean.reshape(b, n)
        for i in range(b):
            order = random.permutation(n)
            q, e = order[:counts[i]], order[counts[i]:counts[i]+k[i]]
            query[i, :counts[i]] = q
            qvalid[i, :counts[i]] = True
            labels[i, :counts[i]] = flat[i, q]
            noisy[i, e] = flat[i, e]
        noisy = noisy.reshape(b, w, w)
        mask = noisy == 2
        t = (1-k/n).astype(np.float32)
        weights = qvalid.astype(np.float64) / 512
        assert qvalid.sum() == 512
    else:
        t = c.rng(seed, step, "time").uniform(.01, 1, b).astype(np.float32)
        t[c.rng(seed, step, "endpoint").random(b) < .02] = 1
        mask = c.rng(seed, step, "mask").random(clean.shape) < t[:, None, None]
        noisy = np.where(mask, 2, clean)
        k = (~mask).sum((1, 2))
        query = np.empty((b, 0), dtype=np.int64)
        qvalid = np.empty((b, 0), dtype=bool)
        labels = np.empty((b, 0), dtype=np.int64)
        weights = mask.astype(np.float64) / t[:, None, None] / c.TOKENS
    paired = c.array_hash(clean, a, ids, origins, flip, d4, coords, noisy, mask, t, query, qvalid, labels, weights)
    return dict(**sc, clean=clean, physical_axes=a, coords=coords, parent=ids, origin=origins,
                d4=d4, spin_flip=flip, noisy=noisy, mask=mask, t=t, k=k,
                queries=query, query_valid=qvalid, labels=labels, loss_weights=weights,
                paired_data_hash=paired, actual_input_hash=c.array_hash(noisy, coords, t))


def bank(parent, ids, seed, geometry_index, width, kind, k):
    ids = np.asarray(ids, dtype=np.int64)
    b = len(ids)
    a = axes(width, kind, b, seed, geometry_index)
    origins = c.rng(seed, geometry_index, "bank_origin").integers(1024, size=(b, 2))
    clean = sampled(parent, ids, a, origins)
    flat = clean.reshape(b, -1)
    random = c.rng(seed, geometry_index, "bank_query_and_evidence")
    q = np.empty((b, 64), dtype=np.int64)
    e = np.empty((b, k), dtype=np.int64)
    noisy = np.full(flat.shape, 2, dtype=np.int8)
    for i in range(b):
        order = random.permutation(width**2)
        q[i], e[i] = order[:64], order[64:64+k]
        noisy[i, e[i]] = flat[i, e[i]]
    return dict(clean=clean, physical_axes=a, input_coordinates=coordinates(a), origin=origins,
                parent=ids, chain=np.asarray(parent.chain_ids)[ids], noisy=noisy.reshape(b, width, width),
                queries=q, evidence=e, labels=flat[np.arange(b)[:, None], q],
                t=np.full(b, 1-k/(width**2), dtype=np.float32), width=np.array(width),
                kind=np.array(kind), k=np.array(k), target_type=np.array("hard"),
                geometry_index=np.array(geometry_index), bank_seed=np.array(seed))


def common_hash(b):
    return c.array_hash(b["noisy"], b["input_coordinates"], b["t"], b["queries"],
                        b["labels"], b["parent"], b["chain"], b["target_type"])


def validate_bank(b):
    n = len(b["t"])
    flat = b["noisy"].reshape(n, -1)
    rows = np.arange(n)[:, None]
    q = b["queries"]
    assert q.dtype.kind in "iu" and np.all(flat[rows, q] == 2)
    assert b["labels"].shape == q.shape
    assert b["input_coordinates"].shape == b["noisy"].shape+(2,)
    assert np.all((flat == 0).sum(1)+(flat == 1).sum(1) == int(b["k"]))
    assert all(len(np.unique(row)) == len(row) for row in q)
    if "clean" in b and str(b["target_type"]) == "hard" and b["clean"].shape == b["noisy"].shape:
        assert np.array_equal(b["clean"].reshape(n, -1)[rows, q], b["labels"])


def parent_subset(parent, frames, expected_per_chain=256):
    groups = [np.flatnonzero(parent.chain_ids == i) for i in range(16)]
    if any(len(g) != expected_per_chain for g in groups):
        raise ValueError("Unexpected reference chain/frame identity")
    return np.concatenate([g[np.asarray(frames)] for g in groups])


def validation_banks(parent):
    groups = [np.flatnonzero(parent.chain_ids == x) for x in np.unique(parent.chain_ids)]
    if len(groups) != 2 or len(parent.spins) != 256:
        raise ValueError("Original two-chain validation split required")
    ids = np.concatenate([g[np.linspace(0, len(g)-1, 32, dtype=int)] for g in groups])
    out = {}
    for gi, kind in enumerate(("continuous", "train_gap")):
        for k in ([2, 32, 512, 115, 1152] if gi == 0 else [2, 32, 512]):
            out[f'{"C" if gi == 0 else "U"}48_k{k}'] = bank(
                parent, ids, c.CONFIG["seeds"]["validation"], 100+gi, 48, kind, k)
    return out


def padding_banks(parent, ids, k):
    base = bank(parent, ids, c.CONFIG["seeds"]["mechanism_and_low_k"], 80, 48, "continuous", k)
    small = {**base, "input_coordinates": base["input_coordinates"]+24}
    noisy = np.full((len(ids), 96, 96), 2, dtype=np.int8)
    noisy[:, 24:72, 24:72] = base["noisy"]
    q = (base["queries"]//48+24)*96+base["queries"]%48+24
    a = np.broadcast_to(np.arange(96), (len(ids), 2, 96)).copy()
    # Do not leave a misleading 48x48 clean/evidence field on the 96x96 input.
    fixed = {key: value for key, value in base.items() if key not in ("clean", "evidence", "physical_axes")}
    fixed.update(width=np.array(96), noisy=noisy, queries=q, input_coordinates=coordinates(a))
    natural = {**fixed, "t": np.full(len(ids), 1-k/9216, dtype=np.float32)}
    return {"small_natural": small, "large_fixed_clock": fixed, "large_natural_clock": natural}


def mechanism_banks(parent, ids):
    ids = np.asarray(ids)
    seed = c.CONFIG["seeds"]["mechanism_and_low_k"]
    origin = np.stack([c.rng(seed, int(i), "factorial_origin").integers(1024, size=2) for i in ids])
    pool = np.stack(np.meshgrid(np.arange(36, 60), np.arange(36, 60), indexing="ij"), -1).reshape(-1, 2)
    shared = np.stack([pool[c.rng(seed, int(i), "factorial_shared_points").permutation(len(pool))[:96]] for i in ids])
    # Query and observed rows are explicit physical identities at every N/D.
    values = (parent.spins[ids[:, None], (origin[:, 0, None]+shared[..., 0]) % 1024,
                           (origin[:, 1, None]+shared[..., 1]) % 1024] > 0).astype(np.int8)
    out = {}
    for domain, lo in ((48, 24), (96, 0)):
        axis = np.arange(lo, lo+domain)
        all_points = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
        full = []
        for j, pid in enumerate(ids):
            anchors = np.array([[lo, lo], [lo+domain-1, lo+domain-1]])
            fixed = np.concatenate([shared[j], anchors])
            keys = fixed[:, 0]*96+fixed[:, 1]
            remaining = all_points[~np.isin(all_points[:, 0]*96+all_points[:, 1], keys)]
            priority = c.rng(seed, int(pid), f"factorial_background_D{domain}").permutation(len(remaining))
            full.append(np.concatenate([fixed, remaining[priority]])[:2304])
        full = np.stack(full)
        for n in (576, 2304):
            coords = full[:, :n].astype(np.float32)[:, None]
            noisy = np.full((len(ids), 1, n), 2, dtype=np.int8)
            noisy[:, 0, 64:96] = values[:, 64:96]
            for clockn in (576, 2304):
                out[f"N{n}_D{domain}_t{clockn}"] = dict(
                    noisy=noisy, input_coordinates=coords, t=np.full(len(ids), 1-32/clockn, dtype=np.float32),
                    queries=np.broadcast_to(np.arange(64), (len(ids), 64)).copy(), labels=values[:, :64],
                    parent=ids, chain=np.asarray(parent.chain_ids)[ids], origin=origin,
                    k=np.array(32), domain=np.array(domain), token_count=np.array(n),
                    kind=np.array("factorial"), target_type=np.array("hard"))
    return out


def gibbs_patterns():
    signs = 2*((np.arange(16)[:, None] >> np.arange(4)) & 1)-1
    beta = np.log(1+np.sqrt(2))/2
    return signs, 1/(1+np.exp(-2*beta*signs.sum(1)))


def fixture_bank():
    signs, target = gibbs_patterns()
    noisy = np.full((16, 8, 8), 2, dtype=np.int8)
    for j, (x, y) in enumerate(((2, 3), (4, 3), (3, 2), (3, 4))):
        noisy[:, x, y] = (signs[:, j]+1)//2
    a = np.broadcast_to(np.arange(8), (16, 2, 8)).copy()
    return dict(noisy=noisy, input_coordinates=coordinates(a), t=np.full(16, 1-4/64, dtype=np.float32),
                queries=np.full((16, 1), 3*8+3, dtype=np.int64), labels=target[:, None],
                parent=np.arange(16), chain=np.zeros(16, dtype=np.int64), k=np.array(4),
                target_type=np.array("soft"), signs=signs)


def oracle_banks(parent, ids):
    ids = np.asarray(ids)
    seed = c.CONFIG["seeds"]["mechanism_and_low_k"]
    signs, target = gibbs_patterns()
    nn = np.array([22*48+23, 24*48+23, 23*48+22, 23*48+24])
    q = 23*48+23
    remaining = np.setdiff1d(np.arange(48**2), np.r_[q, nn])
    origins = np.stack([c.rng(seed, int(i), "oracle_origin").integers(1024, size=2) for i in ids])
    a = np.broadcast_to(np.arange(48), (len(ids), 2, 48)).copy()
    clean = sampled(parent, ids, a, origins).reshape(len(ids), -1)
    priority = np.stack([c.rng(seed, int(i), "oracle_extra_evidence").permutation(remaining) for i in ids])
    b = len(ids)*16
    pid, chain = np.repeat(ids, 16), np.repeat(np.asarray(parent.chain_ids)[ids], 16)
    out = {}
    for k in (4, 32, 512):
        noisy = np.full((len(ids), 16, 48**2), 2, dtype=np.int8)
        for j in range(len(ids)):
            extra = priority[j, :k-4]
            noisy[j, :, extra] = clean[j, extra][:, None]
            noisy[j, :, nn] = ((signs+1)//2).T
        small = noisy.reshape(b, 48, 48)
        large = np.full((b, 96, 96), 2, dtype=np.int8)
        large[:, 24:72, 24:72] = small
        for view, width, x, clockw in (("C48_natural", 48, small, 48),
                                      ("C96_fixed_t48", 96, large, 48),
                                      ("C96_natural", 96, large, 96)):
            axis = np.broadcast_to(np.arange(width), (b, 2, width)).copy()
            coords = coordinates(axis)+(24 if width == 48 else 0)
            query = q if width == 48 else 47*96+47
            out[f"k{k}_{view}"] = dict(noisy=x, input_coordinates=coords,
                t=np.full(b, 1-k/clockw**2, dtype=np.float32), queries=np.full((b, 1), query, dtype=np.int64),
                labels=np.tile(target, len(ids))[:, None], parent=pid, chain=chain,
                pattern=np.tile(np.arange(16), len(ids)), origin=np.repeat(origins, 16, axis=0),
                k=np.array(k), width=np.array(width), target_type=np.array("soft"))
    return out


def low_inputs(layouts):
    """Use the immutable I blueprint supplied by the caller, not a new draw."""
    if layouts["axes"].shape != (80, 2, 2, 48):
        raise ValueError("Not the frozen I 80-pair blueprint")
    patterns = [(1,), (0,), (1, 1), (1, 0), (0, 1), (0, 0)]
    x, xy, ts, qids, pairs, sides, conditions = [], [], [], [], [], [], []
    qx, qy = map(int, layouts["query"])
    for pair, aa in enumerate(layouts["axes"]):
        for side in range(2):
            for condition, pattern in enumerate(patterns):
                noisy = np.full((48, 48), 2, dtype=np.int8)
                for (i, j), value in zip(layouts["visible"], pattern):
                    noisy[i, j] = value
                x.append(noisy)
                xy.append(coordinates(aa[side][None])[0])
                ts.append(1-len(pattern)/2304)
                qids.append(qx*48+qy)
                pairs.append(pair); sides.append(side); conditions.append(condition)
    return dict(noisy=np.array(x), input_coordinates=np.array(xy), t=np.array(ts, dtype=np.float32),
                queries=np.array(qids, dtype=np.int64)[:, None], labels=np.full((960, 1), np.nan),
                parent=np.array(pairs), chain=np.array(sides), pair=np.array(pairs), side=np.array(sides),
                condition=np.array(conditions), target_type=np.array("reference_only"))


def low_reference(parent, layouts, check=lambda: None):
    n = len(parent.spins)
    counts = np.empty((n, 80, 2, 8), dtype=np.int64)
    seed = c.CONFIG["seeds"]["mechanism_and_low_k"]
    origins = np.stack([c.rng(seed, i, "low_reference_translation").integers(1024, size=(256, 2)) for i in range(n)])
    points = np.vstack([layouts["query"], layouts["visible"]])
    for pair in range(80):
        check()
        for side in range(2):
            a = layouts["axes"][pair, side]
            for start in range(0, n, 32):
                rows = np.arange(start, min(start+32, n))
                o = origins[rows]
                code = np.zeros(o.shape[:2], dtype=np.int64)
                for (x, y), bit in zip(points, (4, 2, 1)):
                    code += bit*(parent.spins[rows[:, None], (o[:, :, 0]+a[0, x]) % 1024,
                                              (o[:, :, 1]+a[1, y]) % 1024] > 0)
                counts[rows, pair, side] = np.stack([np.bincount(z, minlength=8) for z in code])
    assert np.all(counts.sum(-1) == 256)
    for i, kind in enumerate(layouts["types"]):
        if str(kind).endswith("negative"):
            assert np.array_equal(counts[:, i, 0], counts[:, i, 1])
    return dict(counts=counts, chain=np.asarray(parent.chain_ids), parent=np.arange(n), origin=origins)
