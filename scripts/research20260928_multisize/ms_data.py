"""Exact retained tasks and local-Markov inputs; no new Monte Carlo."""
from __future__ import annotations

import copy
import numpy as np

import ms_common as c


def subset(bank, ids):
    n = len(bank["query"])
    return {k: a[ids].copy() for k, a in bank.items() if a.ndim and len(a) == n}


def extend_k4(bank, side, natural=False):
    assert side in c.CFG["evaluation_sides"] and side % 2 == 0
    source = np.flatnonzero(bank["k"] == 4)
    offset = (side-4)//2
    n = len(source)
    tokens = np.full((n, 1, side*side), 2, dtype=np.int64)
    query = np.zeros(n, dtype=np.int64)
    for i, row in enumerate(source):
        r, col = divmod(int(bank["query"][row]), 4)
        query[i] = (r+offset)*side+col+offset
        for e in bank["evidence"][row]:
            er, ec = divmod(int(e), 4)
            tokens[i, 0, (er+offset)*side+ec+offset] = bank["tokens"][row, 0, e]
    xy = np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="ij"), -1).reshape(1, 1, side*side, 2).astype(np.float32)
    value = dict(tokens=tokens, query=query, coordinates=np.repeat(xy, n, axis=0),
        t=np.full(n, 1-4/(side*side) if natural else .75, dtype=np.float32),
        target=bank["target"][source].copy(), source_row=source,
        pattern=bank["pattern"][source].copy(), k=np.full(n, 4, dtype=np.int64))
    validate_local(value)
    return value


def validate_local(bank):
    for i, q in enumerate(bank["query"]):
        tok = bank["tokens"][i, 0]
        valid = bank.get("valid", np.ones_like(bank["tokens"], dtype=bool))[i, 0]
        xy = bank["coordinates"][i, 0]
        observed = np.flatnonzero((tok < 2) & valid)
        assert tok[q] == 2 and valid[q] and len(observed) == 4
        assert len(np.unique(xy[valid], axis=0)) == int(valid.sum())
        relative = {tuple(map(float, p)) for p in xy[observed]-xy[q]}
        assert relative == {(-1., 0.), (1., 0.), (0., -1.), (0., 1.)}
        exact = 1/(1+np.exp(-2*c.BETA*(2*tok[observed]-1).sum()))
        assert abs(exact-bank["target"][i]) < 2e-12
    return True


def diagnostic_cell(n, scale, clock, order):
    fixed = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    corners = [(-3, -3), (-3, 4), (4, -3), (4, 4)]
    rest = [(x, y) for x in range(-3, 5) for y in range(-3, 5) if (x, y) not in fixed+corners]
    rest.sort(key=lambda p: (sum(v*v for v in p), p[0], p[1]), reverse=order == "far")
    points = fixed+(corners+rest)[:n-5]
    xy = np.array(points, dtype=np.float32)
    xy[5:] *= scale
    tokens = np.full((16, 1, n), 2, dtype=np.int64)
    bits = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.int64)
    tokens[:, 0, 1:5] = bits
    bank = dict(tokens=tokens, coordinates=np.repeat(xy[None, None], 16, axis=0),
        query=np.zeros(16, dtype=np.int64), t=np.full(16, clock, dtype=np.float32),
        pattern=np.arange(16), target=1/(1+np.exp(-2*c.BETA*(2*bits-1).sum(1))))
    validate_local(bank)
    # Full-range corners remain present at every N, preventing envelope drift with N.
    assert np.array_equal(xy.min(0), np.array([-3, -3])*scale)
    assert np.array_equal(xy.max(0), np.array([4, 4])*scale)
    return bank


def diagnostic_banks():
    result, metadata = {}, {}
    for order in ("near", "far"):
        for n in (16, 32, 64):
            for scale in (1, 2):
                for clock in (.75, .9375):
                    key = "n%d_r%d_t%s_%s" % (n, scale, "075" if clock == .75 else "09375", order)
                    result[key] = diagnostic_cell(n, scale, clock, order)
                    metadata[key] = dict(n=n, scale=scale, clock=clock, order=order,
                        primary_heldout_size_used=False, duplicate_set_at_N64=order == "far" and n == 64)
    for order in ("near", "far"):
        base = copy.deepcopy(result["n16_r1_t075_"+order])
        base["tokens"] = np.concatenate([base["tokens"], np.full((16, 1, 48), 3, dtype=np.int64)], axis=2)
        pad_xy = np.arange(96, dtype=np.float32).reshape(1, 1, 48, 2)+100
        base["coordinates"] = np.concatenate([base["coordinates"], np.repeat(pad_xy, 16, 0)], axis=2)
        base["valid"] = np.concatenate([np.ones((16, 1, 16), bool), np.zeros((16, 1, 48), bool)], axis=2)
        validate_local(base)
        result["pad64_from16_"+order] = base
        metadata["pad64_from16_"+order] = dict(control="PAD", paired_to="n16_r1_t075_"+order)
    return result, metadata


def indices(bank, seed, step):
    random = c.rng(seed, step, "physical_examples")
    return np.concatenate([random.choice(np.flatnonzero((bank["k"] == k) & bank["train"]),
        c.CFG["batch_per_family"], replace=True) for k in (1, 2, 4)]).astype(np.int64)


def view_side(arm, step):
    assert arm in c.CFG["arms"]
    return 6 if arm == "B46" and step % 2 else 4


def logical_and_native(bank, views, ids, side):
    first = subset(bank, ids[:32])
    lookup = {int(row): i for i, row in enumerate(views[side]["source_row"])}
    local = np.array([lookup[int(row)] for row in ids[32:]], dtype=np.int64)
    second = subset(views[side], local)
    fields = ("tokens", "coordinates", "t", "query", "target")
    physical = c.ahash(ids, *[bank[key][ids] for key in fields])
    actual = c.ahash(*[part[key] for part in (first, second) for key in fields])
    assert np.array_equal(np.concatenate([first["target"], second["target"]]), bank["target"][ids])
    assert np.array_equal(np.concatenate([first["t"], second["t"]]), bank["t"][ids])
    return first, second, physical, actual
