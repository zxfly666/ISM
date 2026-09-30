"""Finite open-boundary Ising truth by exhaustive enumeration, not MC."""
from __future__ import annotations

import itertools
import math

import numpy as np

import capability_common as c


def edges(side):
    return [(r*side+s, rr*side+ss)
            for r in range(side) for s in range(side)
            for rr, ss in ((r+1, s), (r, s+1)) if rr < side and ss < side]


def enumerate_ising(side=4, beta=c.BETA):
    n = side*side
    if n > 16:
        raise ValueError("Exhaustive diagnostic is limited to 16 spins")
    index = np.arange(1 << n, dtype=np.uint32)
    bits = ((index[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.int8)
    spins = (2*bits-1).astype(np.int8)
    pairs = np.asarray(edges(side), dtype=np.int64)
    interaction = (spins[:, pairs[:, 0]]*spins[:, pairs[:, 1]]).sum(1)
    logw = float(beta)*interaction
    scale = float(logw.max())
    unnormalized = np.exp(logw-scale)
    weight = unnormalized/unnormalized.sum()
    return dict(bits=bits, spins=spins, weights=weight, interaction=interaction,
                edges=pairs, log_partition=np.array(scale+np.log(unnormalized.sum())),
                beta=np.array(beta), side=np.array(side))


def row_transfer_log_partition(side=4, beta=c.BETA):
    rows = 2*((np.arange(1 << side)[:, None] >> np.arange(side)) & 1)-1
    horizontal = (rows[:, :-1]*rows[:, 1:]).sum(1)
    row_weight = np.exp(beta*horizontal)
    vertical = np.exp(beta*(rows@rows.T))
    current = row_weight.copy()
    for _ in range(side-1):
        current = (current@vertical)*row_weight
    return float(np.log(current.sum()))


def conditional_table(truth, query, evidence):
    evidence = tuple(evidence)
    if query in evidence or len(set(evidence)) != len(evidence):
        raise ValueError("Query leakage or duplicate evidence")
    code = np.zeros(len(truth["bits"]), dtype=np.int64)
    for slot, e in enumerate(evidence):
        code |= truth["bits"][:, e].astype(np.int64) << slot
    denominator = np.bincount(code, weights=truth["weights"], minlength=1 << len(evidence))
    numerator = np.bincount(code, weights=truth["weights"]*truth["bits"][:, query],
                            minlength=1 << len(evidence))
    if not (denominator > 0).all():
        raise ValueError("Undefined exact conditional")
    return numerator/denominator, denominator


def transform_index(index, side, code):
    r, col = divmod(int(index), side)
    if code & 1:
        r, col = col, r
    if code & 2:
        r = side-1-r
    if code & 4:
        col = side-1-col
    return r*side+col


def geometry_key(query, evidence, side=4):
    return min((transform_index(query, side, code),
                tuple(sorted(transform_index(e, side, code) for e in evidence)))
               for code in range(8))


def make_bank(truth):
    side = int(truth["side"])
    if side != 4:
        raise ValueError("Preregistered bank requires side four")
    queries = [r*side+s for r in (1, 2) for s in (1, 2)]
    layouts = {}
    for k in (1, 2, 4):
        rows = []
        for q in queries:
            candidates = ([tuple(sorted((q-side, q-1, q+1, q+side)))] if k == 4 else
                          itertools.combinations([x for x in range(side*side) if x != q], k))
            for e in candidates:
                rows.append((q, tuple(e), geometry_key(q, e, side)))
        layouts[k] = rows
    result = {key: [] for key in ("tokens", "query", "evidence", "pattern", "k",
                                  "target", "pattern_mass", "train", "orbit")}
    split_record = {}
    for k, layouts_k in layouts.items():
        groups = sorted(set(row[2] for row in layouts_k))
        random = np.random.default_rng(np.random.SeedSequence([c.CONFIG["split_seed"], k]))
        holdout = set(int(x) for x in random.permutation(len(groups))[:max(1, len(groups)//3)]) if k != 4 else set()
        group_index = {key: i for i, key in enumerate(groups)}
        split_record[str(k)] = dict(orbits=len(groups), held_out_orbits=sorted(holdout),
                                   layouts=len(layouts_k), held_out_is_new_geometry=k != 4)
        for q, e, key in layouts_k:
            p, mass = conditional_table(truth, q, e)
            for pattern in range(1 << k):
                tokens = np.full(side*side, 2, dtype=np.int8)
                for slot, index in enumerate(e):
                    tokens[index] = (pattern >> slot) & 1
                evidence = np.full(4, -1, dtype=np.int16)
                evidence[:k] = e
                values = dict(tokens=tokens, query=q, evidence=evidence, pattern=pattern, k=k,
                              target=p[pattern], pattern_mass=mass[pattern],
                              train=group_index[key] not in holdout,
                              orbit="k%d_g%03d" % (k, group_index[key]))
                for name, value in values.items():
                    result[name].append(value)
    bank = {key: np.asarray(value) for key, value in result.items()}
    bank["tokens"] = bank["tokens"].astype(np.int64)[:, None, :]
    bank["query"] = bank["query"].astype(np.int64)
    xy = np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="ij"), -1).reshape(1, 1, side*side, 2)
    bank["coordinates"] = np.repeat(xy.astype(np.float32), len(bank["query"]), axis=0)
    bank["t"] = (1-bank["k"]/(side*side)).astype(np.float32)
    bank["row_id"] = np.arange(len(bank["query"]), dtype=np.int64)
    validate_bank(bank)
    return bank, split_record


def validate_bank(bank):
    rows = np.arange(len(bank["query"]))
    if len(rows) != 1864 or not np.all(bank["tokens"][rows, 0, bank["query"]] == 2):
        raise ValueError("Wrong row count or hidden-query violation")
    if not np.array_equal((bank["tokens"] < 2).sum((1, 2)), bank["k"]):
        raise ValueError("Evidence count mismatch")
    for k in (1, 2):
        train = set(bank["orbit"][(bank["k"] == k) & bank["train"]])
        test = set(bank["orbit"][(bank["k"] == k) & ~bank["train"]])
        if not train or not test or train.intersection(test):
            raise ValueError("Geometry leakage")
    return True


def truth_checks(truth, bank):
    partition_error = abs(float(truth["log_partition"])-row_transfer_log_partition())
    flip_error = float(np.max(np.abs(truth["weights"]-truth["weights"][::-1])))
    marginal_error = float(np.max(np.abs(truth["weights"]@truth["spins"].astype(np.float64))))
    gibbs_error = 0.0
    complement_error = 0.0
    for q in (5, 6, 9, 10):
        e = (q-4, q-1, q+1, q+4)
        p, _ = conditional_table(truth, q, e)
        signs = 2*((np.arange(16)[:, None] >> np.arange(4)) & 1)-1
        analytic = 1/(1+np.exp(-2*c.BETA*signs.sum(1)))
        gibbs_error = max(gibbs_error, float(np.max(np.abs(p-analytic))))
    for k in (1, 2, 4):
        m = bank["k"] == k
        p = bank["target"][m].reshape(-1, 1 << k)
        complement_error = max(complement_error, float(np.max(np.abs(p+p[:, ::-1]-1))))
    maxima = dict(row_transfer_logZ_error=partition_error, spin_flip_weight_error=flip_error,
                  zero_field_magnetization_error=marginal_error, exact_Gibbs_probability_error=gibbs_error,
                  conditional_complement_error=complement_error)
    if max(maxima.values()) > 2e-12:
        raise RuntimeError("Exact truth sanity check failed: " + repr(maxima))
    return dict(status="passed", maxima=maxima, states=len(truth["weights"]),
                rows=len(bank["query"]), MC_sampling_error=0.0,
                finite_system_not_infinite_volume=True)


def k4_extension(bank, natural_clock=False):
    selected = np.flatnonzero(bank["k"] == 4)
    tokens = np.full((len(selected), 1, 64), 2, dtype=np.int64)
    q_new = []
    for row, i in enumerate(selected):
        q = int(bank["query"][i]); r, col = divmod(q, 4)
        q_new.append((r+2)*8+col+2)
        for e in bank["evidence"][i]:
            er, ec = divmod(int(e), 4)
            tokens[row, 0, (er+2)*8+ec+2] = bank["tokens"][i, 0, e]
    xy = np.stack(np.meshgrid(np.arange(8), np.arange(8), indexing="ij"), -1).reshape(1, 1, 64, 2).astype(np.float32)
    return dict(tokens=tokens, coordinates=np.repeat(xy, len(selected), 0),
                t=np.full(len(selected), 1-4/(64 if natural_clock else 16), dtype=np.float32),
                query=np.asarray(q_new, dtype=np.int64), target=bank["target"][selected],
                source_row=selected, natural_clock=np.array(natural_clock))
