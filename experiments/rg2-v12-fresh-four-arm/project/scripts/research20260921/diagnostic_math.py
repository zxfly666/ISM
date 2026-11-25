"""NumPy-only diagnostic references, independent of the production G implementation."""
from __future__ import annotations

import numpy as np


def uniform_axis_fft(spins):
    """Per-image open x/y-axis G at index lag; zero-padding prevents wraparound."""
    x = np.asarray(spins, dtype=np.float64)
    n, w, h = x.shape
    if w != h:
        raise ValueError("square grids required")
    fx = np.fft.rfft(x, n=2*w, axis=1)
    fy = np.fft.rfft(x, n=2*w, axis=2)
    cx = np.fft.irfft(fx * fx.conj(), n=2*w, axis=1)[:, :w, :].sum(2)
    cy = np.fft.irfft(fy * fy.conj(), n=2*w, axis=2)[:, :, :w].sum(1)
    counts = 2*w*(w-np.arange(w))
    return (cx+cy)/counts[None, :], counts


def direct_axis(spins):
    """Slow independent slicing reference used only in tests."""
    x = np.asarray(spins, dtype=float)
    w = x.shape[-1]
    return np.stack([np.ones(len(x))] + [
        .5*((x[:, d:, :]*x[:, :-d, :]).mean((1, 2))
            +(x[:, :, d:]*x[:, :, :-d]).mean((1, 2)))
        for d in range(1, w)], axis=1)


def cavity_distribution(full_spins, sites, beta):
    """Exact conditional law given the complete exterior of an internal cavity.

    All internal and cavity-to-exterior nearest-neighbour bonds are included.
    This is NOT an isolated free-boundary Ising patch.
    """
    full = np.asarray(full_spins, dtype=float)
    sites = [tuple(s) for s in sites]
    lookup = {site: i for i, site in enumerate(sites)}
    n = len(sites)
    bits = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(np.int8)
    states = 2*bits.astype(float)-1
    score = np.zeros(len(bits))
    bonds = []
    for i, (x, y) in enumerate(sites):
        if not (0 < x < full.shape[0]-1 and 0 < y < full.shape[1]-1):
            raise ValueError("complete exterior boundary must be inside the stored field")
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            other = (x+dx, y+dy)
            if other in lookup:
                j = lookup[other]
                if i < j:
                    score += states[:, i]*states[:, j]
                    bonds.append((i, j))
            else:
                score += states[:, i]*full[other]
    logp = beta*score
    logp -= logp.max()
    p = np.exp(logp)
    p /= p.sum()
    return bits, p, bonds


def conditional_stages(bits, reference, plan):
    """Exact single-site conditionals for each prefix of a fixed reveal plan."""
    n = bits.shape[1]
    flat = [int(i) for group in plan for i in group]
    if sorted(flat) != list(range(n)):
        raise ValueError("plan must partition the cavity exactly once")
    prior = []
    stages = []
    for group in plan:
        weights = 2**np.arange(len(prior), dtype=np.int64)
        codes = bits[:, prior] @ weights if prior else np.zeros(len(bits), dtype=np.int64)
        assignments = ((np.arange(2**len(prior))[:, None] >> np.arange(len(prior))) & 1).astype(np.int8)
        mass = np.bincount(codes, weights=reference, minlength=len(assignments))
        probabilities = np.stack([np.bincount(codes, weights=reference*bits[:, i],
                                              minlength=len(assignments))/mass for i in group], axis=1)
        stages.append(dict(prior=list(prior), group=list(group), codes=codes,
                           assignments=assignments, probability=probabilities, mass=mass))
        prior.extend(group)
    return stages


def joint_from_conditionals(bits, stages, probabilities):
    logq = np.zeros(len(bits))
    for stage, pp in zip(stages, probabilities):
        q = np.clip(np.asarray(pp, dtype=np.float64)[stage["codes"]], 1e-12, 1-1e-12)
        target = bits[:, stage["group"]]
        logq += (target*np.log(q)+(1-target)*np.log1p(-q)).sum(1)
    q = np.exp(logq)
    if not np.isclose(q.sum(), 1., atol=1e-8):
        raise AssertionError(f"joint distribution not normalized: {q.sum()}")
    return q


def divergence(reference, predicted):
    p, q = np.asarray(reference), np.asarray(predicted)
    return dict(kl=float(np.sum(p*(np.log(p)-np.log(np.clip(q, 1e-300, None))))),
                tv=float(np.abs(p-q).sum()/2))


def kl_decomposition(bits, reference, stages, learned, oracle_joint):
    single = 0.
    for stage, pred in zip(stages, learned):
        p = np.clip(stage["probability"], 1e-12, 1-1e-12)
        q = np.clip(np.asarray(pred, dtype=np.float64), 1e-12, 1-1e-12)
        d = p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))
        single += float((d.sum(1)*stage["mass"]).sum())
    learned_joint = joint_from_conditionals(bits, stages, learned)
    learned_kl = divergence(reference, learned_joint)["kl"]
    factorization = divergence(reference, oracle_joint)["kl"]
    residual = learned_kl - single - factorization
    if abs(residual) > 1e-8:
        raise AssertionError(f"KL decomposition mismatch: {residual}")
    return learned_joint, dict(learned_kl=learned_kl, single_conditional_kl=single,
                              parallel_factorization_kl=factorization, identity_residual=residual)
