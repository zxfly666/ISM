"""Deterministic NumPy designs for frozen-model mechanism interventions."""
from __future__ import annotations

import numpy as np


def axis_frequencies(axis_dim=16, base=10000.0):
    return np.exp(-np.log(base) * np.arange(0, axis_dim, 2) / axis_dim)


def ordered_subset(width, visible_index, count, policy, repetition):
    """Keep query, visible spin and bounding-box corners; vary only unobserved sites.

    The query and visible token are always first and second. Nested selections
    use the same ranking across counts. Selection never depends on spin values.
    """
    total = width * width
    fixed = list(dict.fromkeys([0, visible_index, width-1, total-width, total-1]))
    if not 0 < visible_index < total or not len(fixed) <= count <= total:
        raise ValueError("invalid subset size or visible index")
    candidates = np.setdiff1d(np.arange(total), fixed)
    rng = np.random.default_rng(np.random.SeedSequence([92301, visible_index, repetition]))
    noise = rng.random(len(candidates))
    if policy == "uniform":
        score = noise
    else:
        radius2 = (candidates // width)**2 + (candidates % width)**2
        score = (radius2 if policy == "near" else -radius2).astype(float) + noise * .01
        if policy not in ("near", "far"):
            raise ValueError(policy)
    return np.concatenate([fixed, candidates[np.argsort(score)[:count-len(fixed)]]]).astype(np.int64)


def visible_indices(width, axis, lag):
    if axis not in (0, 1) or not 0 < lag < width:
        raise ValueError("invalid query-visible pair")
    return lag * width if axis == 0 else lag


def nested_visibility(n_samples, total, trial, geometry_id):
    rng = np.random.default_rng(np.random.SeedSequence([92311, trial, geometry_id]))
    return np.argsort(rng.random((n_samples, total)), axis=1)


def exact_visible_mask(order, count):
    if not 0 <= count < order.shape[1]:
        raise ValueError("invalid visible count")
    visible = np.zeros(order.shape, bool)
    np.put_along_axis(visible, order[:, :count], True, axis=1)
    return visible


def bernoulli_kl(p, q):
    p, q = np.asarray(p), np.clip(np.asarray(q), 1e-7, 1-1e-7)
    p = np.clip(p, 1e-12, 1-1e-12)
    return p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))


def per_image_scores(probabilities, labels, masked):
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1-1e-7)
    y, m = np.asarray(labels), np.asarray(masked)
    denominator = m.sum(1)
    if np.any(denominator == 0):
        raise ValueError("every image needs at least one target")
    ce = -(y*np.log(p)+(1-y)*np.log1p(-p))
    return dict(ce=(ce*m).sum(1)/denominator,
                brier=(((p-y)**2)*m).sum(1)/denominator,
                mean_probability=(p*m).sum(1)/denominator,
                mean_label=(y*m).sum(1)/denominator,
                masked=denominator)
