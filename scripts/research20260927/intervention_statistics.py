"""J paired seed x chain x circular-block inference and low-K references.

No training, files, or adaptive replication. Every random role is fixed before
results. C and S share MC weights but have independently resampled seed groups.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import t as student_t
import intervention_common as c


def block_weights(chain, random, block):
    groups = [np.flatnonzero(chain == x) for x in np.unique(chain)]
    weights = np.zeros(len(chain))
    for which in random.integers(len(groups), size=len(groups)):
        group = groups[which]
        n = len(group)
        if n == 0 or block < 1:
            raise ValueError("Empty chain or invalid circular block")
        starts = random.integers(n, size=(n+block-1)//block)
        local = ((starts[:, None]+np.arange(block)) % n).ravel()[:n]
        np.add.at(weights, group[local], 1/(len(groups)*n))
    return weights


def resample(groups, chain, reps, block, role, mode="joint", check=lambda: None):
    """List of [seed, arbitrary outcome axes..., parent]; independent seed groups.

    Equal-size independent groups must NOT receive the same seed multinomial.
    Within a group all methods, metrics and views share a multinomial.
    """
    chain = np.asarray(chain)
    if mode not in ("joint", "seed_only", "mc_only"):
        raise ValueError(mode)
    shapes = [np.asarray(x).shape for x in groups]
    arrays = [np.asarray(x, dtype=np.float64).reshape(s[0], -1, s[-1]) for x, s in zip(groups, shapes)]
    if any(x.shape[-1] != len(chain) or not np.isfinite(x).all() for x in arrays):
        raise ValueError("Nonfinite/misaligned parent risks")
    random = c.rng(c.CONFIG["seeds"]["bootstrap"], block, role+"_"+mode)
    result = [np.empty((reps, x.shape[1])) for x in arrays]
    for start in range(0, reps, 64):
        check()
        b = min(64, reps-start)
        pw = np.stack([block_weights(chain, random, block) if mode != "seed_only"
                       else np.full(len(chain), 1/len(chain)) for _ in range(b)])
        for x, out in zip(arrays, result):
            ns = len(x)
            sw = np.stack([np.bincount(random.integers(ns, size=ns), minlength=ns)/ns
                           if mode != "mc_only" else np.full(ns, 1/ns) for _ in range(b)])
            value = np.zeros((b, x.shape[1]))
            for seed in range(ns):
                value += (pw @ x[seed].T)*sw[:, seed, None]
            out[start:start+b] = value
    return [out.reshape((reps,)+s[1:-1]) for out, s in zip(result, shapes)]


def interval(draw, level=.95):
    return np.quantile(draw, [(1-level)/2, (1+level)/2], axis=0)


def summary(per_seed, draw, level=.95, margin=-.005):
    per_seed, draw = np.asarray(per_seed), np.asarray(draw)
    if per_seed.ndim != 1 or draw.ndim != 1 or not np.isfinite(draw).all():
        raise ValueError("Scalar contrast required")
    mean = float(per_seed.mean())
    se = float(per_seed.std(ddof=1)/np.sqrt(len(per_seed)))
    half = float(student_t.ppf((1+level)/2, len(per_seed)-1)*se)
    ci = interval(draw, level)
    endpoints = np.array([interval(g, level) for g in np.array_split(draw, min(10, len(draw)))])
    return dict(estimate=mean, per_seed=per_seed.tolist(), ci95=interval(draw).tolist(),
        ci_level=level, ci=ci.tolist(), margin=margin, directional=bool(ci[1] < 0),
        practical=bool(ci[1] < margin), retained=bool(ci[1] <= margin),
        bootstrap_sd=float(draw.std(ddof=1)), precision_target_met=bool(draw.std(ddof=1) <= .002),
        paired_t_ci=[mean-half, mean+half], leave_one_seed_out=[float(np.delete(per_seed, i).mean()) for i in range(len(per_seed))],
        endpoint_groups=endpoints.tolist(), endpoint_range=np.ptp(endpoints, axis=0).tolist(),
        empirical_interval_not_a_coverage_guarantee=True)


def geometry_groups(layouts):
    """Equivalence of query-relative observed coordinates AND visible values.

    Does not group on pair label, unobserved layout, or only total span. Each
    (pair,side) has the same prior 1/160. Event masses supply posterior weights.
    """
    patterns = [(1,), (0,), (1, 1), (1, 0), (0, 1), (0, 0)]
    keys, mapping = [], {}
    ids = np.empty((80, 2, 6), dtype=np.int64)
    qx, qy = layouts["query"]
    for pair, axis in enumerate(layouts["axes"]):
        for side, a in enumerate(axis):
            q = np.array([a[0, qx], a[1, qy]])
            for condition, values in enumerate(patterns):
                points = []
                for (x, y), value in zip(layouts["visible"], values):
                    xy = np.array([a[0, x], a[1, y]])-q
                    points.append((int(xy[0]), int(xy[1]), int(value)))
                key = tuple(sorted(points))
                if key not in mapping:
                    mapping[key] = len(keys); keys.append(key)
                ids[pair, side, condition] = mapping[key]
    return ids, [str(key) for key in keys]


def low_distribution(counts, grouping):
    count = np.asarray(counts, dtype=np.float64)
    sym = count+count[..., ::-1]
    total = sym.sum(-1, keepdims=True)
    table = np.divide(sym, total, out=np.zeros_like(sym), where=total > 0)
    codes = np.arange(8)
    patterns = [(1,), (0,), (1, 1), (1, 0), (0, 1), (0, 0)]
    den, num = [], []
    for values in patterns:
        selected = ((codes//2) % 2 == values[0])
        if len(values) == 2:
            selected &= codes % 2 == values[1]
        den.append(table[..., selected].sum(-1))
        num.append(table[..., selected & (codes >= 4)].sum(-1))
    den, num = np.stack(den, -1), np.stack(num, -1)
    p = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    group_den = np.bincount(grouping.ravel(), weights=den.ravel(), minlength=int(grouping.max())+1)
    group_num = np.bincount(grouping.ravel(), weights=num.ravel(), minlength=len(group_den))
    pooled = np.divide(group_num, group_den, out=np.full_like(group_num, np.nan), where=group_den > 0)[grouping]
    return dict(pair_probability=p, pooled_probability=pooled, pattern_mass=den,
                rare=den < .01, undefined=den == 0, symmetrized_joint=table)


def risks(truth, pred):
    pc = np.clip(pred, 1e-12, 1-1e-12)
    return np.stack([-(truth*np.log(pc)+(1-truth)*np.log1p(-pc)),
                     truth*(1-pred)**2+(1-truth)*pred**2], -1)


def low_values(counts, grouping, models):
    """models [method,pair,side,condition]; return per-pair/condition quantities."""
    ref = low_distribution(counts, grouping)
    truth, mass = ref["pair_probability"], ref["pattern_mass"]
    mr = risks(truth[None], models)
    pair_risk = risks(truth, truth)
    pooled_risk = risks(truth, ref["pooled_probability"])
    per_k, reference_k = [], []
    for sl in (slice(0, 2), slice(2, 6)):
        weight = .5*mass[..., sl, None]
        per_k.append(np.where(weight[None] > 0, weight[None]*mr[..., sl, :], 0).sum((-3, -2)))
        reference_k.append(np.stack([
            np.where(weight > 0, weight*pair_risk[..., sl, :], 0).sum((-3, -2)),
            np.where(weight > 0, weight*pooled_risk[..., sl, :], 0).sum((-3, -2))], -2))
    return np.stack(per_k, -2), np.stack(reference_k, -3), ref


def low_bootstrap(counts, chain, grouping, models, reps, check=lambda: None):
    """models [seed,method=B0/CD/CO/SD/SO,pair,side,condition].

    Losses are averaged across seeds; probabilities are averaged only for the
    explicitly named response diagnostic, never before computing proper scores.
    """
    random = c.rng(c.CONFIG["seeds"]["bootstrap"], 8, "low_joint")
    output, references, response, ref_response = [], [], [], []
    for rep in range(reps):
        if rep % 10 == 0:
            check()
        pw = block_weights(np.asarray(chain), random, 8)
        table = np.tensordot(pw, counts, axes=(0, 0))
        ids_c, ids_s = random.integers(6, size=6), random.integers(6, size=6)
        selected = np.concatenate([models[ids_c, :3], models[ids_s, 3:]], axis=1)
        values = [low_values(table, grouping, x) for x in selected]
        output.append(np.mean([x[0] for x in values], axis=0))
        references.append(values[0][1])
        response.append((selected[:, :, :, 1]-selected[:, :, :, 0]).mean(0))
        ref_response.append(values[0][2]["pair_probability"][:, 1]-values[0][2]["pair_probability"][:, 0])
    return dict(model_risk=np.array(output), reference_risk=np.array(references),
                model_response=np.array(response), reference_response=np.array(ref_response))
