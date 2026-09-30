"""Three-spin references and two-query composition; NumPy only.

Zero-field spin-flip symmetry is exact. The reference probabilities obtained
from finite MC histograms are estimates, not exact numerical ground truth.
"""
from __future__ import annotations
import numpy as np

BITS = ((np.arange(8)[:, None] >> np.arange(3)) & 1)
SIGNS = 2 * BITS - 1


def symmetric_distribution(counts):
    counts = np.asarray(counts, dtype=np.float64)
    result = (counts + counts[..., ::-1]) / (2 * counts.sum(-1, keepdims=True))
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("Invalid MC counts")
    return result


def two_query_reference(p):
    # p bit order: query i (bit 0), evidence e (bit 1), query j (bit 2).
    codes = np.array([[[x + 2*a + 4*y for y in (0, 1)] for x in (0, 1)] for a in (0, 1)])
    joint = np.asarray(p)[..., codes]
    return joint / joint.sum((-1, -2), keepdims=True)


def conditional_q(p, a, b):
    return p[..., 1+2*a+4*b] / (p[..., 2*a+4*b]+p[..., 1+2*a+4*b])


def query_specs(n):
    """Each row fixes tokens and time; swapped rows move coordinates ONLY."""
    rows = []
    for a in (0, 1):
        rows.append(dict(kind="base", a=a, reveal=-1, bit=-1, clock="natural", swap=False, t=1-1/n))
        for clock in ("natural", "fixed"):
            for reveal in (0, 2):
                for bit in (0, 1):
                    rows.append(dict(kind="conditional", a=a, reveal=reveal, bit=bit,
                                     clock=clock, swap=False, t=1-(2 if clock == "natural" else 1)/n))
    for a in (0, 1):
        rows.append(dict(kind="swap", a=a, reveal=2, bit=1-a, clock="natural", swap=True, t=1-2/n))
    return rows


def index_of(specs, kind, a, reveal=-1, bit=-1, clock="natural"):
    return next(i for i, s in enumerate(specs) if all(s[k] == v for k, v in
                dict(kind=kind, a=a, reveal=reveal, bit=bit, clock=clock).items()))


def bernoulli(p):
    return np.stack((1-p, p), -1)


def kl(p, q):
    p, q = np.asarray(p), np.asarray(q)
    return np.sum(np.where(p > 0, p*(np.log(np.clip(p, 1e-300, None))-
                                   np.log(np.clip(q, 1e-12, None))), 0), axis=-1)


def compositions(pred, specs, clock="natural"):
    """Return normalized learned joints [..., evidence, i, j]."""
    forwards, reverses, parallels = [], [], []
    for a in (0, 1):
        base = pred[..., index_of(specs, "base", a), :]
        pi, pj = bernoulli(base[..., 0]), bernoulli(base[..., 2])
        j_after_i = np.stack([bernoulli(pred[..., index_of(specs, "conditional", a, 0, x, clock), 2])
                              for x in (0, 1)], -2)
        i_after_j = np.stack([bernoulli(pred[..., index_of(specs, "conditional", a, 2, y, clock), 0])
                              for y in (0, 1)], -1)
        forwards.append(pi[..., :, None] * j_after_i)
        reverses.append(pj[..., None, :] * i_after_j)
        parallels.append(pi[..., :, None] * pj[..., None, :])
    return tuple(np.stack(v, -3) for v in (forwards, reverses, parallels))


def metrics(pred, p, specs):
    """Broadcast leading dimensions, retain the case dimension. No seed pooling."""
    ref = two_query_reference(p)
    forward, reverse, parallel = compositions(pred, specs)
    ff, rr, _ = compositions(pred, specs, "fixed")
    flatref = ref.reshape(*ref.shape[:-2], 4)
    div = lambda q: kl(flatref, q.reshape(*q.shape[:-2], 4)).mean(-1)
    mi = ref.sum(-1)
    mj = ref.sum(-2)
    oracle_parallel = mi[..., :, None] * mj[..., None, :]
    factor = div(oracle_parallel)
    marginal = (kl(mi, parallel.sum(-1)) + kl(mj, parallel.sum(-2))).mean(-1)
    conditional_terms = []
    for q in (forward, reverse):
        first_axis = -1 if q is forward else -2
        # Explicit chain-rule check, independent of flattening joint KL.
        rp, qp = ref.sum(first_axis), q.sum(first_axis)
        if first_axis == -1:
            cond = kl(ref/rp[..., :, None], q/qp[..., :, None])
        else:
            cond = kl(np.swapaxes(ref/rp[..., None, :], -1, -2),
                      np.swapaxes(q/qp[..., None, :], -1, -2))
        conditional_terms.append((kl(rp, qp)+(rp*cond).sum(-1)).mean(-1))
    standard = pred[..., index_of(specs, "conditional", 0, 2, 1), 0]
    swapped = pred[..., index_of(specs, "swap", 0, 2, 1), 0]
    truth0, truth1 = conditional_q(p, 0, 1), conditional_q(p, 1, 0)
    response = swapped-standard
    target = truth1-truth0
    reverse_standard = pred[..., index_of(specs, "conditional", 1, 2, 0), 0]
    reverse_swap = pred[..., index_of(specs, "swap", 1, 2, 0), 0]
    result = dict(
        response_squared_error=(response-target)**2,
        response=response, reference_response=target,
        balanced_conditional_kl=(kl(bernoulli(truth0), bernoulli(standard))+
                                 kl(bernoulli(truth1), bernoulli(swapped)))/2,
        geometry_probability_mae=(abs(standard-truth0)+abs(swapped-truth1))/2,
        sequential_kl=(div(forward)+div(reverse))/2,
        forward_kl=div(forward), reverse_kl=div(reverse), parallel_kl=div(parallel),
        oracle_parallel_kl=factor, marginal_kl=marginal,
        order_tv=(abs(forward-reverse).sum((-1, -2))/2).mean(-1),
        fixed_clock_order_tv=(abs(ff-rr).sum((-1, -2))/2).mean(-1),
        fixed_clock_sequential_kl=(div(ff)+div(rr))/2,
        symmetry_error=abs(standard+reverse_standard-1),
        swap_permutation_error=np.maximum(abs(swapped-reverse_standard),abs(reverse_swap-standard)),
        parallel_identity_residual=abs(div(parallel)-factor-marginal),
        sequential_identity_residual=np.maximum(abs(div(forward)-conditional_terms[0]),
                                                 abs(div(reverse)-conditional_terms[1])))
    return result


def make_cases():
    cases = []
    for name, w, stride in (("continuous48",48,1),("s10_w48",48,10),("continuous96",96,1)):
        ds = [2,4,8,16,19,32,38] + ([64,80] if w == 96 else [])
        for axis in (0, 1):
            for anchor in (0,4):
                for d in ds:
                    points = [[w//2,w//2] for _ in range(3)]
                    for point, loc in zip(points, (anchor,anchor+1,anchor+d)):
                        point[axis] = loc
                    cases.append(dict(geometry=name,width=w,axis=axis,anchor=anchor,d=d,
                        axis_x=(np.arange(w)*stride).tolist(),axis_y=(np.arange(w)*stride).tolist(),
                        rank_sites=points,primary=False,mirror=0))
    for mirror in (0,1):
        gaps = [1]*23+[10]*24
        if mirror: gaps = gaps[::-1]
        # Use symmetric ranks around the change point in either orientation.
        center = 24 if mirror else 23
        axis_values = np.r_[0,np.cumsum(gaps)].tolist()
        for axis in (0,1):
            for d in (1,2,4,8,16):
                points = [[center,center] for _ in range(3)]
                for point, loc in zip(points, (center,center-d,center+d)):
                    point[axis] = loc
                cases.append(dict(geometry="unequal_distance_w48",width=48,axis=axis,
                    anchor=center,d=d,axis_x=axis_values,axis_y=axis_values,
                    rank_sites=points,primary=True,mirror=mirror))
    for i, case in enumerate(cases):
        case["id"] = i
        case["physical_sites"] = [[case["axis_x"][x],case["axis_y"][y]] for x,y in case["rank_sites"]]
    return cases


def case_coordinates(case, representation):
    w = case["width"]
    if representation == "A":
        axes = [np.array(case["axis_x"]), np.array(case["axis_y"])]
    elif representation == "B":
        axes = [np.arange(w),np.arange(w)]
    elif representation == "C":
        random = np.random.default_rng(9235000+w)
        continuous = case["geometry"].startswith("continuous")
        axes = [np.r_[0,np.cumsum(random.choice([1] if continuous else [1,2,4,8],w-1))] for _ in (0,1)]
    else:
        raise ValueError(representation)
    flat = np.stack(np.meshgrid(*axes,indexing="ij"),-1).reshape(-1,2).astype(np.float32)
    sites = [x*w+y for x,y in case["rank_sites"]]
    perm = np.r_[sites,np.setdiff1d(np.arange(w*w),sites)]
    return flat[perm], perm


def case_inputs(case, representation):
    n = case["width"]**2
    coords, _ = case_coordinates(case, representation)
    specs = query_specs(n)
    tokens = np.full((len(specs),1,n),2,np.int64)
    coordinates = np.broadcast_to(coords,(len(specs),1,n,2)).copy()
    for i,s in enumerate(specs):
        tokens[i,0,1] = s["a"]
        if s["reveal"] >= 0: tokens[i,0,s["reveal"]] = s["bit"]
        if s["swap"]: coordinates[i,0,[1,2]] = coordinates[i,0,[2,1]]
    return tokens, coordinates, np.array([s["t"] for s in specs],np.float32), specs
