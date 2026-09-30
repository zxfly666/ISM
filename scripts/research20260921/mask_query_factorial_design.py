"""Four-arm mask/query design; no evaluation data or soft-label oracle."""
import numpy as np
from scipy.spatial import cKDTree
from sparse_conditioning_design import rng, sparse_view, make_cases as _cases

SEEDS = list(range(91001, 91007))
ARMS = ('R00', 'R01', 'R10', 'R11')
MC_SEED = 2026092411
CASE_SEED = 2026092412
GEN_DEFS = [dict(name='continuous128', width=128, kind='continuous', gaps=[1], samples=96),
            dict(name='continuous48', width=48, kind='continuous', gaps=[1], samples=64),
            dict(name='held_s10_w48', width=48, kind='gap', gaps=[10], samples=64)]
from sparse_conditioning_design import COND_DEFS


def make_cases():
    import sparse_conditioning_design as original
    before = original.CASE_SEED
    try:
        original.CASE_SEED = CASE_SEED
        return _cases()
    finally:
        original.CASE_SEED = before


def query_policies(hidden, evidence, coords, random):
    """64 loss slots; exact nearest distance; duplicate slots only when required."""
    assert len(hidden) > 0
    uniform = random.choice(hidden, 64, replace=len(hidden) < 64)
    if len(evidence) == 0:
        return uniform, uniform.copy()
    distances = cKDTree(coords[evidence]).query(coords[hidden], k=1, eps=0, workers=1)[0]
    near = hidden[np.lexsort((hidden, distances))[:min(32, len(hidden))]]
    pool = np.setdiff1d(hidden, near, assume_unique=True)
    if len(pool) == 0:
        pool = hidden
    count = 64 - len(near)
    rest = random.choice(pool, count, replace=len(pool) < count)
    return uniform, np.concatenate([near, rest])


def auxiliary_views(aux, seed, index):
    from ism_diffusion import geometry_study as gs
    clean = aux['clean'].reshape(len(aux['clean']), -1)
    coords = aux['coords']['A'].reshape(len(clean), -1, 2)
    ordinary = aux['noisy'].reshape(clean.shape).copy()
    retries = 0
    for i, row in enumerate(ordinary):
        while not (row == 2).any():
            retries += 1
            if retries > 100:
                raise RuntimeError('No-hidden ordinary mask retry limit')
            _, _, xx = gs.corrupt_batch(aux['clean'][i:i+1], seed, index*1000+retries)
            row[:] = xx.reshape(-1)
    sp = sparse_view(aux['clean'], aux['coords']['A'], seed, index)
    result = {}
    for factor, noisy in enumerate((ordinary, sp['noisy'])):
        q0, q1, ks = [], [], []
        for i, row in enumerate(noisy):
            hidden = np.flatnonzero(row == 2)
            evidence = np.flatnonzero(row != 2)
            random = rng(seed, index*100+i, f'factorial_queries_m{factor}')
            u, near = query_policies(hidden, evidence, coords[i], random)
            q0.append(u); q1.append(near); ks.append(len(evidence))
        k = np.asarray(ks)
        for query_factor, qq in enumerate((q0, q1)):
            queries = np.stack(qq)
            assert (noisy[np.arange(len(clean))[:, None], queries] == 2).all()
            result[f'R{factor}{query_factor}'] = dict(
                noisy=noisy, coords=coords, t=(1-k/clean.shape[1]).astype(np.float32),
                queries=queries, labels=clean[np.arange(len(clean))[:, None], queries],
                k=k, unique_queries=np.array([len(np.unique(x)) for x in queries]))
    return result, retries


def self_test():
    random = np.random.default_rng(2026092419)
    coords = random.normal(size=(81, 2))
    evidence = np.array([0, 12, 33]); hidden = np.setdiff1d(np.arange(81), evidence)
    u, q = query_policies(hidden, evidence, coords, np.random.default_rng(1))
    brute = ((coords[hidden, None]-coords[evidence][None])**2).sum(-1).min(-1)
    expected = hidden[np.lexsort((hidden, brute))[:32]]
    assert np.array_equal(q[:32], expected)
    for size in (1, 10, 32, 50, 64, 70):
        h = np.arange(1, size+1); e = np.array([0])
        a,b = query_policies(h,e,coords,np.random.default_rng(2))
        assert len(a)==len(b)==64 and np.isin(a,h).all() and np.isin(b,h).all()
        if size>=64:
            assert len(np.unique(a))==len(np.unique(b))==64
    a,b = query_policies(np.arange(81),np.array([],dtype=int),coords,np.random.default_rng(3))
    assert np.array_equal(a,b)
    cases = make_cases()
    assert len(cases)==116 and cases[0]['physical_sites']==cases[32]['physical_sites']==cases[64]['physical_sites']
    # Known cell means with interaction 5: 00=0,01=2,10=3,11=10.
    x=np.array([0.,2.,3.,10.])
    assert (np.array([1.,-1.,-1.,1.])@x)==5
    return dict(status='passed',nearest_matches_bruteforce=True,duplicate_boundaries=True,
                zero_evidence_policies_identical=True,cases=116,factorial_algebra=True)
