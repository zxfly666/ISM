"""NumPy-only design and probability identities for the new sparse study."""
from __future__ import annotations
import hashlib
import numpy as np

SEEDS = list(range(91001, 91007))
ARMS = ("T0", "T1", "T2")
WIDTHS = (24, 48, 96)
KS = (1, 2, 4, 8, 16, 32)
MC_SEED = 2026092401
CASE_SEED = 2026092402
GEN_DEFS = [dict(name=f"continuous{w}", width=w, kind="continuous", gaps=[1], samples=n)
            for w, n in ((128, 128), (96, 64), (48, 64))] + [
    dict(name="held_s10_w48", width=48, kind="gap", gaps=[10], samples=64)]
COND_DEFS = [dict(name=f"w{w}_{name}", width=w, **values) for w, name, values in (
    (128, "k8", dict(k=8)), (128, "k32", dict(k=32)),
    (128, "t05", dict(t=.5)), (128, "t095", dict(t=.95)),
    (48, "t05", dict(t=.5)), (48, "t095", dict(t=.95)))]


def rng(seed, index, tag):
    h = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "little")
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(index), h]))


def schedule(seed, step):
    cycle, offset = divmod(step-1, 6)
    cell = int(rng(seed, cycle, "sparse_training_schedule").permutation(6)[offset])
    return WIDTHS[cell % 3], "continuous" if cell < 3 else "gap"


def periodic_correlation(spins):
    spins = np.asarray(spins, np.float64)
    ft = np.fft.rfft2(spins)
    return np.fft.irfft2(ft*ft.conj(), s=spins.shape).real / spins.size


def d4_average(g):
    n = len(g); neg = (-np.arange(n)) % n
    return sum(a for base in (g, g.T) for a in
               (base, base[neg], base[:, neg], base[neg][:, neg])) / 8


def probability_target(g, query_coords, evidence_coords, evidence_bits):
    """Exact zero-field K<=2 identity, evaluated with an estimated valid G."""
    n = len(g); qc = np.asarray(query_coords, np.int64)
    ec = np.asarray(evidence_coords, np.int64)
    signs = 2*np.asarray(evidence_bits, np.int64)-1
    if len(ec) not in (1, 2): raise ValueError("Only K=1/2 has this identity")
    delta = (qc[:, None]-ec[None]) % n
    numerator = (g[delta[..., 0], delta[..., 1]] * signs).sum(-1)
    denominator = 1.
    if len(ec) == 2:
        d = (ec[0]-ec[1]) % n
        denominator += signs.prod()*g[d[0], d[1]]
    if denominator <= 1e-8: raise ValueError("Unresolved conditioning event")
    p = .5*(1+numerator/denominator)
    if not np.isfinite(p).all() or p.min() < -1e-10 or p.max() > 1+1e-10:
        raise ValueError(f"Invalid conditional probability [{p.min()}, {p.max()}]")
    # Only roundoff at an exact probability boundary; not regularization.
    return np.clip(p, 0., 1.)


def sparse_view(clean, coords, seed, index, fixed_k=None):
    clean = np.asarray(clean).reshape(len(clean), -1)
    coords = np.asarray(coords).reshape(len(clean), -1, 2)
    b, n = clean.shape
    random = rng(seed, index, "sparse_evidence_and_queries")
    k = random.choice(KS, b) if fixed_k is None else np.full(b, fixed_k)
    noisy = np.full_like(clean, 2); queries = []; evidence = []
    for i in range(b):
        e = np.sort(random.choice(n, int(k[i]), replace=False))
        hidden = np.setdiff1d(np.arange(n), e, assume_unique=True)
        distance = ((coords[i, hidden, None]-coords[i, e][None])**2).sum(-1).min(-1)
        near = hidden[np.lexsort((hidden, distance))[:32]]
        far = random.choice(np.setdiff1d(hidden, near), 32, replace=False)
        q = np.r_[near, far]
        noisy[i, e] = clean[i, e]
        queries.append(q); evidence.append(e)
    q = np.stack(queries)
    return dict(noisy=noisy, t=(1-k/n).astype(np.float32), queries=q,
                labels=clean[np.arange(b)[:, None], q].astype(np.float64),
                evidence=evidence, k=k, coords=coords)


def make_cases():
    from fixed_geometry_math import make_cases as old_cases
    random = np.random.default_rng(CASE_SEED)
    triplets = []
    for i in range(32):
        if i < 16:
            q = random.integers(4, 44, 2)
            e = q.copy(); e[i % 2] += 1 if (i//2) % 2 else -1
            while True:
                j = random.integers(0, 48, 2)
                if np.linalg.norm(j-q) >= 16: break
            points = np.stack([q, e, j])
        else:
            ids = random.choice(48*48, 3, replace=False)
            points = np.stack([ids//48, ids % 48], -1)
        triplets.append(points.tolist())
    cases = []
    for w in (48, 96, 128):
        for i, points in enumerate(triplets):
            cases.append(dict(id=len(cases), triplet=i, stratum="local" if i<16 else "random",
                geometry=f"matched_continuous{w}", width=w, axis_x=list(range(w)),
                axis_y=list(range(w)), rank_sites=points, physical_sites=points, primary=w==128))
    for c in old_cases():
        if c["primary"]:
            cases.append(dict(c, id=len(cases), primary=False, triplet=-1, stratum="retention"))
    assert len(cases) == 116
    return cases


def self_test():
    random = np.random.default_rng(9723)
    spins = random.choice([-1, 1], (16, 16))
    g = periodic_correlation(spins)
    error = max(abs(g[x,y] - (spins*np.roll(spins, (-x,-y), (0,1))).mean())
                for x in range(16) for y in range(16))
    assert error < 1e-12 and abs(g[0,0]-1) < 1e-12
    # Three-point histogram on the exact same periodic empirical field.
    sites = np.array([[2,3],[4,5],[7,6]])
    values = np.stack([np.roll(spins, -p, (0,1)).ravel() for p in sites], -1)
    values = np.concatenate([values, -values])
    for k in (1, 2):
        for bits in ([0],[1]) if k==1 else ([0,0],[0,1],[1,0],[1,1]):
            selected = (values[:,1:1+k] == (2*np.array(bits)-1)).all(1)
            truth = (values[selected,0] > 0).mean()
            p = probability_target(g, sites[:1], sites[1:1+k], bits)[0]
            assert abs(p-truth) < 1e-12
    sym = d4_average(g); neg = (-np.arange(16)) % 16
    assert np.max(abs(sym-sym.T)) < 1e-12
    assert np.max(abs(sym-sym[neg])) < 1e-12
    cc = np.stack(np.meshgrid(np.arange(16),np.arange(16),indexing="ij"),-1)[None]
    view = sparse_view((spins[None]>0).astype(int), cc, 12, 1, 2)
    assert (view["noisy"] != 2).sum() == 2
    assert len(np.unique(view["queries"])) == 64
    assert (view["noisy"][0,view["queries"][0]] == 2).all()
    cases = make_cases()
    assert cases[0]["physical_sites"] == cases[32]["physical_sites"] == cases[64]["physical_sites"]
    return dict(status="passed",fft_direct_max_error=float(error),conditional_identity=True,
                d4_modular_negative_indices=True,sparse_queries_hidden_unique=True,matched_cases=True)
