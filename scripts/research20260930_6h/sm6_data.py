"""Exact shielded-cavity truth, symmetry-group split and physical pairing."""
from __future__ import annotations

import hashlib
from functools import lru_cache
import numpy as np

import sm6_common as c

CAVITY = np.array(c.EXACT_G8['cavity'], dtype=np.int64)
BOUNDARY = np.array(c.EXACT_G8['boundary_bit_order'], dtype=np.int64)
FIELDS = ('tokens', 'coordinates', 'valid', 't', 'query', 'target', 'row_id')


def transform(xy, code):
    v = np.asarray(xy).copy()
    if code & 1:
        v = v[..., ::-1]
    if code & 2:
        v[..., 0] = 3-v[..., 0]
    if code & 4:
        v[..., 1] = 3-v[..., 1]
    return v


def orbit_key(q, pattern):
    keys = []
    for code in range(8):
        qq = int(np.flatnonzero((CAVITY == transform(CAVITY[q], code)).all(1))[0])
        p = 0
        for i, xy in enumerate(transform(BOUNDARY, code)):
            j = int(np.flatnonzero((BOUNDARY == xy).all(1))[0])
            p |= ((pattern >> i) & 1) << j
        keys += [256*qq+p, 256*qq+(255-p)]
    return min(keys)


@lru_cache(maxsize=1)
def _cavity_bank():
    spins = 2*((np.arange(16)[:, None] >> np.arange(4)) & 1)-1
    bits = (np.arange(256)[:, None] >> np.arange(8)) & 1
    boundary = 2*bits-1
    internal = [(i, j) for i in range(4) for j in range(i+1, 4)
                if np.abs(CAVITY[i]-CAVITY[j]).sum() == 1]
    external = [(i, j) for i in range(4) for j in range(8)
                if np.abs(CAVITY[i]-BOUNDARY[j]).sum() == 1]
    energy = sum(spins[:, i]*spins[:, j] for i, j in internal)[None, :] + np.zeros((256, 16))
    for i, j in external:
        energy += boundary[:, j, None]*spins[None, :, i]
    weights = np.exp(c.BETA*(energy-energy.max(1, keepdims=True)))
    weights /= weights.sum(1, keepdims=True)
    probs = weights @ ((spins+1)/2)
    tokens = np.full((1024, 1, 16), 2, dtype=np.int64)
    query = np.repeat(CAVITY[:, 0]*4+CAVITY[:, 1], 256)
    for q in range(4):
        tokens[q*256:(q+1)*256, 0, BOUNDARY[:, 0]*4+BOUNDARY[:, 1]] = bits
    keys = np.array([orbit_key(q, p) for q in range(4) for p in range(256)], dtype=np.int64)
    ordered = sorted(set(keys.tolist()), key=lambda k: hashlib.sha256(('2026093002|'+str(k)).encode('ascii')).hexdigest())
    split = np.full(1024, 'test', dtype='<U10')
    split[np.isin(keys, ordered[:48])] = 'train'
    split[np.isin(keys, ordered[48:60])] = 'validation'
    result = dict(tokens=tokens, query=query.astype(np.int64), target=probs.T.ravel(),
                  k=np.full(1024, 8, np.int64), row_id=np.arange(1024, dtype=np.int64),
                  orbit_key=keys, split=split, pattern=np.tile(np.arange(256), 4))
    for name, expected in c.EXACT_G8['split_row_counts'].items():
        assert int((split == name).sum()) == expected
        assert set(keys[split == name]) == set(c.EXACT_G8[name+'_orbit_keys'])
    return result


def cavity_bank():
    return {k: v.copy() for k, v in _cavity_bank().items()}


@lru_cache(maxsize=1)
def _parent_bank():
    assert c.sha(c.PARENT) == c.CFG['data']['K1_K2_parent_bank_sha256']
    b = c.load(c.PARENT)
    b['split'] = np.where(b['train'], 'train', 'holdout')
    return b


def parent_bank():
    return {k: v.copy() for k, v in _parent_bank().items()}


def subset(bank, ix):
    n = len(bank['query'])
    return {k: v[ix].copy() for k, v in bank.items() if np.asarray(v).ndim and len(v) == n}


def view(bank, side, arm, placement='center', allocated=None, fixed_native_clock=False):
    n = len(bank['query'])
    assert side >= 4 and side % 2 == 0 and arm in c.CFG['arms']
    if np.any(bank['k'] < 4):
        assert side == 4 and placement == 'center'
    offsets = {'center': ((side-4)//2, (side-4)//2), 'NW': (0, 0), 'NE': (0, side-4),
               'SW': (side-4, 0), 'SE': (side-4, side-4)}
    off = np.asarray(offsets[placement])
    allocated = side*side if allocated is None else allocated
    assert allocated >= side*side
    tokens = np.full((n, 1, allocated), 3, dtype=np.int64)
    tokens[:, :, :side*side] = 2
    valid = np.zeros_like(tokens, dtype=bool)
    valid[:, :, :side*side] = True
    coordinates = np.zeros((n, 1, allocated, 2), dtype=np.float32)
    xy = np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing='ij'), -1).reshape(-1, 2)
    coordinates[:, 0, :side*side] = xy
    for i in range(n):
        evidence = np.flatnonzero(bank['tokens'][i, 0] < 2)
        translated = np.stack(np.divmod(evidence, 4), -1)+off
        tokens[i, 0, translated[:, 0]*side+translated[:, 1]] = bank['tokens'][i, 0, evidence]
    qxy = np.stack(np.divmod(bank['query'], 4), -1)+off
    query = (qxy[:, 0]*side+qxy[:, 1]).astype(np.int64)
    requested = (1-bank['k']/(side*side)).astype(np.float32)
    canonical = c.CFG['arms'][arm]['clock'] == 'constant'
    t = np.full(n, .75, dtype=np.float32) if canonical or fixed_native_clock else requested.copy()
    result = dict(tokens=tokens, coordinates=coordinates, valid=valid, query=query,
                  target=bank['target'].astype(np.float64), t=t, requested_t=requested,
                  row_id=bank['row_id'].copy(), k=bank['k'].copy())
    validate_view(result)
    return result


def validate_view(bank):
    tok, valid, q = bank['tokens'], bank['valid'], bank['query']
    rows = np.arange(len(q))
    assert (tok[rows, 0, q] == 2).all() and valid[rows, 0, q].all()
    assert np.array_equal(((tok < 2) & valid).sum((1, 2)), bank['k'])
    assert (tok[~valid] == 3).all() and np.isfinite(bank['target']).all()
    assert ((bank['target'] > 0) & (bank['target'] < 1)).all()


def common_digest(parts):
    return c.digest(*[p[k] for p in parts for k in ('row_id', 'tokens', 'query', 'target')])


def native_digest(parts):
    return c.digest(*[p[k] for p in parts for k in FIELDS])


class TrainingData:
    def __init__(self):
        self.parent = parent_bank()
        self.g8 = cavity_bank()
        self.small = subset(self.parent, np.flatnonzero(self.parent['k'] < 4))
        self.k4 = subset(self.parent, np.flatnonzero(self.parent['k'] == 4))
        # Family IDs use separate ranges, so no collision with inherited bank row IDs.
        self.g8['row_id'] = self.g8['row_id'] + 10000
        self.large = {key: np.concatenate([self.k4[key], self.g8[key]]) for key in
                      ('tokens', 'query', 'target', 'k', 'row_id', 'split')}
        self.pools = [np.flatnonzero((self.small['k'] == k) & (self.small['split'] == 'train')) for k in (1, 2)]
        self.pools += [np.flatnonzero(self.large['k'] == 4), np.flatnonzero((self.large['k'] == 8) & (self.large['split'] == 'train'))]
        self.cache = {}
        for arm in c.CFG['arms']:
            for phase, side in enumerate(c.CFG['arms'][arm]['valid_side_cycle']):
                allocated = c.CFG['training']['allocated_side_cycle'][phase]**2
                self.cache[arm, phase] = (view(self.small, 4, arm), view(self.large, side, arm, allocated=allocated))

    def indices(self, seed, step):
        r = c.rng(seed, step, 'paired_physical_examples')
        return [r.choice(pool, 32, replace=True) for pool in self.pools]

    def batch(self, seed, step, arm):
        indices = self.indices(seed, step)
        sids, lids = np.concatenate(indices[:2]), np.concatenate(indices[2:])
        phase = (step-1) % 4
        a, b = self.cache[arm, phase]
        parts = [subset(a, sids), subset(b, lids)]
        common = common_digest([subset(self.small, sids), subset(self.large, lids)])
        return parts, dict(paired_data_digest=common, actual_input_digest=native_digest(parts),
                          valid_side=c.CFG['arms'][arm]['valid_side_cycle'][phase],
                          allocated_tokens=sum(p['tokens'].size for p in parts),
                          valid_tokens=sum(int(p['valid'].sum()) for p in parts))


def exact_truth_checks():
    bank = cavity_bank()
    bits = ((np.arange(65536, dtype=np.uint32)[:, None] >> np.arange(16)) & 1).astype(np.int8)
    spins = 2*bits-1
    edges = [(4*r+s, 4*rr+ss) for r in range(4) for s in range(4)
             for rr, ss in ((r+1, s), (r, s+1)) if rr < 4 and ss < 4]
    e = sum(spins[:, i]*spins[:, j] for i, j in edges)
    w = np.exp(c.BETA*(e-e.max()))
    code = sum(bits[:, 4*r+s].astype(np.int64) << i for i, (r, s) in enumerate(BOUNDARY))
    den = np.bincount(code, weights=w, minlength=256)
    ps = np.concatenate([np.bincount(code, weights=w*bits[:, 4*r+s], minlength=256)/den for r, s in CAVITY])
    diff = float(np.max(np.abs(ps-bank['target'])))
    flip = float(np.max(np.abs(bank['target'].reshape(4, 256)+bank['target'].reshape(4, 256)[:, ::-1]-1)))
    assert max(diff, flip) < 2e-12
    # Transformed conditionals must agree, or complement under global flip.
    for q in range(4):
        for p in range(256):
            for code in range(8):
                qq = int(np.flatnonzero((CAVITY == transform(CAVITY[q], code)).all(1))[0])
                pp = 0
                for i, xy in enumerate(transform(BOUNDARY, code)):
                    j = int(np.flatnonzero((BOUNDARY == xy).all(1))[0])
                    pp |= ((p >> i) & 1) << j
                assert abs(bank['target'][256*q+p]-bank['target'][256*qq+pp]) < 2e-12
    parent = parent_bank()
    assert len(parent['query']) == 1864
    k4 = subset(parent, np.flatnonzero(parent['k'] == 4))
    for side in c.CFG['evaluation']['center_sides']:
        b = view(k4, side, 'N')
        signs = ((2*b['tokens']-1)*((b['tokens'] < 2) & b['valid'])).sum((1, 2))
        exact = 1/(1+np.exp(-2*c.BETA*signs))
        assert np.max(np.abs(exact-b['target'])) < 2e-12
    return dict(status='passed', cavity_rows=1024, symmetry_orbits=72,
                full_enumeration_error=diff, flip_error=flip, inherited_bank_sha=c.sha(c.PARENT),
                witness_probabilities=bank['target'][[240, 85]].tolist())


def evaluation_specs():
    # Immutable input manifest; model outputs are only produced after final locking.
    for k in (1, 2):
        for split in ('train', 'holdout'):
            yield dict(key=f'K{k}_{split}', family=f'K{k}', split=split, side=4, placement='center')
    for side in c.CFG['evaluation']['center_sides']:
        yield dict(key=f'K4_N{side}_center', family='K4', split='train', side=side, placement='center')
        for split in (('train', 'validation', 'test') if side in c.CFG['evaluation']['train_validation_center_sides'] else ('test',)):
            yield dict(key=f'G8_{split}_N{side}_center', family='G8', split=split, side=side, placement='center')
    for side in c.CFG['evaluation']['corner_sides']:
        for placement in ('NW', 'NE', 'SW', 'SE'):
            for family, split in (('K4', 'train'), ('G8', 'test')):
                yield dict(key=f'{family}_{split}_N{side}_{placement}', family=family, split=split, side=side, placement=placement, raw_only=True)


def bank_for_spec(spec, arm):
    # Read-only cache: all transformations in controls copy before changing data.
    return _bank_for_spec(spec['family'], spec['split'], spec['side'], spec['placement'], arm)


@lru_cache(maxsize=128)
def _bank_for_spec(family, split, side, placement, arm):
    b = cavity_bank() if family == 'G8' else parent_bank()
    mask = b['split'] == split
    if family != 'G8':
        mask &= b['k'] == int(family[1:])
    b = subset(b, np.flatnonzero(mask))
    return view(b, side, arm, placement)
