"""CPU data adapter for the preregistered fresh geometry factorial.

No checkpoint, evaluation reference, torch, network, or filesystem access.
"""
from __future__ import annotations

import hashlib
import numpy as np
import core_geometry_factorial_design as design


def array_hash(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        a = np.asarray(value)
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


def physical_batch(parent, seed, index, width, kind, batch, augment):
    if parent.lattice_size != 1024:
        raise ValueError('This protocol requires L1024 parents')
    physical = design.axes(width, kind, seed, index, batch)
    ids = design.rng(seed, index, 'parent').integers(0, len(parent.spins), size=batch)
    origin = design.rng(seed, index, 'origin').integers(0, 1024, size=(batch, 2))
    xx = (origin[:, 0, None] + physical[:, 0]) % 1024
    yy = (origin[:, 1, None] + physical[:, 1]) % 1024
    spins = parent.spins[ids[:, None, None], xx[:, :, None], yy[:, None, :]]
    if not np.isin(spins, [-1, 1]).all():
        raise ValueError('Physical parent spins must be -1/+1')
    clean = (spins > 0).astype(np.int64)
    flips = design.rng(seed, index, 'spin_flip').random(batch) < .5 if augment else np.zeros(batch, dtype=bool)
    codes = design.rng(seed, index, 'd4').integers(0, 8, size=batch) if augment else np.zeros(batch, dtype=np.int64)
    clean[flips] = 1 - clean[flips]
    return dict(clean=clean, physical_axes=physical, parent=ids,
                chain=np.asarray(parent.chain_ids)[ids], origin=origin,
                spin_flip=flips, d4=codes, width=width, kind=kind)


def training_batch(parent, seed, step, arm):
    cell = design.schedule(seed, step, arm)
    d = physical_batch(parent, seed, step, cell['width'], cell['kind'], cell['batch'], True)
    clean = d['clean']
    b, w, _ = clean.shape
    n = w*w
    d['coords'] = design.supplied_coordinates(d['physical_axes'], arm, d['d4'])
    # Both coordinate versions are derived from the same sampled physical spins.
    # Including arm in a data RNG key would silently break the G11/G11S control.
    if cell['sparse']:
        k = design.rng(seed, step, 'sparse_k').choice([1,2,4,8,16,32], size=b)
        counts = design.auxiliary_slot_counts(b, seed, step)
        noisy = np.full((b,n), 2, dtype=np.int64)
        queries = np.zeros((b,64), dtype=np.int64)
        valid = np.zeros((b,64), dtype=bool)
        labels = np.zeros((b,64), dtype=np.int64)
        random = design.rng(seed, step, 'sparse_order')
        flat = clean.reshape(b,n)
        for row in range(b):
            order = random.permutation(n)
            q = order[:counts[row]]
            e = order[counts[row]:counts[row]+k[row]]
            noisy[row,e] = flat[row,e]
            queries[row,:counts[row]] = q
            valid[row,:counts[row]] = True
            labels[row,:counts[row]] = flat[row,q]
        t = (1-k/n).astype(np.float32)
        mask = (noisy == 2).reshape(clean.shape)
        noisy = noisy.reshape(clean.shape)
        assert int(valid.sum()) == design.AUX_SLOTS
        assert np.all((noisy != 2).sum((1,2)) == k)
    else:
        t = design.rng(seed, step, 'time').uniform(.01,1.,b).astype(np.float32)
        endpoint = design.rng(seed, step, 'endpoint').random(b) < .02
        t[endpoint] = 1.
        mask = design.rng(seed, step, 'mask').random(clean.shape) < t[:,None,None]
        noisy = np.where(mask,2,clean)
        k = (~mask).sum((1,2))
        queries = np.empty((b,0), dtype=np.int64)
        valid = np.empty((b,0), dtype=bool)
        labels = np.empty((b,0), dtype=np.int64)
    d.update(sparse=cell['sparse'], t=t, mask=mask, noisy=noisy, queries=queries,
             query_valid=valid, labels=labels, k=k, latent_cell=cell['latent_cell'])
    d['paired_data_hash'] = array_hash(clean,d['physical_axes'],d['parent'],d['origin'],
                                      d['spin_flip'],d['d4'],noisy,t,queries,valid,labels)
    d['actual_input_hash'] = array_hash(noisy,d['coords'],t)
    assert clean.size == design.TOKENS
    return d


def conditional_bank(parent, ids, seed, index, width, kind, k):
    """Stateless paired bank; caller chooses validation vs independent test IDs."""
    ids = np.asarray(ids,dtype=np.int64)
    # physical_batch ordinarily chooses random parents. A bank fixes each ID once.
    physical = design.axes(width,kind,seed,index,len(ids))
    origins = design.rng(seed,index,'bank_origin').integers(0,1024,(len(ids),2))
    if parent.lattice_size != 1024 or np.any(ids < 0) or np.any(ids >= len(parent.spins)):
        raise ValueError('Invalid parent bank IDs')
    xx = (origins[:,0,None]+physical[:,0]) % 1024
    yy = (origins[:,1,None]+physical[:,1]) % 1024
    raw = parent.spins[ids[:,None,None],xx[:,:,None],yy[:,None,:]]
    if not np.isin(raw,[-1,1]).all():
        raise ValueError('Invalid reference spins')
    clean = (raw > 0).astype(np.int64)
    obs = design.observations(clean,seed,index,k,64)
    return dict(clean=clean,physical_axes=physical,
                true_coords=design.supplied_coordinates(physical,'G11'),
                summary_coords=design.supplied_coordinates(physical,'G11S'),
                noisy=obs['noisy'].reshape(clean.shape),queries=obs['queries'],
                labels=obs['labels'],evidence=obs['evidence'],t=obs['t'],
                parent=ids,chain=np.asarray(parent.chain_ids)[ids],origin=origins,
                width=width,kind=kind,k=k)
