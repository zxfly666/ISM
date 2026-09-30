"""Timing only: Wolff kernel on disposable copies of OLD training fields.

No new reference chain is produced, no QA pass is sought, and no state returns
to training or evaluation. Formal MC still uses the untouched original sampler.
"""
import time
import numpy as np
from numba import njit
from ism_diffusion.ising_numba import _wolff_cluster_flip


@njit(cache=True)
def timed_work(spins, seed):
    np.random.seed(seed)
    n = spins.size
    marks = np.zeros(n, dtype=np.int64)
    stack = np.empty(n, dtype=np.int64)
    sites = np.empty(n, dtype=np.int64)
    target = 4*n
    updated = 0; marker = 1
    beta = np.log(1+np.sqrt(2))/2
    while updated < target:
        updated += _wolff_cluster_flip(spins, beta, marks, stack, sites, marker)
        marker += 1
    return updated, marker-1


def worker(args):
    seed, spins = args
    # Tiny compile/warmup is timed separately by the parent as elapsed budget.
    timed_work(np.ones((4, 4), dtype=np.int8), 91827)
    start = time.perf_counter()
    updated, clusters = timed_work(spins.copy(), int(seed))
    return dict(seconds=time.perf_counter()-start, updated_spins=int(updated), clusters=int(clusters))
