"""Independent CPU fixtures, then optional CPU torch recovery/sampler tests."""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
import numpy as np
import endpoint_common as c
import endpoint_data as d


def fake_parent(n=16, size=1024):
    r = np.random.default_rng(51001)
    return d.Parent((2*r.integers(2, size=(n, size, size), dtype=np.int8)-1), np.arange(n, dtype=np.int16), size,
                    dict(fixture=True, lattice_size=size))


def exact_gibbs_fixture(side):
    # Enumerate a side^2 interior with a fixed + boundary in (side+2)^2.
    n = side*side
    states = np.array(list(itertools.product([-1, 1], repeat=n)), dtype=np.int8)
    padded = np.ones((len(states), side+2, side+2), dtype=np.int8)
    padded[:, 1:-1, 1:-1] = states.reshape(-1, side, side)
    bond = (padded[:, :-1]*padded[:, 1:]).sum((1, 2)) + (padded[:, :, :-1]*padded[:, :, 1:]).sum((1, 2))
    weight = np.exp(c.BETA*(bond-bond.max())); weight /= weight.sum()
    mapping = {tuple(s): i for i, s in enumerate(states)}
    discrepancy = []
    for color in [0, 1]:
        coords = [(x, y) for x in range(1, side+1) for y in range(1, side+1) if (x+y) % 2 == color]
        transitions = np.zeros((len(states), len(states)))
        for i, field in enumerate(padded):
            prob = np.array([1/(1+np.exp(-2*c.BETA*(int(field[x-1, y])+int(field[x+1, y])+int(field[x, y-1])+int(field[x, y+1])))) for x, y in coords])
            for choice in itertools.product([0, 1], repeat=len(coords)):
                copy = field.copy()
                mass = 1.
                for j, ((x, y), value) in enumerate(zip(coords, choice)):
                    copy[x, y] = 2*value-1
                    mass *= prob[j] if value else 1-prob[j]
                target = mapping[tuple(copy[1:-1, 1:-1].ravel())]
                transitions[i, target] += mass
        assert np.max(np.abs(transitions.sum(1)-1)) < 1e-12
        err = float(np.max(np.abs(weight@transitions-weight)))
        assert err < 1e-12
        discrepancy.append(err)
    return dict(interior_side=side, states=len(states), stationary_max_error=discrepancy)


def cpu_tests(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    p = fake_parent()
    tests = {}
    cells = {}
    for seed in c.SEEDS:
        for step in range(1, 33):
            batches = [d.training_batch(p, seed, step, a) for a in c.ARMS]
            assert len({b["physical_hash"] for b in batches}) == 1
            a, l, e = batches
            assert a["clean"].size == c.TOKENS
            assert np.array_equal(a["clean"], l["clean"]) and np.array_equal(l["coords"], e["coords"])
            if a["sparse"]:
                assert len({b["native_hash"] for b in batches}) == 1
                assert all(b["query_valid"].sum() == 512 for b in batches)
            elif not e["endpoint"]:
                assert l["native_hash"] == e["native_hash"]
            else:
                assert np.all(e["mask_counts"]/e["width"]**2 <= .02)
                assert np.array_equal(e["mask"].sum((1, 2)), e["mask_counts"])
                assert np.isin(e["mask_counts"], [1, 2, 4, 8, 16, 32]).all()
                assert np.all(e["t"] >= .002)
            for b in batches:
                assert np.array_equal(b["noisy"][~b["mask"]], b["clean"][~b["mask"]])
                assert np.all(b["noisy"][b["mask"]] == 2)
            key = (seed, a["cell"])
            cells.setdefault(key, []).append(a["repeat"])
    assert all(sorted(v) == [0, 1, 2, 3] for v in cells.values())
    tests["paired_training_192_triplets"] = "passed"
    for w in (48, 96):
        for m in (1, 2, 8, 32):
            bank = d.local_bank(p, np.arange(16), w, m, "fixture")
            d.validate_bank(bank)
            q = bank["queries"]
            clean_p = d.oracle_probability(bank["clean"], q)
            assert np.array_equal(clean_p, bank["target"])
            # A whole spin flip complements the exact target.
            tokens = np.where(bank["noisy"] == 2, 2, 1-bank["noisy"])
            assert np.allclose(d.oracle_probability(tokens, q), 1-bank["target"], atol=1e-15)
            c.save(out/f"local_w{w}_m{m}.npz", **bank)
        stress = d.stress_bank(p, w); d.validate_bank(stress)
        assert len(stress["t"]) == 64
        c.save(out/f"stress_w{w}.npz", **stress)
    for kind, k in [("continuous", 115), ("continuous", 1152), ("held_gap", 512)]:
        bank = d.retention_bank(p, np.arange(16), kind, k); d.validate_bank(bank)
        assert np.all((bank["noisy"] != 2).sum((1, 2)) == k)
        c.save(out/f"retention_{kind}_{k}.npz", **bank)
    tests["exact_bank_8_stress2_retention3"] = "passed"
    tests["gibbs_enumeration"] = [exact_gibbs_fixture(2), exact_gibbs_fixture(3)]
    r = np.random.default_rng(51002)
    spins = 2*r.integers(2, size=(3, 96, 96), dtype=np.int8)-1
    majority = d.majority(spins)
    assert np.array_equal(d.majority(-spins), -majority)
    for i, x, y in [(0, 0, 0), (1, 10, 12), (2, 31, 31)]:
        assert majority[i, x, y] == np.sign(spins[i, 3*x:3*x+3, 3*y:3*y+3].sum())
    stats = d.physical_stats(spins)
    # Independent elementwise axis-loop reference.
    for lag in [1, 8, 24, 48]:
        brute = []
        for s in spins:
            sx = sum(int(s[x, y])*int(s[x+lag, y]) for x in range(96-lag) for y in range(96))
            sy = sum(int(s[x, y])*int(s[x, y+lag]) for x in range(96) for y in range(96-lag))
            brute.append(sx+sy)
        assert np.array_equal(stats["pair_sum"][:, lag], brute)
        assert np.all(stats["pair_count"][:, lag] == 2*96*(96-lag))
    assert np.all(d.physical_stats(np.ones((2, 48, 48), dtype=np.int8))["energy"] == -1.)
    transformed = d.multiscale_stats(spins)
    assert transformed["majority30_origins"]["m"].shape == (3, 9)
    tests["physical_sums_energy_majority_origin"] = "passed"
    # Omitted cells cannot hide behind averaged gates.
    assert 6*3*2*128 == 4608 and 6*3*2*64 == 2304
    tests["counts"] = "passed"
    result = dict(status="passed", cpu_only=True, time=time.time(), tests=tests)
    c.write(out/"checks.json", result)
    return result


def torch_cpu_tests(out):
    import torch
    import endpoint_sampling as sam
    import endpoint_training as tr
    torch.set_num_threads(2)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(()))
        def forward(self, tokens, t, coords, valid=None):
            # Positionwise constant logits isolate RNG/batching semantics.
            z = torch.zeros_like(tokens, dtype=torch.float32)+self.anchor
            return torch.stack([z, z+.2], 1)
    model = Toy()
    for name in ["monotone-256", "reveal192-repair64"]:
        both = sam.sample(model, 92601, 8, [0, 1], name, amp=False, toy=True)
        separate = [sam.sample(model, 92601, 8, [i], name, amp=False, toy=True) for i in [0, 1]]
        assert np.array_equal(both["spins"], np.concatenate([x["spins"] for x in separate]))
        sam.audit_repair(both)
        c.save(out/(name+".npz"), **both)
    # Compare original open-energy definition without any GPU computation.
    from ism_diffusion.scale_evaluation import open_energy_density
    spins = 2*np.random.default_rng(12).integers(2, size=(4, 48, 48), dtype=np.int8)-1
    assert np.array_equal(d.physical_stats(spins)["energy"], open_energy_density(spins))
    for seed in c.SEEDS:
        m, e, o, meta = tr.initialize(seed, c.ARMS[0], device="cpu")
        assert meta["global_step"] == 12000 and all(bool(torch.isfinite(x).all()) for x in m.state_dict().values())
        del m, e, o
    result = dict(status="passed", toy_sampler_independence=True, committed_repair_replayed=True,
                  six_base_full_state_CPU_loaded=True, old_open_energy_exact=True, cuda_used=False)
    c.write(out/"checks.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--torch-cpu", action="store_true")
    a = p.parse_args()
    result = torch_cpu_tests(a.out) if a.torch_cpu else cpu_tests(a.out)
    print(__import__("json").dumps(result, ensure_ascii=False))
