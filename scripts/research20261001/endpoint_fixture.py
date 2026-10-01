"""Synthetic statistics/rendering integration fixture, NEVER model evidence."""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import endpoint_common as c
import endpoint_statistics as s


def make(root, reference_count=2048):
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    c.write(root/"SYNTHETIC_FIXTURE.json", dict(not_scientific_results=True, purpose="statistics_and_render_integration"))
    random = np.random.default_rng(2026100161)
    def stats(n, width, error=0., origin=False):
        r = np.arange(width//2+1)
        curve = .72/np.maximum(r, 1)**.25; curve[0] = 1
        noise = random.normal(0, .008, size=(n, 1))
        y = curve[None, :]*(1+error+noise); y[:, 0] = 1
        counts = np.broadcast_to(np.maximum(2*width*(width-r), 1), y.shape).copy()
        values = dict(pair_sum=y*counts, pair_count=counts, m=random.normal(0, .003, n),
                      m2=np.full(n, .30), abs_m=np.full(n, .52), energy=np.full(n, -.70))
        if origin:
            values = {k: np.repeat(v[:, None], 9, axis=1) for k, v in values.items()}
        return values
    views = {"full":96, "center48":48, "edge8":96, "majority32":32, "decimation32":32, "majority30_origins":30}
    chain = np.repeat(np.arange(16), reference_count//16)
    for view, width in {**views, "w48":48}.items():
        c.save(root/"reference"/f"stats_{view}.npz", **stats(reference_count, width, origin=view=="majority30_origins"), chain=chain)
    local_chain = np.repeat(np.arange(16), 16)
    c.save(root/"banks/exact_w48_m1.npz", chain=local_chain)
    for si, seed in enumerate(c.SEEDS):
        for ai, arm in enumerate(c.ARMS):
            cell = root/"evaluation"/f"s{seed}_{arm}"
            for width in [48, 96]:
                for mi, m in enumerate([1, 2, 8, 32]):
                    kl = np.full((256, m), .006-ai*.001+si*.00001+mi*.0001)
                    c.save(cell/"conditional"/f"exact_w{width}_m{m}.npz", kl=kl)
                c.save(cell/"conditional"/f"stress_w{width}.npz", probability_error=np.full((64, 1), .02))
            for name in ["C48_K115", "C48_K1152", "H48_K512"]:
                c.save(cell/"conditional"/f"{name}.npz", ce=np.full((256, 64), .50-ai*.0001))
            for step in [4000, 6000, 8000]:
                for m in [1, 2, 8, 32]:
                    c.save(cell/"learning"/f"step{step}_exact_w48_m{m}.npz", kl=np.full((128, m), .004+.0005*(8000-step)/2000))
            for width, n in [(48,64), (96,128)]:
                for method in ["monotone-256", "reveal192-repair64"]:
                    folder = cell/"generation"/f"w{width}_{method}"
                    kinds = ["final"] if method=="monotone-256" else ["prefix", "final", "oracle"]
                    for typ in kinds:
                        error = .35-ai*.08 +si*.001
                        if typ == "final" and method=="reveal192-repair64": error -= .09
                        if typ == "oracle": error = .03
                        for view, vw in (views.items() if width==96 else [("full",48)]):
                            c.save(folder/f"stats_{typ}_{view}.npz", **stats(n, vw, error, view=="majority30_origins"))
        for floor in [.01,.002]:
            c.save(root/"phase0"/f"s{seed}"/f"floor{floor}"/"stats_final_full.npz", **stats(32,96,.35))
            for width in [48,96]:
                for m in [1,2,8,32]:
                    c.save(root/"phase0"/f"s{seed}"/"conditional"/f"phase0_w{width}_m{m}_floor{floor}.npz", kl=np.full((16,m),.015))
    return root


def run(root, full_reps=False):
    start = time.perf_counter(); root=make(root)
    mid = time.perf_counter()
    result=s.analyze(root, fixture=True, full_reps=full_reps)
    stop = time.perf_counter()
    assert -.18 < result["primary"]["P1_training_coverage"]["estimate"] < -.14
    assert -.11 < result["primary"]["P2_committed_repair"]["estimate"] < -.07
    assert result["scientific_success"] and not np.any(result["learning_incomplete"])
    from endpoint_figures import render
    figures=render(root, fixture=True)
    assert len(figures)==6
    receipt=dict(status="passed", fixture_not_science=True, create_seconds=mid-start, statistics_seconds=stop-mid,
                 render_seconds=time.perf_counter()-stop, full_bootstrap_reps=full_reps, figures=6,
                 expected_negative_primary_signs_verified=True)
    c.write(root/"fixture_complete.json", receipt)
    print(__import__("json").dumps(receipt), flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--full-reps", action="store_true")
    a=p.parse_args(); run(a.out, a.full_reps)
