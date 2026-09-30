"""Separate training-only oracle and fresh evaluation MC; no model selection."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
import numpy as np
from ism_diffusion import geometry_study as gs
from ism_diffusion.scale_data import load_parent_split,pack_spins
from sparse_conditioning_design import periodic_correlation,d4_average,MC_SEED,self_test
from run_fixed_geometry_joint_20260923 import npz


def oracle(out):
    began=time.time(); out.mkdir(parents=True,exist_ok=True)
    path=ROOT/"data/level1/parents_l1024.npz"
    parent=load_parent_split(path,"train")
    chains=np.unique(parent.chain_ids);assert chains.tolist()==list(range(6))
    means=[]
    for chain in chains:
        ids=np.flatnonzero(parent.chain_ids==chain)
        total=np.zeros((parent.lattice_size,parent.lattice_size),np.float64)
        for j,i in enumerate(ids): total+=periodic_correlation(parent.spins[i])
        means.append(d4_average(total/len(ids)))
        gs.atomic_json(out/"status.json",dict(stage="training_oracle",chain=int(chain),time=time.time()))
        print(f"training-only correlation chain {chain} complete",flush=True)
    means=np.stack(means)
    loco=(means.sum(0)[None]-means)/(len(chains)-1)
    assert np.max(abs(loco[:,0,0]-1))<1e-12
    npz(out/"training_oracle.npz",loco=loco,chain_means=means,chains=chains)
    gs.atomic_json(out/"oracle_complete.json",dict(status="passed",source=str(path),source_sha256=gs.file_hash(path),
        split="train",chain_ids=chains.tolist(),parents=len(parent.spins),leave_one_entire_chain_out=True,
        sha256=gs.file_hash(out/"training_oracle.npz"),math_tests=self_test(),elapsed_seconds=time.time()-began))


def reference(out):
    from ism_diffusion.ising import generate_independent_chains,BETA_CRITICAL,energy_density,magnetization
    from ism_diffusion.diagnostics import integrated_autocorrelation_time,split_rhat
    out.mkdir(parents=True,exist_ok=True); began=time.time()
    seeds=[int(s.generate_state(1,dtype=np.uint32)[0]) for s in np.random.SeedSequence(MC_SEED).spawn(8)]
    initials=["random"]*4+["plus"]*2+["minus"]*2
    args=dict(lattice_size=1024,chain_seeds=seeds,samples_per_chain=128,burn_in_sweeps=40,
        sweeps_between=4,beta=float(BETA_CRITICAL),adaptation_sweeps=3,pilot_cluster_steps=128,
        workers=4,backend="numba",return_chain_metadata=True,initial_states=initials)
    gs.atomic_json(out/"protocol.json",dict(args,seed=MC_SEED,role="new held-out evaluation only"))
    samples,chain,meta=generate_independent_chains(**args)
    m=magnetization(samples); traces=dict(energy=energy_density(samples),m=m,abs_m=np.abs(m),m2=m*m)
    diagnostics={k:dict(split_rhat=float(split_rhat([v[chain==i] for i in range(8)])),
        chains=[integrated_autocorrelation_time(v[chain==i]) for i in range(8)]) for k,v in traces.items()}
    passed=all(np.isfinite(diagnostics[k]["split_rhat"]) and diagnostics[k]["split_rhat"]<=1.1
        and all(c["ess"]>=16 for c in diagnostics[k]["chains"]) for k in ("energy","abs_m","m2"))
    metadata=dict(lattice_size=1024,beta=float(BETA_CRITICAL),seed=MC_SEED,chain_seeds=seeds,
        chain_metadata=meta,diagnostics=diagnostics,initial_states=initials)
    path=out/"fresh_l1024.npz"
    npz(path,test_target_packed=pack_spins(samples),test_target_chain_id=chain,metadata=np.array(json.dumps(metadata)))
    npz(out/"chain_traces.npz",chain=chain,**traces)
    gs.atomic_json(out/"complete.json",dict(status="passed" if passed else "qa_failed",sha256=gs.file_hash(path),
        diagnostics=diagnostics,elapsed_seconds=time.time()-began,parents=len(samples),seed=MC_SEED))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--mode",choices=["oracle","reference"],required=True)
    a=p.parse_args(); out=Path(a.out)
    try: (oracle if a.mode=="oracle" else reference)(out)
    except Exception as e:
        gs.atomic_json(out/"failure.json",dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
