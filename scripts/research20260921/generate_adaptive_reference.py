"""Eight fresh, chain-isolated confirmation MC references, with atomic output."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from ism_diffusion.ising import generate_independent_chains, BETA_CRITICAL, energy_density, magnetization
from ism_diffusion.diagnostics import integrated_autocorrelation_time, split_rhat
from ism_diffusion.scale_data import pack_spins
from ism_diffusion.geometry_study import atomic_json,file_hash


def main():
    out=ROOT / "artifacts/adaptive_research_20260922/reference"
    out.mkdir(parents=True,exist_ok=True)
    if (out / "complete.json").exists(): return
    seed=2026092301
    seeds=[int(s.generate_state(1,dtype=np.uint32)[0]) for s in np.random.SeedSequence(seed).spawn(8)]
    initials=["random"]*4+["plus"]*2+["minus"]*2
    started=time.time()
    atomic_json(out / "protocol.json",dict(chains=8,samples_per_chain=128,lattice_size=1024,
       beta=BETA_CRITICAL,burn_in_sweeps=40,sweeps_between=4,adaptation_sweeps=3,
       pilot_cluster_steps=128,initial_states=initials,chain_seeds=seeds,seed=seed,workers=4,
       role="new confirmation reference; never used for model selection or training"))
    samples,chain,metadata=generate_independent_chains(lattice_size=1024,chain_seeds=seeds,
        samples_per_chain=128,burn_in_sweeps=40,sweeps_between=4,beta=float(BETA_CRITICAL),
        adaptation_sweeps=3,pilot_cluster_steps=128,workers=4,backend="numba",
        return_chain_metadata=True,initial_states=initials)
    m=magnetization(samples);e=energy_density(samples)
    traces=dict(energy=e,m=m,abs_m=np.abs(m),m2=m*m)
    diagnostics={}
    for name,values in traces.items():
        groups=[values[chain==i] for i in range(8)]
        diagnostics[name]=dict(split_rhat=float(split_rhat(groups)),
                              chains=[integrated_autocorrelation_time(g) for g in groups])
    passed=all(np.isfinite(diagnostics[k]["split_rhat"]) and diagnostics[k]["split_rhat"]<=1.1
               and all(c["ess"]>=16 for c in diagnostics[k]["chains"])
               for k in ("energy","abs_m","m2"))
    meta=dict(lattice_size=1024,beta=float(BETA_CRITICAL),seed=seed,chain_metadata=metadata,
       chain_seeds=seeds,initial_states=initials,diagnostics=diagnostics,
       split_counts=dict(train=0,val=0,test_target=8,test_control=0))
    path=out / "fresh_l1024.npz"
    with path.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f,test_target_packed=pack_spins(samples),test_target_chain_id=chain,
                            metadata=np.array(json.dumps(meta)))
    os.replace(path.with_suffix(".tmp"),path)
    np.savez_compressed(out / "chain_traces.npz",chain=chain,**traces)
    result=dict(status="passed" if passed else "qa_failed",elapsed_seconds=time.time()-started,
                samples=len(samples),sha256=file_hash(path),diagnostics=diagnostics)
    atomic_json(out / "complete.json",result)
    print(json.dumps(result),flush=True)


if __name__=="__main__": main()
