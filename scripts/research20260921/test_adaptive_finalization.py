"""Synthetic analysis tests and exact trajectory noninterference check."""
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).resolve().parent)]
import numpy as np
import torch
from ism_diffusion import geometry_study as gs
from ism_diffusion.scale_data import load_parent_split
from ism_diffusion.scale_evaluation import load_scale_model
import finalize_adaptive_campaign as final
from adaptive_trajectory_diagnostics import smoke_test


def main():
    out=ROOT / "artifacts/adaptive_research_20260922/final_preflight";out.mkdir(parents=True,exist_ok=True)
    random=np.random.default_rng(28291);data={};chain=np.repeat(np.arange(8),128)
    conditional=(np.full((6,3,2,2,1024),.35),np.ones((2,1024)))
    for name in final.PRIMARY:
        n=128;R=24;counts=np.ones((n,R))*100
        sums=random.normal(.4,.02,(6,3,n,R))*counts[None,None]
        # Exactly equal F1/F2 must produce zero paired differences under common draws.
        sums[:,2]=sums[:,1]
        data[name]=dict(sums=sums,blocks=sums.reshape(6,3,-1,16,R).sum(3),counts=counts,
            block_counts=counts.reshape(-1,16,R).sum(1),mc=np.ones_like(counts)*40,
            parent=np.arange(0,1024,8),index=np.arange(25,25+R),
            full_curves=sums.sum(2)/counts.sum(0)[None,None,:],full_mc=np.full(R,.4))
    summary,draws=final.crossed_bootstrap(data,chain,conditional,8,128,872,time.time()+120)
    assert np.array_equal(draws["J"][:,1],draws["J"][:,2]),"paired resampling differs"
    assert summary["ci95"]==[0.,0.] and summary["rejected"]==0
    fake_results=[dict(seed=s,arm=a,J=.1+i*.001+j*.01) for i,s in enumerate(final.SEEDS) for j,a in enumerate(final.ARMS)]
    # Nondegenerate curves for figure rendering test only.
    for d in data.values():d["sums"][:,2]*=1.01
    # Use realistic nondegenerate bootstrap uncertainty for plotting checks.
    for key in final.PRIMARY:draws["curve_"+key]+=random.normal(0,.001,draws["curve_"+key].shape)
    final.figures(fake_results,data,draws,out / "synthetic_figures",include_samples=False)
    parent=load_parent_split(ROOT / "data/level1/parents_l1024.npz","train")
    model,_=load_scale_model(ROOT / "artifacts/geometry_alignment_20260921/training/s91001_A/final.pt",torch.device("cuda"))
    torch.set_num_threads(1)
    instrument=smoke_test(model,parent,out / "trajectory_noninterference")
    result=dict(status="passed",paired_bootstrap_zero_effect_exact=True,valid_resamples=int(draws["valid"].sum()),
                synthetic_figures_rendered=True,trajectory_instrumentation=instrument)
    gs.atomic_json(out / "complete.json",result);print(result,flush=True)


if __name__=="__main__":main()
