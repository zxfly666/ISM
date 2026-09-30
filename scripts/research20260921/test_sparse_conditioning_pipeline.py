"""Prelaunch integration tests. Synthetic/reference-free, or training split only."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
import numpy as np
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import torch
from ism_diffusion import geometry_study as gs
from ism_diffusion.scale_data import load_parent_split
from sparse_conditioning_design import self_test,GEN_DEFS,COND_DEFS,make_cases
from evaluate_sparse_conditioning_20260923 import condition_bank
from finalize_sparse_conditioning_20260923 import scores,plot
from run_sparse_conditioning_20260923 import train_block,base
from run_adaptive_repair_20260922 import restore


def test_scores():
    cases=make_cases();pred=np.ones((3,6,116,2,20,3))*.5
    chain=np.repeat(np.arange(8),128)
    ref=dict(counts=np.ones((116,1024,8),np.int32)*32,chain=chain,reference_p=np.ones((116,8))/8)
    gr=[]
    for d in GEN_DEFS:
        width=d["width"] if d["kind"]=="continuous" else 471
        count=np.ones((1024,width))*10
        gr.append(dict(pair_count=count,pair_sum=count*.3,parent=np.arange(1024),chain=chain,
                       m=np.zeros(1024),m2=np.ones(1024)*.1,abs_m=np.ones(1024)*.2,energy=np.ones(1024)*-.6))
    cond=[];gen=[]
    for a in range(3):
        cond.append([[dict(ce=np.ones(128)*(.6-.1*a),parent=np.arange(128)) for _ in COND_DEFS] for _ in range(6)])
        gen.append([[{k:v[:d["samples"]].copy() for k,v in gr[gi].items() if k not in ("parent","chain")}
                     for gi,d in enumerate(GEN_DEFS)] for _ in range(6)])
    point,keys=scores(cases,pred,cond,gen,ref,gr)
    boot,_=scores(cases,pred,cond,gen,ref,gr,np.ones(1024),np.random.default_rng(123))
    assert abs(point[1,:,keys.index("w48_t05/CE")].mean()-.5)<1e-10
    assert np.max(abs(point[:,:,keys.index("continuous128/G_49_64_NRMSE")]))<1e-10
    assert np.max(abs(point[:,:,:56]))<1e-10
    assert np.max(abs(point-boot))<1e-10
    return len(keys)


def main(out):
    torch.set_num_threads(2);torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    result=self_test();result["statistical_columns"]=test_scores()
    # Broadcast synthetic parent array; not a stored or evaluated MC reference.
    pattern=np.where((np.indices((1024,1024)).sum(0)%3)==0,1,-1).astype(np.int8)
    synthetic=SimpleNamespace(spins=np.broadcast_to(pattern,(1024,1024,1024)),
                              chain_ids=np.repeat(np.arange(8),128),lattice_size=1024)
    for d in COND_DEFS:
        b=condition_bank(synthetic,d)
        assert b["noisy"].shape==(128,1,d["width"]**2)
        assert (b["noisy"][np.arange(128)[:,None],0,b["queries"]]==2).all()
        assert len(np.unique(b["parent"]))==128 and len(np.unique(b["chain"]))==8
        if "k" in d:assert ((b["noisy"]!=2).sum((1,2))==d["k"]).all()
    result["heldout_bank_shapes_hidden_queries_and_chain_balance"]=True
    parent=load_parent_split(ROOT/"data/level1/parents_l1024.npz","train")
    val=load_parent_split(ROOT/"data/level1/parents_l1024.npz","val")
    with np.load(out/"training_oracle.npz") as z:oracle=z["loco"]
    seed=91001;digest=gs.file_hash(base(seed))
    proto=dict(protocol_hash="PRELAUNCH_SCRATCH_ONLY",steps=4,deadline=time.time()+600,
        base_hashes={str(base(seed)):digest},base_validation={str(seed):dict(mean_ce=10.)})
    resumed=out/"scratch_resume_deterministic";direct=out/"scratch_direct_deterministic"
    train_block(resumed,parent,val,oracle,proto,seed,"T1",2)
    saved=torch.load(resumed/"training/s91001_T1/last.pt",map_location="cpu",weights_only=False)
    restored,restored_ema,restored_opt,payload=restore(resumed/"training/s91001_T1/last.pt",gs.MODEL)
    assert all(torch.equal(v,saved["model"][k]) for k,v in ((k,v.cpu()) for k,v in restored.state_dict().items()))
    for k,state in restored_opt.state_dict()["state"].items():
        for name,value in state.items():
            other=saved["optimizer"]["state"][k][name]
            assert torch.equal(value.cpu(),other) if torch.is_tensor(value) else value==other
    del restored,restored_ema,restored_opt,payload,saved;torch.cuda.empty_cache()
    train_block(resumed,parent,val,oracle,proto,seed,"T1",4)
    train_block(direct,parent,val,oracle,proto,seed,"T1",4)
    p=torch.load(resumed/"training/s91001_T1/final.pt",map_location="cpu",weights_only=False)
    q=torch.load(direct/"training/s91001_T1/final.pt",map_location="cpu",weights_only=False)
    error=max(float((p["model"][k]-q["model"][k]).abs().max()) for k in p["model"])
    assert error<1e-5 and p["stream_digests"]==q["stream_digests"]
    assert p["step"]==36004 and gs.file_hash(base(seed))==digest
    result.update(checkpoint_roundtrip_max_error=error,resume_streams_identical=True,old_checkpoint_unchanged=True,
        scratch_only_updates=8,heldout_MC_used=False,status="passed",finished=time.time(),
        exact_state_and_optimizer_roundtrip=True,deterministic_algorithms_in_this_test=True,
        production_note="BF16 production is not promised bitwise deterministic")
    gs.atomic_json(out/"pipeline_tests.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args();out=Path(a.out)
    try:main(out)
    except Exception as e:
        gs.atomic_json(out/"pipeline_failure.json",dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
