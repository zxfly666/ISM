"""Scratch-only GPU tests and runtime forecast, never saving trained weights."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
import numpy as np
import torch
from ism_diffusion import geometry_study as gs
from ism_diffusion.scale_data import load_parent_split
from ism_diffusion.scale_diffusion import CoordinateAbsorbingDiffusion
from run_adaptive_repair_20260922 import restore
from sparse_conditioning_design import ARMS,WIDTHS,GEN_DEFS,self_test,make_cases
from sparse_conditioning_core import views,update
from fixed_geometry_math import case_inputs,metrics
from run_fixed_geometry_joint_20260923 import predict


def main(out):
    torch.set_num_threads(2);torch.set_float32_matmul_precision("high")
    parent=load_parent_split(ROOT/"data/level1/parents_l1024.npz","train")
    validation=load_parent_split(ROOT/"data/level1/parents_l1024.npz","val")
    with np.load(out/"training_oracle.npz") as z: oracle=z["loco"]
    base=ROOT/"artifacts/adaptive_research_20260922/repair/training/s91001_F0/final.pt"
    before=gs.file_hash(base); bench={}; checks=self_test();timings=[]
    for arm in ARMS:
        model,ema,opt,p=restore(base,gs.MODEL);model.train()
        records=[];torch.cuda.reset_peak_memory_stats()
        for step in range(1,19):
            torch.cuda.synchronize();t0=time.perf_counter()
            packed=views(parent,oracle,91001,step)
            stats=update(model,ema,opt,packed,step,8000,arm)
            torch.cuda.synchronize();elapsed=time.perf_counter()-t0
            if step>6:records.append(dict(width=packed[0]["width"],kind=packed[0]["kind"],seconds=elapsed))
            assert stats["tokens"]==18432
        bench[arm]=dict(mean_seconds=float(np.mean([r["seconds"] for r in records])),
                       peak_gib=torch.cuda.max_memory_allocated()/2**30,records=records)
        print(json.dumps(dict(arm=arm,benchmark=bench[arm])),flush=True)
        del model,ema,opt,p;torch.cuda.empty_cache()
    # Identical input streams, distinct labels only for eligible K<=2 rows.
    aa=views(parent,oracle,91001,5);bb=views(parent,oracle,91001,5)
    assert gs.array_hash(aa[2]["noisy"],aa[2]["queries"],aa[2]["soft"])==gs.array_hash(bb[2]["noisy"],bb[2]["queries"],bb[2]["soft"])
    checks["paired_deterministic_views"]=True
    # A small update with hard labels encoded as soft probabilities must match T1.
    a=views(parent,oracle,91001,1);a[2]["soft"]=a[2]["labels"].copy()
    states=[]
    for arm in ("T1","T2"):
        model,ema,opt,p=restore(base,gs.MODEL)
        update(model,ema,opt,a,1,8000,arm)
        states.append({k:v.cpu().clone() for k,v in model.state_dict().items()})
        del model,ema,opt,p;torch.cuda.empty_cache()
    err=max(float((states[0][k]-states[1][k]).abs().max()) for k in states[0])
    assert err<1e-6;checks["hard_soft_identity_weight_error"]=err
    model,ema,opt,p=restore(base,gs.MODEL);ema.eval();del model,opt,p
    gen={};sampler=CoordinateAbsorbingDiffusion()
    for w in (48,96,128):
        cc=gs.coordinate_arrays(16,w,"continuous",(1,),101,w)[0]["A"]
        coord=torch.as_tensor(cc,device="cuda");valid=torch.ones((16,w,w),device="cuda",dtype=torch.bool)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            sampler.sample(ema,coord,valid,steps=2,generator=torch.Generator(device="cuda").manual_seed(987))
            torch.cuda.synchronize();t0=time.perf_counter()
            result=sampler.sample(ema,coord,valid,steps=16,generator=torch.Generator(device="cuda").manual_seed(987))
            torch.cuda.synchronize();seconds=time.perf_counter()-t0
        assert ((result==0)|(result==1)).all()
        gen[str(w)]=dict(shard16_steps16_seconds=seconds,forecast_per_image_256steps=seconds,
                         peak_gib=torch.cuda.max_memory_allocated()/2**30)
        print(json.dumps(dict(generation_width=w,benchmark=gen[str(w)])),flush=True)
        del coord,valid,result;torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("highest")
    diag={}
    for w in (48,96,128):
        case=next(c for c in make_cases() if c["width"]==w)
        x,c,t,spec=case_inputs(case,"A")
        torch.cuda.synchronize();t0=time.perf_counter()
        pp=predict(ema,x,c,t,1 if w==128 else 2)
        diag[str(w)]=time.perf_counter()-t0
        mm=metrics(pp,np.ones(8)/8,spec)
        assert float(mm["swap_permutation_error"])<2e-5
        assert float(mm["parallel_identity_residual"])<1e-10
    torch.set_float32_matmul_precision("high")
    t0=time.perf_counter();gs.evaluate(ema,validation,"A",gs.VAL_DEFS);valcost=time.perf_counter()-t0
    stepcost=6*sum(x["mean_seconds"] for x in bench.values())*1.2
    generation=18*sum(d["samples"]*gen[str(d["width"])]["forecast_per_image_256steps"] for d in GEN_DEFS)*1.2
    conditional=18*2*sum(diag[str(c["width"])] for c in make_cases())*1.25+600
    forecasts={str(n):n*stepcost+generation+conditional+(18*n/1000+6)*valcost+1800 for n in (4000,6000,8000)}
    choices=[int(n) for n,v in forecasts.items() if v<10*3600]
    assert gs.file_hash(base)==before
    answer=dict(status="passed" if choices else "budget_gate_failed",selected_steps=max(choices) if choices else None,
        tests=checks,training=bench,generation=gen,diagnostics_case_seconds=diag,val_seconds=valcost,
        forecasts_seconds=forecasts,base_unchanged=True,base_sha256=before,created=time.time(),
        microbatch={24:8,48:4,96:1},generation_shard=16,new_training_saved=False,
        note="Short 16-step sampling is timing only, not an experimental result")
    gs.atomic_json(out/"gpu_preflight.json",answer);print(json.dumps(answer),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args();out=Path(a.out)
    try:main(out)
    except Exception as e:
        gs.atomic_json(out/"gpu_failure.json",dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
