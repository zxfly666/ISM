"""Guarded adaptive continuation: wait for capacity decision, then six paired arms.

Only explicitly documented development rules adapt model size/compute. No
confirmation results are read when selecting training steps or lambda.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).resolve().parent)]
import numpy as np
import torch
from ism_diffusion import geometry_study as gs
from ism_diffusion.context_repair_core import construct,pack_views,tensor_batch,objective,update,self_test
from ism_diffusion.scale_data import load_parent_split
from adaptive_repair_evaluation import GEN_DEFS,evaluate_checkpoint,prepare_reference_evidence

OUT=ROOT / "artifacts/adaptive_research_20260922"
OLD=ROOT / "artifacts/geometry_alignment_20260921"
SEEDS=list(range(91001,91007));ARMS=("F0","F1","F2")


def status(stage,**kw):
    gs.atomic_json(OUT / "repair/status.json",dict(stage=stage,time=time.time(),**kw))
    print(json.dumps(dict(stage=stage,**kw)),flush=True)


def base_path(size,seed):
    return OLD / f"training/s{seed}_A/final.pt" if size=="S" else OUT / f"capacity/training/s{seed}_M/final.pt"


def restore(path,config):
    p=torch.load(path,map_location="cpu",weights_only=False)
    assert p["config"]["model"]==config
    model=construct(config,99199);model.load_state_dict(p["model"],strict=True)
    ema=copy.deepcopy(model).eval().requires_grad_(False);ema.load_state_dict(p["ema"],strict=True)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-5,betas=(.9,.95),weight_decay=.05,fused=True)
    opt.load_state_dict(p["optimizer"])
    return model,ema,opt,p


def run_benchmark(path,config,parent):
    results={};calibration=[]
    for arm in ARMS:
        model,ema,opt,p=restore(path,config);model.train()
        times=[];torch.cuda.reset_peak_memory_stats()
        for w in gs.WIDTHS:
            for kind in ("continuous","gap"):
                b=gs.make_batch(parent,18432//w**2,w,kind,gs.GAPS,921022,24001+w,True)
                for rep in range(5):
                    torch.cuda.synchronize();start=time.perf_counter()
                    update(model,ema,opt,b,921022,rep+1,12000,arm,1.)
                    torch.cuda.synchronize()
                    if rep>=2:times.append(time.perf_counter()-start)
        results[arm]=dict(mean_seconds=float(np.mean(times)),p90_seconds=float(np.quantile(times,.9)),
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        del model,ema,opt,p;torch.cuda.empty_cache()
    return results


def calibrate(size,config,parent):
    records=[]
    for seed in SEEDS:
        model,ema,opt,p=restore(base_path(size,seed),config);model.train()
        for w in gs.WIDTHS:
            for kind in ("continuous","gap"):
                b=gs.make_batch(parent,18432//w**2,w,kind,gs.GAPS,2026092302,w+(100 if kind=="gap" else 0),True)
                t,mask,noisy=gs.corrupt_batch(b["clean"],2026092302,w+(100 if kind=="gap" else 0))
                views=pack_views(b,t,mask,noisy,2026092302,w,rho=.25)
                data=tensor_batch(views)
                _,parts=objective(model,data,"F2",1.)
                parameters=list(model.parameters())
                a=torch.autograd.grad(parts["supervised"],parameters,retain_graph=True,allow_unused=True)
                c=torch.autograd.grad(parts["js"],parameters,allow_unused=True)
                na=float(torch.sqrt(sum((v.float()**2).sum() for v in a if v is not None)))
                nc=float(torch.sqrt(sum((v.float()**2).sum() for v in c if v is not None)))
                dot=float(sum((x.float()*y.float()).sum() for x,y in zip(a,c) if x is not None and y is not None))
                records.append(dict(seed=seed,width=w,kind=kind,supervised_norm=na,js_norm=nc,
                                    ratio=nc/max(na,1e-12),cosine=dot/max(na*nc,1e-12),input_hash=views["paired_hash"]))
        del model,ema,opt,p;torch.cuda.empty_cache()
    ratio=float(np.median([r["ratio"] for r in records]))
    if not math.isfinite(ratio) or ratio<=1e-12: raise RuntimeError("vanishing/invalid JS calibration")
    chosen=min([.25,1.,4.,16.],key=lambda x:(abs(math.log(max(x*ratio,1e-20)/.1)),x))
    return dict(lambda_value=chosen,median_unscaled_gradient_ratio=ratio,
                attained_ratio=chosen*ratio,records=records,
                note="training-only, fixed quarter-retention calibration; not test tuning")


def save_checkpoint(path,model,ema,opt,payload,step,config,protocol,seed,arm,elapsed):
    value=dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
        config=dict(model=config,variant="A",seed=seed),step=24000+step,repair_step=step,
        repair_arm=arm,protocol_hash=protocol["protocol_hash"],elapsed_seconds=elapsed,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
        numpy_rng=np.random.get_state(),python_rng=random.getstate(),
        base_checkpoint_sha256=payload.get("base_checkpoint_sha256"),
        rng_strategy="stateless original seed at global update 24000+repair_step")
    tmp=path.with_suffix(".tmp")
    torch.save(value,tmp);os.replace(tmp,path)


def train_block(size,config,parent,validation,seed,arm,target,protocol,deadline):
    dest=OUT / f"repair/training/s{seed}_{arm}"
    dest.mkdir(parents=True,exist_ok=True)
    last=dest / "last.pt"
    model,ema,opt,p=restore(last if last.exists() else base_path(size,seed),config)
    start=int(p.get("repair_step",0))+1
    previous=float(p.get("elapsed_seconds",0)) if "repair_step" in p else 0.
    if "repair_step" in p and p["protocol_hash"]!=protocol["protocol_hash"]:
        raise RuntimeError("cannot resume different repair protocol")
    p["base_checkpoint_sha256"]=gs.file_hash(base_path(size,seed))
    if "repair_step" in p:
        torch.set_rng_state(p["torch_rng"]);torch.cuda.set_rng_state_all(p["cuda_rng"])
        np.random.set_state(p["numpy_rng"]);random.setstate(p["python_rng"])
    began=time.monotonic();model.train()
    with (dest / "train.jsonl").open("a",encoding="utf-8",buffering=1) as log:
        for step in range(start,target+1):
            if time.time()>=deadline:
                save_checkpoint(last,model,ema,opt,p,step-1,config,protocol,seed,arm,previous+time.monotonic()-began)
                raise TimeoutError("campaign budget stop")
            width,kind=gs.schedule(seed,24000+step)
            b=gs.make_batch(parent,18432//width**2,width,kind,gs.GAPS,seed,24000+step,True)
            stats=update(model,ema,opt,b,seed,step,protocol["steps"],arm,protocol["lambda"])
            if step%20==0 or step==1:
                log.write(json.dumps(dict(step=step,seed=seed,arm=arm,width=width,kind=kind,
                    elapsed=previous+time.monotonic()-began,**stats))+"\n")
        risk=gs.balanced_risk(gs.evaluate(ema,validation,"A",gs.VAL_DEFS))
        log.write(json.dumps(dict(step=target,seed=seed,arm=arm,validation=risk))+"\n")
    elapsed=previous+time.monotonic()-began
    save_checkpoint(last,model,ema,opt,p,target,config,protocol,seed,arm,elapsed)
    monitor=dest / "validation_history.json"
    history=json.loads(monitor.read_text()) if monitor.exists() else []
    history=[r for r in history if r["step"]<target]
    history.append(dict(step=target,ce=risk["mean_ce"]))
    gs.atomic_json(monitor,history)
    baseline=protocol["base_validation"][str(seed)]["mean_ce"]
    if len(history)>=2 and all(r["ce"]>baseline+.03 for r in history[-2:]):
        raise RuntimeError(f"CE safety stop seed={seed} arm={arm}; two consecutive >base+0.03; saved last.pt")
    if target in (4000,8000,protocol["steps"]):
        save_checkpoint(dest / f"step{target:05d}.pt",model,ema,opt,p,target,config,protocol,seed,arm,elapsed)
    if target==protocol["steps"]:
        save_checkpoint(dest / "final.pt",model,ema,opt,p,target,config,protocol,seed,arm,elapsed)
        gs.atomic_json(dest / "complete.json",dict(status="trained",steps=target,
                       elapsed_seconds=elapsed,validation=risk,sha256=gs.file_hash(dest / "final.pt")))
    result=dict(seed=seed,arm=arm,step=target,validation=risk,elapsed=elapsed)
    del model,ema,opt,p;torch.cuda.empty_cache()
    return result


def bootstrap_summary(results):
    from scipy.stats import t as student_t
    pairs=[]
    for seed in SEEDS:
        ff={r["arm"]:r for r in results if r["seed"]==seed}
        pairs.append(dict(seed=seed,F0=ff["F0"]["J"],F1=ff["F1"]["J"],F2=ff["F2"]["J"],
                          F2_minus_F1=ff["F2"]["J"]-ff["F1"]["J"]))
    d=np.array([r["F2_minus_F1"] for r in pairs])
    upper=float(d.mean()+student_t.ppf(.95,5)*d.std(ddof=1)/np.sqrt(6))
    return dict(paired=pairs,mean_F2_minus_F1=float(d.mean()),paired_t_one_sided95_upper=upper,
                limitation="conditional-on-reference paired t summary; crossed MC/seed bootstrap and curve bands are separate required follow-up, not claimed complete here")


def main():
    (OUT / "repair").mkdir(parents=True,exist_ok=True)
    lock=(OUT / "repair/queue.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    preflight=json.loads((OUT / "repair_preflight/complete.json").read_text())
    if preflight["status"]!="passed": raise RuntimeError("real-device preflight not passed")
    tests=self_test("cpu");gs.atomic_json(OUT / "repair/unit_tests.json",tests)
    torch.set_num_threads(4);torch.set_float32_matmul_precision("high")
    campaign=json.loads((OUT / "campaign.json").read_text());deadline=campaign["deadline"]
    status("waiting_capacity",deadline=deadline)
    while not (OUT / "capacity/decision.json").exists():
        if time.time()>=deadline: raise TimeoutError("capacity did not complete before deadline")
        p=OUT / "capacity/status.json"
        if p.exists() and json.loads(p.read_text())["stage"] in ("failed","budget_stop"):
            raise RuntimeError("capacity failed; do not restart blindly")
        time.sleep(15)
    decision=json.loads((OUT / "capacity/decision.json").read_text())
    size=decision["selected_size"];config=decision["model"]
    status("capacity_analyzed",decision=decision)
    parent=load_parent_split(ROOT / "data/level1/parents_l1024.npz","train")
    validation=load_parent_split(ROOT / "data/level1/parents_l1024.npz","val")
    if size=="M":
        import run_capacity_20260922 as capacity
        gs.MODEL=copy.deepcopy(config);gs.new_model=capacity.build_model
        c=json.loads((OUT / "capacity/protocol.json").read_text())
        for seed in SEEDS[2:]:
            status("additional_base_training",seed=seed,size=size)
            rr=gs.train_one(parent,validation,OUT / f"capacity/training/s{seed}_M","A",seed,c,deadline)
            if rr["status"]!="trained": raise TimeoutError("base training budget stop")
    proto_path=OUT / "repair/protocol.json"
    if proto_path.exists():
        protocol=json.loads(proto_path.read_text())
        for rel,digest in protocol["source_hashes"].items():
            if gs.file_hash(ROOT / rel)!=digest: raise RuntimeError("frozen repair source changed: "+rel)
    else:
        status("repair_benchmark",selected_size=size)
        bench=run_benchmark(base_path(size,91001),config,parent)
        gs.atomic_json(OUT / "repair/benchmark.json",bench)
        status("training_only_lambda_calibration")
        calibration=calibrate(size,config,parent)
        gs.atomic_json(OUT / "repair/lambda_calibration.json",calibration)
        base_validation={}
        for seed in SEEDS:
            model,ema,opt,p=restore(base_path(size,seed),config)
            base_validation[str(seed)]=gs.balanced_risk(gs.evaluate(ema,validation,"A",gs.VAL_DEFS))
            del model,ema,opt,p;torch.cuda.empty_cache()
        # Extrapolate only measured development sampling rates at selected size.
        capeval=[json.loads((OUT / f"capacity/evaluation/s{s}_{size}/complete.json").read_text()) for s in (91001,91002)]
        rate96=np.mean([r["generation"]["continuous96"]["sampling_seconds"]/128 for r in capeval])
        rate48=np.mean([r["generation"]["held_s10_w48"]["sampling_seconds"]/128 for r in capeval])
        rate32=np.mean([r["generation"]["held_gap57_w32"]["sampling_seconds"]/128 for r in capeval])
        rate64=rate48*(.5130/.2343)
        generation_forecast=18*sum(d["samples"]*({96:rate96,64:rate64,48:rate48,32:rate32}[d["width"]]) for d in GEN_DEFS)*1.12
        step_cost=sum(bench[a]["mean_seconds"] for a in ARMS)*6*1.15
        remaining=deadline-time.time()
        choices=[n for n in (12000,10000,8000) if n*step_cost+generation_forecast+2.5*3600<=remaining]
        if not choices: raise RuntimeError("balanced >=8000-step repair does not fit remaining budget; requires decision")
        sources=["ism_diffusion/context_repair_core.py","scripts/research20260921/run_adaptive_repair_20260922.py",
                 "scripts/research20260921/adaptive_repair_evaluation.py","ism_diffusion/geometry_study.py",
                 "ism_diffusion/scale_model.py","ism_diffusion/scale_diffusion.py"]
        protocol=dict(version="adaptive_repair_v1",selected_size=size,model=config,seeds=SEEDS,
           arms=list(ARMS),steps=max(choices),lambda_value=calibration["lambda_value"],
           generation=GEN_DEFS,conditional_samples=512,deadline=deadline,
           forecast_seconds=max(choices)*step_cost+generation_forecast+2.5*3600,
           source_hashes={f:gs.file_hash(ROOT / f) for f in sources},
           base_hashes={str(s):gs.file_hash(base_path(size,s)) for s in SEEDS},
           base_validation=base_validation,
           deviations=["capacity pilot precedes repair; selected size follows frozen development rule",
             "lambda calibrated with fixed 1/4 retention rather than mixed retention",
             "fixed-step 1000-update blocks, final EMA; train-field geometry and corruption unchanged",
             "crossed bootstrap/full diagnostic completion follows main generation, not inferred from t interval"])
        protocol["lambda"]=protocol.pop("lambda_value")
        protocol["protocol_hash"]=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
        gs.atomic_json(proto_path,protocol)
    status("repair_training",protocol=protocol)
    # Interleave all 18 cells at 1000-update boundaries, not just the favored arm.
    for end in range(1000,protocol["steps"]+1,1000):
        for seed in SEEDS:
            for arm in ARMS:
                done=OUT / f"repair/training/s{seed}_{arm}/last.pt"
                if done.exists():
                    p=torch.load(done,map_location="cpu",weights_only=False)
                    completed=int(p.get("repair_step",0));del p
                    if completed>=end:continue
                status("repair_training",seed=seed,arm=arm,target_step=end,total_steps=protocol["steps"])
                rr=train_block(size,config,parent,validation,seed,arm,end,protocol,deadline)
                print(json.dumps(rr),flush=True)
    status("waiting_reference")
    while not (OUT / "reference/complete.json").exists():
        if time.time()>=deadline:raise TimeoutError("reference incomplete")
        time.sleep(15)
    refstatus=json.loads((OUT / "reference/complete.json").read_text())
    if refstatus["status"]!="passed":raise RuntimeError("new reference failed QA, no historical fallback")
    fresh=load_parent_split(OUT / "reference/fresh_l1024.npz","test_target")
    prepare_reference_evidence(fresh,OUT / "repair/reference_evidence",protocol["generation"])
    del fresh
    results=[]
    for seed in SEEDS:
        for arm in ARMS:
            status("confirmation_evaluation",seed=seed,arm=arm)
            results.append(evaluate_checkpoint(OUT / f"repair/training/s{seed}_{arm}/final.pt",
               OUT / "reference/fresh_l1024.npz",OUT / f"repair/evaluation/s{seed}_{arm}",seed,arm,protocol,deadline))
    summary=bootstrap_summary(results)
    summary.update(status="main_complete_requires_final_analysis",models=len(results),
        generation_samples=sum(d["samples"] for d in GEN_DEFS)*len(results),
        completed=time.time(),elapsed_hours=(time.time()-campaign["started"])/3600,
        remaining_hours=(deadline-time.time())/3600,protocol_hash=protocol["protocol_hash"],
        next="agent reviews all task outcomes, performs crossed uncertainty/figures/backups; supplementary experiments within remaining budget")
    gs.atomic_json(OUT / "repair/main_summary.json",summary)
    status("main_complete_requires_final_analysis",summary=summary)


if __name__=="__main__":
    try:main()
    except TimeoutError as exc:status("budget_stop",error=str(exc))
    except Exception as exc:
        status("failed",error=repr(exc),traceback=traceback.format_exc());raise
