"""Two-seed, source-frozen capacity screen; preserves the historical experiment.

Uses the established training update/data law, but explicitly constructs and
checks the selected architecture. No passwords or network operations here.
"""
from __future__ import annotations
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import numpy as np
import torch
from ism_diffusion import geometry_study as gs
from ism_diffusion.scale_model import CoordinateDenseDenoiser, CoordinateDenoiserConfig
from ism_diffusion.scale_data import load_parent_split
from ism_diffusion.scale_evaluation import load_scale_model
from evaluate_study import generate_and_score, write_csv

OUT = ROOT / "artifacts/adaptive_research_20260922"
OLD = ROOT / "artifacts/geometry_alignment_20260921"
M = dict(gs.MODEL, d_model=192, n_heads=6)
SEEDS = [91001, 91002]
GEN = [dict(name="continuous96", width=96, kind="continuous", gaps=[1], samples=128),
       dict(name="held_gap57_w32", width=32, kind="gap", gaps=[5, 7], samples=128),
       dict(name="held_s10_w48", width=48, kind="gap", gaps=[10], samples=128)]


def build_model(seed, device="cuda"):
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
    model = CoordinateDenseDenoiser(CoordinateDenoiserConfig(**M)).to(device)
    assert sum(p.numel() for p in model.parameters()) == 4439170
    assert model.blocks[0].attention.head_dim == 32
    return model


def status(stage, **fields):
    gs.atomic_json(OUT / "capacity/status.json", dict(stage=stage, time=time.time(), **fields))
    print(json.dumps(dict(stage=stage, **fields)), flush=True)


def preflight(parent, validation):
    dest = OUT / "capacity/preflight.json"
    if dest.exists():
        return json.loads(dest.read_text())
    model = build_model(92199)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9,.95), weight_decay=.05, fused=True)
    timings = []
    torch.cuda.reset_peak_memory_stats()
    for width in gs.WIDTHS:
        for kind in ("continuous", "gap"):
            batch = gs.make_batch(parent, gs.TOKENS_PER_UPDATE // width**2,
                                  width, kind, gs.GAPS, 92199, width, True)
            for k in range(5):
                torch.cuda.synchronize(); before=time.perf_counter()
                gs.train_update(model, ema, opt, batch, "A", 92199, k+1, 24000, len(batch["clean"]))
                torch.cuda.synchronize()
                if k>=2: timings.append(time.perf_counter()-before)
    model.eval()
    xx = torch.full((2,1,12),2,device="cuda",dtype=torch.long)
    xx[:,0,1] = 1
    cc = torch.randn((2,1,12,2),device="cuda")
    tt = torch.full((2,),.8,device="cuda")
    with torch.no_grad():
        base = model(xx,tt,cc).float().softmax(1)
        xp=torch.cat([xx,torch.full((2,1,4),3,device="cuda",dtype=torch.long)],-1)
        cp=torch.cat([cc,torch.zeros((2,1,4,2),device="cuda")],-2)
        padded=model(xp,tt,cp,xp.ne(3)).float().softmax(1)[...,:12]
        error=float((base-padded).abs().max())
    assert error<3e-5, error
    # Verify serialized architecture and strict EMA loading, without retaining
    # this warmed-up model as a scientific initialization.
    test_path=OUT / "capacity/preflight_model.pt"
    torch.save(dict(model=model.state_dict(),ema=ema.state_dict(),config=dict(model=M)),test_path)
    loaded,payload=load_scale_model(test_path,torch.device("cuda"))
    assert sum(p.numel() for p in loaded.parameters())==4439170
    # This file is retained as preflight provenance, not reused for training.
    result=dict(status="passed",parameters=4439170,model=M,
                mean_update_seconds=float(np.mean(timings)),
                p90_update_seconds=float(np.quantile(timings,.9)),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                pad_probability_error=error,torch_version=torch.__version__,
                device=torch.cuda.get_device_name(),source_sha256=gs.file_hash(Path(__file__)))
    gs.atomic_json(dest,result)
    del model,ema,opt,loaded,payload
    torch.cuda.empty_cache()
    return result


def evaluate_capacity(path, reference, size, seed, deadline):
    dest=OUT / "capacity/evaluation" / f"s{seed}_{size}"
    dest.mkdir(parents=True,exist_ok=True)
    done=dest / "complete.json"
    if done.exists(): return json.loads(done.read_text())
    model,_=load_scale_model(path,torch.device("cuda"))
    records=gs.evaluate(model,reference,"A",gs.TEST_DEFS[:2],samples=256,seed=923333,masks=2,microbatch=8)
    write_csv(dest / "conditional.csv",records)
    risk=gs.balanced_risk(records)
    generations={d["name"]:generate_and_score(model,reference,"A",seed,d,
                   dest / "generation" / d["name"],deadline) for d in GEN}
    long_errors=[generations[d["name"]]["errors"]["context_25_plus" if d["kind"]=="continuous" else "physical_129_plus"]["nrmse"] for d in GEN]
    result=dict(size=size,seed=seed,checkpoint_sha256=gs.file_hash(path),
                conditional=risk,generation=generations,J=float(np.mean(long_errors)),
                reference_role="old independent reference reused for DEVELOPMENT screen, not new confirmation")
    gs.atomic_json(done,result)
    del model;torch.cuda.empty_cache()
    return result


def main():
    (OUT / "capacity").mkdir(parents=True,exist_ok=True)
    lock=(OUT / "capacity/queue.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.set_float32_matmul_precision("high")
    campaign_path=OUT / "campaign.json"
    if not campaign_path.exists():
        now=time.time()
        gs.atomic_json(campaign_path,dict(started=now,deadline=now+20*3600,
            total_budget_hours=20,status="active",authorization="user requested adaptive continued experiments",
            server_shutdown=False,automatic_failed_job_restart=False))
    campaign=json.loads(campaign_path.read_text());deadline=campaign["deadline"]
    cfg_path=OUT / "capacity/protocol.json"
    if cfg_path.exists():
        config=json.loads(cfg_path.read_text())
        if config["source_sha256"] != gs.file_hash(Path(__file__)):
            raise RuntimeError("Capacity source changed after freeze")
    else:
        config=dict(version="capacity_20260922_v1",seeds=SEEDS,steps=24000,model=M,
            microbatch={str(w):gs.TOKENS_PER_UPDATE//w**2 for w in gs.WIDTHS},
            source_sha256=gs.file_hash(Path(__file__)),checkpoint_interval=1000,
            dataset_sha256=gs.file_hash(ROOT / "data/level1/parents_l1024.npz"),
            generation=GEN,decision_rule="M only if both seeds J improve, mean relative J improves >=10%, and each seed CE <= S+0.002; otherwise S. Development rule, not significance.",
            inherited_training="same named clean/geometry/time/MASK streams, objective, optimizer and schedule as historical S; equal updates, not equal FLOPs")
        config["protocol_hash"]=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        gs.atomic_json(cfg_path,config)
    if config["dataset_sha256"]!="1f7de1ec81e82ebcfcbc4134d5670ee35711230f05d00cd037c7b9e575ef8934":
        raise RuntimeError("Wrong parent data")
    # Explicit compatibility adapter around the established trainer. Both the
    # constructor and serialized MODEL config are set and verified together.
    gs.MODEL=copy.deepcopy(M);gs.new_model=build_model
    parent=load_parent_split(ROOT / "data/level1/parents_l1024.npz","train")
    validation=load_parent_split(ROOT / "data/level1/parents_l1024.npz","val")
    status("preflight")
    bench=preflight(parent,validation)
    assert bench["mean_update_seconds"]*24000*2<6*3600,"Unexpected throughput; do not exhaust campaign"
    for seed in SEEDS:
        status("training",seed=seed,size="M",benchmark=bench)
        result=gs.train_one(parent,validation,OUT / f"capacity/training/s{seed}_M","A",seed,config,deadline)
        if result["status"]!="trained":
            status("budget_stop",result=result);return
    ref=load_parent_split(OLD / "reference/fresh_l1024.npz","test_target")
    results=[]
    for seed in SEEDS:
        for size in ("S","M"):
            status("evaluation",seed=seed,size=size)
            path=(OLD / f"training/s{seed}_A/final.pt" if size=="S" else OUT / f"capacity/training/s{seed}_M/final.pt")
            results.append(evaluate_capacity(path,ref,size,seed,deadline))
    paired=[]
    for seed in SEEDS:
        small=next(x for x in results if x["seed"]==seed and x["size"]=="S")
        medium=next(x for x in results if x["seed"]==seed and x["size"]=="M")
        paired.append(dict(seed=seed,S_J=small["J"],M_J=medium["J"],
            relative_J_gain=(small["J"]-medium["J"])/max(small["J"],1e-12),
            CE_difference=medium["conditional"]["mean_ce"]-small["conditional"]["mean_ce"]))
    promote=(all(x["relative_J_gain"]>0 and x["CE_difference"]<=.002 for x in paired)
             and np.mean([x["relative_J_gain"] for x in paired])>=.10)
    decision=dict(status="complete",selected_size="M" if promote else "S",paired=paired,
        model=M if promote else dict(M,d_model=128,n_heads=4),completed=time.time(),
        interpretation="two-seed development choice; not a confirmatory superiority claim",
        next="six-seed F0/F1/F2 context repair, using remaining campaign budget and a separately frozen protocol")
    gs.atomic_json(OUT / "capacity/decision.json",decision)
    status("complete",decision=decision)


if __name__=="__main__":
    try: main()
    except Exception as exc:
        status("failed",error=repr(exc),traceback=traceback.format_exc())
        raise
