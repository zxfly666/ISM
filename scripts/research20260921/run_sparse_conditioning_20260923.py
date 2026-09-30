"""Bounded three-arm campaign. New output, frozen sources, no automatic restart."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
import numpy as np
import torch
from ism_diffusion import geometry_study as gs
from ism_diffusion.scale_data import load_parent_split
from run_adaptive_repair_20260922 import restore
from sparse_conditioning_design import SEEDS,ARMS,GEN_DEFS,COND_DEFS,MC_SEED,make_cases,self_test
from sparse_conditioning_core import views,update
from evaluate_sparse_conditioning_20260923 import log,budget,prepare_reference,evaluate_model

DOC=ROOT/"docs/research_reboot_20260921/SPARSE_CONDITIONING_TRAINING_PROTOCOL_20260923_ZH.md"


def base(seed):return ROOT/f"artifacts/adaptive_research_20260922/repair/training/s{seed}_F0/final.pt"


def freeze(out,validation):
    pre=json.loads((out/"preflight/gpu_preflight.json").read_text())
    assert pre["tests"]["status"]=="passed" and pre["base_unchanged"]
    assert pre["tests"]["hard_soft_identity_weight_error"]<1e-6
    # Recalculate ONLY costs after the documented reduction of generation counts.
    stepcost=6*sum(x["mean_seconds"] for x in pre["training"].values())*1.2
    generation=18*sum(d["samples"]*pre["generation"][str(d["width"])]["forecast_per_image_256steps"] for d in GEN_DEFS)*1.2
    conditional=18*2*sum(pre["diagnostics_case_seconds"][str(c["width"])] for c in make_cases())*1.25+600
    forecasts={str(n):n*stepcost+generation+conditional+(18*n/1000+6)*pre["val_seconds"]+1800 for n in (4000,6000,8000)}
    choices=[int(n) for n,v in forecasts.items() if v<10*3600]
    if not choices:raise RuntimeError("Revised minimum balanced design still fails budget; do not start")
    baseline={}
    for seed in SEEDS:
        model,ema,opt,p=restore(base(seed),gs.MODEL);assert p["step"]==36000
        baseline[str(seed)]=gs.balanced_risk(gs.evaluate(ema,validation,"A",gs.VAL_DEFS))
        del model,ema,opt,p;torch.cuda.empty_cache()
    sources=sorted(list((ROOT/"ism_diffusion").glob("*.py"))+list(Path(__file__).parent.glob("*.py")))
    snapshot=out/"source";snapshot.mkdir(exist_ok=False)
    hashes={}
    for p in sources:
        rel=p.relative_to(ROOT);target=snapshot/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target);hashes[str(rel)]=gs.file_hash(p)
    shutil.copy2(DOC,out/"protocol.md")
    began=time.time();steps=max(choices)
    protocol=dict(version="sparse_conditioning_v1_budget_revised_before_training",started=began,deadline=began+12*3600,
        steps=steps,model=gs.MODEL,seeds=SEEDS,arms=ARMS,generation=GEN_DEFS,conditional=COND_DEFS,
        forecasts_seconds=forecasts,forecast_hours=forecasts[str(steps)]/3600,source_hashes=hashes,
        base_hashes={str(base(s)):gs.file_hash(base(s)) for s in SEEDS},base_validation=baseline,
        oracle=json.loads((out/"preflight/oracle_complete.json").read_text()),mc_seed=MC_SEED,
        train_source_sha256=gs.file_hash(ROOT/"data/level1/parents_l1024.npz"),
        protocol_sha256=gs.file_hash(out/"protocol.md"),microbatch={24:8,48:4,96:1},generation_shard=16,
        initial_budget_gate=pre["status"],budget_revision="reduce generation counts uniformly before training; preserve 6 seeds, 3 arms and 256-step sampler")
    protocol["protocol_hash"]=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    gs.atomic_json(out/"run_protocol.json",protocol);gs.atomic_json(out/"cases.json",make_cases())
    return protocol


def save(path,model,ema,opt,p,step,protocol,seed,arm,elapsed,digest):
    data=dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
        config=dict(model=gs.MODEL,variant="A",seed=seed),step=36000+step,sparse_step=step,arm=arm,
        protocol_hash=protocol["protocol_hash"],base_checkpoint_sha256=protocol["base_hashes"][str(base(seed))],
        elapsed_seconds=elapsed,stream_digests=digest,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
        numpy_rng=np.random.get_state(),python_rng=random.getstate())
    tmp=path.with_suffix(".tmp");torch.save(data,tmp);os.replace(tmp,path)


def train_block(out,parent,validation,oracle,protocol,seed,arm,target):
    dest=out/"training"/f"s{seed}_{arm}";dest.mkdir(parents=True,exist_ok=True)
    last=dest/"last.pt";model,ema,opt,p=restore(last if last.exists() else base(seed),gs.MODEL)
    start=int(p.get("sparse_step",0))+1
    previous=float(p["elapsed_seconds"]) if "sparse_step" in p else 0.
    digest=p.get("stream_digests",dict(main_hash="",aux_clean_hash="",sparse_input_hash=""))
    if "sparse_step" in p:
        assert p["protocol_hash"]==protocol["protocol_hash"]
        torch.set_rng_state(p["torch_rng"]);torch.cuda.set_rng_state_all(p["cuda_rng"])
        np.random.set_state(p["numpy_rng"]);random.setstate(p["python_rng"])
    began=time.monotonic();model.train();completed=start-1
    try:
        with (dest/"train.jsonl").open("a",encoding="utf-8",buffering=1) as logfile:
            for step in range(start,target+1):
                budget(protocol["deadline"])
                packed=views(parent,oracle,seed,step)
                stats=update(model,ema,opt,packed,step,protocol["steps"],arm)
                completed=step
                for key in digest:digest[key]=hashlib.sha256((digest[key]+str(step)+stats[key]).encode()).hexdigest()
                if step==1 or step%20==0:
                    logfile.write(json.dumps(dict(step=step,seed=seed,arm=arm,width=packed[0]["width"],
                        kind=packed[0]["kind"],elapsed=previous+time.monotonic()-began,**stats))+"\n")
            budget(protocol["deadline"])
            risk=gs.balanced_risk(gs.evaluate(ema,validation,"A",gs.VAL_DEFS))
            logfile.write(json.dumps(dict(step=target,validation=risk))+"\n")
    except Exception:
        save(dest/"stopped_state.pt",model,ema,opt,p,completed,protocol,seed,arm,
             previous+time.monotonic()-began,digest);raise
    elapsed=previous+time.monotonic()-began
    save(last,model,ema,opt,p,target,protocol,seed,arm,elapsed,digest)
    hp=dest/"validation_history.json";history=json.loads(hp.read_text()) if hp.exists() else []
    history.append(dict(step=target,ce=risk["mean_ce"]));gs.atomic_json(hp,history)
    basece=protocol["base_validation"][str(seed)]["mean_ce"]
    if len(history)>=2 and all(x["ce"]>basece+.05 for x in history[-2:]):
        raise RuntimeError(f"Whole campaign CE safety stop {seed} {arm}; two consecutive >base+0.05")
    if target==protocol["steps"]:
        save(dest/"final.pt",model,ema,opt,p,target,protocol,seed,arm,elapsed,digest)
        gs.atomic_json(dest/"complete.json",dict(status="trained",steps=target,elapsed_seconds=elapsed,
            stream_digests=digest,validation=risk,sha256=gs.file_hash(dest/"final.pt")))
    del model,ema,opt,p;torch.cuda.empty_cache()
    return dict(seed=seed,arm=arm,step=target,validation=risk,elapsed_seconds=elapsed)


def main(out):
    lock=(out/"run.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/"run_protocol.json").exists() or (out/"training").exists():
        raise RuntimeError("Existing campaign detected; no automatic restart")
    smoke=json.loads((out/"preflight/pipeline_tests.json").read_text());assert smoke["status"]=="passed"
    torch.set_num_threads(2);torch.set_float32_matmul_precision("high")
    parent=load_parent_split(ROOT/"data/level1/parents_l1024.npz","train")
    validation=load_parent_split(ROOT/"data/level1/parents_l1024.npz","val")
    oracle_status=json.loads((out/"preflight/oracle_complete.json").read_text())
    assert oracle_status["status"]=="passed" and oracle_status["split"]=="train"
    assert gs.file_hash(out/"preflight/training_oracle.npz")==oracle_status["sha256"]
    with np.load(out/"preflight/training_oracle.npz") as z:oracle=z["loco"]
    protocol=freeze(out,validation)
    log(out,"protocol_frozen",steps=protocol["steps"],forecast_hours=protocol["forecast_hours"],deadline=protocol["deadline"])
    mc=None
    try:
        ref=out/"reference";ref.mkdir(exist_ok=False)
        with (ref/"queue.log").open("w") as f:
            mc=subprocess.Popen([sys.executable,"-u",str(Path(__file__).parent/"prepare_sparse_conditioning_data.py"),
                "--out",str(ref),"--mode","reference"],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1"))
        gs.atomic_json(out/"processes.json",dict(main_pid=os.getpid(),reference_pid=mc.pid,started=protocol["started"]))
        for target in range(1000,protocol["steps"]+1,1000):
            for seed in SEEDS:
                for arm in ARMS:
                    budget(protocol["deadline"])
                    if mc.poll() not in (None,0):raise RuntimeError("Independent MC process failed; stop campaign")
                    qa=ref/"complete.json"
                    if qa.exists() and json.loads(qa.read_text())["status"]!="passed":raise RuntimeError("New MC QA failed; no alternate seed/fallback")
                    log(out,"training",seed=seed,arm=arm,target_step=target,total_steps=protocol["steps"])
                    result=train_block(out,parent,validation,oracle,protocol,seed,arm,target)
                    print(json.dumps(dict(stage="training_block_complete",**result)),flush=True)
        for seed in SEEDS:
            digests=[json.loads((out/"training"/f"s{seed}_{a}"/"complete.json").read_text())["stream_digests"] for a in ARMS]
            assert digests[0]==digests[1]==digests[2]
        del parent,validation,oracle
        log(out,"waiting_reference")
        while mc.poll() is None:
            budget(protocol["deadline"]);time.sleep(10)
        assert mc.returncode==0
        qa=json.loads((ref/"complete.json").read_text());assert qa["status"]=="passed"
        assert gs.file_hash(ref/"fresh_l1024.npz")==qa["sha256"]
        fresh=load_parent_split(ref/"fresh_l1024.npz","test_target")
        prepare_reference(fresh,out,make_cases(),protocol["deadline"])
        for seed in SEEDS:
            for arm in ARMS:
                budget(protocol["deadline"]);log(out,"evaluation",seed=seed,arm=arm)
                evaluate_model(out/"training"/f"s{seed}_{arm}"/"final.pt",fresh,
                    out/"evaluation"/f"s{seed}_{arm}",out,seed,arm,protocol["deadline"])
        del fresh
        log(out,"statistical_analysis")
        from finalize_sparse_conditioning_20260923 import main as finalize
        summary=finalize(out,protocol["deadline"])
        log(out,"remote_complete_requires_backup_and_visual_review",elapsed_hours=summary["elapsed_hours"])
    finally:
        # Only the child process group started by THIS campaign is eligible.
        if mc is not None and mc.poll() is None:
            os.killpg(mc.pid,signal.SIGTERM);mc.wait(timeout=30)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args();out=Path(a.out).resolve()
    try:main(out)
    except Exception as e:
        gs.atomic_json(out/"failure.json",dict(error=repr(e),traceback=traceback.format_exc(),time=time.time(),
            budget_stop=isinstance(e,TimeoutError),automatic_restart=False))
        log(out,"failed_or_budget_stop",error=repr(e));raise
