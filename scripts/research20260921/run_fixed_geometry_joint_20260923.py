"""Frozen-model geometric response and low-dimensional joint composition.

Writes only its new output directory. No optimizer, MC generation, sampling
schedule changes, frequency edits, or writes to earlier experiments.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
import numpy as np
import torch
from ism_diffusion.geometry_study import atomic_json,file_hash,array_hash
from ism_diffusion.scale_data import load_parent_split
from ism_diffusion.scale_evaluation import load_scale_model
from fixed_geometry_math import make_cases,case_inputs,query_specs,metrics,symmetric_distribution

ARMS=("A","B","C","F0","F1","F2")
SEEDS=list(range(91001,91007))
PROTOCOL=ROOT/"docs/research_reboot_20260921/FIXED_GEOMETRY_JOINT_PROTOCOL_20260923_ZH.md"
REFERENCE=ROOT/"artifacts/adaptive_research_20260922/reference/fresh_l1024.npz"


def checkpoint(seed,arm):
    sub="geometry_alignment_20260921" if arm in ("A","B","C") else "adaptive_research_20260922/repair"
    return ROOT/f"artifacts/{sub}/training/s{seed}_{arm}/final.pt"


def log(out,phase,**kw):
    row=dict(phase=phase,time=time.time(),pid=os.getpid(),**kw)
    atomic_json(out/"status.json",row)
    print(json.dumps(row),flush=True)


def npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".tmp")
    with temp.open("wb") as f: np.savez_compressed(f,**arrays)
    os.replace(temp,path)


def budget(deadline):
    if time.time()>=deadline: raise TimeoutError("Two-hour budget reached; saved units retained, no automatic restart")


@torch.inference_mode()
def predict(model,tokens,coords,t,batch):
    result=[]
    for start in range(0,len(tokens),batch):
        x=torch.as_tensor(tokens[start:start+batch],device="cuda")
        c=torch.as_tensor(coords[start:start+batch],device="cuda")
        tt=torch.as_tensor(t[start:start+batch],device="cuda")
        pp=model(x,tt,c).float().softmax(1)[:,1,0,:3]
        if not bool(torch.isfinite(pp).all()): raise AssertionError("Nonfinite prediction")
        result.append(pp.cpu().numpy())
    return np.concatenate(result)


def math_tests():
    # Load the repository's explicit unit tests without requiring pytest remotely.
    import importlib.util
    spec=importlib.util.spec_from_file_location("fixed_tests",ROOT/"tests/test_fixed_geometry_math.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    names=sorted(n for n in vars(module) if n.startswith("test_"))
    for name in names: getattr(module,name)()
    return names


def preflight(out):
    tests=math_tests()
    model,_=load_scale_model(checkpoint(91001,"A"),torch.device("cuda"))
    model.requires_grad_(False)
    cases=make_cases()
    bench={};checks={}
    for w in (48,96):
        case=next(c for c in cases if c["width"]==w)
        tokens,coords,t,specs=case_inputs(case,"A")
        batch=4 if w==48 else 2
        predict(model,tokens[:batch],coords[:batch],t[:batch],batch)
        torch.cuda.synchronize();start=time.perf_counter()
        base=predict(model,tokens,coords,t,batch)
        seconds=time.perf_counter()-start
        bench[str(w)]=dict(case_seconds=seconds,batch=batch,cases=sum(c["width"]==w for c in cases))
        moved=predict(model,tokens,coords+np.array([13.,-7.],np.float32),t,batch)
        checks[f"translation_w{w}"]=float(abs(base-moved).max())
        # Permute the unknown background only; first three query/evidence slots stay.
        perm=np.r_[np.arange(3),np.random.default_rng(230923+w).permutation(np.arange(3,w*w))]
        reordered=predict(model,tokens[:,:,perm],coords[:,:,perm],t,batch)
        checks[f"background_permutation_w{w}"]=float(abs(base-reordered).max())
        pp=np.ones(8)/8
        checked=metrics(base,pp,specs)
        checks[f"swap_equivalence_w{w}"]=float(checked["swap_permutation_error"])
        checks[f"parallel_identity_w{w}"]=float(checked["parallel_identity_residual"])
        checks[f"sequential_identity_w{w}"]=float(checked["sequential_identity_residual"])
    del model
    torch.cuda.empty_cache()
    forecast=36*sum(v["case_seconds"]*v["cases"] for v in bench.values())*1.3+300
    result=dict(status="passed" if max(checks.values())<2e-5 and forecast<7200 else "failed",
                math_tests=tests,checks=checks,benchmark=bench,forecast_seconds=forecast,
                gpu=torch.cuda.get_device_name(),torch=torch.__version__,python=platform.python_version(),
                precision="FP32, highest matmul, no autocast",model_parameters=1976706)
    atomic_json(out/"preflight.json",result)
    if result["status"]!="passed": raise RuntimeError("Preflight gate failed; full run not started")
    return result


def reference(out,cases,deadline):
    log(out,"reference_loading")
    parent=load_parent_split(REFERENCE,"test_target")
    with (REFERENCE.parent/"complete.json").open() as f: qa=json.load(f)
    assert qa["status"]=="passed"
    assert file_hash(REFERENCE)==qa["sha256"]
    n=len(parent.spins);assert n==1024 and len(np.unique(parent.chain_ids))==8
    origins=np.random.default_rng(9235101).integers(parent.lattice_size,size=(n,256,2))
    ids=np.arange(n)[:,None]
    counts=np.zeros((len(cases),n,8),np.int32)
    for i,case in enumerate(cases):
        budget(deadline)
        sites=np.array(case["physical_sites"])
        values=[]
        for site in sites:
            xx=(origins[...,0]+site[0])%parent.lattice_size
            yy=(origins[...,1]+site[1])%parent.lattice_size
            values.append((parent.spins[ids,xx,yy]>0).astype(np.int8))
        codes=values[0]+2*values[1]+4*values[2]
        for code in range(8): counts[i,:,code]=(codes==code).sum(1)
        if i%16==0: log(out,"reference_counts",case=i,total=len(cases))
    assert np.all(counts.sum(-1)==256)
    p=symmetric_distribution(counts.sum(1))
    event=p[:,[0,1,2,3,4,5,6,7]].reshape(-1,8)
    # Every three-spin state must be represented in pooled reference; no clipping.
    assert event.min()>0
    npz(out/"reference.npz",counts=counts,origins=origins,chain=parent.chain_ids,parent=np.arange(n),
        state_bit_order=np.array(["i","e","j"]),reference_p=p)
    atomic_json(out/"reference_complete.json",dict(status="complete",source=str(REFERENCE),
        source_sha256=qa["sha256"],parents=n,chains=8,origins_per_parent=256,
        minimum_joint_state_probability=float(event.min()),
        source_qa=qa,reference_sha256=file_hash(out/"reference.npz"),
        warning="MC reference reused from prior campaign; exact symmetry, estimated moments; exploratory diagnostic"))
    return p


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--preflight-only",action="store_true")
    args=parser.parse_args()
    out=Path(args.output).resolve()
    # Fresh outputs only. Explicit lock prevents accidental duplicated processes.
    out.mkdir(parents=True,exist_ok=False)
    lock=(out/"run.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();deadline=started+7200
    os.environ.setdefault("OMP_NUM_THREADS","2")
    torch.set_num_threads(2);torch.set_float32_matmul_precision("highest")
    try:
        shutil.copy2(PROTOCOL,out/"protocol.md")
        src=out/"source";src.mkdir()
        for name in ("run_fixed_geometry_joint_20260923.py","fixed_geometry_math.py","analyze_fixed_geometry_joint_20260923.py"):
            shutil.copy2(Path(__file__).parent/name,src/name)
        tests=ROOT/"tests/test_fixed_geometry_math.py";shutil.copy2(tests,src/tests.name)
        for name in ("scale_model.py","scale_evaluation.py","scale_data.py","geometry_study.py","model.py"):
            shutil.copy2(ROOT/"ism_diffusion"/name,src/name)
        log(out,"preflight")
        gate=preflight(out)
        if args.preflight_only:
            log(out,"preflight_complete",elapsed_seconds=time.time()-started,**gate)
            return
        cases=make_cases()
        atomic_json(out/"cases.json",cases)
        atomic_json(out/"query_specs.json",{str(w):query_specs(w*w) for w in (48,96)})
        hashes={f"s{s}_{a}":file_hash(checkpoint(s,a)) for s in SEEDS for a in ARMS}
        atomic_json(out/"run_protocol.json",dict(started=started,deadline=deadline,budget_hours=2,
            seeds=SEEDS,arms=ARMS,cases=len(cases),inputs_per_case=20,
            expected_models=36,expected_forward_inputs=36*len(cases)*20,
            checkpoints=hashes,source_hashes={f.name:file_hash(f) for f in src.iterdir()},
            protocol_sha256=file_hash(out/"protocol.md"),forecast_seconds=gate["forecast_seconds"],
            new_training=False,old_campaign_unmodified=True))
        p=reference(out,cases,deadline)
        completed=[]
        for seed in SEEDS:
            for arm in ARMS:
                budget(deadline)
                dest=out/"models"/f"s{seed}_{arm}";dest.mkdir(parents=True)
                model,payload=load_scale_model(checkpoint(seed,arm),torch.device("cuda"))
                model.requires_grad_(False)
                assert not model.training and sum(x.numel() for x in model.parameters())==1976706
                rep=arm if arm in ("A","B","C") else "A"
                predictions=[];input_hashes=[];mrows=[];t0=time.time()
                for i,case in enumerate(cases):
                    budget(deadline)
                    tokens,coords,t,specs=case_inputs(case,rep)
                    pp=predict(model,tokens,coords,t,gate["benchmark"][str(case["width"])]["batch"])
                    mm=metrics(pp,p[i],specs)
                    for key in ("swap_permutation_error","parallel_identity_residual","sequential_identity_residual"):
                        if float(mm[key])>2e-5: raise AssertionError(f"{key}: {mm[key]}")
                    predictions.append(pp);input_hashes.append(array_hash(tokens,coords,t))
                    mrows.append(dict(case=i,**{k:float(v) for k,v in mm.items()}))
                    if i%16==0: log(out,"model_evaluation",seed=seed,arm=arm,case=i,total=len(cases),complete_models=len(completed))
                npz(dest/"predictions.npz",probabilities=np.stack(predictions),input_sha256=np.array(input_hashes))
                atomic_json(dest/"metrics.json",mrows)
                after=file_hash(checkpoint(seed,arm))
                assert after==hashes[f"s{seed}_{arm}"]
                atomic_json(dest/"complete.json",dict(seed=seed,arm=arm,cases=len(cases),inputs=len(cases)*20,
                    checkpoint_sha256=after,checkpoint_unchanged=True,elapsed_seconds=time.time()-t0,
                    predictions_sha256=file_hash(dest/"predictions.npz"),parameters=1976706,
                    checkpoint_step=payload.get("step",payload.get("steps")),native_representation=rep))
                completed.append(f"s{seed}_{arm}")
                del model,payload;torch.cuda.empty_cache()
                log(out,"model_complete",cell=completed[-1],complete_models=len(completed),elapsed_seconds=time.time()-started)
        atomic_json(out/"inference_complete.json",dict(status="complete",models=completed,checkpoint_unchanged=True,
            elapsed_seconds=time.time()-started,finished=time.time()))
        log(out,"statistical_analysis",complete_models=len(completed))
        budget(deadline)
        subprocess.run([sys.executable,str(Path(__file__).parent/"analyze_fixed_geometry_joint_20260923.py"),
                        "--root",str(out)],check=True,timeout=max(1,deadline-time.time()))
        log(out,"remote_complete_requires_backup_and_visual_review",complete_models=len(completed),
            elapsed_seconds=time.time()-started)
    except Exception as exc:
        atomic_json(out/"failure.json",dict(error=repr(exc),traceback=traceback.format_exc(),
            time=time.time(),elapsed_seconds=time.time()-started,budget_stop=isinstance(exc,(TimeoutError,subprocess.TimeoutExpired))))
        log(out,"failed_or_budget_stopped",error=repr(exc))
        raise


if __name__=="__main__": main()
