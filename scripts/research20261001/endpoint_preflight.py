"""One irreversible GPU preflight clock; only validation/synthetic inputs."""
from __future__ import annotations
import copy
import hashlib
import json
import platform
import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import endpoint_common as c
import endpoint_data as d
import endpoint_training as tr
import endpoint_sampling as sam


def setup_torch():
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before this GPU process starts")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    # The default fused CUDA backward was not bitwise reproducible for 13/18
    # inherited branches. Enforce the SAME setting in preflight and production.
    torch.use_deterministic_algorithms(True, warn_only=False)


def same(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    return a==b


def exact_restore(val,out):
    rows=[]
    for seed in c.SEEDS:
        hashes=[]
        for arm in c.ARMS:
            c.check();m,e,o,meta=tr.initialize(seed,arm)
            hashes.append([meta[k] for k in ["initial_raw_hash","initial_ema_hash","base_sha256"]])
            b=d.training_batch(val,seed,1,arm);tr.update(m,e,o,b,1)
            meta.update(step=1,global_step=12001)
            path=out/f"restore_s{seed}_{arm}.pt"
            tr.save_state(path,m,e,o,meta,"technical_preflight_not_science")
            # Uninterrupted next update, then exact restore and same next update.
            b=d.training_batch(val,seed,2,arm);left=tr.update(m,e,o,b,2)
            mm,ee,oo,rr=tr.restore(path,seed,arm,"technical_preflight_not_science")
            right=tr.update(mm,ee,oo,b,2)
            equality=dict(rows=left==right,model=same(m.state_dict(),mm.state_dict()),
                          ema=same(e.state_dict(),ee.state_dict()),optimizer=same(o.state_dict(),oo.state_dict()))
            c.write(out/f"restore_s{seed}_{arm}.json",dict(seed=seed,arm=arm,equality=equality,
                    row_differences={k:[left[k],right[k]] for k in left if left[k]!=right[k]},
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
            assert all(equality.values()), (seed,arm,equality)
            tr.check_optimizer(oo,12002)
            rows.append(dict(seed=seed,arm=arm,next_update_model_ema_optimizer_exact=True,checkpoint_sha256=c.sha(path)))
            del m,e,o,mm,ee,oo;torch.cuda.empty_cache()
        assert hashes[0]==hashes[1]==hashes[2]
    c.write(out/"restore.json",dict(status="passed",branches=rows));return rows


def training_timing(val,out):
    rows=[]
    for arm in c.ARMS:
        m,e,o,meta=tr.initialize(c.SEEDS[0],arm)
        for step in range(1,33):
            c.check();b=d.training_batch(val,c.SEEDS[0],step,arm)
            tr.update(m,e,o,b,step);tr.update(m,e,o,b,step)
            durations=[]
            for _ in range(4):
                torch.cuda.synchronize();start=time.perf_counter()
                b=d.training_batch(val,c.SEEDS[0],step,arm)
                rec=tr.update(m,e,o,b,step)
                torch.cuda.synchronize();durations.append(time.perf_counter()-start)
            rows.append(dict(arm=arm,width=b["width"],kind=b["kind"],sparse=b["sparse"],endpoint=b["endpoint"],
                             repeat=b["repeat"],durations=durations,mean_seconds=float(np.mean(durations)),
                             maximum_seconds=max(durations),loss=rec["loss"],grad_norm=rec["grad_norm"]))
        del m,e,o;torch.cuda.empty_cache()
    assert len(rows)==96
    c.write(out/"training_timing.json",dict(validation_only=True,rows=rows))
    return rows


def prediction_checks(val,out):
    model=tr.load_ema(c.base_path(c.SEEDS[0]),c.SEEDS[0])
    ids=d.select_parents(val,32);rows=[];differences=[];ces=[];timings={};metrics=[]
    for width in [48,96]:
        for masks in [1,2,8,32]:
            b=d.local_bank(val,ids,width,masks,"preflight_validation")
            c.save(out/f"validation_w{width}_m{masks}.npz",**b)
            torch.cuda.synchronize();start=time.perf_counter();fp=tr.predict(model,b)
            torch.cuda.synchronize();sec=time.perf_counter()-start
            bf=tr.predict(model,b,amp=True);dp=float(np.abs(fp["probability"]-bf["probability"]).max())
            dc=float(bf["ce"].mean()-fp["ce"].mean())
            differences.append(dp);ces.extend((bf["ce"].mean(1)-fp["ce"].mean(1)).tolist())
            rows.append(dict(width=width,masks=masks,inputs=32,max_probability_difference=dp,mean_ce_difference=dc,fp32_seconds=sec))
            timings.setdefault(str(width),[]).append(sec/32)
            c.save(out/f"precision_w{width}_m{masks}.npz",fp32=fp["probability"],bf16=bf["probability"],target=b["target"])
    # Time retained empirical CE separately rather than assuming local K0 shape cost.
    for kind,k in [("continuous",115),("continuous",1152),("held_gap",512)]:
        b=d.retention_bank(val,ids,kind,k);torch.cuda.synchronize();start=time.perf_counter()
        tr.predict(model,b);torch.cuda.synchronize();sec=time.perf_counter()-start
        timings.setdefault("retention",[]).append(sec/32)
    result=dict(status="passed" if max(differences)<=.005 and abs(float(np.mean(ces)))<=.001 else "failed",
                inputs=256,rows=rows,max_probability_difference=max(differences),absolute_mean_ce_difference=abs(float(np.mean(ces))),
                per_input_seconds={k:max(v) for k,v in timings.items()},base_sha256=c.sha(c.base_path(c.SEEDS[0])))
    c.write(out/"precision.json",result)
    del model;torch.cuda.empty_cache()
    if result["status"]!="passed":raise RuntimeError("BF16 precision gate failed; no automatic precision/protocol change")
    return result


def sampling_timing(out):
    model=tr.load_ema(c.base_path(c.SEEDS[0]),c.SEEDS[0]);rows=[]
    for w in [48,96]:
        for method in ["monotone-256","reveal192-repair64"]:
            c.check();torch.cuda.synchronize();start=time.perf_counter()
            # Preflight image identity domain is disjoint from all scientific generation.
            z=sam.sample(model,c.SEEDS[0],w,np.arange(2000000,2000016),method)
            torch.cuda.synchronize();sample_seconds=time.perf_counter()-start
            auditstart=time.perf_counter();sam.audit_repair(z)
            stats={}
            for typ,key in [("final","spins"),("prefix","prefix_spins"),("oracle","oracle_spins")]:
                if key not in z:continue
                for name,value in d.multiscale_stats(z[key]).items():
                    c.save(out/"sampler"/f"w{w}_{method}_{typ}_{name}.npz",**value)
            path=out/"sampler"/f"w{w}_{method}_shard.npz";c.save(path,**z)
            rows.append(dict(width=w,sampler=method,images=16,network_calls=256,sampling_seconds=sample_seconds,
                             statistics_io_audit_seconds=time.perf_counter()-auditstart,shard_bytes=path.stat().st_size,
                             purpose="technical_timing_not_scientific_generation"))
            c.write(out/"sampler"/f"w{w}_{method}_timing.json",rows[-1])
    del model;torch.cuda.empty_cache()
    c.write(out/"sampling_timing.json",dict(rows=rows));return rows


def run(cpu_receipt,fixture_receipt,continue_existing=False):
    assert not (c.OUT/"run.lock").exists() and not (c.OUT/"run_protocol.json").exists()
    cpu=c.read(cpu_receipt);fixture=c.read(fixture_receipt)
    assert cpu["status"]=="passed" and fixture["status"]=="passed" and fixture["full_bootstrap_reps"]
    c.OUT.mkdir(parents=True,exist_ok=True)
    if continue_existing:
        failure=c.read(c.OUT/"preflight_failure.json")
        assert "exact_restore" in failure["traceback"] and failure["exception"]=="AssertionError()"
        assert (c.EXPORT/"restore_diagnosis_v2.json").exists()
        diagnosis=c.read(c.EXPORT/"restore_diagnosis_v2.json")["results"]["production_bf16"]
        assert diagnosis["initial_loaded_state_exact"] and diagnosis["raw_max_abs_diff"]==0 and diagnosis["optimizer_exact"]
        start=c.read(c.OUT/"budget.json")["started"]
        assert (c.OUT/"preflight_failure_v2.json").exists()
        # A bounded all-branch diagnostic demonstrated the specific runtime fix.
        # This is a continuation of the original clock, never a new experiment.
        evidence=c.read(c.EXPORT/"restore_diagnosis_v4.json")
        strict=[r for r in evidence["rows"] if r["deterministic"]]
        assert {(r["seed"],r["arm"]) for r in strict}=={(s,a) for s in c.SEEDS for a in c.ARMS}
        assert all(all(r[k] for k in ["rows_equal","model_exact","ema_exact","optimizer_exact"])
                   and r["grad_max_diff"]==0 for r in strict)
        out=c.OUT/"preflight";recovery=out/"recovery_v3";recovery.mkdir(exist_ok=False)
    else:
        # No CUDA context or model evaluation occurs above this immutable clock creation.
        assert not (c.OUT/"budget.json").exists()
        start=time.time();c.write(c.OUT/"budget.json",dict(started=start,deadline=start+43200,wall_seconds=43200,
                                                        clock_includes_preflight_waits_statistics_backup=True))
        out=c.OUT/"preflight";out.mkdir(exist_ok=False);recovery=out
    c.set_clock();c.check()
    setup_torch();torch.cuda.reset_peak_memory_stats()
    c.write(out/("environment_v3.json" if continue_existing else "environment.json"),dict(torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version(),
                                       device=torch.cuda.get_device_name(),resources=c.resources(),tf32=False,autocast="bfloat16",
                                       cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),original_budget_preserved=True,
                                       deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
    val=d.load_parent(c.DATA,"val");assert len(val.spins)==256
    c.log("preflight_restore_start",continuing_original_clock=continue_existing);restore=exact_restore(val,recovery)
    c.log("preflight_precision_start");precision=prediction_checks(val,out)
    c.log("preflight_training_timing_start");train=training_timing(val,out)
    c.log("preflight_sampling_timing_start");sampling=sampling_timing(out)
    # Reconstructing a new input stream and physical re-audit are separately budgeted.
    audit_start=time.perf_counter()
    for step in range(1,129):
        for arm in c.ARMS:d.training_batch(val,c.SEEDS[0],step,arm)
    reconstruction_seconds=(time.perf_counter()-audit_start)/384*144000
    train_seconds=sum(float(np.mean([r["maximum_seconds"] for r in train if r["arm"]==a]))*48000 for a in c.ARMS)*1.15
    generation_seconds=0.;stat_io=0.
    for row in sampling:
        shards=6*3*(8 if row["width"]==96 else 4)
        if row["width"]==96 and row["sampler"]=="monotone-256":shards+=24
        generation_seconds+=row["sampling_seconds"]*shards*1.15
        # One recomputation audit after recording all physical outputs.
        stat_io+=row["statistics_io_audit_seconds"]*shards*2*1.15
    cond=precision["per_input_seconds"]
    conditional_seconds=(48000*cond["48"]+20352*cond["96"]+13824*cond["retention"])*1.15
    analysis_seconds=fixture["statistics_seconds"]*1.25+fixture["render_seconds"]+reconstruction_seconds*1.25+300
    elapsed=time.time()-start
    components=dict(preflight_elapsed=elapsed,train=train_seconds,mc_overlap_max=6500.,
                    generation=generation_seconds,physical_statistics_and_reaudit=stat_io,
                    conditional=conditional_seconds,analysis_input_audit_render=analysis_seconds,
                    checkpoint_io_and_misc=900.,backup_visual=2700.)
    forecast=elapsed+max(train_seconds,6500.)+generation_seconds+stat_io+conditional_seconds+analysis_seconds+900+2700
    # Conservative simultaneous checkpoint/output/archive footprint plus 2 GiB reserve.
    peak_extra_bytes=10*1024**3
    available=shutil.disk_usage(c.ROOT).free
    result=dict(status="passed" if forecast<=37800 and available>=peak_extra_bytes+2*1024**3 else "failed",
                created=time.time(),elapsed=elapsed,forecast_seconds=forecast,launch_max_seconds=37800,
                components=components,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                disk_free=available,estimated_extra_peak_bytes=peak_extra_bytes,required_reserve_bytes=2*1024**3,
                cpu_receipt_sha256=c.sha(cpu_receipt),fixture_receipt_sha256=c.sha(fixture_receipt),
                validation_only=True,formal_training_started=False,original_failure_preserved=continue_existing,
                recovery_contract="exact production BF16 next update and raw/EMA/optimizer; no tolerance widening")
    c.write(out/"summary.json",result);c.log("preflight_complete",result=result)
    return result
