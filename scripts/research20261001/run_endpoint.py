"""Authorized endpoint study. Explicit modes; exclusive run lock; no implicit resume."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
import endpoint_common as c


def freeze(cpu_receipt,fixture_receipt,visual_receipt):
    cpu_receipt,fixture_receipt,visual_receipt=map(lambda x:Path(x).resolve(),[cpu_receipt,fixture_receipt,visual_receipt])
    assert not (c.OUT/"run.lock").exists()
    p=c.read(c.OUT/"preflight/summary.json");b=c.read(c.OUT/"budget.json")
    assert p["status"]=="passed"
    forecast=p["forecast_seconds"]+time.time()-p["created"]
    if forecast>37800:raise RuntimeError("Elapsed waits exceed launch forecast gate; no automatic budget reset")
    cpu=c.read(cpu_receipt)
    assert cpu["status"]=="passed" and c.read(fixture_receipt)["status"]=="passed"
    for name,digest in cpu["source_hashes"].items():assert c.sha(Path(__file__).parent/name)==digest
    visual=c.read(visual_receipt);assert visual["status"]=="passed" and visual["actual_figures_viewed"]==6
    from endpoint_training import base_hashes
    bases={str(seed):dict(path=c.base_path(seed).relative_to(c.ROOT).as_posix(),sha256=c.sha(c.base_path(seed))) for seed in c.SEEDS}
    assert all(bases[str(seed)]["sha256"]==base_hashes()[seed] for seed in c.SEEDS)
    effective=copy.deepcopy(c.CFG)
    effective.update(lifecycle="authorized_frozen_before_formal_training",execution_authorized=True,
                     design_snapshot_preserved=True,
                     implementation_clarifications={
                         "phase0_order":"Historical coverage after freeze; frozen-model science after all18 finals locked, sharing fresh evaluation MC",
                         "energy":"negative mean open bond product, same historical per-bond unit; MC QA retains periodic per-site energy",
                         "stress_background_selection":"4 fixed backgrounds: chain0/4/8/12 midpoint, all16 neighbor patterns; finite test not population max",
                         "origin30_long_band":[9,15],
                         "conditional_blocks":{"main":2,"sensitivity":[1,4]},
                         "missing_mask_updates":"Ordinary random masks may contain zero MASK; ordinary objective unchanged",
                         "seed_order":"six inherited lineages;18 branches in fixed seed-major arm order,1000-step chunks",
                         "sampler_timing":"same256 calls; actual wall seconds also retained",
                         "technical_fix_history":"Before any GPU work, CPU differential test corrected new energy formula to historical per-bond units; failing v1 retained",
                         "coarse_graining":"nine offsets nested within original image; no new independent replicates or RG training"})
    effective["runtime_environment_contract"]={"CUBLAS_WORKSPACE_CONFIG":":4096:8","TF32":False,
                                              "torch_use_deterministic_algorithms":True,"warn_only":False,
                                              "GPU_recovery":"exact production BF16; original failed preflight retained; no tolerance widening"}
    c.write(c.OUT/"effective_config.json",effective)
    protocol=dict(study=c.STUDY,created=time.time(),authorization="ENDPOINT_MASK_SAMPLER_DISENTANGLEMENT_AUTHORIZATION_20261001_ZH.md",
                  sources=c.sources(),base_checkpoints=bases,deadline=b["deadline"],budget_started=b["started"],
                  effective_config_sha256=c.sha(c.OUT/"effective_config.json"),
                  preflight_summary_sha256=c.sha(c.OUT/"preflight/summary.json"),
                  cpu_receipt=dict(path=str(Path(cpu_receipt).relative_to(c.ROOT)),sha256=c.sha(cpu_receipt)),
                  software_receipt=dict(path=str(Path(fixture_receipt).relative_to(c.ROOT)),sha256=c.sha(fixture_receipt)),
                  visual_receipt=dict(path=str(Path(visual_receipt).relative_to(c.ROOT)),sha256=c.sha(visual_receipt)),
                  launch_forecast_at_freeze=forecast,scientific_samples_not_reduced=True)
    protocol["protocol_hash"]=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
    c.write(c.OUT/"run_protocol.json",protocol)
    from endpoint_management import archive
    paths=[c.ROOT/name for name in protocol["sources"]]
    # Preflight restore scratch weights are redundant technical fixtures; formal finals never excluded.
    paths += [p for p in c.OUT.rglob("*") if p.is_file() and p.suffix!=".pt" and not p.name.endswith(".tmp") and p.name!="status.json"]
    for directory in [Path(cpu_receipt).parent,Path(fixture_receipt).parent]:
        paths += [p for p in directory.rglob("*") if p.is_file()]
    paths += [Path(visual_receipt)]
    # Preserve the technical failure investigation in the initial archive. These
    # files are management evidence, not extra scientific samples/checkpoints.
    for pattern in ["restore_diagnosis_v*.json","diagnose_gpu_restore_v*.py",
                    "software_environment.json","cpu_visual_equivalence_v*.json"]:
        paths += [p for p in c.EXPORT.glob(pattern) if p.is_file()]
    receipt=archive(c.ROOT,c.EXPORT,"initial_source_protocol_preflight_v1",paths)
    c.write(c.OUT/"initial_export.json",dict(archive_sha256=receipt["archive_sha256"],members=receipt["members"],time=time.time()))
    print(json.dumps(dict(status="frozen_not_started",protocol_hash=protocol["protocol_hash"],archive_bytes=receipt["archive_bytes"],deadline=b["deadline"])),flush=True)


def start_reference():
    remaining=c.DEADLINE-time.time();assert remaining>0
    command=["timeout","--signal=KILL",f"{remaining:.3f}s",sys.executable,"-u",str(Path(__file__)),"--mode","reference"]
    output=(c.OUT/"reference.stdout").open("xb")
    child=subprocess.Popen(command,cwd=c.ROOT,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    output.close()
    c.write(c.OUT/"reference_process.json",dict(timeout_pid=child.pid,pgid=child.pid,command=command,deadline=c.DEADLINE,time=time.time()))
    return child


def check_reference(child):
    c.check()
    if child is None:
        # Separately authorized training phase has no MC/evaluation dependency.
        return
    if (c.OUT/"reference_failure.json").exists():raise RuntimeError("Independent reference failed; see reference_failure.json")
    code=child.poll()
    if code is not None and (code!=0 or not (c.OUT/"reference/complete.json").exists()):
        raise RuntimeError(f"Reference exited without QA completion, exit={code}")


def train(protocol,child,prepare_diagnostics=True):
    import torch
    import endpoint_data as d
    import endpoint_training as tr
    parent=d.load_parent(c.DATA,"train");assert len(parent.spins)==768
    if prepare_diagnostics:
        from endpoint_evaluation import old_coverage_audit,validation_banks
        old_coverage_audit(parent);validation_banks()
    start=time.time();finals={}
    for block in range(8):
        for seed in c.SEEDS:
            for arm in c.ARMS:
                check_reference(child);c.check_sources(protocol["sources"])
                folder=c.OUT/"training"/f"s{seed}_{arm}"
                if block==0:
                    folder.mkdir(parents=True,exist_ok=False)
                    m,e,o,meta=tr.initialize(seed,arm)
                    c.write(folder/"initial.json",dict(**meta,protocol_hash=protocol["protocol_hash"],time=time.time()))
                else:
                    m,e,o,meta=tr.restore(folder/"last.pt",seed,arm,protocol["protocol_hash"])
                    assert meta["step"]==block*1000
                t0=time.time();c.log("training_chunk_start",seed=seed,arm=arm,from_step=block*1000,to_step=(block+1)*1000)
                with (folder/"log.jsonl").open("a",encoding="utf-8",buffering=1) as handle:
                    for step in range(block*1000+1,(block+1)*1000+1):
                        c.check()
                        if step%100==1:check_reference(child)
                        batch=d.training_batch(parent,seed,step,arm)
                        row=tr.update(m,e,o,batch,step)
                        for field,key in [("physical","physical_hash"),("native","native_hash"),("supervised","supervision_hash")]:
                            meta[field+"_digest"]=c.increment(meta[field+"_digest"],row[key])
                        meta.update(step=step,global_step=12000+step)
                        row.update(seed=seed,arm=arm,step=step,global_step=12000+step,time=time.time())
                        handle.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
                meta["elapsed_seconds"]+=time.time()-t0
                tr.check_optimizer(o,12000+meta["step"])
                tr.save_state(folder/"last.pt",m,e,o,meta,protocol["protocol_hash"],exclusive=block==0)
                if meta["step"] in [4000,6000]:
                    path=folder/f"ema_{meta['step']}.pt"
                    with path.open("xb") as f:
                        torch.save(dict(study=c.STUDY,protocol_hash=protocol["protocol_hash"],seed=seed,arm=arm,
                                        step=meta["step"],global_step=meta["global_step"],ema=e.state_dict(),metadata=copy.deepcopy(meta)),f)
                if block==7:
                    path=folder/"final.pt";tr.save_state(path,m,e,o,meta,protocol["protocol_hash"])
                    digest=c.sha(path);finals[path.relative_to(c.ROOT).as_posix()]=digest
                    c.write(folder/"complete.json",dict(status="complete",step=8000,global_step=20000,final_sha256=digest,metadata=meta,time=time.time()))
                c.log("training_chunk_complete",seed=seed,arm=arm,step=meta["step"],loss=row["loss"],grad_norm=row["grad_norm"],training_final_complete=len(finals))
                del m,e,o;torch.cuda.empty_cache()
    assert len(finals)==18
    c.write(c.OUT/"final_lock.json",dict(files=finals,locked=time.time(),training_seconds=time.time()-start,updates=144000))
    c.log("all_training_finals_locked",finals=18,additional_updates=144000)


def run():
    c.set_clock();c.check()
    protocol=c.read(c.OUT/"run_protocol.json");c.check_sources(protocol["sources"])
    assert protocol["deadline"]==c.DEADLINE
    preflight=c.read(c.OUT/"preflight/summary.json")
    total=preflight["forecast_seconds"]+time.time()-preflight["created"]
    if preflight["status"]!="passed" or total>37800:raise RuntimeError("Final launch time gate did not pass")
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG")==":4096:8"
    backup=c.read(c.OUT/"initial_backup_confirmation.json")
    assert backup["status"]=="passed" and backup["archive_sha256"]==c.read(c.OUT/"initial_export.json")["archive_sha256"]
    with (c.OUT/"run.lock").open("x",encoding="utf-8") as f:
        json.dump(dict(study=c.STUDY,pid=os.getpid(),time=time.time(),protocol_hash=protocol["protocol_hash"]),f)
    started=time.time()
    c.write(c.OUT/"formal_started.json",dict(started=started,pid=os.getpid(),pgid=os.getpgrp(),parent_pid=os.getppid(),
                                           budget_started=protocol["budget_started"],deadline=c.DEADLINE,forecast_including_elapsed=total))
    from endpoint_preflight import setup_torch
    setup_torch();child=None
    try:
        child=start_reference();train(protocol,child)
        while not (c.OUT/"reference/complete.json").exists():
            check_reference(child);c.log("waiting_fixed_reference");time.sleep(20)
        check_reference(child)
        from endpoint_evaluation import evaluate_all
        c.log("frozen_evaluation_start");evaluate_all(protocol)
        from endpoint_statistics import analyze
        c.log("statistics_start");summary=analyze(c.OUT)
        from endpoint_figures import render
        c.log("render_start");render(c.OUT)
        from endpoint_management import audit_all,package_science
        c.log("final_cpu_audit_start");audit_all()
        ended=time.time()
        c.write(c.OUT/"final_summary.json",dict(status="remote_complete_requires_backup_and_visual_review",scientific_success=summary["scientific_success"],
                                              formal_started=started,science_completed=ended,formal_seconds=ended-started,
                                              total_budget_seconds_used=ended-protocol["budget_started"],deadline=c.DEADLINE,
                                              primary=summary["primary"],analysis_summary="analysis/summary.json",finals=18,formal_images=6912,
                                              frozen_phase0_images=384,reference_parents=2048,prediction_inputs=82176,figures=6))
        c.log("scientific_outputs_complete");receipt=package_science()
        c.write(c.OUT/"remote_export_complete.json",dict(time=time.time(),archive_sha256=receipt["archive_sha256"],bytes=receipt["archive_bytes"],members=receipt["members"]))
        c.log("remote_export_complete_requires_local_closure",archive_bytes=receipt["archive_bytes"])
    except BaseException as exc:
        try:c.write(c.OUT/"failure.json",dict(time=time.time(),exception=repr(exc),traceback=traceback.format_exc(),deadline=c.DEADLINE))
        finally:
            if child is not None and child.poll() is None:
                # Only the child process group created by this invocation; no unrelated jobs.
                os.killpg(child.pid,signal.SIGTERM)
        raise


def reference():
    c.set_clock();c.check()
    assert (c.OUT/"run.lock").exists()
    try:
        from endpoint_evaluation import new_reference
        new_reference()
    except BaseException as exc:
        c.write(c.OUT/"reference_failure.json",dict(time=time.time(),exception=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--mode",required=True,choices=["preflight","preflight_continue","freeze","run","reference"])
    p.add_argument("--cpu-receipt");p.add_argument("--fixture-receipt");p.add_argument("--visual-receipt")
    a=p.parse_args()
    if a.mode in ("preflight","preflight_continue"):
        from endpoint_preflight import run as preflight
        try:preflight(a.cpu_receipt,a.fixture_receipt,a.mode=="preflight_continue")
        except BaseException as exc:
            c.write(c.OUT/("preflight_failure_v3.json" if a.mode=="preflight_continue" else "preflight_failure.json"),dict(time=time.time(),exception=repr(exc),traceback=traceback.format_exc(),formal_started=False))
            raise
    elif a.mode=="freeze":freeze(a.cpu_receipt,a.fixture_receipt,a.visual_receipt)
    elif a.mode=="run":run()
    else:reference()
