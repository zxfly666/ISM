"""Single bounded execution. Existing output/lock forbids re-launch/resume."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import gc
import hashlib
import io
import json
import platform
import shutil
import time
import traceback
import unittest
import numpy as np
import torch
import ms_common as c
import ms_data as d
import ms_model as m
import ms_evaluation as ev


def phase(name, **extra):
    row=dict(phase=name,time=time.time(),**extra)
    c.write(c.OUT/"status.json",row,replace=True)
    print(json.dumps(row,ensure_ascii=False),flush=True)


def prepare():
    if c.OUT.exists():
        raise FileExistsError("Study already exists: no repeated prepare, budget reset or automatic resume")
    c.disk()
    c.OUT.mkdir(parents=True)
    now=time.time()
    c.write(c.OUT/"budget.json",dict(started=now,deadline=now+c.CFG["hard_seconds"],hard_seconds=c.CFG["hard_seconds"]))
    c.write(c.OUT/"run.lock",dict(started=now,study=c.CFG["study"],process_id=__import__("os").getpid(),restart_allowed=False))
    phase("prepare")
    c.write(c.OUT/"authorization.json",dict(user_request="请你继续",scope="execute prior recommended dense factor diagnostic and paired single/multi-size control",
        no_old_run_restart=True,no_new_architecture=True,no_18h_run=True,hard_seconds=c.CFG["hard_seconds"]))
    parent=c.parent_verification()
    c.write(c.OUT/"parent_verification.json",parent)
    sources=c.source_snapshot()
    proto=dict(study=c.CFG["study"],config=c.CFG,sources=sources,parent=parent,created=time.time(),
        budget=c.read(c.OUT/"budget.json"),python=sys.version,platform=platform.platform(),
        torch=torch.__version__,numpy=np.__version__,device="cpu",precision="FP32",disk_free=c.disk())
    proto["protocol_hash"]=hashlib.sha256(json.dumps(proto,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    c.write(c.OUT/"run_protocol.json",proto)
    bank=c.load(c.PARENT/"bank.npz")
    c.save(c.OUT/"banks/base.npz",**bank)
    for side in (4,6,8):
        for natural in (False,True):
            if side==4 and natural:
                continue
            key="k4_%d_%s"%(side,"natural" if natural else "fixed")
            c.save(c.OUT/"banks"/(key+".npz"),**d.extend_k4(bank,side,natural))
    shutil.copyfile(c.PARENT/"exact_states.npz",c.OUT/"exact_states.npz")
    import test_multisize
    output=io.StringIO()
    tests=unittest.defaultTestLoader.loadTestsFromModule(test_multisize)
    results=unittest.TextTestRunner(stream=output,verbosity=2).run(tests)
    c.write(c.OUT/"software_tests.json",dict(passed=results.wasSuccessful(),tests=results.testsRun,log=output.getvalue()))
    assert results.wasSuccessful(),output.getvalue()
    return bank,proto


def preflight(bank,proto):
    phase("factor_diagnostic")
    ev.factor_diagnostics(proto["parent"])
    phase("recovery_and_timing")
    views={s:c.load(c.OUT/"banks"/("k4_%d_fixed.npz"%s)) for s in (4,6)}
    tensors=m.prepare_tensors(bank,views)
    a,b=m.fresh(91991),m.fresh(91991)
    assert m.model_hash(a[0])==m.model_hash(b[0])
    first=d.indices(bank,91991,1)
    ra=m.update(*a,tensors,first,1,4)
    rb=m.update(*b,tensors,first,1,4)
    assert ra["loss"]==rb["loss"] and ra["grad_norm"]==rb["grad_norm"]
    assert m.same_state(a[0].state_dict(),b[0].state_dict())
    assert m.same_state(a[1].state_dict(),b[1].state_dict())
    assert m.same_state(a[2].state_dict(),b[2].state_dict())
    fixture=c.OUT/"preflight/recovery_fixture.pt"
    m.checkpoint(fixture,a,91991,"preflight",1,proto["protocol_hash"])
    restored,payload=m.restore(fixture)
    ids=d.indices(bank,91991,2)
    ra=m.update(*a,tensors,ids,2,6)
    rr=m.update(*restored,tensors,ids,2,6)
    assert ra["loss"]==rr["loss"] and ra["grad_norm"]==rr["grad_norm"]
    assert m.same_state(a[0].state_dict(),restored[0].state_dict())
    assert m.same_state(a[1].state_dict(),restored[1].state_dict())
    assert m.same_state(a[2].state_dict(),restored[2].state_dict())
    c.write(c.OUT/"preflight/recovery.json",dict(passed=True,raw_EMA_AdamW_next_update_exact=True,
        paired_initialization_and_N16_update_exact=True,next_N36_update_exact=True,steps=[1,2],torch_rng_restored=True))
    del a,b,restored
    gc.collect()
    timing={}
    for side in (4,6):
        state=m.fresh(91990+side)
        step=0
        for _ in range(c.CFG["timing_warmup"]):
            step+=1
            m.update(*state,tensors,d.indices(bank,91990+side,step),step,side)
        repeats=[]
        for _ in range(c.CFG["timing_repeats"]):
            start=time.perf_counter()
            for _ in range(c.CFG["timing_steps_per_repeat"]):
                step+=1
                ids=d.indices(bank,91990+side,step)
                d.logical_and_native(bank,views,ids,side)
                m.update(*state,tensors,ids,step,side)
            repeats.append((time.perf_counter()-start)/c.CFG["timing_steps_per_repeat"])
        timing[str(side)]=dict(seconds_per_update=repeats,conservative=max(repeats))
        del state
        gc.collect()
    elapsed=time.time()-c.read(c.OUT/"budget.json")["started"]
    estimate=c.CFG["timing_safety_multiplier"]*(9216*timing["4"]["conservative"]+3072*timing["6"]["conservative"])
    total=elapsed+estimate+c.CFG["post_training_reserve_seconds"]
    gate=dict(passed=total<c.CFG["hard_seconds"],elapsed=elapsed,timing=timing,
        predicted_training_seconds=estimate,predicted_total_seconds=total,
        reserve_seconds=c.CFG["post_training_reserve_seconds"],fixed_update_counts={"N16":9216,"N36":3072},time=time.time())
    c.write(c.OUT/"preflight/budget_gate.json",gate)
    c.write(c.OUT/"input_freeze.json",{p.relative_to(c.OUT).as_posix():c.sha(p) for p in sorted(c.OUT.rglob("*.npz"))})
    c.check_sources(proto["sources"])
    if not gate["passed"]:
        phase("stopped_before_training_budget_gate",gate=gate)
        return False
    phase("preflight_passed",predicted_total_seconds=total,hard_deadline=c.read(c.OUT/"budget.json")["deadline"])
    return True


def train(bank,proto):
    views={s:c.load(c.OUT/"banks"/("k4_%d_fixed.npz"%s)) for s in (4,6)}
    tensors=m.prepare_tensors(bank,views)
    finals={}
    for seed in c.CFG["seed_labels"]:
        states={arm:m.fresh(seed) for arm in c.CFG["arms"]}
        assert m.model_hash(states["A4"][0])==m.model_hash(states["B46"][0])
        logs={}
        steps={arm:0 for arm in states}
        for arm,state in states.items():
            folder=c.OUT/"training"/("s%d_%s"%(seed,arm))
            c.write(folder/"initial.json",dict(seed=seed,arm=arm,initialization_seed=c.initial_seed(seed),
                model_hash=m.model_hash(state[0]),time=time.time(),from_scratch=True))
            logs[arm]=(folder/"train.jsonl").open("x",encoding="utf-8")
        try:
            for block in range(0,c.CFG["steps"],c.CFG["interleave_block"]):
                for arm,state in states.items():
                    cell="s%d_%s"%(seed,arm)
                    folder=c.OUT/"training"/cell
                    for step in range(block+1,block+c.CFG["interleave_block"]+1):
                        ids=d.indices(bank,seed,step)
                        side=d.view_side(arm,step)
                        _,_,physical,native=d.logical_and_native(bank,views,ids,side)
                        row=m.update(*state,tensors,ids,step,side)
                        row.update(seed=seed,arm=arm,time=time.time(),paired_data_digest=physical,
                            native_full_precision_digest=native,batch_row_digest=c.ahash(ids))
                        logs[arm].write(json.dumps(row,allow_nan=False)+"\n")
                        steps[arm]=step
                        if step%32==0:
                            logs[arm].flush()
                    logs[arm].flush()
                    m.checkpoint(folder/"last.pt",state,seed,arm,steps[arm],proto["protocol_hash"],replace=True)
                    if steps[arm]==c.CFG["steps"]:
                        m.checkpoint(folder/"final.pt",state,seed,arm,steps[arm],proto["protocol_hash"])
                        finals[cell]=c.sha(folder/"final.pt")
                        c.write(folder/"complete.json",dict(steps=steps[arm],time=time.time(),final_sha256=finals[cell],
                            raw_hash=m.model_hash(state[0]),ema_hash=m.model_hash(state[1]),ability_not_yet_evaluated=True))
                    phase("training",cell=cell,step=steps[arm],loss=row["loss"],finite=True,finals=len(finals))
                    c.disk()
                    c.check_sources(proto["sources"])
        except BaseException:
            for arm,state in states.items():
                if steps[arm]>0:
                    m.checkpoint(c.OUT/"training"/("s%d_%s"%(seed,arm))/"interrupted.pt",state,seed,arm,steps[arm],proto["protocol_hash"])
            raise
        finally:
            for handle in logs.values():
                handle.close()
        del states
        gc.collect()
    assert len(finals)==6
    c.write(c.OUT/"all_models_locked.json",dict(time=time.time(),finals=finals,steps_per_model=c.CFG["steps"],
        primary_predictions_before_lock=0,total_training_updates=12288))
    phase("training_complete",finals=6)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",action="store_true",required=True)
    parser.parse_args()
    m.configure()
    if c.OUT.exists():
        raise FileExistsError("Existing study cannot be restarted or budget-reset")
    try:
        bank,proto=prepare()
        if not preflight(bank,proto):
            return
        train(bank,proto)
        phase("final_evaluation")
        ev.final_evaluation()
        phase("auditing")
        import ms_audit
        ms_audit.audit()
        import ms_delivery
        ms_delivery.report()
        phase("science_and_audit_complete_pending_backup",time_remaining=c.deadline()-time.time())
        ms_delivery.package()
        print(json.dumps({"phase":"closed","backup_verified":True,"time":time.time()},ensure_ascii=False),flush=True)
    except BaseException as error:
        c.write(c.OUT/"failure.json",dict(time=time.time(),type=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
        phase("failed_no_automatic_retry",error=str(error))
        raise


if __name__=="__main__":
    main()
