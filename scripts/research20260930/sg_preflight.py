"""Single-attempt correctness/capacity/timing gates, never a formal training run."""
from __future__ import annotations

import copy
import io
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import sg_common as c
import sg_data as d
import sg_model as m
import sg_evaluation as ev
import sg_analysis as analysis
import sg_management as management


def model_checks(root, device):
    root.mkdir(parents=True,exist_ok=True)
    data=d.TrainingData(); state=m.fresh(2026093004,device)
    parts,_=data.batch(93001,1,'A')
    # Small software-only batch, no held-out pattern or main test output.
    parts=[d.subset(p,np.arange(4)) for p in parts]
    m.update_parts(state,parts,1)
    path=root/'restore_fixture.pt'
    m.checkpoint(path,state,2026093004,'scratch',1,'preflight')
    m.update_parts(state,parts,2)
    restored,payload=m.restore(path,device)
    m.update_parts(restored,parts,2)
    assert m.equal_state(state[0].state_dict(),restored[0].state_dict())
    assert m.equal_state(state[1].state_dict(),restored[1].state_dict())
    assert m.equal_state(state[2].state_dict(),restored[2].state_dict())
    b=d.subset(data.g8,np.flatnonzero(data.g8['split']=='train')[:8])
    base=d.view(b,4,'A'); pad=d.view(b,4,'A',allocated=144)
    left=m.fresh(2026093004,device); right=m.fresh(2026093004,device)
    _,gl=m.update_parts(left,[base],1,keep_gradient=True)
    _,gr=m.update_parts(right,[pad],1,keep_gradient=True)
    grad_diff=max(float((gl[k]-gr[k]).abs().max()) for k in gl)
    param_diff=max(float((l-r).abs().max()) for l,r in zip(left[0].parameters(),right[0].parameters()))
    assert grad_diff<=2e-5 and param_diff<=2e-5,(grad_diff,param_diff)
    assert float(gr['token_embedding.weight'][3].abs().max())==0.
    model=state[0]; reference=c.metrics(m.predict(model,base),base['target'])['probability']
    alterations={'PAD':pad}
    shifted={k:v.copy() for k,v in base.items()};shifted['coordinates']+=np.array([17,-11],np.float32);alterations['translation']=shifted
    perm=np.arange(16)[::-1].copy();inv=np.argsort(perm)
    reordered={k:v.copy() for k,v in base.items()}
    for field in ('tokens','coordinates','valid'):reordered[field]=base[field][:,:,perm].copy()
    reordered['query']=inv[base['query']];alterations['permutation']=reordered
    delta={name:float(np.max(np.abs(c.metrics(m.predict(model,bk),bk['target'])['probability']-reference))) for name,bk in alterations.items()}
    assert max(delta.values())<=2e-5,delta
    canonical=d.view(b,4,'D'); requested=canonical.copy();requested['requested_t']=np.full(8,.99,np.float32)
    assert np.array_equal(m.predict(model,canonical),m.predict(model,requested))
    result=dict(status='passed',device=device,parameters=sum(p.numel() for p in model.parameters()),
                next_update_raw_ema_adam_bitwise=True,pad_gradient_error=grad_diff,pad_parameter_error=param_diff,
                invalid_pad_gradient=0,negative_control_probability_changes=delta,
                canonical_external_clock_invariance='structural_not_accuracy')
    c.write(root/'model_checks.json',result)
    return result


def fixture(root):
    root.mkdir(parents=True,exist_ok=False)
    rows=[]
    for j in ev.jobs():
        bank=ev.input_bank(j);y=bank['target']
        # Deliberately synthetic and clearly labelled: this validates plumbing only.
        magnitude={'A':.50,'B':.32,'C':.20,'D':.08}[j['arm']]
        offset=magnitude*(1+(j['seed']-93003.5)*.015)*(24000/j['step'])
        logits=np.stack([np.log1p(-y),np.log(y)+offset],1)
        rows.append(ev.emit_prediction(root,j,bank,logits,'SYNTHETIC_FIXTURE_NOT_A_CHECKPOINT'))
    controls=[dict(seed=seed,arm=arm,kind='synthetic_control_fixture',passed=True,max_probability_change=0.)
              for seed in c.CFG['seed_labels'] for arm in 'ABCD']
    verification=management.audit_evaluation(root,rows,fixture=True)
    summary=analysis.analyze(root,rows,controls,{'fixture_only':1.},fixture=True)
    c.write(root/'fixture_audit.json',dict(**verification,not_a_neural_result=True,manifest_jobs=len(rows)))
    return dict(status='passed',jobs=len(rows),headline=summary['gates'],not_a_neural_result=True)


def prepare_cpu():
    root=c.OUT/'cpu_preparation'
    root.mkdir(parents=True,exist_ok=False)
    started=time.time()
    truth=d.exact_truth_checks();c.write(root/'truth_checks.json',truth)
    model=model_checks(root/'model','cpu')
    fx=fixture(root/'fixture')
    # Backup writer test uses actual file bytes and independently verifies each member.
    paths=[root/'truth_checks.json',root/'model/model_checks.json',root/'fixture/final_summary.json']
    archive=management.archive(c.ROOT,root/'archive_test','member_test',paths)
    receipt=management.verify(root/'archive_test/member_test.tar.gz',root/'archive_test/member_test.manifest.json',root/'archive_test/member_test.verification.json')
    result=dict(status='passed',time=time.time(),seconds=time.time()-started,truth=truth,model=model,fixture=fx,archive=receipt,
                GPU_budget_started=False,formal_models_started=False)
    c.write(root/'complete.json',result)
    return result


def environment():
    import scipy
    import matplotlib
    capture=io.StringIO()
    old=sys.stdout
    try:
        sys.stdout=capture;np.show_config()
    finally:sys.stdout=old
    return dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,
                scipy=scipy.__version__,matplotlib=matplotlib.__version__,cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),torch_config=torch.__config__.show(),numpy_config=capture.getvalue(),
                threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled(),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32,cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                rng='PCG64/SeedSequence; addressed per root,seed,step,role',device=torch.cuda.get_device_name())


def gpu():
    root=c.OUT/'preflight'
    repair=root.exists()
    if repair:
        notice=c.read(root/'software_repair_authorization_v1.json')
        assert notice['reason']=='restore_optimizer_step_device_before_capacity'
        assert not list(root.glob('capacity_*')) and not (c.OUT/'run.lock').exists()
        assert not (root/'model_repair_v1').exists()
        c.deadline()
    else:
        root.mkdir(parents=True,exist_ok=False)
    assert c.AUTH.is_file() and c.CFG['study'] in c.AUTH.read_text(encoding='utf-8')
    assert c.read(c.OUT/'cpu_preparation/complete.json')['status']=='passed'
    before=c.resources();assert 'RTX 4090' in before['gpu']
    c.write(root/('resources_before_repair_v1.json' if repair else 'resources_before.json'),before)
    # This is the sole original clock. It precedes all GPU correctness/capacity/timing work.
    if repair:
        started=c.read(c.OUT/'budget.json')['started']
    else:
        started=time.time();c.write(c.OUT/'budget.json',dict(started=started,deadline=started+43200,hard_seconds=43200,waiting_counts=True))
    c.status('gpu_preflight')
    device='cuda';assert torch.cuda.is_available()
    c.write(root/('environment_repair_v1.json' if repair else 'environment.json'),environment())
    correctness=model_checks(root/('model_repair_v1' if repair else 'model'),device)
    data=d.TrainingData()
    pools=np.flatnonzero(data.g8['split']=='train')
    chosen=pools[np.floor(np.arange(64)*len(pools)/64).astype(int)]
    tiny=d.subset(data.g8,chosen)
    capacities=[]
    for arm in ('A','B'):
        state=m.fresh(c.CFG['scratch_seed'],device)
        bank=d.view(tiny,4,arm)
        logpath=root/f'capacity_{arm}.jsonl'
        with logpath.open('x',encoding='utf-8') as log:
            for step in range(1,2049):
                c.deadline()
                row=m.update_parts(state,[bank],step,total=2048,warmup=128)
                log.write(json.dumps(row,allow_nan=False)+'\n')
        logits=m.predict(state[0],bank)
        val=c.summary(logits,bank['target']);val.update(arm=arm,steps=2048,seed=c.CFG['scratch_seed'])
        c.save(root/f'capacity_{arm}_prediction.npz',logits=logits,target=bank['target'],row_id=bank['row_id'])
        c.write(root/f'capacity_{arm}.json',val);capacities.append(val)
        if val['max_kl']>.02 or val['max_probability_error']>.10:
            raise RuntimeError('Single-attempt train-only capacity gate failed: '+arm)
        del state
    c.write(root/'capacity_complete.json',dict(status='passed',rows=capacities))
    timings=[]
    for arm in 'ABCD':
        for phase in range(4):
            state=m.fresh(c.CFG['scratch_seed'],device)
            # Use training-only inputs. Step addresses set the desired size phase.
            for i in range(8):m.update(state,data,c.CFG['scratch_seed'],phase+1+4*i,arm)
            repeats=[]
            for rep in range(3):
                start=time.perf_counter()
                with (root/f'timing_{arm}_{phase}_{rep}.jsonl').open('x',encoding='utf-8') as log:
                    for i in range(32):m.update(state,data,c.CFG['scratch_seed'],phase+1+4*(8+32*rep+i),arm,log)
                torch.cuda.synchronize();repeats.append((time.perf_counter()-start)/32)
            timings.append(dict(arm=arm,phase=phase,seconds_per_update=repeats,conservative=max(repeats)))
            del state
    c.write(root/'update_timing.json',timings)
    # Measure the complete1000-update block and real checkpoint overhead once.
    state=m.fresh(c.CFG['scratch_seed'],device);start=time.perf_counter()
    with (root/'block_timing.jsonl').open('x',encoding='utf-8') as log:
        for step in range(1,1001):m.update(state,data,c.CFG['scratch_seed'],step,'D',log)
    raw_block=time.perf_counter()-start
    start=time.perf_counter();m.checkpoint(root/'timing_checkpoint.pt',state,c.CFG['scratch_seed'],'scratch',1000,'timing')
    checkpoint_seconds=time.perf_counter()-start
    m.checkpoint(root/'timing_snapshot.pt',state,c.CFG['scratch_seed'],'scratch',1000,'timing',full=False)
    snapshot_bytes=(root/'timing_snapshot.pt').stat().st_size
    # Train-only rows at evaluation shapes, never formal held-out model performance.
    sample=d.subset(data.g8,pools[:32]); eval_times={}
    for side in c.CFG['evaluation']['center_sides']:
        b=d.view(sample,side,'A');m.predict(state[0],b)
        reps=[]
        for _ in range(3):
            start=time.perf_counter();m.predict(state[0],b);torch.cuda.synchronize();reps.append((time.perf_counter()-start)/32)
        eval_times[str(side)]=max(reps)
    jobs=list(ev.jobs()); eval_projection=0.;input_rows=0
    for job in jobs:
        b=ev.input_bank(job);input_rows+=len(b['query']);eval_projection+=len(b['query'])*eval_times[str(job['spec']['side'])]
    # All planned control forwards, with padded shapes conservatively costed at side24.
    control_rows=24*2*(64+168)*(4+2)
    eval_projection=1.35*(eval_projection+control_rows*eval_times['24'])+300
    train_projection=1.35*sum(t['conservative']*36000 for t in timings)+576*checkpoint_seconds
    # Source-code interpretation: each arm-phase has6 seeds*24000/4=36000 updates.
    state_bytes=(root/'timing_checkpoint.pt').stat().st_size
    # Recompute the predeclared storage gate from real serializer/fixture/log byte counts.
    fixture_bytes=sum(p.stat().st_size for p in (c.OUT/'cpu_preparation/fixture').rglob('*') if p.is_file())
    preflight_bytes=sum(p.stat().st_size for folder in (root,c.OUT/'cpu_preparation') for p in folder.rglob('*') if p.is_file())
    max_log_line=max(len(line) for p in root.glob('timing_*.jsonl') for line in p.read_bytes().splitlines())+128
    estimated_logs=576000*max_log_line
    science_bytes=24*state_bytes+48*snapshot_bytes+estimated_logs+2*fixture_bytes+64*1024**2
    storage_estimate=int(1.35*(science_bytes*2.02+24*state_bytes+2*state_bytes+2*preflight_bytes))
    free=c.resources()['disk_free']
    actual_requirement=storage_estimate+2*1024**3
    elapsed=time.time()-started
    total=elapsed+train_projection+eval_projection+1800+2700+2700
    result=dict(time=time.time(),elapsed_seconds=elapsed,train_seconds=train_projection,
                evaluation_seconds=eval_projection,analysis_audit_seconds=1800,backup_seconds=2700,reserve_seconds=2700,
                conservative_total_seconds=total,deadline=started+43200,training_jobs=24,updates=576000,
                evaluation_jobs=len(jobs),evaluation_rows=input_rows,checkpoint_bytes=state_bytes,
                snapshot_bytes=snapshot_bytes,fixture_bytes=fixture_bytes,estimated_log_bytes=estimated_logs,
                checkpoint_seconds=checkpoint_seconds,measured_block_seconds=raw_block,
                storage_estimate_bytes=storage_estimate,storage_required_with_margin=actual_requirement,
                disk_free=free,initial_20GiB_planning_target_met=free>=20*1024**3,
                storage_gate_actual_bytes=free>=actual_requirement,capacity=capacities,
                max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),environment=c.read(root/'environment.json'),
                passed=total<=43200 and free>=actual_requirement)
    c.write(root/'launch_gate.json',result)
    if not result['passed']:
        raise RuntimeError('Predeclared resource/time budget gate failed; no formal launch')
    c.write(root/'complete.json',dict(status='passed',time=time.time(),gate_sha=c.sha(root/'launch_gate.json')))
    c.status('gpu_preflight_passed_pending_backup_and_lock',gate_sha=c.sha(root/'launch_gate.json'))
    return result
