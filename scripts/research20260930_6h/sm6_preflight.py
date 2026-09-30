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

import sm6_common as c
import sm6_data as d
import sm6_model as m
import sm6_evaluation as ev
import sm6_analysis as analysis
import sm6_management as management


def model_checks(root, device):
    root.mkdir(parents=True,exist_ok=True)
    data=d.TrainingData(); state=m.fresh(c.CFG['scratch_seed'],device)
    parts,_=data.batch(93061,1,'N')
    # Small software-only batch, no held-out pattern or main test output.
    parts=[d.subset(p,np.arange(4)) for p in parts]
    m.update_parts(state,parts,1)
    path=root/'restore_fixture.pt'
    m.checkpoint(path,state,c.CFG['scratch_seed'],'scratch',1,'preflight')
    m.update_parts(state,parts,2)
    restored,payload=m.restore(path,device)
    m.update_parts(restored,parts,2)
    assert m.equal_state(state[0].state_dict(),restored[0].state_dict())
    assert m.equal_state(state[1].state_dict(),restored[1].state_dict())
    assert m.equal_state(state[2].state_dict(),restored[2].state_dict())
    b=d.subset(data.g8,np.flatnonzero(data.g8['split']=='train')[:8])
    base=d.view(b,4,'N'); pad=d.view(b,4,'N',allocated=144)
    left=m.fresh(2026093064,device); right=m.fresh(2026093064,device)
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
    canonical=d.view(b,4,'W'); requested=canonical.copy();requested['requested_t']=np.full(8,.99,np.float32)
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
        magnitude={'N':.50,'W':.08}[j['arm']]
        offset=magnitude*(1+(j['seed']-93063.5)*.015)*(8000/j['step'])
        logits=np.stack([np.log1p(-y),np.log(y)+offset],1)
        rows.append(ev.emit_prediction(root,j,bank,logits,'SYNTHETIC_FIXTURE_NOT_A_CHECKPOINT'))
    controls=[]
    for seed in c.CFG['seed_labels']:
        for arm in 'NW':
            for meta,base,changed in ev.control_banks(seed,arm):
                y=base['target'];z=np.stack([np.log1p(-y),np.log(y)+.1],1)
                controls.append(ev.emit_control(root,meta,base,changed,z,z,'SYNTHETIC_FIXTURE_NOT_A_CHECKPOINT'))
    management.audit_controls(root,controls,fixture=True)
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
    root=c.OUT/'preflight';root.mkdir(parents=True,exist_ok=False)
    assert c.AUTH.is_file() and c.CFG['study'] in c.AUTH.read_text(encoding='utf-8')
    assert c.read(c.OUT/'cpu_preparation/complete.json')['status']=='passed'
    assert c.read(c.OUT/'preflight_visual_review.json')['actually_viewed_pngs']==5
    before=c.resources();assert 'RTX 4090' in before['gpu']
    assert before['disk_free']>=6*1024**3
    c.write(root/'resources_before.json',before)
    started=time.time()
    c.write(c.OUT/'budget.json',dict(started=started,deadline=started+21600,hard_seconds=21600,waiting_counts=True))
    c.status('gpu_preflight')
    device='cuda';assert torch.cuda.is_available()
    c.write(root/'environment.json',environment())
    frozen=c.source_snapshot();c.write(root/'preflight_sources.json',frozen)
    correctness=model_checks(root/'model',device)
    data=d.TrainingData();pools=np.flatnonzero(data.g8['split']=='train')
    chosen=pools[np.floor(np.arange(64)*len(pools)/64).astype(int)]
    tiny=d.subset(data.g8,chosen);state=m.fresh(c.CFG['scratch_seed'],device);bank=d.view(tiny,4,'W')
    with (root/'capacity_canonical.jsonl').open('x',encoding='utf-8') as log:
        for step in range(1,2049):
            c.deadline();row=m.update_parts(state,[bank],step,total=2048,warmup=128)
            log.write(json.dumps(row,allow_nan=False)+'\n')
    logits=m.predict(state[0],bank);capacity=c.summary(logits,bank['target'])
    capacity.update(arm='canonical_scratch',steps=2048,seed=c.CFG['scratch_seed'],train_only=True)
    c.save(root/'capacity_prediction.npz',logits=logits,target=bank['target'],row_id=bank['row_id'])
    c.write(root/'capacity.json',capacity)
    if capacity['max_kl']>.02 or capacity['max_probability_error']>.10:
        raise RuntimeError('Single-attempt canonical capacity gate failed; no retry')
    c.write(root/'capacity_complete.json',dict(status='passed',capacity=capacity))
    del state
    timings=[]
    for arm in 'NW':
        for phase in range(4):
            state=m.fresh(c.CFG['scratch_seed'],device)
            for i in range(8):m.update(state,data,c.CFG['scratch_seed'],phase+1+4*i,arm)
            repeats=[]
            for rep in range(3):
                torch.cuda.synchronize();start=time.perf_counter()
                with (root/f'timing_{arm}_{phase}_{rep}.jsonl').open('x',encoding='utf-8') as log:
                    for i in range(32):m.update(state,data,c.CFG['scratch_seed'],phase+1+4*(8+32*rep+i),arm,log)
                torch.cuda.synchronize();repeats.append((time.perf_counter()-start)/32)
            timings.append(dict(arm=arm,phase=phase,seconds_per_update=repeats,conservative=max(repeats)))
            del state
    c.write(root/'update_timing.json',timings)
    state=m.fresh(c.CFG['scratch_seed'],device);torch.cuda.synchronize();start=time.perf_counter()
    with (root/'block_timing.jsonl').open('x',encoding='utf-8') as log:
        for step in range(1,1001):m.update(state,data,c.CFG['scratch_seed'],step,'W',log)
    torch.cuda.synchronize();raw_block=time.perf_counter()-start
    start=time.perf_counter();m.checkpoint(root/'timing_checkpoint.pt',state,c.CFG['scratch_seed'],'scratch',1000,'timing')
    checkpoint_seconds=time.perf_counter()-start
    m.checkpoint(root/'timing_snapshot.pt',state,c.CFG['scratch_seed'],'scratch',1000,'timing',full=False)
    state_bytes=(root/'timing_checkpoint.pt').stat().st_size
    snapshot_bytes=(root/'timing_snapshot.pt').stat().st_size
    sample=d.subset(data.g8,pools[:32]);eval_times={}
    for side in c.CFG['evaluation']['center_sides']:
        b=d.view(sample,side,'W');m.predict(state[0],b)
        reps=[]
        for _ in range(3):
            torch.cuda.synchronize();start=time.perf_counter();m.predict(state[0],b);torch.cuda.synchronize()
            reps.append((time.perf_counter()-start)/32)
        eval_times[str(side)]=max(reps)
    # Explicitly benchmark every PAD shape, including actual invalid keys.
    pad_times={}
    for side,allocated in c.CFG['evaluation']['control_PAD_side_allocated_pairs']:
        b=d.view(sample,side,'W',allocated=allocated);m.predict(state[0],b);reps=[]
        for _ in range(3):
            torch.cuda.synchronize();start=time.perf_counter();m.predict(state[0],b);torch.cuda.synchronize()
            reps.append((time.perf_counter()-start)/32)
        pad_times[str(side)]=max(reps)
    c.write(root/'evaluation_timing.json',dict(per_row_side=eval_times,padded_side=pad_times,train_only=True))
    plan=list(ev.jobs());eval_cost=sum(len(ev.input_bank(j)['query'])*eval_times[str(j['spec']['side'])] for j in plan)
    input_rows=sum(len(ev.input_bank(j)['query']) for j in plan)
    control_cost=0.;control_rows=0
    for seed in c.CFG['seed_labels']:
        for arm in 'NW':
            for meta,base,changed in ev.control_banks(seed,arm):
                n=len(base['query']);side=str(meta['side']);control_rows+=n
                control_cost+=n*(eval_times[side]+(pad_times[side] if meta['kind']=='invalid_PAD' else eval_times[side]))
    assert len(plan)==600 and input_rows==129600 and control_rows==13920
    eval_projection=max(900.,1.35*(eval_cost+control_cost)+300.)
    checkpoint_allowance=1.35*132*checkpoint_seconds
    by_types=1.35*sum(v['conservative']*12000 for v in timings)+checkpoint_allowance
    by_block=1.35*96*raw_block+checkpoint_allowance
    train_projection=max(by_types,by_block)
    fixture_bytes=sum(p.stat().st_size for p in (c.OUT/'cpu_preparation/fixture').rglob('*') if p.is_file())
    preflight_bytes=sum(p.stat().st_size for folder in (root,c.OUT/'cpu_preparation') for p in folder.rglob('*') if p.is_file())
    max_line=max(len(line) for p in root.glob('timing_*.jsonl') for line in p.read_bytes().splitlines())+128
    estimated_logs=96000*max_line
    science_bytes=12*state_bytes+24*snapshot_bytes+estimated_logs+2*fixture_bytes+64*1024**2
    storage_estimate=int(1.35*(science_bytes*2.02+12*state_bytes+2*preflight_bytes))
    free=c.resources()['disk_free'];requirement=storage_estimate+2*1024**3
    elapsed=time.time()-started;total=elapsed+train_projection+eval_projection+1200+1800+1800
    result=dict(time=time.time(),elapsed_seconds=elapsed,train_seconds=train_projection,
        train_by_types=by_types,train_by_block=by_block,evaluation_seconds=eval_projection,
        analysis_audit_seconds=1200,backup_seconds=1800,reserve_seconds=1800,
        conservative_total_seconds=total,deadline=started+21600,training_jobs=12,updates=96000,
        evaluation_jobs=600,evaluation_rows=input_rows,control_pairs=120,control_rows=control_rows,
        checkpoint_bytes=state_bytes,snapshot_bytes=snapshot_bytes,fixture_bytes=fixture_bytes,
        estimated_log_bytes=estimated_logs,estimated_science_bytes=int(science_bytes),
        checkpoint_seconds=checkpoint_seconds,measured_block_seconds=raw_block,
        storage_estimate_bytes=storage_estimate,storage_required_with_margin=requirement,
        disk_free=free,storage_gate_actual_bytes=free>=requirement,capacity=capacity,
        max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),environment=c.read(root/'environment.json'),
        transfer_verification_gate='mandatory_separate_measurement_before_lock',
        passed=total<=21600 and free>=requirement)
    c.check_sources(frozen);c.write(root/'launch_gate.json',result)
    if not result['passed']:raise RuntimeError('Predeclared time/storage gate failed; no launch or automatic scope reduction')
    c.write(root/'complete.json',dict(status='passed',time=time.time(),gate_sha=c.sha(root/'launch_gate.json')))
    c.status('gpu_preflight_passed_pending_transfer_backup_lock',gate_sha=c.sha(root/'launch_gate.json'))
    return result
