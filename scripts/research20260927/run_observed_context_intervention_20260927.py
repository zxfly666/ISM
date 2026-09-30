"""Guarded, single-launch J18h execution. No launch on import.

CPU implementation checks precede the clock. The first GPU preflight creates
one immutable absolute budget. Freeze and launch cannot reset or bypass it.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/research20260921')]
import intervention_common as c


def sources():
    return {p.relative_to(c.ROOT).as_posix():c.sha(p) for p in c.source_files()}


def environment():
    env=os.environ.copy()
    env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',MPLBACKEND='Agg',
        PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/'scripts/research20260921'),str(Path(__file__).parent)]))
    return env


def resources():
    memory=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    current=int(Path('/sys/fs/cgroup/memory.current').read_text())
    result=dict(time=time.time(),disk_free=shutil.disk_usage(c.ROOT).free,
        memory_limit=memory,memory_current=current,cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True))
    if result['disk_free'] < 12*1024**3: raise RuntimeError('Insufficient remote free disk for safe J outputs')
    if memory!='max' and int(memory)-current < 16*1024**3: raise RuntimeError('Insufficient cgroup RAM headroom')
    return result


def stop(signum, frame):
    raise TimeoutError('J18h watchdog/SIGTERM stop; no new budget')


def start_process(mode, record):
    budget=c.read(c.OUT/'budget.json')
    remaining=math.floor(budget['deadline']-time.time())
    if remaining <= 0: raise TimeoutError('Original J18h deadline elapsed')
    log=c.OUT/(mode+'.log')
    if log.exists() or (c.OUT/record).exists(): raise FileExistsError('Existing process/launch evidence')
    with log.open('xb') as out:
        p=subprocess.Popen(['timeout','--signal=TERM','--kill-after=15',str(remaining),sys.executable,
            str(Path(__file__).resolve()),'--mode',mode],cwd=ROOT,env=environment(),
            stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    result=dict(pid=p.pid,pgid=p.pid,mode=mode,time=time.time(),deadline=budget['deadline'],
        command=['timeout','--signal=TERM','--kill-after=15',str(remaining),sys.executable,str(Path(__file__).resolve()),'--mode',mode])
    c.write(c.OUT/record,result)
    print(json.dumps(result),flush=True)
    return p


def cpu_prepare():
    c.verified_authorization()
    if (c.OUT/'budget.json').exists() or (c.OUT/'run.lock').exists():
        raise RuntimeError('Already charged/started; CPU preparation cannot replace prior state')
    if c.OUT.exists(): raise FileExistsError('J output already exists; inspect, do not overwrite')
    res=resources()
    if c.sha(c.DATA)!=c.CONFIG['training_common']['parent_file_sha256']:
        raise RuntimeError('Original parent library SHA mismatch')
    for i,digest in enumerate(c.CONFIG['base_final_sha256']):
        if c.sha(c.base_checkpoint(i))!=digest: raise RuntimeError('Base SHA mismatch')
    old=ROOT/'artifacts/geometry_identification_20260926/run_protocol.json'
    if c.sha(old)!=c.CONFIG['base_run_protocol_sha256']:raise RuntimeError('Old protocol changed')
    # New seed namespace must have no formal source in any other run protocol.
    used=[]
    for p in (ROOT/'artifacts').glob('*/run_protocol.json'):
        text=p.read_text(encoding='utf-8')
        if any(str(seed) in text for seed in [*c.COHORTS['fresh']['initialization_seeds'],c.CONFIG['seeds']['mc']]):
            used.append(str(p))
    if used: raise RuntimeError(('Seed namespace already used',used))
    c.OUT.mkdir(); out=c.OUT/'cpu_preparation';out.mkdir()
    before=sources()
    for name in ('test_intervention_cpu.py','test_intervention_model.py','test_intervention_statistics.py'):
        start=time.time()
        result=subprocess.run([sys.executable,str(Path(__file__).with_name(name))],cwd=ROOT,env=environment(),
            text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300)
        c.write(out/(name+'.json'),dict(exit_code=result.returncode,stdout=result.stdout,seconds=time.time()-start))
        if result.returncode:raise RuntimeError('CPU suite failed: '+name)
    scratch=ROOT/'artifacts/observed_context_intervention_18h_exports_20260927/software_scratch_cpu_v1/complete.json'
    if not scratch.exists() or c.read(scratch)['status']!='passed_software_only':
        raise RuntimeError('Full CPU software scratch receipt missing')
    c.write(out/'software_scratch_receipt.json',dict(path=str(scratch),sha256=c.sha(scratch),receipt=c.read(scratch)))
    if sources()!=before:raise RuntimeError('Source modified during CPU checks')
    c.write(out/'complete.json',dict(status='passed_CPU_only',resources=res,files=before,
        no_GPU_or_new_MC=True,time=time.time()))
    print('CPU_PREPARATION_PASSED_NO_BUDGET',flush=True)


def begin_preflight():
    c.verified_authorization()
    ready=c.read(c.OUT/'cpu_preparation/complete.json')
    if ready['status']!='passed_CPU_only' or ready['files']!=sources():
        raise RuntimeError('CPU checks stale: verify source before spending GPU budget')
    if (c.OUT/'budget.json').exists() or (c.OUT/'preflight_process.json').exists():
        raise RuntimeError('GPU clock already exists; no automatic retry/reset')
    c.write(c.OUT/'resource_before_gpu.json',resources())
    now=time.time()
    c.write(c.OUT/'budget.json',dict(study=c.CONFIG['study'],started=now,deadline=now+64800,
        hard_seconds=64800,launch_gate_seconds=63000,includes_preflight_waiting_and_backup=True))
    start_process('preflight','preflight_process.json')


def preflight():
    from intervention_preflight import run
    c.verify_sources(c.read(c.OUT/'cpu_preparation/complete.json'))
    check=c.Deadline(c.read(c.OUT/'budget.json')['deadline'])
    c.write(c.OUT/'preflight_worker.json',dict(pid=os.getpid(),pgid=os.getpgrp(),time=time.time()))
    result=run(c.OUT/'preflight_v1',check)
    c.verify_sources(c.read(c.OUT/'cpu_preparation/complete.json'))
    c.write(c.OUT/'preflight_computed.json',dict(time=time.time(),result=result,
        status='computed_passed_awaiting_visual_backup_budget'))
    print('PREFLIGHT_COMPUTED_NEEDS_VISUAL_TRANSFER_BUDGET',flush=True)


def transfer_probe():
    import tarfile
    import numpy as np
    if (c.OUT/'transfer_probe.manifest.json').exists():raise FileExistsError('Probe already exists')
    files=[]; r=np.random.default_rng(793215)
    # New disposable I/O data, not a redownload of any historical checkpoint.
    for i in range(8):
        p=c.OUT/'transfer_probe'/f'random_{i}.npz'
        c.save(p,software_transfer_probe=r.random((512,1024)))
        files.append(p)
    archive=c.EXPORTS/'transfer_probe_v1.tar.gz';c.EXPORTS.mkdir(exist_ok=True)
    began=time.perf_counter();rows=[]
    with archive.open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz',compresslevel=2) as tar:
            for p in files:
                rel=p.relative_to(ROOT).as_posix();tar.add(p,arcname=rel,recursive=False)
                rows.append(dict(path=rel,bytes=p.stat().st_size,sha256=c.sha(p)))
    c.write(c.OUT/'transfer_probe.manifest.json',dict(stage='transfer_probe',archive=archive.name,
        bytes=archive.stat().st_size,sha256=c.sha(archive),files=rows,members=len(rows),
        remote_export_seconds=time.perf_counter()-began))
    print(json.dumps(dict(archive=str(archive),manifest=str(c.OUT/'transfer_probe.manifest.json'))))


def freeze():
    c.verified_authorization()
    if (c.OUT/'run_protocol.json').exists() or (c.OUT/'run.lock').exists():raise FileExistsError('Already frozen/launched')
    budget=c.read(c.OUT/'budget.json');c.Deadline(budget['deadline'])()
    ready=c.read(c.OUT/'cpu_preparation/complete.json');c.verify_sources(ready)
    computed=c.read(c.OUT/'preflight_v1/computed_checks.json')
    visual=c.read(c.OUT/'preflight_visual_review.json')
    receipt=c.read(c.OUT/'local_transfer_receipt.json')
    if computed['status']!='passed_computation_needs_visual_transfer_budget_gate' or visual['status']!='passed' or receipt['status']!='passed':
        raise RuntimeError('Incomplete preflight computation/visual/transfer gate')
    if visual['figures']!=10 or receipt['local_C_free']<2*1024**3 or receipt['local_D_free']<12*1024**3:
        raise RuntimeError('Missing visual review or local disk capacity')
    if receipt['probe_archive_sha256']!=c.read(c.OUT/'transfer_probe.manifest.json')['sha256']:
        raise RuntimeError('Transfer receipt does not match exported probe')
    # Conservative full new scientific backup size upper bound 8 GiB; no timing
    # credit from reusing already-local old base weights. Measured full roundtrip
    # includes server export, SFTP and local member/whole-archive SHA.
    B=max(2700.,receipt['end_to_end_seconds']*(8*1024**3)/receipt['probe_bytes'])
    parts={k:float(computed[k]) for k in ('T','M','F','L','A')}
    E=time.time()-budget['started']; total=c.budget_estimate(E=E,B=B,**parts)
    res=resources()
    decision=dict(time=time.time(),elapsed_seconds=E,projected_backup_bytes=8*1024**3,B=B,**parts,
        total_seconds=total,deadline=budget['deadline'],passed=total<=63000,resources=res)
    c.write(c.OUT/'launch_budget_gate.json',decision)
    if total>63000:raise RuntimeError('Conservative total exceeds17.5h gate; no scope reduction or launch')
    protocol=dict(study=c.CONFIG['study'],config=c.CONFIG,files=ready['files'],authorization_sha256=c.sha(c.AUTH_PATH),
        budget=budget,launch_gate=decision,evaluation_identities=c.evaluation_identities(),
        base_final_sha256=c.CONFIG['base_final_sha256'],base_backup_sources=receipt['base_backup_sources'],
        software_checks='CPU suites + synthetic plumbing + actual GPU plumbing + exact recovery + fixed capability',
        source_not_an_independent_review=True)
    protocol['protocol_hash']=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    c.write(c.OUT/'run_protocol.json',protocol)
    from intervention_pipeline import copy_blueprint
    copy_blueprint(c.OUT)
    print('FROZEN_REQUIRES_VERIFIED_INITIAL_BACKUP_BEFORE_LAUNCH',flush=True)


def launch():
    protocol=c.read(c.OUT/'run_protocol.json');c.verify_sources(protocol)
    budget=c.read(c.OUT/'budget.json');check=c.Deadline(budget['deadline']);check()
    if (c.OUT/'failure.json').exists() or list(c.OUT.glob('preflight_failure*')):
        raise RuntimeError('Unresolved preflight failure; no automatic launch')
    initial=c.read(c.OUT/'initial_backup_receipt.json')
    if initial['status']!='passed' or not initial['verified_members']:
        raise RuntimeError('Initial local package not verified')
    gate=protocol['launch_gate']
    projected=c.budget_estimate(E=time.time()-budget['started'],**{k:gate[k] for k in ('T','M','F','L','A','B')})
    if projected>63000:raise RuntimeError('Elapsed waiting/backup now exceeds launch gate; no reset')
    c.write(c.OUT/'run.lock',dict(time=time.time(),protocol_sha256=c.sha(c.OUT/'run_protocol.json'),unique_launch=True))
    start_process('run','processes.json')


def reference():
    import intervention_pipeline as pipe
    c.verify_sources(c.read(c.OUT/'run_protocol.json'))
    c.write(c.OUT/'reference_worker.json',dict(pid=os.getpid(),pgid=os.getpgrp(),time=time.time()))
    pipe.reference(c.OUT,c.Deadline(c.read(c.OUT/'budget.json')['deadline']))


def run():
    from intervention_preflight import configure
    import intervention_pipeline as pipe
    import intervention_analysis as analysis
    configure();protocol=c.read(c.OUT/'run_protocol.json');c.verify_sources(protocol)
    budget=c.read(c.OUT/'budget.json');check=c.Deadline(budget['deadline']);check()
    if not (c.OUT/'run.lock').exists():raise RuntimeError('Use guarded launch')
    c.write(c.OUT/'run_started.json',dict(time=time.time(),pid=os.getpid(),pgid=os.getpgrp(),deadline=budget['deadline']))
    mc=start_process('reference','reference_process.json')
    try:
        training=pipe.train(c.OUT,protocol,check)
        while mc.poll() is None:
            check()
            if (c.OUT/'reference_failure.json').exists():raise RuntimeError('Reference failed')
            time.sleep(5)
        if mc.returncode!=0:raise RuntimeError('Reference/watchdog returned nonzero')
        reference=c.read(c.OUT/'reference/complete.json')
        c.verify_sources(protocol)
        runtime=dict(training_seconds=training,reference_seconds=reference['elapsed_seconds'])
        runtime.update(pipe.evaluate(c.OUT,check));c.write(c.OUT/'runtime.json',runtime)
        pipe.status(c.OUT,'analysis')
        gates=analysis.run(c.OUT,check);check();c.verify_sources(protocol)
        counts=dict(finals=len(list((c.OUT/'training').glob('*/final.pt'))),
            ema=len(list((c.OUT/'training').glob('*/ema_*.pt'))),
            evaluation_predictions=len(list((c.OUT/'evaluation').glob('*/*/*.npz'))),
            validation_predictions=len(list((c.OUT/'training').glob('*/validation/*/*.npz'))),
            evaluation_complete=len(list((c.OUT/'evaluation').glob('*/complete.json'))),
            png=len(list((c.OUT/'analysis/figures').glob('*.png'))),pdf=len(list((c.OUT/'analysis/figures').glob('*.pdf'))))
        if counts!=dict(finals=24,ema=48,evaluation_predictions=888,validation_predictions=768,evaluation_complete=42,png=10,pdf=10):
            raise RuntimeError(('Incomplete final science',counts))
        locked=c.read(c.OUT/'finals_locked.json')
        for ident in c.evaluation_identities():
            if c.sha(ROOT/ident['checkpoint'])!=locked['checkpoints'][ident['cell']]:raise RuntimeError('Final weight changed')
        c.write(c.OUT/'final_summary.json',dict(status='remote_complete_requires_backup_and_visual_review',time=time.time(),
            counts=counts,gates=gates,elapsed_from_budget_seconds=time.time()-budget['started'],deadline=budget['deadline']))
        # Exclude only duplicate/temporary weights and live administrative files;
        # all full finals, probabilities, input banks and analysis are included.
        excluded={'last.pt','status.json','run.lock','manifest.json','preflight.log','run.log','reference.log'}
        manifest={}
        for p in c.OUT.rglob('*'):
            rel=p.relative_to(c.OUT)
            if not p.is_file() or p.name in excluded or p.name.endswith('.tmp'):continue
            if rel.parts[0].startswith(('preflight','cpu_preparation','transfer_probe')):continue
            manifest[rel.as_posix()]=dict(bytes=p.stat().st_size,sha256=c.sha(p))
        c.write(c.OUT/'manifest.json',manifest)
        pipe.status(c.OUT,'remote_complete_requires_backup_and_visual_review',files=len(manifest))
    finally:
        if mc.poll() is None:
            os.killpg(mc.pid,signal.SIGTERM)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,
        choices=['cpu-prepare','begin-preflight','preflight','transfer-probe','freeze','launch','reference','run'])
    args=p.parse_args();signal.signal(signal.SIGTERM,stop)
    try:
        globals()[args.mode.replace('-','_')]()
    except BaseException:
        if c.OUT.exists():
            name='reference_failure.json' if args.mode=='reference' else ('failure.json' if args.mode=='run' else f'{args.mode}_failure_{time.time_ns()}.json')
            c.write(c.OUT/name,dict(time=time.time(),mode=args.mode,traceback=traceback.format_exc()))
        raise
