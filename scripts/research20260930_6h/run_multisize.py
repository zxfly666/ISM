"""Explicit prepare/preflight/lock/run modes, single lock and original deadline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import sm6_common as c


def lock():
    c.deadline()
    assert c.read(c.OUT/'preflight/complete.json')['status']=='passed'
    gate=c.read(c.OUT/'preflight/launch_gate.json');assert gate['passed']
    assert (c.OUT/'preflight_visual_review.json').is_file()
    visual=c.read(c.OUT/'preflight_visual_review.json')
    assert visual['status']=='passed' and visual['actually_viewed_pngs']==5
    receipt=c.read(c.OUT/'initial_backup_receipt.json')
    assert receipt['status']=='passed' and receipt['verified_locally'] is True
    transfer=c.read(c.OUT/'transfer_verification_gate.json')
    assert transfer['status']=='passed' and transfer['measured_transfer_bytes']>0
    assert transfer['local_C_free']>=2*1024**3 and transfer['local_D_free']>=gate['storage_required_with_margin']
    gate['backup_seconds']=max(1800,transfer['backup_projection_seconds'])
    c.check_sources(c.read(c.OUT/'preflight/preflight_sources.json'))
    now=time.time();original=c.read(c.OUT/'budget.json')
    remaining_est=gate['train_seconds']+gate['evaluation_seconds']+gate['analysis_audit_seconds']+gate['backup_seconds']+gate['reserve_seconds']
    if now+remaining_est>original['deadline']:
        raise RuntimeError('Waiting included: launch budget no longer fits original deadline')
    proto=dict(study=c.CFG['study'],authorized=True,authorization_source=c.AUTH.relative_to(c.ROOT).as_posix(),
               sources=c.source_snapshot(),config=c.CFG,created=now,deadline=original['deadline'],
               budget_start=original['started'],gate_sha=c.sha(c.OUT/'preflight/launch_gate.json'),
               transfer_gate_sha=c.sha(c.OUT/'transfer_verification_gate.json'),
               initial_backup_receipt_sha=c.sha(c.OUT/'initial_backup_receipt.json'),
               precision='FP32 deterministic; same-device bitwise recovery; cross-stack equivalence not bitwise promised')
    proto['protocol_hash']=hashlib.sha256(json.dumps(proto,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    c.write(c.OUT/'run_protocol.json',proto)
    return dict(status='locked_not_running',protocol_hash=proto['protocol_hash'],deadline=proto['deadline'])


def run():
    import sm6_data as d
    import sm6_model as m
    import sm6_evaluation as ev
    import sm6_analysis as analysis
    import sm6_management as management
    import torch
    proto=c.read(c.OUT/'run_protocol.json');c.check_sources(proto['sources']);c.deadline()
    with (c.OUT/'run.lock').open('x',encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(),started=time.time(),protocol_hash=proto['protocol_hash']),f)
    started=time.time();c.write(c.OUT/'formal_started.json',dict(started=started,pid=os.getpid(),protocol_hash=proto['protocol_hash']))
    data=d.TrainingData();states={};handles={}
    # All fresh identities are explicit. No old weights are restored or reused.
    for seed in c.CFG['seed_labels']:
        hashes=[]
        for arm in 'NW':
            state=m.fresh(seed,'cuda');states[seed,arm]=state
            folder=c.OUT/'training'/f's{seed}_{arm}';folder.mkdir(parents=True,exist_ok=False)
            h=m.model_hash(state[0]);hashes.append(h)
            c.write(folder/'initial.json',dict(seed=seed,arm=arm,raw_hash=h,ema_hash=m.model_hash(state[1]),initial_rng=c.init_seed(seed),time=time.time()))
            handles[seed,arm]=(folder/'log.jsonl').open('x',encoding='utf-8')
        assert len(set(hashes))==1
    try:
        completed_blocks=0; block_times=[]
        for block in range(8):
            # Deterministic rotation, independent of loss/validation and all GPU timing outcomes.
            pairs=[(s,a) for s in c.CFG['seed_labels'] for a in 'NW']
            offset=block%12;pairs=pairs[offset:]+pairs[:offset]
            for seed,arm in pairs:
                block_started=time.time();c.deadline();c.check_sources(proto['sources']);state=states[seed,arm];folder=c.OUT/'training'/f's{seed}_{arm}'
                for step in range(1000*block+1,1000*(block+1)+1):
                    row=m.update(state,data,seed,step,arm,handles[seed,arm])
                handles[seed,arm].flush()
                m.checkpoint(folder/'last.pt',state,seed,arm,step,proto['protocol_hash'],replace=True)
                if step in (4000,6000):m.checkpoint(folder/f'step_{step}.pt',state,seed,arm,step,proto['protocol_hash'],full=False)
                if step==8000:
                    identity=m.checkpoint(folder/'final.pt',state,seed,arm,step,proto['protocol_hash'])
                    c.write(folder/'complete.json',dict(**identity,final_sha256=c.sha(folder/'final.pt'),time=time.time()))
                c.status('training',seed=seed,arm=arm,step=step,blocks_done=block,
                         completed=len(list(c.OUT.glob('training/*/complete.json'))),latest_loss=row['loss'],latest_grad=row['grad_norm'])
                completed_blocks+=1;block_times.append(time.time()-block_started)
                gate=c.read(c.OUT/'preflight/launch_gate.json')
                backup=max(1800,c.read(c.OUT/'transfer_verification_gate.json')['backup_projection_seconds'])
                remaining_train=1.35*(96-completed_blocks)*max(block_times[-min(12,len(block_times)):])
                forecast=remaining_train+gate['evaluation_seconds']+1200+backup+1800
                c.write(c.OUT/'budget_forecast.json',dict(time=time.time(),completed_blocks=completed_blocks,
                    remaining_train_seconds=remaining_train,conservative_remaining_seconds=forecast,
                    deadline=proto['deadline'],fits=time.time()+forecast<=proto['deadline']),replace=True)
                if time.time()+forecast>proto['deadline']:
                    raise TimeoutError('Measured remaining workload no longer fits original deadline; checkpoint preserved')
    finally:
        for f in handles.values():f.close()
    training_end=time.time();del states;torch.cuda.empty_cache()
    finals={p.relative_to(c.OUT).as_posix():c.sha(p) for p in c.OUT.glob('training/*/final.pt')}
    assert len(finals)==12
    c.write(c.OUT/'finals_locked.json',dict(time=time.time(),finals=finals,training_seconds=training_end-started))
    c.check_sources(proto['sources'])
    rows,controls=ev.run()
    evaluation_end=time.time();c.status('audit',predictions=len(rows))
    training_audit=management.audit_training();c.write(c.OUT/'training_audit.json',training_audit)
    eval_audit=management.audit_evaluation(c.OUT,rows);c.write(c.OUT/'evaluation_audit.json',eval_audit)
    control_audit=management.audit_controls(c.OUT,controls);c.write(c.OUT/'control_audit.json',control_audit)
    # Deserializing all final optimizer states is a restore audit, not extra training.
    recovered=[]
    for rel,h in finals.items():
        state,payload=m.restore(c.OUT/rel,'cpu')
        assert payload['step']==8000 and state[2].state
        for part in (payload['raw'],payload['ema']):
            assert all(torch.isfinite(v).all() for v in part.values())
        recovered.append(dict(path=rel,sha256=h,step=payload['step']))
    c.write(c.OUT/'restore_audit.json',dict(status='passed',finals=recovered,not_new_training=True))
    audit_end=time.time()
    timing=dict(training=training_end-started,evaluation=evaluation_end-training_end,audit=audit_end-evaluation_end,
                preflight_and_wait=started-proto['budget_start'])
    analysis.analyze(c.OUT,rows,controls,timing)
    c.check_sources(proto['sources']);c.deadline()
    c.status('remote_complete_requires_backup_and_visual_review',time_completed=time.time(),finals=12,predictions=len(rows))
    package=management.package_science()
    c.write(c.EXPORT/'remote_export_complete.json',dict(time=time.time(),archive=package['archive'],archive_sha256=package['archive_sha256'],
              elapsed_budget_to_export=time.time()-proto['budget_start'],deadline=proto['deadline']))
    c.deadline()
    return dict(status='remote_export_complete_requires_local_SHA_and_visual_review',archive=package['archive'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('prepare','preflight','lock','run'),required=True)
    args=parser.parse_args()
    import sm6_model as m
    import sm6_preflight as pre
    m.configure()
    try:
        result={'prepare':pre.prepare_cpu,'preflight':pre.gpu,'lock':lock,'run':run}[args.mode]()
        print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
    except Exception as error:
        failure=dict(time=time.time(),mode=args.mode,error=repr(error),traceback=traceback.format_exc(),
                     no_automatic_retry=True,formal_run_lock=(c.OUT/'run.lock').exists())
        # Unique failure evidence: never overwrite an earlier failure or reset the budget.
        c.write(c.OUT/(args.mode+'_failure_'+str(time.time_ns())+'.json'),failure)
        c.status(args.mode+'_failed',error=repr(error))
        print(json.dumps(failure,ensure_ascii=False),flush=True)
        raise


if __name__=='__main__':main()
