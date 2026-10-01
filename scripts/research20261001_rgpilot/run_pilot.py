"""One immutable RG-data training pilot. Explicit preflight then one locked run."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import copy
import hashlib
import json
import signal
import shutil
import time
import traceback
from pathlib import Path
import numpy as np
import torch
import common as c
import model_data as m
import training as tr
import evaluation as ev

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)

def stop_handler(signum,frame):
    # A flag is checked at update boundaries; do not interrupt between optimizer and bookkeeping.
    c.REQUEST_STOP=True

def verify_inherited():
    ep=c.read(c.ROOT/'artifacts/endpoint_mask_sampler_evaluation_20261001/run_protocol.json')
    assert ep['protocol_hash']=='a11d627b753068762bd126070b739dd945ad2a82ccf8afe9b82935602e6863d3'
    for path,digest in ep['sources'].items():
        assert c.sha(c.ROOT/path)==digest,path
    budget=c.read(c.ROOT/'docs/research_reboot_20260921/ENDPOINT_MASK_SAMPLER_DISENTANGLEMENT_BUDGET_20261001.json')
    expected={r['lineage']:r['sha256'] for r in budget['base_final_hashes_from_existing_manifests_not_reloaded_in_this_design_check']}
    for seed in c.SEEDS:assert c.sha(c.base(seed))==expected[seed]
    ref=c.read(c.ROOT/'artifacts/endpoint_mask_sampler_evaluation_20261001/reference/complete.json')
    science=c.read(c.ROOT/'artifacts/endpoint_mask_sampler_evaluation_20261001/science_manifest.json')
    row=next(r for r in science['files'] if r['path']==c.TEST_DATA.relative_to(c.ROOT).as_posix())
    assert c.sha(c.TEST_DATA)==row['sha256'] and c.TEST_DATA.stat().st_size==row['bytes']
    return dict(previous_sources=len(ep['sources']),base_checkpoints=3,reference_sha256=row['sha256'])

def identity_set(parent):
    return {hashlib.sha256(np.packbits(s>0,axis=-1,bitorder='little').tobytes()).hexdigest() for s in parent.spins}

def validate_parent_splits(train,val,test):
    a,b,d=identity_set(train),identity_set(val),identity_set(test)
    assert len(a)==len(train.spins) and len(b)==len(val.spins) and len(d)==len(test.spins)
    assert not a&b and not a&d and not b&d
    assert len(np.unique(test.chain_ids))==16
    return dict(train_parents=len(a),learning_parents=len(b),test_parents=len(d),test_chains=16,
        duplicate_whole_lattices=0,train_digest=c.digest(np.array(sorted(a))),
        learning_digest=c.digest(np.array(sorted(b))),test_digest=c.digest(np.array(sorted(d))))

def test_toy():
    x=np.array([[[1,1,-1],[1,-1,1],[-1,1,-1]]],dtype=np.int8)
    assert c.data.majority(x).item()==1 and c.data.majority(-x).item()==-1
    r=np.random.default_rng(123);x=r.choice([-1,1],size=(4,12,12)).astype(np.int8)
    expected=np.empty((4,4,4),dtype=np.int8)
    for b in range(4):
        for i in range(4):
            for j in range(4):expected[b,i,j]=np.sign(x[b,3*i:3*i+3,3*j:3*j+3].sum())
    assert np.array_equal(c.data.majority(x),expected)
    assert np.array_equal(c.data.majority(-x),-expected)
    for k in range(4):assert np.array_equal(c.data.majority(np.rot90(x,k,axes=(1,2))),np.rot90(expected,k,axes=(1,2)))
    return dict(majority_slow_reference=True,spin_flip=True,rotation_equivariance=True)

def preflight():
    c.deadline();stamp=time.strftime('%Y%m%d_%H%M%S',time.gmtime())
    folder=c.OUT/'preflight'/stamp;folder.mkdir(parents=True,exist_ok=False)
    start=time.time()
    try:
        inherited=verify_inherited();toy=test_toy()
        train=c.data.load_parent(c.TRAIN_DATA,'train')
        val,vids=c.chain_select(c.TRAIN_DATA,'val',c.CFG['learning_parents'])
        test,tids=c.chain_select(c.TEST_DATA,'test_target',c.CFG['test_parents'])
        splits=validate_parent_splits(train,val,test)
        for cycle in range(1,33):
            a=m.train_batch(train,c.SEEDS[0],cycle,'core')
            assert a['native_hash']==m.train_batch(train,c.SEEDS[0],cycle,'core')['native_hash']
            b=m.train_batch(train,c.SEEDS[0],cycle,'aux',True)
            d=m.train_batch(train,c.SEEDS[0],cycle,'aux',False)
            assert b['selection_hash']==d['selection_hash'] and b['mask_time_hash']==d['mask_time_hash']
            assert b['level']==1 and d['level']==0 and b['clean'].size==d['clean'].size==18432
        initial_hashes={};identity_max=0.
        for seed in c.SEEDS:
            model,ema,opt=m.initialize(seed)
            old_model=m.CoordinateDenseDenoiser(m.CoordinateDenoiserConfig(**m.MODEL)).cuda().eval()
            payload=torch.load(c.base(seed),map_location='cpu',weights_only=False)
            old_model.load_state_dict(payload['ema'])
            batch=m.train_batch(train,seed,4,'core');x=torch.as_tensor(batch['noisy'][:1],device='cuda',dtype=torch.long)
            t=torch.as_tensor(batch['t'][:1],device='cuda');xy=torch.as_tensor(batch['coords'][:1],device='cuda')
            with torch.no_grad():
                a=model(x,t,xy,0);b=old_model(x,t,xy)
                assert torch.equal(a,b),float((a-b).abs().max())
                assert torch.equal(a,model(x,t,xy,1))
            initial_hashes[str(seed)]=m.model_hash(model)
            del old_model,payload,model,ema,opt
        # Two identical short scratch trajectories must be bitwise equal; not reused for formal work.
        trajectory=[]
        for repeat in range(2):
            model,ema,opt=m.initialize(c.SEEDS[0]);losses=[]
            for cycle in range(1,5):
                b=m.train_batch(train,c.SEEDS[0],cycle,'aux',True)
                losses.append(tr.update(model,ema,opt,b,cycle))
            trajectory.append(dict(raw=m.model_hash(model),ema=m.model_hash(ema),losses=losses))
            del model,ema,opt
        assert trajectory[0]==trajectory[1]
        timings={}
        for role,coarse in [('core',False),('aux',False),('aux',True)]:
            model,ema,opt=m.initialize(c.SEEDS[0]);times=[]
            for cycle in range(1,33):
                torch.cuda.synchronize();beg=time.perf_counter()
                b=m.train_batch(train,c.SEEDS[0],cycle,role,coarse)
                tr.update(model,ema,opt,b,cycle);torch.cuda.synchronize()
                if cycle>8:times.append(time.perf_counter()-beg)
            timings[role+('_rg' if coarse else '_fine')]=dict(mean=float(np.mean(times)),max=float(np.max(times)))
            del model,ema,opt
        model,ema,opt=m.initialize(c.SEEDS[0]);prediction={}
        for width in (32,48,96):
            if width==96:
                bank=c.data.local_bank(test,np.arange(16),96,8,'rgpilot_preflight');bank['level']=np.array(0);bank['target_kind']=bank['target_type']
            else:bank=m.empirical_bank(test,16,width,.5,width==32,'rgpilot_preflight_'+str(width))
            m.predict(ema,bank);torch.cuda.synchronize();beg=time.perf_counter()
            pred=m.predict(ema,bank);torch.cuda.synchronize()
            prediction[str(width)]=(time.perf_counter()-beg)/16
            assert all(np.isfinite(v).all() for v in pred.values())
        train_seconds=(3*c.CFG['cycles']*(3*timings['core_fine']['mean']+timings['aux_fine']['mean']+timings['aux_rg']['mean']))*1.35
        # Exact prospective bank counts, including initial, learning, and all scale-label ablations.
        eval_seconds=(9216*prediction['96']+22464*prediction['48']+20160*prediction['32'])*1.35
        reserve=1200.  # input audits, checkpoints, CPU bootstrap, plots and remote package
        forecast=train_seconds+eval_seconds+reserve
        free=shutil.disk_usage(c.ROOT).free
        result=dict(status='passed',time=time.time(),elapsed=time.time()-start,inherited=inherited,toy=toy,
            splits=splits,initial_hashes=initial_hashes,zero_scale_identity_bitwise=True,
            repeat_scratch_trajectory_bitwise=True,paired_aux_choices=True,strict_environment=c.environment(),
            timings=timings,fp32_prediction_seconds_per_parent=prediction,forecast_training_seconds=train_seconds,
            forecast_evaluation_seconds=eval_seconds,reserve_seconds=reserve,forecast_total_seconds=forecast,
            expected_finish=time.time()+forecast,compute_deadline=c.CFG['compute_deadline'],free_disk_bytes=free,
            time_gate=time.time()+forecast<c.CFG['compute_deadline'],disk_gate=free>3*1024**3,
            peak_cuda_allocated=torch.cuda.max_memory_allocated(),validation_ids=vids.tolist(),test_ids=tids.tolist())
        assert result['time_gate'] and result['disk_gate'],result
        c.write(folder/'summary.json',result)
        c.write(c.OUT/'preflight_passed.json',dict(path=str((folder/'summary.json').relative_to(c.ROOT)),sha256=c.sha(folder/'summary.json'),time=time.time()))
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except BaseException as error:
        c.write(folder/'failure.json',dict(error=str(error),traceback=traceback.format_exc(),time=time.time()))
        raise

def freeze():
    c.deadline();passed=c.read(c.OUT/'preflight_passed.json')
    path=c.ROOT/passed['path'];assert c.sha(path)==passed['sha256']
    forecast=c.read(path);assert time.time()+forecast['forecast_total_seconds']<c.CFG['compute_deadline']
    proto=dict(study=c.CFG['study'],config=c.CFG,sources=c.records(c.source_files()),
        preflight=passed,created=time.time(),forecast=forecast['forecast_total_seconds'],
        status='frozen_for_one_run',old_results_unchanged=True,initialization=c.CFG['initialization'])
    proto['protocol_hash']=hashlib.sha256(json.dumps(proto,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    c.write(c.OUT/'run_protocol.json',proto)
    initial_paths=[p for p in c.source_files() if p.suffix in ['.py','.json','.md']]+[c.OUT/'run_protocol.json',path,c.OUT/'preflight_passed.json']
    package=c.archive(c.ROOT,c.EXPORT,'rgpilot_initial_v1',initial_paths)
    print(json.dumps(dict(protocol_hash=proto['protocol_hash'],package=package['archive'],bytes=package['archive_bytes']),ensure_ascii=False))

def package(proto):
    c.check_sources(proto)
    paths=[p for p in c.OUT.rglob('*') if p.is_file() and p.name not in ['last.pt','formal.stdout','status.json','run.jsonl']
           and not p.name.endswith('.tmp') and '__pycache__' not in p.parts]
    science=dict(files=c.records(paths+[c.ROOT/r['path'] for r in proto['sources']]),time=time.time(),
        exclusions=[str(p.relative_to(c.ROOT)) for p in c.OUT.rglob('last.pt')],
        inherited_large_inputs_reuse_existing_verified_archives=True,administrative_closure_required=True)
    c.write(c.OUT/'science_manifest.json',science)
    result=c.archive(c.ROOT,c.EXPORT,'rgpilot_complete_v1',paths+[c.OUT/'science_manifest.json'])
    return result

def run():
    c.deadline();proto=c.read(c.OUT/'run_protocol.json');c.check_sources(proto)
    c.write(c.OUT/'run.lock',dict(pid=os.getpid(),pgid=os.getpgrp(),started=time.time(),protocol_hash=proto['protocol_hash']))
    signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
    begun=time.time();states=[]
    try:
        c.log('formal_started',pid=os.getpid(),pgid=os.getpgrp(),compute_deadline=c.CFG['compute_deadline'])
        c.write(c.OUT/'formal_started.json',dict(pid=os.getpid(),pgid=os.getpgrp(),started=begun,compute_deadline=c.CFG['compute_deadline']))
        train=c.data.load_parent(c.TRAIN_DATA,'train')
        val,vids=c.chain_select(c.TRAIN_DATA,'val',c.CFG['learning_parents'])
        test,tids=c.chain_select(c.TEST_DATA,'test_target',c.CFG['test_parents'])
        split=validate_parent_splits(train,val,test)
        c.write(c.OUT/'data_manifest.json',dict(**split,validation_ids=vids.tolist(),test_ids=tids.tolist(),
            train_source_sha256=c.sha(c.TRAIN_DATA),test_source_sha256=c.sha(c.TEST_DATA)))
        test_paths=m.make_banks(test,len(test.spins),c.OUT/'banks/test')
        learning_paths=m.make_banks(val,len(val.spins),c.OUT/'banks/learning',learning=True)
        c.log('banks_complete',test_banks=len(test_paths),learning_banks=len(learning_paths))
        for seed in c.SEEDS:
            for arm in c.ARMS:
                state=tr.make_state(seed,arm);states.append(state)
                c.write(c.OUT/f'training/s{seed}_{arm}/initialization.json',state['metadata'])
                if arm=='fine_only':
                    ip=c.OUT/f'initial/s{seed}_ema.pt';isha=tr.save_state(ip,state,proto['protocol_hash'],ema_only=True)
                    ev.evaluate(state['ema'],seed,'initial','initial','test',test_paths,isha)
                    ev.evaluate(state['ema'],seed,'initial','initial','learning',learning_paths,isha)
            assert len({s['metadata']['initial_hash'] for s in states if s['seed']==seed})==1
        tr.train(states,train,proto)
        c.write(c.OUT/'training_complete.json',dict(time=time.time(),updates=sum(s['updates'] for s in states),checkpoints=9))
        c.log('training_complete',updates=sum(s['updates'] for s in states))
        ev.evaluate_finals(states,proto,test_paths,learning_paths)
        c.log('evaluation_complete')
        training_audit=tr.audit_training(train,proto)
        ev.audit_banks(test,val)
        pred_audit=ev.audit_predictions()
        result=ev.analyze();ev.figures(result)
        c.check_sources(proto)
        c.write(c.OUT/'final_summary.json',dict(status='scientific_pilot_complete_needs_local_backup_visual_review_publication',
            time=time.time(),started=begun,wall_seconds=time.time()-begun,training_audit=training_audit['status'],
            prediction_audit=pred_audit['status'],analysis_sha256=c.sha(c.OUT/'analysis/summary.json'),
            no_confirmatory_claim=True,updates=c.CFG['total_updates'],final_checkpoints=9))
        c.log('scientific_complete')
        result=package(proto)
        c.write(c.OUT/'remote_export_complete.json',dict(time=time.time(),archive=result['archive'],bytes=result['archive_bytes'],sha256=result['archive_sha256']))
        c.log('remote_export_complete',bytes=result['archive_bytes'])
    except BaseException as error:
        saved=[]
        for state in states:
            try:
                path=c.OUT/f'training/s{state["seed"]}_{state["arm"]}/emergency.pt'
                saved.append(dict(path=str(path.relative_to(c.ROOT)),sha256=tr.save_state(path,state,proto['protocol_hash'])))
            except BaseException as save_error:saved.append(dict(seed=state['seed'],arm=state['arm'],error=str(save_error)))
        c.write(c.OUT/'failure.json',dict(time=time.time(),error=str(error),traceback=traceback.format_exc(),
            updates=sum(s['updates'] for s in states),saved=saved,automatic_restart=False))
        c.log('formal_failure',error=str(error),updates=sum(s['updates'] for s in states));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['preflight','freeze','run'],required=True)
    args=ap.parse_args();{'preflight':preflight,'freeze':freeze,'run':run}[args.mode]()
