"""Budget-first GPU gate. Scratch only: NEVER starts the formal campaign.

Even a passed budget gate is not launch approval: full runner/scratch/MC/statistical
pipeline gates must pass separately. The first invocation starts the hard clock.
"""
from __future__ import annotations
import json, os, sys, time, traceback, hashlib, signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
import numpy as np
import torch
from ism_diffusion import geometry_study as gs
from ism_diffusion.scale_data import load_parent_split
from ism_diffusion.scale_evaluation import load_scale_model
import core_geometry_factorial_design as d
import core_geometry_factorial_data as data
import core_geometry_factorial_training as tr
import core_geometry_factorial_statistics as stats
import evaluate_study as ev

OUT=ROOT/'artifacts/core_geometry_factorial_20260925'
SCRATCH=OUT/'preflight_buffered_v2'
DEADLINE=float('inf')

def check():
    if time.time()>=DEADLINE: raise TimeoutError('Original ten-hour hard deadline')

def stopped(*_): raise TimeoutError('External preflight watchdog')

def log(stage,**kw):
    row=dict(stage=stage,time=time.time(),**kw)
    gs.atomic_json(SCRATCH/'status.json',row)
    print(json.dumps(row),flush=True)

def elapsed_measure(fn):
    check();torch.cuda.synchronize();started=time.perf_counter();v=fn()
    torch.cuda.synchronize();return time.perf_counter()-started,v

def assert_tree(a,b):
    if torch.is_tensor(a):
        assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: assert_tree(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): assert_tree(x,y)
    else: assert a==b

def main():
    global DEADLINE
    assert json.loads((OUT/'preflight/budget_gate.json').read_text())['status']=='budget_gate_failed'
    assert not (OUT/'run.lock').exists()
    SCRATCH.mkdir(exist_ok=False)
    with (SCRATCH/'preflight.lock').open('x') as f: f.write(str(os.getpid()))
    original_budget=json.loads((OUT/'budget.json').read_text())
    started=original_budget['started'];DEADLINE=original_budget['deadline'];check()
    log('begin_buffered_logging_recheck',pid=os.getpid(),deadline=DEADLINE,
        reason='same updates and log contents; keep log handle open, flush every 20 steps; no clock reset')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    source=ROOT/'data/level1/parents_l1024.npz'
    parent=load_parent_split(source,'train');val=load_parent_split(source,'val')
    manifest=parent.metadata['split_manifest']
    assert not set(manifest['train'])&set(manifest['val'])
    from ism_diffusion.ising import BETA_CRITICAL
    assert abs(parent.metadata['beta']-BETA_CRITICAL)<1e-14
    gs.atomic_json(SCRATCH/'input_audit.json',dict(parent_file_sha256=gs.file_hash(source),
        split_manifest=manifest,beta=parent.metadata['beta'],mc_seed=d.MC_SEED,
        note='array-local chain IDs overlap across splits; physical split_manifest IDs do not',
        design=d.self_test(),bootstrap=stats.self_test()))
    seed=d.SEEDS[0]
    # Deterministic FP32 math-SDPA validates exact continuation, separately from
    # normal BF16 flash throughput (whose nondeterminism is measured below).
    from torch.nn.attention import sdpa_kernel,SDPBackend
    step=next(i for i in range(1,33) if d.schedule(seed,i,'G11')['width']==16
        and d.schedule(seed,i,'G11')['kind']=='continuous' and not d.schedule(seed,i,'G11')['sparse'])
    b=data.training_batch(parent,seed,step,'G11')
    bs=data.training_batch(parent,seed,step,'G11S')
    assert b['paired_data_hash']==bs['paired_data_hash'] and b['actual_input_hash']==bs['actual_input_hash']
    m,e,o,meta=tr.fresh_state(seed)
    same_initial=[]
    for arm in d.ARMS:
        mm=gs.new_model(seed);same_initial.append(gs.model_hash(mm));del mm
    assert len(set(same_initial+[meta['initial_hash']]))==1
    torch.use_deterministic_algorithms(True)
    with sdpa_kernel(SDPBackend.MATH):
        a=tr.update(m,e,o,b,1,amp=False);meta['step']=1
        assert gs.model_hash(m)!=meta['initial_hash']
        tr.save_state(SCRATCH/'resume.pt',m,e,o,meta,seed,'G11','scratch',immutable=True)
        tr.update(m,e,o,bs,2,amp=False)
        mm,ee,oo,met=tr.resume_state(SCRATCH/'resume.pt',seed,'G11','scratch')
        tr.update(mm,ee,oo,b,2,amp=False)
        assert_tree(m.state_dict(),mm.state_dict());assert_tree(e.state_dict(),ee.state_dict())
        assert_tree(o.state_dict(),oo.state_dict())
    torch.use_deterministic_algorithms(False)
    # Same-seed no-old-checkpoint initialization and optimizer mutation verified.
    gs.atomic_json(SCRATCH/'recovery_test.json',dict(status='passed',step1=a,
        five_arm_initial_hash=same_initial[0],model_ema_optimizer_exact=True,
        mode='deterministic FP32 math SDPA',old_checkpoint_used=False,
        torch_version=torch.__version__,cuda=torch.version.cuda))
    del m,e,o,mm,ee,oo;torch.cuda.empty_cache()
    m,e,o,meta=tr.fresh_state(seed)
    cells=[]
    for arm in ['G11','G11S']:
        for width in d.WIDTHS:
            for kind in ['continuous','train_gap']:
                for sparse in [False,True]:
                    if arm=='G11S' and kind=='continuous': continue
                    steps=[s for s in range(1,3000) if (lambda c:c['width']==width and c['kind']==kind and c['sparse']==sparse)(d.schedule(seed,s,arm))][:25]
                    assert len(steps)==25
                    times=[];peak=0
                    training_log=(SCRATCH/'timing_train.jsonl').open('a')
                    for j,s in enumerate(steps):
                        torch.cuda.reset_peak_memory_stats()
                        def one():
                            batch=data.training_batch(parent,seed,s,arm)
                            r=tr.update(m,e,o,batch,s)
                            # Account for per-step digest and serialized logging overhead.
                            hashlib.sha256((r['paired_data_hash']+r['actual_input_hash']).encode()).hexdigest()
                            training_log.write(json.dumps(r)+'\n')
                            if (j+1)%20==0:training_log.flush()
                            return r
                        duration,r=elapsed_measure(one)
                        peak=max(peak,torch.cuda.max_memory_allocated())
                        if j>=5: times.append(duration)
                    close_started=time.perf_counter();training_log.close()
                    close_seconds=time.perf_counter()-close_started
                    times=[x+close_seconds/len(times) for x in times]
                    cell=dict(arm=arm,width=width,kind=kind,sparse=sparse,n=len(times),
                        mean=float(np.mean(times)),p90=float(np.quantile(times,.9)),
                        conservative=max(float(np.mean(times)),float(np.quantile(times,.9))),
                        raw_seconds=times,peak_bytes=peak,last_finite_update=r)
                    cells.append(cell);gs.atomic_json(SCRATCH/'training_timing.json',cells)
                    log('training_benchmark',arm=arm,width=width,kind=kind,sparse=sparse,
                        mean=cell['mean'],p90=cell['p90'])
    # BF16 normal backend determinism is not assumed; quantify a duplicate update.
    m1,e1,o1,_=tr.fresh_state(seed);m2,e2,o2,_=tr.fresh_state(seed)
    tr.update(m1,e1,o1,b,1);tr.update(m2,e2,o2,b,1)
    bf16_error=max(float((x-y).abs().max()) for x,y in zip(m1.parameters(),m2.parameters()))
    gs.atomic_json(SCRATCH/'bf16_replay.json',dict(max_abs_parameter_difference=bf16_error,
        exact=bf16_error==0,note='Measured, not a promise of cross-hardware bitwise repeatability'))
    del m1,e1,o1,m2,e2,o2;torch.cuda.empty_cache()
    # Check a complete recoverable state and standard EMA loader.
    meta['step']=2500
    save_seconds,_=elapsed_measure(lambda:tr.save_state(SCRATCH/'throughput.pt',m,e,o,meta,seed,'G11','scratch'))
    load_seconds,loaded=elapsed_measure(lambda:load_scale_model(SCRATCH/'throughput.pt',torch.device('cuda')))
    loaded_model,payload=loaded;assert gs.model_hash(loaded_model)==gs.model_hash(e)
    del loaded_model,payload;torch.cuda.empty_cache()
    ids=np.arange(8,dtype=np.int64);pred=[];bank_seconds=[];npz_seconds=[]
    for g,(name,w,kind) in enumerate(d.GEOMETRIES):
        for arm in ['G11','G11S']:
            for k in d.evaluation_ks(w):
                t=time.perf_counter();bank=data.conditional_bank(val,ids,2026092522,g,w,kind,k);bank_seconds.append((w,time.perf_counter()-t))
                for warm in range(2): tr.predictions(e,bank,arm,check=check)
                seconds,p=elapsed_measure(lambda:tr.predictions(e,bank,arm,check=check))
                t=time.perf_counter();ev.atomic_npz(SCRATCH/'prediction_scratch.npz',**p);npz_seconds.append(time.perf_counter()-t)
                pred.append(dict(geometry=name,width=w,kind=kind,k=k,arm=arm,seconds_per_parent=seconds/8))
            log('prediction_benchmark',geometry=name,arm=arm)
    gs.atomic_json(SCRATCH/'prediction_timing.json',pred)
    # A full 16-image, 256-step production sampler shard. Only old val data.
    e.eval();torch.set_float32_matmul_precision('high')
    definition=dict(name='core_geometry_W96',width=96,kind='continuous',gaps=[1],samples=16)
    seconds,gen=elapsed_measure(lambda:ev.generate_and_score(e,val,'A',seed,definition,SCRATCH/'generation',DEADLINE,16))
    files=list((SCRATCH/'generation').glob('shard_*.npz'));assert len(files)==1
    assert np.load(files[0])['spins'].shape==(16,96,96)
    torch.set_float32_matmul_precision('highest')
    # Exact frequency over all 12000 updates, not a uniform arm-size shortcut.
    lookup={(x['arm'],x['width'],x['kind'],x['sparse']):x['conservative'] for x in cells}
    per_arm={}
    for arm in d.ARMS:
        cost=0.
        for seed2 in d.SEEDS:
            for s in range(1,d.STEPS+1):
                c=d.schedule(seed2,s,arm)
                a='G11S' if arm=='G11S' and c['kind']=='train_gap' else 'G11'
                cost+=lookup[a,c['width'],c['kind'],c['sparse']]
        per_arm[arm]=cost
    train=sum(per_arm.values())
    inference=0.
    for arm in d.ARMS:
        a='G11S' if arm=='G11S' else 'G11'
        inference+=6*128*sum(x['seconds_per_parent'] for x in pred if x['arm']==a)
    validation=30*3*2*64*max(x['seconds_per_parent'] for x in pred if x['width']==48)
    storage=(30*12+30*3)*save_seconds+(30*11+30)*load_seconds
    prediction_storage=900*max(npz_seconds)*16 # 8-parent measured -> 128-parent conservative scale
    reference_banks=30*max(v for w,v in bank_seconds)*16*2
    generation=120*seconds
    raw=train+inference+validation+storage+prediction_storage+reference_banks+generation
    # 1h independent MC allowance with overlap against the measured training only.
    mc_residual=max(0.,3600-train)
    forecast=time.time()-started+1.2*raw+mc_residual+1800
    result=dict(status='budget_gate_passed_pending_full_pipeline' if forecast<=34200 else 'budget_gate_failed',
        started=started,deadline=DEADLINE,predicted_seconds=forecast,predicted_hours=forecast/3600,
        gate_seconds=34200,hard_seconds=36000,per_arm_raw_training_seconds=per_arm,
        raw=dict(training=train,predictions=inference,validation=validation,checkpoint_io=storage,
            prediction_io=prediction_storage,bank_preparation=reference_banks,generation=generation),
        safety_multiplier=1.2,mc_residual_seconds=mc_residual,statistics_reserve_seconds=1800,
        elapsed_preflight_seconds=time.time()-started,shard16_full256_seconds=seconds,
        checkpoint_save_seconds=save_seconds,checkpoint_load_seconds=load_seconds,
        device=torch.cuda.get_device_name(),formal_training_started=False,
        new_mc_started=False,full_pipeline_scratch_passed=False,
        implementation_only_change='persistent log handle, flush every20; same records, same model/data/loss/update',
        original_failure_preserved='preflight/budget_gate.json',clock_reset=False,
        source_sha256={p.name:gs.file_hash(p) for p in Path(__file__).parent.glob('*core_geometry*.py')})
    gs.atomic_json(SCRATCH/'budget_gate.json',result);log('budget_gate',**result)

if __name__=='__main__':
    signal.signal(signal.SIGTERM,stopped)
    try: main()
    except Exception:
        if SCRATCH.exists(): gs.atomic_json(SCRATCH/'failure.json',dict(time=time.time(),traceback=traceback.format_exc(),formal_training_started=False))
        raise
