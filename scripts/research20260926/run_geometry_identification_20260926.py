"""Authorized I experiment: CPU tests, single preflight, freeze, run, reference.

No automatic restart, no old-checkpoint initialization, no scientific downscaling.
"""
import argparse,json,os,signal,subprocess,sys,time,traceback,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent),str(ROOT/'scripts/research20260921')]
import numpy as np
import torch
from ism_diffusion import geometry_study as gs
from ism_diffusion.scale_data import load_parent_split,ParentSplit,pack_spins
from ism_diffusion.scale_evaluation import load_scale_model
import identification_common as c
import identification_data as d
import identification_training as tr
import identification_statistics as st
import identification_analysis as an

PREF=c.OUT/'preflight_v1'

def stop(*_):raise TimeoutError('External watchdog or termination')
def measure(fn):
    c.check();torch.cuda.synchronize();began=time.perf_counter();value=fn();torch.cuda.synchronize()
    return time.perf_counter()-began,value

def val_banks(parent):
    groups=[np.flatnonzero(parent.chain_ids==x) for x in np.unique(parent.chain_ids)]
    ids=np.concatenate([g[np.linspace(0,len(g)-1,64//len(groups),dtype=int)] for g in groups]);assert len(ids)==64
    banks=[]
    for gi,kind in enumerate(['continuous','train_gap']):
        for k in [2,32,512]:banks.append(d.conditional_bank(parent,ids,2026092622,100+gi,48,kind,k))
    for k in [1152,115]:banks.append(d.conditional_bank(parent,ids,2026092622,100,48,'continuous',k))
    return banks

def reference():
    from ism_diffusion.ising import generate_independent_chains,BETA_CRITICAL,magnetization,energy_density
    from ism_diffusion.diagnostics import integrated_autocorrelation_time,split_rhat
    assert (c.OUT/'run.lock').exists();protocol=c.read(c.OUT/'run_protocol.json');c.verify_sources(protocol)
    out=c.OUT/'reference';out.mkdir(exist_ok=False);started=time.time()
    seeds=[int(x.generate_state(1,dtype=np.uint32)[0]) for x in np.random.SeedSequence(c.CONFIG['mc']['seed']).spawn(16)]
    args=dict(lattice_size=1024,chain_seeds=seeds,samples_per_chain=128,burn_in_sweeps=40,sweeps_between=4,
              beta=float(BETA_CRITICAL),adaptation_sweeps=3,pilot_cluster_steps=128,workers=4,backend='numba',
              initial_states=['random']*8+['plus']*4+['minus']*4,return_chain_metadata=True)
    c.write(out/'protocol.json',dict(args,master_seed=c.CONFIG['mc']['seed'],role='evaluation_only'))
    spins,chain,metadata=generate_independent_chains(**args);c.check()
    md=dict(lattice_size=1024,beta=float(BETA_CRITICAL),seed=c.CONFIG['mc']['seed'],chain_metadata=metadata,chain_seeds=seeds)
    c.save(out/'fresh_l1024.npz',test_target_packed=pack_spins(spins),test_target_chain_id=chain,metadata=np.array(json.dumps(md)))
    parent=ParentSplit(spins,chain,1024,md);parts=[];origins=[]
    for start in range(0,2048,16):
        c.check();ids=np.arange(start,start+16);axis=np.broadcast_to(np.arange(96),(16,2,96)).copy()
        origin=np.stack([c.rng(c.CONFIG['role_seeds']['generation'],int(i),'reference_crop').integers(1024,size=2) for i in ids])
        crop=2*d.sampled(parent,ids,axis,origin)-1;parts.append(tr.physical_stats(crop));origins.append(origin)
    stats={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    c.save(out/'generation_reference.npz',**stats,chain=chain,parent=np.arange(2048),origin=np.concatenate(origins))
    m=magnetization(spins);trace=dict(energy=energy_density(spins),m=m,abs_m=np.abs(m),m2=m*m,
                                    G25_48=(stats['pair_sum'][:,25:49]/stats['pair_count'][:,25:49]).mean(1))
    diag={k:dict(split_rhat=float(split_rhat([v[chain==i] for i in range(16)])),chains=[integrated_autocorrelation_time(v[chain==i]) for i in range(16)]) for k,v in trace.items()}
    passed=all(np.isfinite(diag[k]['split_rhat']) and diag[k]['split_rhat']<=1.1 and all(v['ess']>=16 for v in diag[k]['chains']) for k in ['energy','abs_m','m2','G25_48'])
    c.save(out/'chain_traces.npz',chain=chain,**trace)
    c.write(out/'qa.json',dict(status='passed' if passed else 'qa_failed',diagnostics=diag,parents=2048,elapsed_seconds=time.time()-started))
    if not passed:raise RuntimeError('Independent reference QA failed; no adaptive extension')
    layouts=c.load(c.OUT/'design/low_layouts.npz')
    low=d.low_reference(parent,layouts,c.check);c.save(out/'low_joint_counts.npz',**low)
    c.write(out/'complete.json',dict(status='passed',parents=2048,chains=16,elapsed_seconds=time.time()-started,
        files={p.name:c.sha(p) for p in out.glob('*.npz')},qa_sha256=c.sha(out/'qa.json')))

def train(protocol):
    parent=load_parent_split(c.DATA,'train');val=load_parent_split(c.DATA,'val');banks=val_banks(val)
    for j,b in enumerate(banks):c.save(c.OUT/'validation_banks'/f'v{j}.npz',**b)
    for end in range(1000,12001,1000):
        for seed,arm in protocol['cell_order']:
            c.check()
            if (c.OUT/'reference_failure.json').exists():raise RuntimeError('Reference failed; stop formal training')
            out=c.OUT/'training'/f's{seed}_{arm}';out.mkdir(parents=True,exist_ok=True)
            if end==1000:
                m,e,o,meta=tr.fresh(seed);c.write(out/'initial.json',dict(seed=seed,arm=arm,initial_hash=meta['initial_hash'],fresh=True))
            else:m,e,o,meta=tr.restore(out/'last.pt',seed,arm,protocol['protocol_hash'])
            assert meta['step']==end-1000;c.log('training',seed=seed,arm=arm,start=meta['step'],target=end);began=time.perf_counter()
            with (out/'train.jsonl').open('a',buffering=65536) as f:
                for step in range(meta['step']+1,end+1):
                    c.check();batch=d.training_batch(parent,seed,step,arm);r=tr.update(m,e,o,batch,step)
                    meta['step']=step;meta['paired_data_digest']=c.increment_digest(meta['paired_data_digest'],r['paired_data_hash']);meta['actual_input_digest']=c.increment_digest(meta['actual_input_digest'],r['actual_input_hash'])
                    f.write(json.dumps(dict(step=step,time=time.time(),**r))+'\n')
                    if step%20==0:f.flush()
            meta['elapsed_seconds']+=time.perf_counter()-began;tr.save_state(out/'last.pt',m,e,o,meta,seed,arm,protocol['protocol_hash'])
            if end in [4000,8000,12000]:
                vals=[];per_parent=[]
                for b in banks:
                    reps=4 if arm=='I-R' and str(b['kind'])!='continuous' else 1
                    pp=[tr.predict(e,b,arm,r) for r in range(reps)]
                    one=np.mean([np.stack([v['ce'].mean(1),v['brier'].mean(1)],-1) for v in pp],axis=0)
                    vals.append(one.mean(0));per_parent.append(one)
                c.save(out/f'validation_{end}.npz',metrics=np.array(vals),parent_risks=np.array(per_parent))
                if end<12000:
                    p=out/f'ema_{end}.pt';torch.save(dict(ema=e.state_dict(),config=dict(model=gs.MODEL),step=end,seed=seed,arm=arm),p)
                else:
                    tr.save_state(out/'final.pt',m,e,o,meta,seed,arm,protocol['protocol_hash'],True)
                    c.write(out/'complete.json',dict(status='complete',steps=end,metadata=meta,final_sha256=c.sha(out/'final.pt'),ema_sha256=gs.model_hash(e)))
            c.write(out/'status.json',dict(step=end,seed=seed,arm=arm,time=time.time()));del m,e,o;torch.cuda.empty_cache()
    for seed in c.SEEDS:
        v=[c.read(c.OUT/'training'/f's{seed}_{a}'/'complete.json') for a in c.ARMS]
        assert len({x['metadata']['initial_hash'] for x in v})==1 and len({x['metadata']['paired_data_digest'] for x in v})==1
    c.log('training_complete',models=18)

def make_banks(parent,out,small=False):
    out.mkdir(parents=True,exist_ok=False)
    for gi,g in enumerate(c.CONFIG['conditional_geometries']):
        if small:ids=np.arange(2)
        elif g['name']=='H48':ids=np.arange(2048)
        else:ids=np.concatenate([np.flatnonzero(parent.chain_ids==i)[::8] for i in range(16)])
        previous=None
        for k in g['ks']:
            c.check();b=d.conditional_bank(parent,ids,c.CONFIG['role_seeds']['evaluation'],gi,g['width'],g['kind'],k);d.validate_bank(b)
            if previous is not None:assert np.array_equal(previous['queries'],b['queries']) and np.array_equal(previous['labels'],b['labels'])
            previous=b;c.save(out/f'{g["name"]}_k{k}.npz',**b)
    c.write(out/'complete.json',dict(status='complete',banks=37,software_scratch=small,files={p.name:c.sha(p) for p in out.glob('*.npz')}))

def evaluate(protocol):
    qa=c.read(c.OUT/'reference/complete.json');assert qa['status']=='passed'
    for name,sha in qa['files'].items():assert c.sha(c.OUT/'reference'/name)==sha
    parent=load_parent_split(c.OUT/'reference/fresh_l1024.npz','test_target');make_banks(parent,c.OUT/'banks')
    ids=np.concatenate([np.flatnonzero(parent.chain_ids==i)[::8] for i in range(16)])
    layouts=c.load(c.OUT/'design/low_layouts.npz')
    padding={k:d.padding_banks(parent,ids,k) for k in [2,32,512]}
    for k,banks in padding.items():
        for v,b in enumerate(banks):c.save(c.OUT/'padding_banks'/f'k{k}_v{v}.npz',**b)
    for seed,arm in protocol['cell_order']:
        c.check();ck=c.OUT/'training'/f's{seed}_{arm}'/'final.pt';old=c.sha(ck);model,payload=load_scale_model(ck,torch.device('cuda'));del payload
        out=c.OUT/'evaluation'/f's{seed}_{arm}';out.mkdir(parents=True,exist_ok=False);number=0
        torch.set_float32_matmul_precision('highest')
        for g in c.CONFIG['conditional_geometries']:
            for k in g['ks']:
                b=c.load(c.OUT/'banks'/f'{g["name"]}_k{k}.npz');reps=4 if arm=='I-R' and g['kind']!='continuous' else 1
                for r in range(reps):c.save(out/'conditional'/f'{g["name"]}_k{k}_r{r}.npz',**tr.predict(model,b,arm,r));number+=1
            c.log('conditional_evaluation',seed=seed,arm=arm,geometry=g['name'],files=number)
        for k,banks in padding.items():
            for v,b in enumerate(banks):c.save(out/'padding'/f'k{k}_v{v}.npz',**tr.predict(model,b,arm))
        low=[]
        for r in range(4 if arm=='I-R' else 1):
            bank=d.low_inputs(layouts,arm,r);pred=tr.predict(model,bank,arm,r);p=pred['probability'].reshape(80,2,6)
            if arm!='I-F':assert np.max(np.abs(p[:,0]-p[:,1]))<1e-6
            low.append(p)
        c.save(out/'low_k.npz',probability=np.stack(low,-1),layout_sha256=np.array(c.sha(c.OUT/'design/low_layouts.npz')),native_arm=np.array(arm))
        torch.set_float32_matmul_precision('high');tr.generate(model,out/'generation',seed);torch.set_float32_matmul_precision('highest')
        assert c.sha(ck)==old
        c.write(out/'complete.json',dict(status='complete',predictions=number,padding=9,low_pairs=80,generation_images=128,checkpoint_sha256=old))
        del model;torch.cuda.empty_cache()

def tree_equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:tree_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):tree_equal(x,y)
    else:assert a==b

def cpu_tests():
    from types import SimpleNamespace
    parent=SimpleNamespace(lattice_size=1024,spins=np.where(np.indices((2,1024,1024)).sum(0)%2,1,-1).astype(np.int8),chain_ids=np.array([0,1]))
    for seed in c.SEEDS:
        for step in range(1,65):
            b=[d.training_batch(parent,seed,step,a) for a in c.ARMS];assert len({x['paired_data_hash'] for x in b})==1
            assert all(x['clean'].size==18432 for x in b)
            if b[0]['kind']=='continuous':assert len({x['actual_input_hash'] for x in b})==1
    layouts=d.low_layouts()
    for a in ['I-S','I-R']:
        b=d.low_inputs(layouts,a);z=b['input_coordinates'].reshape(80,2,6,48,48,2);assert np.array_equal(z[:,0],z[:,1])
    for k in [2,32,512]:
        pp=d.padding_banks(parent,np.arange(2),k)
        assert np.array_equal(pp[0]['labels'],pp[1]['labels']) and np.array_equal(pp[0]['t'],pp[1]['t'])
    value=st.low_metrics(np.full((80,2,8),32),np.full((6,3,80,2,6,4),.5));assert np.max(np.abs(value[:,:,0]))<1e-12 and np.max(np.abs(value[:,:,5:]))<1e-12
    # Independent enumeration checks production mapping, entropy and posterior mixing.
    table=np.array([[(1+(2*(j//4)-1)*(2*((j//2)%2)-1)*x+(2*(j//4)-1)*(2*(j%2)-1)*y+(2*((j//2)%2)-1)*(2*(j%2)-1)*z)/8 for j in range(8)] for x,y,z in [(.4,.2,.3),(.1,.1,.05)]])
    p,pv,coarse=st.low_distribution(table)
    for j,visible in enumerate([(1,None),(0,None),(1,1),(1,0),(0,1),(0,0)]):
        eligible=[i for i in range(8) if (i//2)%2==visible[0] and (visible[1] is None or i%2==visible[1])]
        den=table[:,eligible].sum(1);num=table[:,[i for i in eligible if i//4==1]].sum(1)
        assert np.allclose(pv[:,j],den) and np.allclose(p[:,j],num/den) and np.allclose(coarse[j],num.sum()/den.sum())
    oracle=np.stack([p,np.broadcast_to(coarse,p.shape),np.broadcast_to(coarse,p.shape)])[None,:,None,:,:,None]
    metric=st.low_metrics(table[None],oracle)
    assert np.max(np.abs(metric[...,5:]))<1e-12 and np.all(metric[...,0]>0)
    # Unequal visible-pattern probabilities require posterior geometry weights.
    from check_geometry_identification_design import main as design_check
    design_check()
    chain=np.repeat(np.arange(16),16);v=np.zeros((6,3,256));v[:,1]=.01;v[:,2]=.02
    boot=st.bootstrap(v,chain,32,8,'software_test');assert np.allclose(boot,[0,.01,.02])
    assert np.array_equal(boot,st.bootstrap(v,chain,32,8,'software_test'))
    out=c.OUT/'cpu_tests';out.mkdir(parents=True,exist_ok=False)
    c.save(out/'low_layouts.npz',**layouts)
    c.write(out/'complete.json',dict(status='passed_CPU_only',training_data_pair_checks=6*64,old_data_audit_sha256=c.sha(c.OUT/'old_audit_complete.json'),
                                    source_sha256={p.relative_to(ROOT).as_posix():c.sha(p) for p in c.source_files()}))
    print('CPU_TESTS_PASSED',flush=True)

def preflight():
    assert not (c.OUT/'run.lock').exists() and not PREF.exists()
    assert c.read(c.OUT/'old_audit_complete.json')['status']=='passed_local_independent_implementation_audit'
    cpu=c.read(c.OUT/'cpu_tests/complete.json');assert cpu['status']=='passed_CPU_only'
    for path,sha in cpu['source_sha256'].items():assert c.sha(ROOT/path)==sha
    PREF.mkdir(exist_ok=False)
    if not (c.OUT/'budget.json').exists():
        started=time.time();c.write(c.OUT/'budget.json',dict(started=started,deadline=started+43200,hard_seconds=43200,plan_sha256=c.sha(c.PLAN_PATH)),exclusive=True)
    budget=c.read(c.OUT/'budget.json');c.DEADLINE=budget['deadline'];c.check()
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('Deterministic CUDA workspace environment missing')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');c.log('preflight_started',deadline=c.DEADLINE,pid=os.getpid())
    parent=load_parent_split(c.DATA,'train');val=load_parent_split(c.DATA,'val');split=parent.metadata['split_manifest']
    assert not set(split['train'])&set(split['val'])
    from ism_diffusion.ising import BETA_CRITICAL
    assert abs(parent.metadata['beta']-BETA_CRITICAL)<1e-14
    seed=c.SEEDS[0];step=next(s for s in range(1,33) if (lambda x:x['width']==16 and x['kind']=='continuous' and not x['sparse'])(d.schedule(seed,s)))
    b=d.training_batch(parent,seed,step,'I-F');initial=[]
    for arm in c.ARMS:
        m,e,o,meta=tr.fresh(seed);initial.append(meta['initial_hash']);del m,e,o
    assert len(set(initial))==1
    from torch.nn.attention import sdpa_kernel,SDPBackend
    m,e,o,meta=tr.fresh(seed);torch.use_deterministic_algorithms(True)
    with sdpa_kernel(SDPBackend.MATH):
        tr.update(m,e,o,b,1,amp=False);assert gs.model_hash(m)!=meta['initial_hash'];meta['step']=1
        tr.save_state(PREF/'restore_fixture.pt',m,e,o,meta,seed,'I-F','scratch',True)
        tr.update(m,e,o,b,2,amp=False);mm,ee,oo,_=tr.restore(PREF/'restore_fixture.pt',seed,'I-F','scratch')
        tr.update(mm,ee,oo,b,2,amp=False);tree_equal(m.state_dict(),mm.state_dict());tree_equal(e.state_dict(),ee.state_dict());tree_equal(o.state_dict(),oo.state_dict())
    torch.use_deterministic_algorithms(False)
    c.write(PREF/'recovery.json',dict(status='passed',model_ema_optimizer_exact=True,mode='FP32 deterministic math SDPA',initial_hash=initial[0]))
    del m,e,o,mm,ee,oo;torch.cuda.empty_cache();m,e,o,meta=tr.fresh(seed)
    timing=[]
    for arm in c.ARMS:
        for w in c.WIDTHS:
            for kind in ['continuous','train_gap']:
                for sparse in [False,True]:
                    indexes=[i for i in range(1,1601) if (lambda z:z['width']==w and z['kind']==kind and z['sparse']==sparse)(d.schedule(seed,i))][:25]
                    assert len(indexes)==25;times=[];peak=0
                    with (PREF/'timing_train.jsonl').open('a') as f:
                        for j,index in enumerate(indexes):
                            def work():
                                bb=d.training_batch(parent,seed,index,arm);r=tr.update(m,e,o,bb,index)
                                c.increment_digest('',r['paired_data_hash']);f.write(json.dumps(r)+'\n')
                                if (j+1)%20==0:f.flush()
                                return r
                            torch.cuda.reset_peak_memory_stats();seconds,_=measure(work);peak=max(peak,torch.cuda.max_memory_allocated())
                            if j>=5:times.append(seconds)
                    timing.append(dict(arm=arm,width=w,kind=kind,sparse=sparse,raw_seconds=times,mean=float(np.mean(times)),p90=float(np.quantile(times,.9)),conservative=max(float(np.mean(times)),float(np.quantile(times,.9))),peak_bytes=peak))
                    c.write(PREF/'training_timing.json',timing);c.log('preflight_training',arm=arm,width=w,kind=kind,sparse=sparse,seconds=timing[-1]['conservative'])
    # BF16 repeatability recorded separately, not guaranteed across hardware.
    m1,e1,o1,_=tr.fresh(seed);m2,e2,o2,_=tr.fresh(seed);tr.update(m1,e1,o1,b,1);tr.update(m2,e2,o2,b,1)
    error=max(float((x-y).abs().max()) for x,y in zip(m1.parameters(),m2.parameters()));del m1,e1,o1,m2,e2,o2;torch.cuda.empty_cache()
    c.write(PREF/'bf16_replay.json',dict(max_abs_parameter_difference=error,exact=error==0))
    meta['step']=1200;save_s,_=measure(lambda:tr.save_state(PREF/'timing.pt',m,e,o,meta,seed,'I-F','scratch'))
    load_s,loaded=measure(lambda:load_scale_model(PREF/'timing.pt',torch.device('cuda')));del loaded;torch.cuda.empty_cache()
    pred=[];bank_times=[];write_times=[];ids=np.arange(8)
    for gi,g in enumerate(c.CONFIG['conditional_geometries']):
        for k in g['ks']:
            begin=time.perf_counter();bb=d.conditional_bank(val,ids,c.CONFIG['role_seeds']['evaluation'],gi,g['width'],g['kind'],k);bank_times.append((time.perf_counter()-begin)/8);d.validate_bank(bb)
            for arm in c.ARMS:
                tr.predict(e,bb,arm);seconds,p=measure(lambda:tr.predict(e,bb,arm))
                pred.append(dict(geometry=g['name'],width=g['width'],k=k,arm=arm,seconds_per_parent=seconds/8))
                path=PREF/'timing_predictions'/f'{g["name"]}_{k}_{arm}.npz';begin=time.perf_counter();c.save(path,**p);write_times.append((time.perf_counter()-begin)/8)
        c.log('preflight_predictions',geometry=g['name'])
    c.write(PREF/'prediction_timing.json',pred)
    layouts=d.low_layouts();tiny_parent=ParentSplit(val.spins[:16],val.chain_ids[:16],1024,val.metadata)
    begin=time.perf_counter();low=d.low_reference(tiny_parent,layouts,c.check);low_cpu_per_parent=(time.perf_counter()-begin)/16
    c.save(PREF/'low_reference_software.npz',**low)
    # Every native low-K/padding view is exercised with the finite-update scratch EMA.
    for arm in c.ARMS:
        pp=tr.predict(e,d.low_inputs(layouts,arm),arm)['probability'].reshape(80,2,6)
        if arm!='I-F':assert np.max(np.abs(pp[:,0]-pp[:,1]))<1e-6
    for k in [2,32,512]:
        for bb in d.padding_banks(val,np.arange(2),k):
            pp=[tr.predict(e,bb,arm)['probability'] for arm in c.ARMS];assert all(np.array_equal(pp[0],p) for p in pp[1:])
    # All 1098 production file identities, with two OLD validation parents each.
    scratch=PREF/'scratch_pipeline';make_banks(val,scratch/'banks',small=True);files=0
    for seed2 in c.SEEDS:
        for arm in c.ARMS:
            for g in c.CONFIG['conditional_geometries']:
                for k in g['ks']:
                    bb=c.load(scratch/'banks'/f'{g["name"]}_k{k}.npz')
                    for rep in range(4 if arm=='I-R' and g['kind']!='continuous' else 1):
                        pp=tr.predict(e,bb,arm,rep);c.save(scratch/'evaluation'/f's{seed2}_{arm}'/'conditional'/f'{g["name"]}_k{k}_r{rep}.npz',**pp);files+=1
    assert files==1098
    # Independent image sampling equivalence with a content-dependent toy model.
    class Toy(torch.nn.Module):
        def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.zeros(()))
        def forward(self,x,t,coord,valid):
            v=.13*coord[...,0]+.2*t[:,None,None]+.1*(x==1);return torch.stack([-v,v],1)
    toy=Toy().cuda();seeds=[tr.image_seed(c.SEEDS[0],i) for i in range(3)]
    x=tr.sample_images(toy,seeds,width=8,steps=16,amp=False);y=np.concatenate([tr.sample_images(toy,[s],width=8,steps=16,amp=False) for s in seeds]);assert np.array_equal(x,y)
    # Actual model logits have no cross-image coupling; numeric batching tolerance measured.
    bb=d.conditional_bank(val,np.arange(2),2026092621,0,48,'continuous',32)
    bat=tr.predict(e,bb,'I-F',batch_size=2)['probability'];sep=tr.predict(e,bb,'I-F',batch_size=1)['probability'];batch_error=float(np.max(np.abs(bat-sep)));assert batch_error<2e-5
    torch.set_float32_matmul_precision('high');gen_s,gen=measure(lambda:tr.generate(e,PREF/'generation',c.SEEDS[0],16));torch.set_float32_matmul_precision('highest')
    from ism_diffusion.ising import generate_independent_chains
    spins,chain,metadata=generate_independent_chains(lattice_size=16,chain_seeds=[832101,832102],samples_per_chain=4,burn_in_sweeps=2,sweeps_between=1,beta=float(BETA_CRITICAL),adaptation_sweeps=1,pilot_cluster_steps=16,workers=1,backend='numba',return_chain_metadata=True,initial_states=['random','plus'])
    assert spins.shape==(8,16,16);c.save(PREF/'MC_software_only.npz',spins=spins,chain=chain)
    # Full file-reader integration uses finite-update EMA and old validation data.
    # Copied generation/validation fixtures are explicitly NOT18 trained models.
    c.save(scratch/'reference/low_joint_counts.npz',**low)
    c.save(scratch/'reference/generation_reference.npz',**gen,chain=np.zeros(16,dtype=int))
    pad_cache={}
    for arm in c.ARMS:
        for k in [2,32,512]:
            for v,bb in enumerate(d.padding_banks(val,np.arange(2),k)):pad_cache[arm,k,v]=tr.predict(e,bb,arm)
    low_cache={a:np.stack([tr.predict(e,d.low_inputs(layouts,a,r),a,r)['probability'].reshape(80,2,6) for r in range(4 if a=='I-R' else 1)],-1) for a in c.ARMS}
    validation_fixture=np.full((8,2),.4)
    for seed2 in c.SEEDS:
        for arm in c.ARMS:
            out=scratch/'evaluation'/f's{seed2}_{arm}'
            for k in [2,32,512]:
                for v in range(3):c.save(out/'padding'/f'k{k}_v{v}.npz',**pad_cache[arm,k,v])
            c.save(out/'low_k.npz',probability=low_cache[arm])
            shutil.copytree(PREF/'generation',out/'generation')
            c.write(out/'complete.json',dict(status='complete',software_fixture_not_science=True))
            for step2 in [4000,8000,12000]:c.save(scratch/'training'/f's{seed2}_{arm}'/f'validation_{step2}.npz',metrics=validation_fixture)
    collected=an.collect(scratch,generation_images=16)
    assert collected['main'].shape==(6,3,6,2,2) and collected['secondary'].shape==(6,3,31,2,2)
    c.write(scratch/'complete.json',dict(status='reader_integration_passed',conditional_files=files,not_science=True,fixture_model='one finite-update EMA, duplicated identities; old val only'))
    an.analyze(PREF/'statistical_fixture',arrays=an.fixture_arrays(),scratch=True)
    raw_train=6*12000*sum(.75*np.mean([v['conservative'] for v in timing if v['arm']==arm and not v['sparse']])+.25*np.mean([v['conservative'] for v in timing if v['arm']==arm and v['sparse']]) for arm in c.ARMS)
    times={w:max(v['seconds_per_parent'] for v in pred if v['width']==w) for w in [48,96]};inference=0.;parents=0
    for g in c.CONFIG['conditional_geometries']:
        nfiles=6*(2+(1 if g['kind']=='continuous' else 4))*len(g['ks']);parents+=nfiles*g['parents'];inference+=nfiles*g['parents']*times[g['width']]
    padding=18*256*3*(times[48]+2*times[96]);low_forward=34560*times[48]
    validation=6*3*64*(8+8+17)*times[48]
    checkpoint_io=(18*12+18*3)*save_s+(18*12)*load_s
    prediction_io=parents*max(write_times);banks=20224*max(bank_times)
    low_cpu=2048*low_cpu_per_parent
    raw=dict(training=raw_train,predictions=inference,padding=padding,low_forward=low_forward,generation=144*gen_s,
             validation=validation,checkpoint_io=checkpoint_io,prediction_io=prediction_io,bank_preparation=banks,low_reference_cpu=low_cpu)
    # Reference MC+table are concurrent with training; do not double-count low table.
    overlap_reference=5400+low_cpu;raw_serial=sum(raw.values())-low_cpu
    residual=max(0.,overlap_reference-raw_train)
    projected=time.time()-budget['started']+1.2*raw_serial+residual+2700
    gate=dict(status='budget_passed_pending_visual_freeze' if projected<=41400 else 'budget_gate_failed',
              started=budget['started'],deadline=budget['deadline'],predicted_seconds=projected,predicted_hours=projected/3600,
              elapsed_preflight_seconds=time.time()-budget['started'],raw=raw,raw_serial=raw_serial,mc_residual=residual,
              safety_multiplier=1.2,statistics_reserve_seconds=2700,shard16_seconds=gen_s,
              image_rng_batch_independence_toy=True,model_batch_logit_max_error=batch_error,
              full_scratch_prediction_files=files,full_pipeline_scratch_passed=True,new_training_started=False,new_mc_started=False,
              source_sha256={p.relative_to(ROOT).as_posix():c.sha(p) for p in c.source_files()})
    c.write(PREF/'budget_gate.json',gate);c.log('preflight_complete',**gate)

def current_forecast(gate):return gate['predicted_seconds']+time.time()-gate['started']-gate['elapsed_preflight_seconds']

def freeze():
    assert not (c.OUT/'run_protocol.json').exists() and not (c.OUT/'run.lock').exists()
    gate=c.read(PREF/'budget_gate.json');assert gate['status']=='budget_passed_pending_visual_freeze' and gate['full_pipeline_scratch_passed']
    assert c.read(PREF/'visual_review.json')['status']=='passed'
    for p,h in gate['source_sha256'].items():assert c.sha(ROOT/p)==h
    forecast=current_forecast(gate)
    if forecast>41400:raise RuntimeError(f'Budget gate expired: {forecast/3600:.4f}h')
    c.save(c.OUT/'design/low_layouts.npz',**d.low_layouts())
    protocol=dict(study=c.CONFIG['study'],config=c.CONFIG,started=gate['started'],deadline=gate['deadline'],
                  launch_predicted_seconds=forecast,cell_order=[[s,a] for s in c.SEEDS for a in c.ARMS],
                  files={p.relative_to(ROOT).as_posix():c.sha(p) for p in c.source_files()},
                  layout_sha256=c.sha(c.OUT/'design/low_layouts.npz'),old_audit_sha256=c.sha(c.OUT/'old_audit_complete.json'),
                  implementation_details=dict(low_bootstrap_reps=2000,secondary_bootstrap_reps=10000,generation_sensitivity_reps=5000,
                  sampling_categorical='binary_inverse_CDF_same_S0_law_independent_image_uniform_streams',torch_threads=4,
                  paired_stream_root=2026092621,validation_seed=2026092622,preflight='all48 training cells,all37 banks,1098 file identities,8 figures'))
    protocol['protocol_hash']=__import__('hashlib').sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    c.write(c.OUT/'run_protocol.json',protocol,exclusive=True);print(json.dumps(dict(status='frozen_ready',predicted_hours=forecast/3600,deadline=c.DEADLINE)),flush=True)

def run():
    protocol=c.read(c.OUT/'run_protocol.json');c.verify_sources(protocol);c.check();gate=c.read(PREF/'budget_gate.json')
    if current_forecast(gate)>41400:raise RuntimeError('Launch budget gate no longer passes')
    with (c.OUT/'run.lock').open('x') as f:json.dump(dict(pid=os.getpid(),started=time.time(),protocol_hash=protocol['protocol_hash']),f)
    remain=max(1,int(c.DEADLINE-time.time()));mc_log=(c.OUT/'mc.log').open('x')
    mc=subprocess.Popen(['timeout','--signal=TERM','--kill-after=10',str(remain),sys.executable,'-u',str(Path(__file__).resolve()),'--mode','reference'],stdout=mc_log,stderr=subprocess.STDOUT,start_new_session=True)
    c.write(c.OUT/'processes.json',dict(main=os.getpid(),main_pgid=os.getpgrp(),mc_watchdog=mc.pid,mc_pgid=mc.pid,started=time.time(),deadline=c.DEADLINE))
    try:
        train(protocol)
        while mc.poll() is None:c.check();time.sleep(3)
        if mc.returncode:raise RuntimeError('Reference subprocess failed')
        evaluate(protocol);c.log('statistics');result=an.analyze(c.OUT);c.verify_sources(protocol)
        for seed in c.SEEDS:
            for arm in c.ARMS:
                out=c.OUT/'training'/f's{seed}_{arm}';v=c.read(out/'complete.json');assert v['steps']==12000 and c.sha(out/'final.pt')==v['final_sha256']
        count=dict(training_models=len(list((c.OUT/'training').glob('*/complete.json'))),predictions=len(list((c.OUT/'evaluation').glob('*/conditional/*.npz'))),
                   generation_shards=len(list((c.OUT/'evaluation').glob('*/generation/shard_*.npz'))),padding_files=len(list((c.OUT/'evaluation').glob('*/padding/*.npz'))))
        assert count==dict(training_models=18,predictions=1098,generation_shards=144,padding_files=162)
        c.write(c.OUT/'final_summary.json',dict(status='remote_complete_requires_backup_and_visual_review',elapsed_hours=(time.time()-protocol['started'])/3600,
                source_data_checkpoints_unchanged=True,generation_images=2304,**count,**result))
        c.write(c.OUT/'manifest.json',c.manifests(c.OUT));c.log('remote_complete_requires_backup_and_visual_review')
    finally:
        if mc.poll() is None:
            try:os.killpg(mc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        mc_log.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['cpu_tests','preflight','freeze','run','reference'],required=True);args=parser.parse_args()
    if (c.OUT/'budget.json').exists():c.DEADLINE=c.read(c.OUT/'budget.json')['deadline']
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');signal.signal(signal.SIGTERM,stop)
    try:globals()[args.mode]()
    except Exception:
        filename='reference_failure.json' if args.mode=='reference' else ('failure.json' if args.mode=='run' else f'{args.mode}_failure_{time.time_ns()}.json')
        c.write(c.OUT/filename,dict(mode=args.mode,time=time.time(),traceback=traceback.format_exc()));raise
