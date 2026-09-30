"""Frozen-weight development diagnostics and independent generation confirmation."""
from __future__ import annotations
import argparse, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
import numpy as np
import torch
from ism_diffusion import geometry_study as gs
from ism_diffusion.scale_data import load_parent_split,pack_spins
from ism_diffusion.scale_evaluation import load_scale_model,open_energy_density
from ism_diffusion.scale_diffusion import CoordinateAbsorbingDiffusion
from adaptive_repair_evaluation import confirmation_batch
from analyze_fixed_geometry_joint_20260923 import mc_weights
import evaluate_study as ev

OUT=ROOT/'artifacts/generation_bridge_20260924'
OLD=ROOT/'artifacts/mask_query_factorial_20260924'
DOC=ROOT/'docs/research_reboot_20260921/GENERATION_BRIDGE_PROTOCOL_20260924_ZH.md'
SEEDS=list(range(91001,91007)); ARMS=['R00','R10','R11']
MCSEED=2026092441; DEADLINE=float('inf')
GENSAMPLES=224
NSHARDS=GENSAMPLES//16

def check():
    if time.time()>=DEADLINE: raise TimeoutError('Bridge hard deadline; do not restart automatically')

def timed_out(signum,frame):
    raise TimeoutError('External deadline/termination signal; preserved partial artifacts')

def save(path,**data):
    path.parent.mkdir(parents=True,exist_ok=True);ev.atomic_npz(path,**data)

def log(stage,**data):
    row=dict(stage=stage,time=time.time(),**data);gs.atomic_json(OUT/'status.json',row)
    print(json.dumps(row),flush=True)

def checkpoint(s,a): return OLD/'training'/f's{s}_{a}'/'final.pt'

def definitions():
    result=[]
    for w in [48,96,128]:
        for k in sorted(set([0,1,2,8,32,128,512,2048,w*w//4,w*w//2,3*w*w//4])):
            result.append(dict(name=f'w{w}_k{k}',width=w,k=k))
    return result

def make_bank(parent,w,k,n=64):
    assert 0<=k<w*w-64 and n%8==0
    ids=np.array([np.flatnonzero(parent.chain_ids==i%8)[(i//8)*len(np.flatnonzero(parent.chain_ids==i%8))//(n//8)] for i in range(n)])
    random=np.random.default_rng(2026092442+w)
    origin=random.integers(parent.lattice_size,size=(n,2))
    axis=np.arange(w);xx=(origin[:,0,None]+axis)%parent.lattice_size;yy=(origin[:,1,None]+axis)%parent.lattice_size
    clean=(parent.spins[ids[:,None,None],xx[:,:,None],yy[:,None,:]]>0).astype(np.int64).reshape(n,-1)
    grid=np.stack(np.meshgrid(axis,axis,indexing='ij'),-1).astype(np.float32)
    noisy=np.full_like(clean,2);queries=[];dist=[]
    from scipy.spatial import cKDTree
    for i in range(n):
        order=random.permutation(w*w);q=order[:64];v=order[64:64+k]
        noisy[i,v]=clean[i,v];queries.append(q)
        dist.append(cKDTree(grid.reshape(-1,2)[v]).query(grid.reshape(-1,2)[q])[0] if k else np.full(64,-1.))
    queries=np.array(queries);labels=clean[np.arange(n)[:,None],queries]
    assert np.all(noisy[np.arange(n)[:,None],queries]==2)
    assert np.all((noisy!=2).sum(1)==k)
    return dict(noisy=noisy.reshape(n,w,w),coords=np.broadcast_to(grid,(n,w,w,2)).copy(),
                t=np.full(n,1-k/(w*w),np.float32),queries=queries,labels=labels,
                parent=ids,chain=parent.chain_ids[ids],origin=origin,distance=np.array(dist))

@torch.inference_mode()
def predictions(model,bank,clock='natural',limit=None):
    n=len(bank['t']) if limit is None else limit;result=[]
    for start in range(0,n,2):
        check();sl=slice(start,min(start+2,n))
        t=bank['t'][sl] if clock=='natural' else np.full(len(bank['t'][sl]),.5,np.float32)
        logits=model(torch.as_tensor(bank['noisy'][sl],device='cuda'),torch.as_tensor(t,device='cuda'),torch.as_tensor(bank['coords'][sl],device='cuda'))
        p=logits.float().softmax(1)[:,1].flatten(1).gather(1,torch.as_tensor(bank['queries'][sl],device='cuda')).cpu().numpy()
        assert np.isfinite(p).all();result.append(p)
    return np.concatenate(result)

def reference():
    from ism_diffusion.ising import generate_independent_chains,BETA_CRITICAL,magnetization,energy_density
    from ism_diffusion.diagnostics import integrated_autocorrelation_time,split_rhat
    out=OUT/'reference';out.mkdir(exist_ok=False);began=time.time()
    seeds=[int(s.generate_state(1,dtype=np.uint32)[0]) for s in np.random.SeedSequence(MCSEED).spawn(8)]
    args=dict(lattice_size=1024,chain_seeds=seeds,samples_per_chain=256,burn_in_sweeps=40,sweeps_between=4,
        beta=float(BETA_CRITICAL),adaptation_sweeps=3,pilot_cluster_steps=128,workers=4,backend='numba',
        return_chain_metadata=True,initial_states=['random']*4+['plus']*2+['minus']*2)
    gs.atomic_json(out/'protocol.json',dict(args,seed=MCSEED,role='new evaluation only'))
    spins,chain,meta=generate_independent_chains(**args)
    m=magnetization(spins);traces=dict(energy=energy_density(spins),m=m,abs_m=np.abs(m),m2=m*m)
    metadata=dict(lattice_size=1024,seed=MCSEED,beta=float(BETA_CRITICAL),chain_metadata=meta,chain_seeds=seeds)
    save(out/'fresh_l1024.npz',test_target_packed=pack_spins(spins),test_target_chain_id=chain,metadata=np.array(json.dumps(metadata)))
    from ism_diffusion.scale_data import ParentSplit
    parent=ParentSplit(spins,chain,1024,metadata);parts=[];ids=[];origins=[];energy=[]
    for start in range(0,len(spins),16):
        check();b=confirmation_batch(parent,16,128,'continuous',[1],2026092443,start)
        crop=(2*b['clean']-1).astype(np.int8);parts.append(ev.physical_axis_statistics(crop,b['axes']))
        ids.extend(b['parent']);origins.extend(b['origin']);energy.extend(open_energy_density(crop))
    stats=ev.merge_statistics(parts);order=np.argsort(ids)
    assert np.array_equal(np.array(ids)[order],np.arange(len(spins)))
    stats={k:v[order] for k,v in stats.items()};stats['energy']=np.array(energy)[order]
    save(out/'generation_reference.npz',**stats,parent=np.arange(len(spins)),chain=chain,origin=np.array(origins)[order])
    traces['G49_64']=(stats['pair_sum'][:,49:65]/stats['pair_count'][:,49:65]).mean(1)
    diag={k:dict(split_rhat=float(split_rhat([v[chain==i] for i in range(8)])),
                 chains=[integrated_autocorrelation_time(v[chain==i]) for i in range(8)]) for k,v in traces.items()}
    passed=all(np.isfinite(diag[k]['split_rhat']) and diag[k]['split_rhat']<=1.1 and all(c['ess']>=16 for c in diag[k]['chains']) for k in ['energy','abs_m','m2','G49_64'])
    save(out/'chain_traces.npz',chain=chain,**traces)
    gs.atomic_json(out/'complete.json',dict(status='passed' if passed else 'qa_failed',parents=len(spins),seed=MCSEED,
        diagnostics=diag,elapsed_seconds=time.time()-began,files={p.name:gs.file_hash(p) for p in out.glob('*.npz')}))

def preflight():
    OUT.mkdir(exist_ok=False);started=time.time()
    gs.atomic_json(OUT/'budget.json',dict(started=started,deadline=started+12*3600))
    scratch=OUT/'preflight';scratch.mkdir()
    parent=load_parent_split(OLD/'reference/fresh_l1024.npz','test_target')
    model,_=load_scale_model(checkpoint(91001,'R00'),torch.device('cuda'));model.eval();model.requires_grad_(False)
    condition_seconds={}
    for w in [48,96,128]:
        b=make_bank(parent,w,32);bb=make_bank(parent,w,512)
        assert np.array_equal(b['labels'],bb['labels']) and np.array_equal(b['queries'],bb['queries'])
        assert np.all((b['noisy']==2)|(b['noisy']==bb['noisy']))
        logits=predictions(model,b,limit=2)
        torch.cuda.synchronize();t=time.time();pp=predictions(model,b,limit=8);torch.cuda.synchronize()
        condition_seconds[w]=(time.time()-t)/8
        assert np.allclose(pp[:2],logits,atol=2e-5)
    b=confirmation_batch(parent,16,128,'continuous',[1],2026092444,0)
    coordinates=torch.tensor(b['coords']['A'],device='cuda');valid=torch.ones((16,128,128),device='cuda',dtype=torch.bool)
    torch.set_float32_matmul_precision('high');torch.cuda.synchronize();t=time.time()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        tokens=CoordinateAbsorbingDiffusion().sample(model,coordinates,valid,steps=32,
            generator=torch.Generator(device='cuda').manual_seed(2026092445))
    torch.cuda.synchronize();sampling_seconds=time.time()-t
    assert ((tokens==0)|(tokens==1)).all()
    save(scratch/'sampler_smoke.npz',spins=(tokens.cpu().numpy()*2-1).astype(np.int8))
    fake=np.ones((2,8,8),np.int8);axes=[np.broadcast_to(np.arange(8),(2,8))]*2
    stat=ev.physical_axis_statistics(fake,axes);assert np.all(stat['pair_sum']==stat['pair_count'])
    truth={**stat,'chain':np.array([0,1]),'energy':np.zeros(2)}
    toy={k:np.repeat(v[:1],256,axis=0) for k,v in truth.items() if k!='chain'}
    # gen_values uses physical bands through 64: construct an exact full-width fixture.
    truth['pair_sum']=truth['pair_count']=np.ones((2,128))
    toy['pair_sum']=toy['pair_count']=np.ones((256,128))
    assert np.allclose(gen_values([[toy]*6]*3,truth),0)
    seconds=sum(condition_seconds[d['width']]*64*(2 if d['k'] in [2,32,512] else 1) for d in definitions())*18
    predicted=(time.time()-started+1.25*(sampling_seconds*8*288+seconds)+7200)/3600
    result=dict(status='passed' if predicted<=11.5 else 'budget_failed',seconds_per_condition_sample=condition_seconds,
        sampling_16_images_32_steps_seconds=sampling_seconds,predicted_hours=predicted,
        query_labels_hidden_valid=True,nested_mask_test=True,repeated_prediction_test=True,statistics_test=True,
        device=torch.cuda.get_device_name(),formal_samples_per_model=256)
    gs.atomic_json(scratch/'complete.json',result);print(json.dumps(result),flush=True)

def diagnostics():
    parent=load_parent_split(OLD/'reference/fresh_l1024.npz','test_target')
    banks=OUT/'development_banks';banks.mkdir()
    for d in definitions():save(banks/(d['name']+'.npz'),**make_bank(parent,d['width'],d['k']))
    for s in SEEDS:
        for a in ARMS:
            path=checkpoint(s,a);before=gs.file_hash(path)
            model,_=load_scale_model(path,torch.device('cuda'));model.eval();model.requires_grad_(False)
            torch.set_float32_matmul_precision('highest');dest=OUT/'diagnostics'/f's{s}_{a}';dest.mkdir(parents=True)
            cells=0
            for d in definitions():
                with np.load(banks/(d['name']+'.npz')) as z:b=dict(z)
                for clock in ['natural']+(['common'] if d['k'] in [2,32,512] else []):
                    pp=predictions(model,b,clock).astype(float);y=b['labels']
                    ce=-(y*np.log(np.clip(pp,1e-12,1))+(1-y)*np.log(np.clip(1-pp,1e-12,1)))
                    tt=b['t'] if clock=='natural' else np.full_like(b['t'],.5)
                    save(dest/(d['name']+'_'+clock+'.npz'),probability=pp,ce=ce,brier=(pp-y)**2,
                        parent=b['parent'],chain=b['chain'],input_hash=np.array(gs.array_hash(b['noisy'],b['coords'],tt,b['queries'])))
                    cells+=1
                log('condition_map',seed=s,arm=a,condition=d['name'])
            assert cells==42 and gs.file_hash(path)==before
            gs.atomic_json(dest/'complete.json',dict(status='complete',cells=cells,checkpoint_unchanged=True,sha256=before))
            del model;torch.cuda.empty_cache()

def gen_values(g,ref,mcw=None,shardw=None):
    rw=np.ones(len(ref['chain'])) if mcw is None else mcw
    rg=rw@ref['pair_sum']/np.maximum(rw@ref['pair_count'],1)
    vals=np.empty((3,6,7))
    for a in range(3):
        for s in range(6):
            data=g[a][s];w=np.ones(len(data['m'])) if shardw is None else shardw[s]
            curve=w@data['pair_sum']/np.maximum(w@data['pair_count'],1)
            for i,(lo,hi) in enumerate([(1,8),(9,48),(49,64)]):
                ix=slice(lo,hi+1);vals[a,s,i]=np.sqrt(np.mean((curve[ix]-rg[ix])**2))/np.sqrt(np.mean(rg[ix]**2))
            for i,k in enumerate(['m','m2','abs_m','energy'],3):
                vals[a,s,i]=np.average(data[k],weights=w)-np.average(ref[k],weights=rw)
    return vals

def analyze():
    out=OUT/'analysis';out.mkdir()
    ref=dict(np.load(OUT/'reference/generation_reference.npz'));g=[]
    for a in ARMS:
        rows=[]
        for s in SEEDS:
            folder=OUT/'generation'/f's{s}_{a}';data=dict(np.load(folder/'statistics.npz'))
            r={k[6:]:v for k,v in data.items() if k.startswith('model_') and k!='model_G'}
            files=sorted(folder.glob('shard_*.npz'));assert len(files)==NSHARDS
            r['energy']=np.concatenate([open_energy_density(np.load(f)['spins']) for f in files]);rows.append(r)
        g.append(rows)
    point=gen_values(g,ref);save(out/'generation_point.npz',values=point,arms=np.array(ARMS))
    uncertainty={};draw_store={}
    for mode,block,reps in [('joint',8,2000),('joint',4,1000),('joint',16,1000),('fixed_models',8,1000),('seed_only',8,1000),('mc_only',8,1000),('sampling_only',8,1000)]:
        random=np.random.default_rng(2026092450+len(uncertainty));boot=[]
        for i in range(reps):
            check()
            rw=mc_weights(ref['chain'],block,random) if mode in ['joint','fixed_models','mc_only'] else None
            sw=np.array([np.repeat(np.bincount(random.integers(NSHARDS,size=NSHARDS),minlength=NSHARDS),16) for _ in SEEDS]) if mode in ['joint','fixed_models','sampling_only'] else None
            idx=random.integers(6,size=6) if mode in ['joint','seed_only'] else np.arange(6)
            boot.append(gen_values(g,ref,rw,sw)[:,idx].mean(1))
        boot=np.array(boot);assert np.isfinite(boot).all();key=f'{mode}_block{block}'
        save(out/(key+'.npz'),values=boot)
        cc={}
        for label,u,v in [('R10-R00',1,0),('R11-R10',2,1)]:
            delta=boot[:,u]-boot[:,v];estimate=point.mean(1)[u]-point.mean(1)[v]
            cc[label]=dict(estimate=estimate.tolist(),ci95=np.quantile(delta,[.025,.975],axis=0).tolist(),ci9875=np.quantile(delta,[.00625,.99375],axis=0).tolist())
        uncertainty[key]=cc;draw_store[key]=boot
        log('statistics',mode=mode,block=block)
    gs.atomic_json(out/'generation_uncertainty.json',uncertainty)
    names=[];allvals=[];distrows=[]
    for d in definitions():
        bank=dict(np.load(OUT/'development_banks'/(d['name']+'.npz')))
        for clock in ['natural']+(['common'] if d['k'] in [2,32,512] else []):
            name=d['name']+'_'+clock;v=[];hashes=[]
            for a in ARMS:
                av=[]
                for s in SEEDS:
                    z=dict(np.load(OUT/'diagnostics'/f's{s}_{a}'/(name+'.npz')))
                    hashes.append(str(z['input_hash']));av.append(np.stack([z['ce'].mean(1),z['brier'].mean(1)],-1))
                    for label,sel in [('near',(bank['distance']>=0)&(bank['distance']<=2)),('middle',(bank['distance']>2)&(bank['distance']<=8)),('far',bank['distance']>8)]:
                        if sel.any():distrows.append(dict(condition=name,arm=a,seed=s,band=label,queries=int(sel.sum()),ce=float(z['ce'][sel].mean()),brier=float(z['brier'][sel].mean())))
                v.append(av)
            assert len(set(hashes))==1;allvals.append(v);names.append(name)
    x=np.array(allvals);assert x.shape==(42,3,6,64,2);save(out/'conditional_point.npz',values=x,conditions=np.array(names))
    ev.write_csv(out/'distance_descriptive.csv',distrows)
    chain=bank['chain'];rng=np.random.default_rng(2026092458);boot=[]
    for i in range(2000):
        check();w=mc_weights(chain,2,rng);idx=rng.integers(6,size=6)
        val=np.einsum('caspd,p->casd',x,w)/w.sum();boot.append(val[:,:,idx].mean(2))
    boot=np.array(boot);save(out/'conditional_bootstrap.npz',values=boot,conditions=np.array(names))
    rows=[]
    for label,u,v in [('R10-R00',1,0),('R11-R10',2,1)]:
        dd=boot[:,:,u]-boot[:,:,v];ci=np.quantile(dd,[.025,.975],axis=0)
        pt=x.mean((2,3));delta=pt[:,u]-pt[:,v]
        for i,n in enumerate(names):
            for j,m in enumerate(['CE','Brier']):rows.append(dict(condition=n,contrast=label,metric=m,estimate=float(delta[i,j]),lo95=float(ci[0,i,j]),hi95=float(ci[1,i,j]),scope='exploratory_pointwise'))
    ev.write_csv(out/'conditional_contrasts.csv',rows)
    plot(out,point,x,names,uncertainty,g,ref)
    return uncertainty['joint_block8']

def plot(out,point,x,names,u,g,ref):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.linewidth':1.8,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42})
    colors=['#767676','#0F4D92','#42949E'];fig,axs=plt.subplots(1,3,figsize=(15,4.5))
    for ax,w in zip(axs,[48,96,128]):
        defs=[d for d in definitions() if d['width']==w];ks=[d['k'] for d in defs];ii=[names.index(d['name']+'_natural') for d in defs]
        for a in range(3):ax.plot(ks,x[ii,a].mean((1,2))[:,0],'-o',color=colors[a],label=ARMS[a])
        ax.set(xscale='symlog',xlabel=f'Visible K (W={w})',ylabel='MC-context CE (nats)');ax.legend()
    fig.tight_layout(pad=2)
    for ext in ['png','pdf']:fig.savefig(out/f'coverage_map.{ext}',dpi=300)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for s in range(6):axs[0].plot(range(3),point[:,s,2],color='#CFCECE',lw=1)
    for a in range(3):axs[0].scatter(np.full(6,a),point[a,:,2],color=colors[a])
    axs[0].set(xticks=range(3),xticklabels=ARMS,ylabel='W128 G49-64 NRMSE')
    for i,label in enumerate(['R10-R00','R11-R10']):
        d=u['joint_block8'][label];v=d['estimate'][2];lo,hi=np.array(d['ci9875'])[:,2]
        axs[1].errorbar(v,i,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',capsize=4)
    axs[1].axvline(0,color='#767676');axs[1].set(yticks=[0,1],yticklabels=['R10-R00','R11-R10'],xlabel='Difference; 98.75% interval')
    fig.tight_layout(pad=2)
    for ext in ['png','pdf']:fig.savefig(out/f'generation_confirmation.{ext}',dpi=300)
    plt.close(fig)

def run():
    global DEADLINE
    budget=json.loads((OUT/'budget.json').read_text());DEADLINE=budget['deadline'];check()
    review=json.loads((OUT/'preflight/budget_review.json').read_text())
    assert review['status']=='passed' and review['samples_per_model']==GENSAMPLES
    lock=os.open(OUT/'run.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(lock,str(os.getpid()).encode());os.close(lock)
    paths=sorted((ROOT/'ism_diffusion').glob('*.py'))+sorted((ROOT/'scripts/research20260921').glob('*.py'))+[DOC]
    sources={str(p.relative_to(ROOT)):gs.file_hash(p) for p in paths}
    weights={str(checkpoint(s,a).relative_to(ROOT)):gs.file_hash(checkpoint(s,a)) for s in SEEDS for a in ARMS}
    assert all(h==json.loads((ROOT/p).with_name('complete.json').read_text())['sha256'] for p,h in weights.items())
    protocol=dict(**budget,arms=ARMS,seeds=SEEDS,mc_seed=MCSEED,generation_samples_per_model=GENSAMPLES,
        source_hashes=sources,checkpoint_hashes=weights,development_reference_sha256=gs.file_hash(OLD/'reference/fresh_l1024.npz'))
    gs.atomic_json(OUT/'run_protocol.json',protocol)
    snapshot=OUT/'source';snapshot.mkdir()
    import shutil
    for p in paths:
        dest=snapshot/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    mc_log=(OUT/'mc.log').open('w');mc=subprocess.Popen([sys.executable,str(Path(__file__)),'--mode','reference'],stdout=mc_log,stderr=subprocess.STDOUT,start_new_session=True)
    gs.atomic_json(OUT/'processes.json',dict(main=os.getpid(),mc=mc.pid,deadline=DEADLINE))
    log('started',pid=os.getpid(),mc_pid=mc.pid)
    try:
        diagnostics();log('diagnostics_complete',models=18)
        while mc.poll() is None:check();time.sleep(5)
        assert mc.returncode==0 and json.loads((OUT/'reference/complete.json').read_text())['status']=='passed','MC QA/worker failure'
        parent=load_parent_split(OUT/'reference/fresh_l1024.npz','test_target')
        torch.set_float32_matmul_precision('high');ev.make_batch=confirmation_batch
        for s in SEEDS:
            for a in ARMS:
                check();path=checkpoint(s,a);before=gs.file_hash(path)
                model,_=load_scale_model(path,torch.device('cuda'));model.eval();model.requires_grad_(False)
                log('generation',seed=s,arm=a)
                d=dict(name='continuous128',width=128,kind='continuous',gaps=[1],samples=GENSAMPLES)
                ev.generate_and_score(model,parent,'A',s+2026092500,d,OUT/'generation'/f's{s}_{a}',DEADLINE,16)
                assert gs.file_hash(path)==before;del model;torch.cuda.empty_cache()
        result=analyze()
        assert all(gs.file_hash(ROOT/p)==h for p,h in sources.items())
        assert all(gs.file_hash(ROOT/p)==h for p,h in weights.items())
        assert len(list((OUT/'diagnostics').glob('*/complete.json')))==18
        assert len(list((OUT/'generation').glob('*/complete.json')))==18
        assert len(list((OUT/'generation').glob('*/shard_*.npz')))==18*NSHARDS
        gs.atomic_json(OUT/'final_summary.json',dict(status='remote_complete_requires_backup_and_visual_review',models=18,
            diagnostic_cells=756,generation_samples=18*GENSAMPLES,shards=18*NSHARDS,checkpoint_unchanged=True,source_unchanged=True,
            primary=result,elapsed_hours=(time.time()-budget['started'])/3600,
            limitations=['fixed checkpoints, not new training replication','development MC reused for condition map only','pointwise conditional intervals exploratory','single-source uncertainty intervals are not additive variance fractions']))
        gs.atomic_json(OUT/'manifest.json',dict(files=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=gs.file_hash(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['queue.log','mc.log','status.json','manifest.json','run.lock']]))
        log('remote_complete_requires_backup_and_visual_review')
    finally:
        if mc.poll() is None:
            os.killpg(mc.pid,signal.SIGTERM)
            try:mc.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(mc.pid,signal.SIGKILL);mc.wait()
        mc_log.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['preflight','reference','run'],required=True);args=p.parse_args()
    torch.set_num_threads(2)
    signal.signal(signal.SIGTERM,timed_out)
    try:
        if args.mode!='preflight':DEADLINE=json.loads((OUT/'budget.json').read_text())['deadline']
        {'preflight':preflight,'reference':reference,'run':run}[args.mode]()
    except BaseException as e:
        name='budget_stop.json' if isinstance(e,TimeoutError) else 'failure.json'
        dest=OUT/('reference_'+name if args.mode=='reference' else name)
        gs.atomic_json(dest,dict(mode=args.mode,error=repr(e),traceback=traceback.format_exc(),time=time.time()))
        raise
