"""Fresh five-arm geometry factorial; explicit gates, single run, hard clock."""
from __future__ import annotations
import argparse,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
import numpy as np
import torch
from ism_diffusion import geometry_study as gs
from ism_diffusion.scale_data import load_parent_split,pack_spins,ParentSplit
from ism_diffusion.scale_evaluation import load_scale_model,open_energy_density
from ism_diffusion.scale_diffusion import CoordinateAbsorbingDiffusion
from adaptive_repair_evaluation import confirmation_batch
import evaluate_study as ev
import core_geometry_factorial_design as d
import core_geometry_factorial_data as da
import core_geometry_factorial_training as tr
import core_geometry_factorial_analysis as an

OUT=ROOT/'artifacts/core_geometry_factorial_20260925'
DOC=ROOT/'docs/research_reboot_20260921/CORE_GEOMETRY_FACTORIAL_PROTOCOL_20260925_ZH.md'
AMENDMENT=ROOT/'docs/research_reboot_20260921/CORE_GEOMETRY_FACTORIAL_BUDGET_V2_20260925_ZH.md'
PREFLIGHT=OUT/'preflight_authorized12h_v3_envfix'
DATA=ROOT/'data/level1/parents_l1024.npz'
DEADLINE=float('inf')

def read(p): return json.loads(p.read_text())
def save(p,**a):p.parent.mkdir(parents=True,exist_ok=True);ev.atomic_npz(p,**a)
def check():
    if time.time()>=DEADLINE:raise TimeoutError('Authorized v2 twelve-hour deadline')
def interrupt(*_):raise TimeoutError('Watchdog/termination signal')
def log(stage,**kw):
    row=dict(stage=stage,time=time.time(),**kw);gs.atomic_json(OUT/'status.json',row);print(json.dumps(row),flush=True)
def digest(old,value):return hashlib.sha256((old+value).encode()).hexdigest()
def validation_banks(parent):
    ids=np.concatenate([np.flatnonzero(parent.chain_ids==c)[np.linspace(0,len(np.flatnonzero(parent.chain_ids==c))-1,32,dtype=int)] for c in np.unique(parent.chain_ids)])
    assert len(ids)==64
    return [da.conditional_bank(parent,ids,2026092522,0,48,'continuous',k) for k in [1152,115]]
def verify_sources(protocol):
    for path,sha in protocol['files'].items():
        if gs.file_hash(ROOT/path)!=sha:raise RuntimeError('Frozen file changed: '+path)
def source_files():
    return sorted([p for folder in ['ism_diffusion','scripts/research20260921'] for p in (ROOT/folder).glob('*.py')]+[DOC,AMENDMENT,DATA])

def reference():
    from ism_diffusion.ising import generate_independent_chains,BETA_CRITICAL,magnetization,energy_density
    from ism_diffusion.diagnostics import integrated_autocorrelation_time,split_rhat
    out=OUT/'reference';out.mkdir(exist_ok=False);started=time.time()
    seeds=[int(s.generate_state(1,dtype=np.uint32)[0]) for s in np.random.SeedSequence(d.MC_SEED).spawn(8)]
    args=dict(lattice_size=1024,chain_seeds=seeds,samples_per_chain=128,burn_in_sweeps=40,sweeps_between=4,
        beta=float(BETA_CRITICAL),adaptation_sweeps=3,pilot_cluster_steps=128,workers=4,backend='numba',
        return_chain_metadata=True,initial_states=['random']*4+['plus']*2+['minus']*2)
    gs.atomic_json(out/'protocol.json',dict(args,seed=d.MC_SEED,role='independent evaluation only'))
    spins,chain,meta=generate_independent_chains(**args);check()
    m=magnetization(spins);traces=dict(energy=energy_density(spins),m=m,abs_m=np.abs(m),m2=m*m)
    metadata=dict(lattice_size=1024,beta=float(BETA_CRITICAL),seed=d.MC_SEED,chain_metadata=meta,chain_seeds=seeds)
    save(out/'fresh_l1024.npz',test_target_packed=pack_spins(spins),test_target_chain_id=chain,metadata=np.array(json.dumps(metadata)))
    parent=ParentSplit(spins,chain,1024,metadata);parts=[];ids=[];origins=[];energies=[]
    for start in range(0,len(spins),16):
        check();b=confirmation_batch(parent,16,96,'continuous',[1],2026092513,start)
        crop=(2*b['clean']-1).astype(np.int8);parts.append(ev.physical_axis_statistics(crop,b['axes']))
        ids.extend(b['parent']);origins.extend(b['origin']);energies.extend(open_energy_density(crop))
    stats=ev.merge_statistics(parts);order=np.argsort(ids)
    assert np.array_equal(np.asarray(ids)[order],np.arange(1024))
    stats={k:v[order] for k,v in stats.items()};stats['energy']=np.array(energies)[order]
    save(out/'generation_reference.npz',**stats,parent=np.arange(1024),chain=chain,origin=np.array(origins)[order])
    traces['G25_48']=(stats['pair_sum'][:,25:49]/stats['pair_count'][:,25:49]).mean(1)
    diag={k:dict(split_rhat=float(split_rhat([v[chain==c] for c in range(8)])),
        chains=[integrated_autocorrelation_time(v[chain==c]) for c in range(8)]) for k,v in traces.items()}
    passed=all(np.isfinite(diag[k]['split_rhat']) and diag[k]['split_rhat']<=1.1 and all(x['ess']>=16 for x in diag[k]['chains']) for k in ['energy','abs_m','m2','G25_48'])
    save(out/'chain_traces.npz',chain=chain,**traces)
    gs.atomic_json(out/'complete.json',dict(status='passed' if passed else 'qa_failed',parents=1024,seed=d.MC_SEED,
        elapsed_seconds=time.time()-started,diagnostics=diag,files={p.name:gs.file_hash(p) for p in out.glob('*.npz')}))

def make_banks(parent,out):
    out.mkdir(exist_ok=False);ids=np.concatenate([np.flatnonzero(parent.chain_ids==c)[::8] for c in range(8)])
    assert len(ids)==128 and all(np.sum(parent.chain_ids[ids]==c)==16 for c in range(8))
    from scipy.spatial import cKDTree
    for g,(name,w,kind) in enumerate(d.GEOMETRIES):
        prior=None
        for ki,k in enumerate(d.evaluation_ks(w)):
            check();b=da.conditional_bank(parent,ids,2026092512,g,w,kind,k)
            if prior is not None:
                assert np.array_equal(prior['labels'],b['labels']) and np.array_equal(prior['queries'],b['queries'])
            if ki==0:prior=b
            distances=[]
            for i in range(128):
                xy=b['true_coords'][i].reshape(-1,2)
                distances.append(cKDTree(xy[b['evidence'][i]]).query(xy[b['queries'][i]])[0])
            b['distance']=np.array(distances)
            save(out/f'g{g}_k{ki}.npz',**b)
    gs.atomic_json(out/'complete.json',dict(status='complete',banks=30,parents=128,
        files={p.name:gs.file_hash(p) for p in out.glob('*.npz')}))

def train_all(protocol):
    parent=load_parent_split(DATA,'train');val=load_parent_split(DATA,'val')
    vb=validation_banks(val);vf=OUT/'validation_banks';vf.mkdir()
    for k,b in enumerate(vb):save(vf/f'k{k}.npz',**b)
    ph=protocol['protocol_hash'];order=protocol['cell_order']
    for stop in range(1000,d.STEPS+1,1000):
        for seed,arm in order:
            check();ref=OUT/'reference/complete.json'
            if ref.exists() and read(ref)['status']!='passed':raise RuntimeError('Independent MC QA failed')
            cell=OUT/'training'/f's{seed}_{arm}';cell.mkdir(parents=True,exist_ok=True)
            if stop==1000:
                m,e,o,meta=tr.fresh_state(seed)
                gs.atomic_json(cell/'initial.json',dict(seed=seed,arm=arm,sha256=meta['initial_hash'],fresh=True))
            else:m,e,o,meta=tr.resume_state(cell/'last.pt',seed,arm,ph)
            assert meta['step']==stop-1000
            log('training',seed=seed,arm=arm,start=meta['step'],target=stop)
            started=time.perf_counter()
            with (cell/'train.jsonl').open('a',buffering=65536) as handle:
                for step in range(meta['step']+1,stop+1):
                    check();batch=da.training_batch(parent,seed,step,arm);r=tr.update(m,e,o,batch,step)
                    meta['paired_data_digest']=digest(meta['paired_data_digest'],r['paired_data_hash'])
                    meta['actual_input_digest']=digest(meta['actual_input_digest'],r['actual_input_hash'])
                    meta['step']=step;handle.write(json.dumps(dict(step=step,time=time.time(),width=batch['width'],kind=batch['kind'],**r))+'\n')
                    if step%20==0:handle.flush()
            meta['elapsed_seconds']+=time.perf_counter()-started
            tr.save_state(cell/'last.pt',m,e,o,meta,seed,arm,ph)
            if stop in [4000,8000,12000]:
                validation=[]
                for b in vb:
                    vp=tr.predictions(e,b,arm,check=check)
                    validation.append([vp[key].mean() for key in ['ce','brier']])
                validation=np.array(validation)
                save(cell/f'validation_{stop}.npz',metrics=validation)
                if stop<12000:
                    tmp=cell/f'ema_{stop}.pt.tmp';torch.save(dict(ema=e.state_dict(),config=dict(model=gs.MODEL),step=stop,seed=seed,arm=arm),tmp)
                    os.replace(tmp,cell/f'ema_{stop}.pt')
                else:
                    tr.save_state(cell/'final.pt',m,e,o,meta,seed,arm,ph,immutable=True)
                    gs.atomic_json(cell/'complete.json',dict(status='complete',steps=stop,metadata=meta,
                        final_sha256=gs.file_hash(cell/'final.pt'),ema_sha256=gs.model_hash(e)))
            gs.atomic_json(cell/'status.json',dict(step=stop,seed=seed,arm=arm,time=time.time()))
            del m,e,o;torch.cuda.empty_cache()
    for seed in d.SEEDS:
        cells=[read(OUT/'training'/f's{seed}_{a}'/'complete.json') for a in d.ARMS]
        assert len({c['metadata']['initial_hash'] for c in cells})==1
        assert cells[3]['metadata']['paired_data_digest']==cells[4]['metadata']['paired_data_digest']
    log('training_complete',models=30)

def evaluate_all(protocol):
    qa=read(OUT/'reference/complete.json');assert qa['status']=='passed'
    for name,sha in qa['files'].items():assert gs.file_hash(OUT/'reference'/name)==sha
    parent=load_parent_split(OUT/'reference/fresh_l1024.npz','test_target')
    make_banks(parent,OUT/'banks')
    for seed,arm in protocol['cell_order']:
        check();ck=OUT/'training'/f's{seed}_{arm}'/'final.pt';before=gs.file_hash(ck)
        model,payload=load_scale_model(ck,torch.device('cuda'));del payload
        out=OUT/'evaluation'/f's{seed}_{arm}';out.mkdir(parents=True,exist_ok=False)
        torch.set_float32_matmul_precision('highest')
        for g in range(6):
            for k in range(5):
                check();b=dict(np.load(OUT/'banks'/f'g{g}_k{k}.npz'))
                pred=tr.predictions(model,b,arm,check=check);save(out/f'g{g}_k{k}.npz',**pred)
            log('conditional_evaluation',seed=seed,arm=arm,geometry=g)
        torch.set_float32_matmul_precision('high')
        definition=dict(name='core_geometry_W96',width=96,kind='continuous',gaps=[1],samples=64)
        ev.generate_and_score(model,parent,'A',seed,definition,out/'generation',DEADLINE,16)
        torch.set_float32_matmul_precision('highest')
        assert gs.file_hash(ck)==before and len(list(out.glob('g*_k*.npz')))==30
        shards=list((out/'generation').glob('shard_*.npz'));assert len(shards)==4
        assert sum(len(np.load(p)['spins']) for p in shards)==64
        gs.atomic_json(out/'complete.json',dict(status='complete',predictions=30,generation_images=64,
            checkpoint_sha256=before,native_coordinate_mode='span_only' if arm=='G11S' else 'physical',
            continuous_generation_native_encodings_identical=True))
        del model;torch.cuda.empty_cache()

def analysis():
    values=np.empty((6,5,6,5,128,2));learning=np.empty((6,5,3,2,2));g=[];distrows=[]
    for s,seed in enumerate(d.SEEDS):
        gg=[]
        for a,arm in enumerate(d.ARMS):
            out=OUT/'evaluation'/f's{seed}_{arm}'
            assert read(out/'complete.json')['status']=='complete'
            for geo in range(6):
                for k in range(5):
                    z=dict(np.load(out/f'g{geo}_k{k}.npz'));values[s,a,geo,k]=np.stack([z['ce'].mean(1),z['brier'].mean(1)],-1)
                    b=dict(np.load(OUT/'banks'/f'g{geo}_k{k}.npz'))
                    expected=da.array_hash(b['noisy'],b['true_coords'],b['t'],b['queries'],b['labels'])
                    assert str(z['common_data_hash'])==expected
                    for label,mask in [('near',b['distance']<=2),('middle',(b['distance']>2)&(b['distance']<=8)),('far',b['distance']>8)]:
                        if mask.any():distrows.append(dict(seed=seed,arm=arm,geometry=geo,condition=k,band=label,
                            queries=int(mask.sum()),ce=float(z['ce'][mask].mean()),brier=float(z['brier'][mask].mean())))
            for j,step in enumerate([4000,8000,12000]):learning[s,a,j]=np.load(OUT/'training'/f's{seed}_{arm}'/f'validation_{step}.npz')['metrics']
            z=dict(np.load(out/'generation/statistics.npz'));one={k[6:]:v for k,v in z.items() if k.startswith('model_') and k!='model_G'}
            one['energy']=np.concatenate([open_energy_density(np.load(p)['spins']) for p in sorted((out/'generation').glob('shard_*.npz'))]);gg.append(one)
        g.append(gg)
    # Raw sampler identities must match within every five-arm seed.
    for seed in d.SEEDS:
        for start in range(0,64,16):
            sources=[np.load(OUT/'evaluation'/f's{seed}_{a}'/'generation'/f'shard_{start:05d}.npz') for a in d.ARMS]
            for key in ['axis_x','axis_y','input_coordinates','parent','chain','origin','seed','sample_start']:
                assert all(np.array_equal(sources[0][key],z[key]) for z in sources[1:])
            for z in sources:z.close()
    ref=dict(np.load(OUT/'reference/generation_reference.npz'));chain=np.load(OUT/'banks/g0_k0.npz')['chain']
    result=an.analyze_arrays(OUT/'analysis',values,chain,g,ref,learning,check)
    ev.write_csv(OUT/'analysis/distance_descriptive.csv',distrows)
    return result

def scratch():
    """Full synthetic-shape pipeline assembly, plus real scratch GPU prediction/shard."""
    dst=OUT/'preflight_pipeline_v3';dst.mkdir(exist_ok=False);check()
    gate=read(PREFLIGHT/'budget_gate.json');assert gate['status']=='budget_gate_passed_pending_full_pipeline'
    parent=load_parent_split(DATA,'val');vb=validation_banks(parent)
    model,payload=load_scale_model(PREFLIGHT/'throughput.pt',torch.device('cuda'))
    for k,b in enumerate(vb):
        save(dst/f'validation_bank{k}.npz',**b)
        for arm in d.ARMS:
            pred=tr.predictions(model,b,arm,check=check);save(dst/f'{arm}_prediction{k}.npz',**pred)
    # Exercise every bank layout using OLD bridge fields, taking 128 per chain.
    # These are explicitly scratch inputs, never the new confirmation reference.
    old=load_parent_split(ROOT/'artifacts/generation_bridge_20260924/reference/fresh_l1024.npz','test_target')
    ids=np.concatenate([np.flatnonzero(old.chain_ids==c)[:128] for c in range(8)])
    old_subset=ParentSplit(old.spins[ids],old.chain_ids[ids],1024,old.metadata)
    make_banks(old_subset,dst/'banks')
    del old,old_subset
    for file in sorted((dst/'banks').glob('g*_k*.npz')):
        b=dict(np.load(file));small={k:(v[:2] if isinstance(v,np.ndarray) and v.ndim>0 and len(v)==128 else v) for k,v in b.items()}
        for seed in d.SEEDS:
            for arm in d.ARMS:
                pred=tr.predictions(model,small,arm,check=check)
                save(dst/'evaluation'/f's{seed}_{arm}'/file.name,**pred)
    assert len(list((dst/'evaluation').glob('*/g*_k*.npz')))==900
    from ism_diffusion.ising import generate_independent_chains,BETA_CRITICAL
    tiny,chain,tiny_meta=generate_independent_chains(lattice_size=16,chain_seeds=[831021,831022],samples_per_chain=4,
        burn_in_sweeps=2,sweeps_between=1,beta=float(BETA_CRITICAL),adaptation_sweeps=1,pilot_cluster_steps=16,
        workers=1,backend='numba',return_chain_metadata=True,initial_states=['random','plus'])
    assert tiny.shape==(8,16,16) and len(np.unique(chain))==2
    save(dst/'MC_software_smoke_only.npz',spins=tiny,chain=chain,metadata=np.array(json.dumps(tiny_meta)))
    # Actual update and full-state restore are certified by this exact module's
    # budget preflight recovery receipt, not simulated here.
    assert read(PREFLIGHT/'recovery_test.json')['status']=='passed'
    torch.set_float32_matmul_precision('high')
    ev.generate_and_score(model,parent,'A',d.SEEDS[0],dict(name='core_geometry_W96',width=96,
        kind='continuous',gaps=[1],samples=16),dst/'generation',DEADLINE,16)
    torch.set_float32_matmul_precision('highest')
    # Full required array axes, 20 bootstrap repetitions only for a non-scientific
    # software fixture. Formal repetitions remain the frozen 20k/10k values.
    result=an.fixture(dst/'analysis',check)
    assert len(list((dst/'analysis').glob('*.png')))==6
    assert len(list((dst/'analysis').glob('*.pdf')))==6
    assert len(list(dst.glob('*_prediction*.npz')))==10
    assert len(list((dst/'evaluation').glob('*/g*_k*.npz')))==900
    files={p.relative_to(dst).as_posix():gs.file_hash(p) for p in dst.rglob('*') if p.is_file()}
    gs.atomic_json(dst/'manifest.json',files)
    gs.atomic_json(dst/'complete.json',dict(status='passed_scratch_pipeline',source='old validation + synthetic statistical fixture',
        real_update_and_restore_receipt='../preflight_authorized12h_v3_envfix/recovery_test.json',
        new_mc_not_used=True,formal_training_not_started=True,files=len(files),
        all_30_banks_tested=True,scratch_prediction_files=900,parents_per_scratch_file=2,
        scratch_same_model_all_cells=True,MC_smoke='L16 two chains software only, not a QA claim',
        source_sha256={p.relative_to(ROOT).as_posix():gs.file_hash(p) for p in source_files() if p!=DATA}))
    print(json.dumps(dict(status='scratch_complete',files=len(files))),flush=True)

def freeze():
    assert not (OUT/'run_protocol.json').exists() and not (OUT/'run.lock').exists()
    gate=read(PREFLIGHT/'budget_gate.json');pipeline=read(OUT/'preflight_pipeline_v3/complete.json')
    assert gate['status']=='budget_gate_passed_pending_full_pipeline' and pipeline['status']=='passed_scratch_pipeline'
    assert read(OUT/'preflight_pipeline_v3/visual_review.json')['status']=='passed'
    for p,sha in pipeline['source_sha256'].items():assert gs.file_hash(ROOT/p)==sha
    forecast=gate['predicted_seconds']+(time.time()-gate['started']-gate['elapsed_preflight_seconds'])
    fixture_timing=read(OUT/'preflight_pipeline_v3/analysis/timing.json')
    forecast+=max(0.,fixture_timing['projected_full_statistics_seconds']-gate['statistics_reserve_seconds'])
    if forecast>41400:raise RuntimeError(f'Launch budget gate failed with implementation elapsed: {forecast/3600:.4f}h')
    protocol=d.plan();protocol.update(status='frozen_ready',started=gate['started'],deadline=gate['deadline'],
        budget_version=2,hard_seconds=43200,launch_gate_seconds=41400,launch_forecast_max_seconds=41400,
        budget_amendment=AMENDMENT.relative_to(ROOT).as_posix(),
        runtime_budget=read(OUT/'budget_v2.json'),
        launch_predicted_seconds=forecast,files={p.relative_to(ROOT).as_posix():gs.file_hash(p) for p in source_files()},
        cell_order=[[s,a] for s in d.SEEDS for a in d.ARMS],
        log_buffering='persistent per1000 block handle; flush every20; all step records retained',
        cpu_threads=4,float32_precision_training='highest',sampler='S0_cos_squared_256_temperature1_no_MC_correction')
    protocol['protocol_hash']=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    gs.atomic_json(OUT/'run_protocol.json',protocol);print(json.dumps(dict(status='frozen_ready',predicted_hours=forecast/3600,deadline=DEADLINE)),flush=True)

def run():
    protocol=read(OUT/'run_protocol.json');verify_sources(protocol);check()
    # freeze already included elapsed time; use original gate to avoid double counting.
    gate=read(PREFLIGHT/'budget_gate.json')
    current_forecast=gate['predicted_seconds']+time.time()-gate['started']-gate['elapsed_preflight_seconds']
    current_forecast+=max(0.,read(OUT/'preflight_pipeline_v3/analysis/timing.json')['projected_full_statistics_seconds']-gate['statistics_reserve_seconds'])
    if current_forecast>41400:raise RuntimeError('Budget gate no longer passes at actual launch')
    with (OUT/'run.lock').open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time(),protocol_hash=protocol['protocol_hash']),f)
    mc_log=(OUT/'mc.log').open('x')
    mc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--mode','reference'],stdout=mc_log,stderr=subprocess.STDOUT,start_new_session=False)
    gs.atomic_json(OUT/'processes.json',dict(main=os.getpid(),mc=mc.pid,started=time.time(),deadline=DEADLINE))
    try:
        train_all(protocol)
        while mc.poll() is None:check();time.sleep(5)
        if mc.returncode:raise RuntimeError('Reference process failed; inspect reference_failure.json')
        assert read(OUT/'reference/complete.json')['status']=='passed'
        evaluate_all(protocol);log('statistics');result=analysis();verify_sources(protocol)
        for s in d.SEEDS:
            for a in d.ARMS:
                cell=OUT/'training'/f's{s}_{a}';r=read(cell/'complete.json')
                assert gs.file_hash(cell/'final.pt')==r['final_sha256'] and r['steps']==12000
        assert len(list((OUT/'evaluation').glob('*/complete.json')))==30
        assert len(list((OUT/'evaluation').glob('*/generation/shard_*.npz')))==120
        gs.atomic_json(OUT/'final_summary.json',dict(status='remote_complete_requires_backup_and_visual_review',
            elapsed_hours=(time.time()-protocol['started'])/3600,training_models=30,predictions=900,
            generation_images=1920,generation_shards=120,source_data_checkpoints_unchanged=True,**result))
        manifest={p.relative_to(OUT).as_posix():dict(bytes=p.stat().st_size,sha256=gs.file_hash(p)) for p in OUT.rglob('*')
            if p.is_file() and p.name not in ['manifest.json','last.pt','queue.log','status.json'] and not any(x.startswith('preflight') for x in p.relative_to(OUT).parts)}
        gs.atomic_json(OUT/'manifest.json',manifest);log('remote_complete_requires_backup_and_visual_review')
    finally:
        if mc.poll() is None:mc.terminate()
        mc_log.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['scratch','freeze','run','reference'],required=True);args=p.parse_args()
    DEADLINE=read(OUT/'budget_v2.json')['deadline'];torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    signal.signal(signal.SIGTERM,interrupt)
    try:globals()[args.mode]()
    except Exception:
        name='reference_failure.json' if args.mode=='reference' else ('failure.json' if args.mode=='run' else args.mode+'_failure_'+str(time.time_ns())+'.json')
        gs.atomic_json(OUT/name,dict(mode=args.mode,time=time.time(),traceback=traceback.format_exc()));raise
