"""Frozen final-EMA development evaluation and paired, descriptive estimates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import phase_a as a
import phase_b as b

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def wasserstein(x,y):
    x=np.sort(x);y=np.sort(y);support=np.sort(np.r_[x,y]);width=np.diff(support)
    return float(np.sum(width*abs(np.searchsorted(x,support[:-1],side='right')/len(x)-np.searchsorted(y,support[:-1],side='right')/len(y))))

def entries(training):
    results=[]
    for seed in b.CONFIG['seeds']:
        for arm in b.CONFIG['arms']:
            p=training/f's{seed}_{arm}/final_ema.pt';done=json.loads((p.parent/'complete.json').read_text())
            assert done['updates']==12000 and done['ema_sha']==a.file_sha(p)
            results.append(dict(key=f's{seed}_{arm}_final',seed=seed,arm=arm,stage='final',file=p.relative_to(a.ROOT).as_posix(),
                                sha256=done['ema_sha'],source_sha256=done['ema_sha']))
    return results

def audit_training(training,out):
    audit=[]
    clock=json.loads((a.ROOT/'runs/B_full_clock_audit.json').read_text())
    expected_endpoints={(v['seed'],v['role']):v['endpoint_samples'] for v in clock['rows']}
    for seed in b.CONFIG['seeds']:
        logs={};initial={}
        for arm in b.CONFIG['arms']:
            folder=training/f's{seed}_{arm}'
            initial[arm]=json.loads((folder/'initialization.json').read_text())
            rows=[json.loads(line) for line in (folder/'updates.jsonl').read_text().splitlines()]
            assert [v['step'] for v in rows]==list(range(1,12001))
            assert all(np.isfinite([v['loss'],v['unclipped_grad_norm'],v['lr']]).all() for v in rows)
            assert all(v['clipped']==(v['unclipped_grad_norm']>1) and abs(v['lr']-b.lr(v['step']))<1e-15 for v in rows)
            for level,role in ((0,'common'),(1,'aux')):
                selected=[v for v in rows if v['level']==level]
                assert len(selected)==6000 and sum(v['endpoint_samples'] for v in selected)==expected_endpoints[seed,role]
                assert all(sum(v['width']==w for v in selected)==1500 for w in (16,24,32,48))
            logs[arm]=rows
            checkpoints=[]
            for step in (4000,8000,12000):
                path=folder/f'step{step:05d}.pt';p=torch.load(path,map_location='cpu',weights_only=False)
                assert (p['seed'],p['arm'],p['step'])==(seed,arm,step)
                assert p['config']==b.CONFIG and p['source_sha256']==a.file_sha(Path(b.__file__))
                for name in ('model','ema'):
                    assert all(bool(torch.isfinite(v).all()) for v in p[name].values())
                for v in p['optimizer']['state'].values():
                    assert int(v['step'])==step
                    assert bool(torch.isfinite(v['exp_avg']).all()) and bool(torch.isfinite(v['exp_avg_sq']).all())
                if step==12000:
                    final=torch.load(folder/'final_ema.pt',map_location='cpu',weights_only=True)
                    b.compare(p['ema'],final)
                checkpoints.append(dict(step=step,sha256=a.file_sha(path)))
            audit.append(dict(seed=seed,arm=arm,updates=12000,common=6000,aux=6000,tokens=12000*18432,
                             initial_hash=initial[arm]['model_hash'],checkpoints=checkpoints))
        assert len({v['model_hash'] for v in initial.values()})==1
        assert all(v['model_hash']==v['ema_hash'] and v['optimizer_state_entries']==0 for v in initial.values())
        for index in range(12000):
            rows={arm:logs[arm][index] for arm in b.CONFIG['arms']}
            assert len({v['selection_hash'] for v in rows.values()})==1
            assert len({v['noise_hash'] for v in rows.values()})==1
            if index%2==0:assert len({v['input_hash'] for v in rows.values()})==1
            else:
                assert rows['F']['spin_hash']==rows['P']['spin_hash']
                assert len({rows[v]['coordinate_hash'] for v in ('P','D','M')})==1
    a.write_json(out/'training_audit.json',dict(status='passed',models=audit,all_96000_update_records_checked=True,
                  paired_initializations_checked=True,all_pairing_hashes_checked=True,checkpoint_optimizer_and_EMA_checks=True))
    print('FULL TRAINING AUDIT PASSED: 96000 updates, 24 checkpoints',flush=True)

def statistics(out,n):
    seeds=b.CONFIG['seeds'];arms=b.CONFIG['arms'];allrows=[];comparisons=[];draws={}
    # One independent MC draw is shared by every arm and seed per replicate.
    z=load(a.ROOT/'inputs/mc_W144_centered.npz')
    reference={w:load(out/f'MC_reference_W{w}.npz') for w in (96,32)}
    parents=reference[96]['parent_ids'];assert np.array_equal(parents,reference[32]['parent_ids'])
    chain=z['chain'][parents]
    # Bootstrap the actual 256-field reference used by the point estimate,
    # rather than silently adding 768 unused historical fields to inference.
    groups=np.stack([np.flatnonzero(chain==c) for c in range(1,16,2)])
    assert groups.shape==(8,32)
    generator=np.random.default_rng(np.random.SeedSequence([2026112506]))
    repeats=10000;block=8
    seed_draw=generator.integers(0,2,size=(repeats,2));sample_draw=generator.integers(0,n,size=(repeats,2,n))
    chosen_chains=generator.integers(0,8,size=(repeats,8));starts=generator.integers(0,32,size=(repeats,8,4))
    mcids=groups[chosen_chains[:,:,None,None],(starts[:,:,:,None]+np.arange(block))%32].reshape(repeats,256)
    for steps in (64,256):
        for path,w,lo in [('fine',96,25),('B',32,9)]:
            gg=[];mm=[];ee=[]
            for seed in seeds:
                gs=[];ms=[];es=[]
                for arm in arms:
                    data=load(out/'metrics'/f's{seed}_{arm}_final'/f'{path}_K{steps}.npz')
                    gs.append(data['g']);ms.append(data['m']);es.append(data['e'])
                gg.append(gs);mm.append(ms);ee.append(es)
            gg=np.array(gg);mm=np.array(mm);ee=np.array(ee)
            actual_ref=load(out/f'MC_reference_W{w}.npz');ref=actual_ref['g'].mean(0)[lo:]
            point=np.linalg.norm(gg.mean(2)[:,:,lo:]-ref,axis=-1)/np.linalg.norm(ref)
            for si,seed in enumerate(seeds):
                for ai,arm in enumerate(arms):
                    allrows.append(dict(seed=seed,arm=arm,steps=steps,path=path,width=w,n=n,G_error=float(point[si,ai]),
                            mean_m=float(mm[si,ai].mean()),mean_abs_m=float(abs(mm[si,ai]).mean()),mean_e=float(ee[si,ai].mean()),
                            W1_signed_m=wasserstein(mm[si,ai],actual_ref['m']),W1_abs_m=wasserstein(abs(mm[si,ai]),abs(actual_ref['m'])),
                            W1_energy=wasserstein(ee[si,ai],actual_ref['e'])))
            errors=np.empty((repeats,4))
            # Bounded chunks avoid materializing [10000,2,4,n,r] simultaneously.
            for start in range(0,repeats,100):
                ix=slice(start,start+100);reference_mean=reference[w]['g'][mcids[ix],lo:].mean(1)
                sampled=gg[seed_draw[ix,:,None,None],np.arange(4)[None,None,:,None],sample_draw[ix,:,None,:],lo:].mean(3)
                errors[ix]=(np.linalg.norm(sampled-reference_mean[:,None,None,:],axis=-1)/np.linalg.norm(reference_mean,axis=-1)[:,None,None]).mean(1)
            for control in ('D','F','P'):
                delta=errors[:,arms.index('M')]-errors[:,arms.index(control)]
                seed_values=point[:,arms.index('M')]-point[:,arms.index(control)]
                q=np.quantile(delta,[.025,.975]);tag=f'{path}_K{steps}_M_minus_{control}';draws[tag]=delta
                comparisons.append(dict(path=path,steps=steps,comparison='M-'+control,paired_seed_values=seed_values.tolist(),
                    estimate=float(seed_values.mean()),descriptive_joint_bootstrap95=q.tolist(),both_seeds_improve=bool((seed_values<0).all()),
                    confirmation=False,note='Two development seeds; intervals describe exploratory variability and do not satisfy stage-C rules'))
    a.npz(out/'development_bootstrap_draws.npz',**draws,seed_draw=seed_draw,sample_draw=sample_draw,
          mc_parent_draw=parents[mcids],reference_parent_ids=parents)
    a.write_json(out/'development_statistics.json',dict(per_seed=allrows,comparisons=comparisons,repeats=repeats,block=block,
                 rng=2026112506,reference_n=256,reference_chain_pool=list(range(1,16,2)),new_confirmation_data_used=False,
                 method='joint training-seed / paired-image / independent MC-chain plus circular block; nonlinear errors recomputed each draw'))

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--statistics-only',action='store_true');args=p.parse_args();training=args.training.absolute();out=args.out.absolute();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    models=entries(training)
    identity=dict(stage='B development',n=32,steps=[64,256],precision='FP32, TF32 disabled',batch=1,rng_root=a.RNG_ROOT,
                  models=models,source=a.file_sha(Path(__file__)),phase_a_source=a.file_sha(Path(a.__file__)),torch=str(torch.__version__))
    with a.run_lock(out/'run.lock'):
        if (out/'identity.json').exists():assert json.loads((out/'identity.json').read_text())==identity
        a.write_json(out/'identity.json',identity)
        if not args.statistics_only:
            audit_training(training,out)
            a.midpoint_exact(models,torch.device('cuda'),out,8)
            a.generate_matrix(models,torch.device('cuda'),out,32,1)
            a.summarize_a1(models,out,32)
        statistics(out,32)
        a.write_json(out/'complete.json',dict(stage='B',models=8,generation_images=512,exact_cells=48,time=time.time(),no_C_launched=True))

if __name__=='__main__':main()
