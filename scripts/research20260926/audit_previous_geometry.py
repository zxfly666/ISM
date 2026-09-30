"""Independent, CPU-only audit of immutable G archives. Never imports G analysis.

New audit outputs only; no model loading, training, physics sampling or network.
"""
from pathlib import Path
import csv, hashlib, io, json, os, tarfile, time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BACKUP = Path('D:/ISM_research_backups/20260925_core_geometry_factorial')
OUT = ROOT/'artifacts/geometry_identification_20260926_local_audit'
ARMS = ['G00','G10','G01','G11','G11S']
SEEDS = list(range(92501,92507))


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def ah(*arrays):
    h=hashlib.sha256()
    for x in arrays:
        a=np.asarray(x);h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes(order='C'))
    return h.hexdigest()


def readz(t,m):
    with np.load(io.BytesIO(t.extractfile(m).read()),allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def jwrite(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def old_rng(seed,block,role):
    tag=np.frombuffer(hashlib.sha256(role.encode()).digest()[:16],dtype='<u4')
    return np.random.default_rng(np.random.SeedSequence([2026092521,seed,block,*map(int,tag)]))


def weights(random,chain,block):
    groups=[np.where(chain==v)[0] for v in np.unique(chain)]
    count=np.zeros(len(chain),np.int64)
    for c in random.integers(len(groups),size=len(groups)):
        ids=groups[c];n=len(ids)
        starts=random.integers(n,size=(n+block-1)//block)
        draw=np.concatenate([(start+np.arange(block))%n for start in starts])[:n]
        count+=np.bincount(ids[draw],minlength=len(chain))
    return count/count.sum()


def genmetrics(g,ref,pw,iw):
    rg=(pw@ref['pair_sum'])/np.maximum(pw@ref['pair_count'],1e-100)
    rmean=np.array([pw@ref[k] for k in ['m','m2','abs_m','energy']])
    out=np.empty((6,5,7))
    for s in range(6):
        for a in range(5):
            z=g[s][a];w=iw[s]/iw[s].sum()
            curve=(w@z['pair_sum'])/np.maximum(w@z['pair_count'],1e-100)
            for j,(lo,hi) in enumerate([(1,8),(9,24),(25,48)]):
                out[s,a,j]=np.sqrt(np.mean((curve[lo:hi+1]-rg[lo:hi+1])**2))/np.sqrt(np.mean(rg[lo:hi+1]**2))
            out[s,a,3:]=np.array([w@z[k] for k in ['m','m2','abs_m','energy']])-rmean
    return out


def main():
    OUT.mkdir(exist_ok=False);started=time.time()
    protocol=BACKUP/'final_review_20260925/run_protocol.json'
    p=json.loads(protocol.read_text(encoding='utf-8'))
    source_check={k:dict(expected=v,actual=sha(ROOT/k)) for k,v in p['files'].items()}
    assert all(v['expected']==v['actual'] for v in source_check.values())
    jwrite(OUT/'source_check.json',source_check)
    banks={};bank_blobs={};identity={}
    with tarfile.open(BACKUP/'banks_complete_v1.tar.gz') as t:
        for m in t:
            if not m.name.endswith('.npz'):continue
            z=readz(t,m);name=Path(m.name).stem;banks[name]=z
            n=len(z['parent']);noisy=z['noisy'].reshape(n,-1);clean=z['clean'].reshape(n,-1)
            q=z['queries'];e=z['evidence'];rows=np.arange(n)[:,None]
            assert np.all(noisy[rows,q]==2) and np.array_equal(z['labels'],clean[rows,q])
            assert np.array_equal(noisy[rows,e],clean[rows,e])
            assert all(len(np.unique(np.r_[q[i],e[i]]))==q.shape[1]+e.shape[1] for i in range(n))
            assert np.all((noisy!=2).sum(1)==int(z['k']))
            assert np.allclose(z['t'],1-int(z['k'])/int(z['width'])**2)
            identity[name]=ah(z['noisy'],z['true_coords'],z['t'],q,z['labels'])
    assert len(banks)==30
    values=np.empty((6,5,6,5,128,2));g=[[None]*5 for _ in range(6)]
    maximum_ce=0.;maximum_brier=0.;native_ok=0;generation_pairing={}
    evidence=tarfile.open(OUT/'review_evidence.tar.gz','w:gz',compresslevel=6)
    for path in [protocol,*[ROOT/'scripts/research20260921'/f'core_geometry_factorial_{n}.py' for n in ['design','data','training','statistics','analysis']],ROOT/'scripts/research20260921/run_core_geometry_factorial_20260925.py',ROOT/'docs/research_reboot_20260921/CORE_GEOMETRY_FACTORIAL_PROTOCOL_20260925_ZH.md']:
        evidence.add(path,arcname='source/'+path.name,recursive=False)
    for s,seed in enumerate(SEEDS):
        for a,arm in enumerate(ARMS):
            with tarfile.open(BACKUP/f'evaluation_s{seed}_{arm}_v1.tar.gz') as t:
                energies=[];shards=0
                for m in t:
                    if not m.name.endswith('.npz'):continue
                    name=Path(m.name).stem;z=readz(t,m)
                    if name.startswith('g') and '_k' in name:
                        b=banks[name];geo,ki=map(int,name.replace('g','').split('_k'))
                        assert np.array_equal(z['parent'],b['parent']) and np.array_equal(z['chain'],b['chain'])
                        assert str(z['common_data_hash'])==identity[name]
                        coord=b['summary_coords'] if arm=='G11S' else b['true_coords']
                        assert str(z['actual_input_hash'])==ah(b['noisy'],coord,b['t'],b['queries'])
                        native_ok+=1
                        prob=z['probability'];assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
                        prob=np.clip(prob,1e-12,1-1e-12);y=b['labels']
                        ce=np.where(y==1,-np.log(prob),-np.log1p(-prob));br=(z['probability']-y)**2
                        maximum_ce=max(maximum_ce,float(np.max(np.abs(ce-z['ce']))))
                        maximum_brier=max(maximum_brier,float(np.max(np.abs(br-z['brier']))))
                        values[s,a,geo,ki,:,0]=ce.mean(1);values[s,a,geo,ki,:,1]=br.mean(1)
                        if (geo==2 and a in (1,3,4) or geo==3 and a in (2,3)) and ki<3:
                            blob=t.extractfile(m).read();info=tarfile.TarInfo('probabilities/'+f's{seed}_{arm}/'+Path(m.name).name);info.size=len(blob);evidence.addfile(info,io.BytesIO(blob))
                    elif name=='statistics':
                        g[s][a]={k[6:]:v for k,v in z.items() if k.startswith('model_') and k!='model_G'}
                    elif name.startswith('shard_'):
                        shards+=1;spin=z['spins'].astype(np.float64)
                        assert spin.shape==(16,96,96) and np.isin(spin,[-1,1]).all()
                        energies.append(-((spin[:,1:,:]*spin[:,:-1,:]).sum((1,2))+(spin[:,:,1:]*spin[:,:,:-1]).sum((1,2)))/(2*96*95))
                        pair={k:ah(z[k]) for k in ['axis_x','axis_y','input_coordinates','parent','chain','origin','seed','sample_start']}
                        key=f'{seed}/{name}'
                        if key in generation_pairing:assert generation_pairing[key]==pair
                        else:generation_pairing[key]=pair
                assert shards==4
                g[s][a]['energy']=np.concatenate(energies)
            print(json.dumps(dict(stage='raw_probability_and_generation_audit',seed=seed,arm=arm)),flush=True)
    evidence.close()
    assert native_ok==900 and maximum_ce<1e-12 and maximum_brier<1e-12
    prior=np.load(BACKUP/'final_review_20260925/analysis/conditional_point.npz')
    risk_error=float(np.max(np.abs(values-prior['values'])));assert risk_error<1e-12
    chain=banks['g2_k0']['chain']
    diffs=np.stack([(values[:,3,2,:3,:,0]-values[:,1,2,:3,:,0]).mean(1),
                    (values[:,3,3,:3,:,0]-values[:,2,3,:3,:,0]).mean(1),
                    (values[:,3,2,:3,:,0]-values[:,4,2,:3,:,0]).mean(1)],axis=1)
    np.savez_compressed(OUT/'parent_risks_recomputed.npz',values=values,primary_parent_differences=diffs,chain=chain)
    bootstrap_checks={}
    for block,nrep in [(2,20000),(1,256),(4,256)]:
        random=old_rng(2026092531,block,'conditional_bootstrap_joint');new=[]
        old=np.load(BACKUP/f'final_review_20260925/analysis/conditional_bootstrap_block{block}.npz')['primary']
        for r in range(nrep):
            sw=np.bincount(random.integers(6,size=6),minlength=6)/6
            pw=weights(random,chain,block)
            new.append(np.tensordot(sw,diffs,axes=(0,0))@pw)
        new=np.asarray(new);err=float(np.max(np.abs(new-old[:nrep])));assert err<1e-12
        bootstrap_checks[str(block)]=dict(reps_verified=nrep,max_absolute_error=err,full_formal_replay=block==2)
        if block==2:np.savez_compressed(OUT/'independent_primary_bootstrap.npz',primary=new)
    with (OUT/'per_seed_chain_K.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['seed','chain','K','G11_minus_G11S_CE','exploratory'])
        for s,seed in enumerate(SEEDS):
            for ki,k in enumerate([2,32,512]):
                for cc in np.unique(chain):writer.writerow([seed,int(cc),k,float((values[s,3,2,ki,chain==cc,0]-values[s,4,2,ki,chain==cc,0]).mean()),True])
    with tarfile.open(BACKUP/'reference_complete_v1.tar.gz') as t:
        m=next(m for m in t if m.name.endswith('generation_reference.npz'));ref=readz(t,m)
    genpoint=genmetrics(g,ref,np.ones(len(ref['chain']))/len(ref['chain']),np.ones((6,64)))
    saved=np.load(BACKUP/'final_review_20260925/analysis/generation_point.npz')
    key='values' if 'values' in saved.files else 'per_seed'
    assert np.max(np.abs(genpoint-saved[key]))<1e-12
    sensitivity={}
    for unit in ['image','shard']:
        random=np.random.default_rng(26092610 if unit=='image' else 26092611);draw=[]
        for i in range(2000):
            pw=weights(random,ref['chain'],8)
            iw=np.array([np.bincount(random.integers(64,size=64),minlength=64) if unit=='image' else np.repeat(np.bincount(random.integers(4,size=4),minlength=4),16) for _ in range(6)])
            ss=random.integers(6,size=6);vv=genmetrics(g,ref,pw,iw)[ss].mean(0)
            draw.append(vv[3]-vv[4])
        draw=np.asarray(draw);sensitivity[unit]=dict(reps=2000,ci95=np.quantile(draw,[.025,.975],axis=0).tolist(),role='exploratory_audit_only_old_formal_shard_rule_unchanged')
    np.savez_compressed(OUT/'generation_per_image_statistics.npz',**{f's{s}_a{a}_{k}':v for s in range(6) for a in range(5) for k,v in g[s][a].items()},**{'reference_'+k:v for k,v in ref.items()})
    jwrite(OUT/'generation_resampling_sensitivity.json',sensitivity)
    # Sampler draws use a single shard stream, but disjoint tensor entries;
    # the frozen network has only within-image attention, LayerNorm, no BatchNorm.
    model_source=(ROOT/'ism_diffusion/scale_model.py').read_text(encoding='utf-8')
    assert 'BatchNorm' not in model_source
    jwrite(OUT/'complete.json',dict(status='passed_local_independent_implementation_audit',
        elapsed_seconds=time.time()-started,frozen_source_data_files=len(source_check),predictions=native_ok,
        max_ce_error=maximum_ce,max_brier_error=maximum_brier,parent_risk_error=risk_error,
        primary_means=diffs.mean((0,2)).tolist(),bootstrap_replay=bootstrap_checks,
        generation_rng_review='single generator per shard; distinct per-image tensor draws; no shared latent draw found; model has within-image attention and LayerNorm, no BatchNorm',
        generation_sensitivity='2000 each image/shard; audit only, not an amendment to original CI',
        limitation='Independent code in same assistant audit; not external peer reproduction; block1/4 verify256 prefixes, block2 all20000',
        new_training_started=False,new_mc_started=False,
        evidence_package_sha256=sha(OUT/'review_evidence.tar.gz')))
    print((OUT/'complete.json').read_text(),flush=True)


if __name__=='__main__':main()
