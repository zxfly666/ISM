"""Frozen, paired supplements to RG-2 v1.2 stage A; no training side effects."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

import phase_a as a
from diagnostic_math import direct_axis, uniform_axis_fft

ROOT=a.ROOT
OLD=Path('/root/shared-nvme/ISM_research_20260921')
SEED=2026112502

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def csvout(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def mc():
    z=load(ROOT/'inputs/mc_W144_centered.npz')
    return 2*np.unpackbits(z['fine_bits'],axis=-1,bitorder='little').astype(np.int8)-1,z['chain']

def select(chain,n):
    groups=[np.flatnonzero(chain==c) for c in np.unique(chain)]
    if n%len(groups):raise ValueError('Require equal chain weights')
    return np.concatenate([g[:n//len(groups)] for g in groups])

def physical_coords(axes):
    n,_,w=axes.shape
    c=np.empty((n,w,w,2),dtype=np.float32)
    c[...,0]=axes[:,0,:,None];c[...,1]=axes[:,1,None,:]
    return c

def verify_original(path,manifest,rel):
    actual=a.file_sha(path)
    if rel not in manifest:raise ValueError(f'Missing historical manifest: {rel}')
    entry=manifest[rel]
    expected=entry['sha256'] if isinstance(entry,dict) else entry
    if actual!=expected:raise ValueError(f'Historical SHA mismatch: {path}')
    return actual

def prepare(out):
    banks=out/'banks';banks.mkdir(parents=True,exist_ok=True)
    source=OLD/'artifacts/geometry_identification_20260926'
    manifest=json.loads((source/'manifest.json').read_text())
    provenance=[]
    # Retain the complete historical 256-parent banks; H48 is balanced down
    # from 2048 to the first 16 consecutive parents in each of its 16 chains.
    names=[f'{g}_k{k}' for g in ('C48','U48','H48','B48') for k in (2,32,512)]
    names += [f'{g}_k{k}' for g in ('C96','H96') for k in (2,32,512,4608)]
    files=[('banks/'+n+'.npz','geometry_'+n) for n in names]
    files += [(f'padding_banks/k{k}_v{v}.npz',f'container_k{k}_v{v}') for k in (2,32,512) for v in range(3)]
    for rel,name in files:
        src=source/rel;sha=verify_original(src,manifest,rel);z=load(src)
        ids=select(z['chain'],256)
        c=z.get('input_coordinates')
        if c is None:c=physical_coords(z['physical_axes'])
        payload={k:z[k][ids] for k in ('noisy','queries','labels','t','parent','chain','origin')}
        payload.update(coords=c[ids],source_rows=ids,source_sha=np.array(sha))
        assert np.all(np.take_along_axis(payload['noisy'].reshape(256,-1),payload['queries'],1)==2)
        a.npz(banks/(name+'.npz'),**payload)
        provenance.append(dict(name=name,source=str(src),sha256=sha,n=256,selection='first 16 consecutive parents per chain'))
    # Validate all three container views actually preserve visible evidence,
    # query identities, physical coordinates and labels. Clock changes only v2.
    for k in (2,32,512):
        s,f,n=[load(banks/f'container_k{k}_v{v}.npz') for v in range(3)]
        assert np.array_equal(s['noisy'],f['noisy'][:,24:72,24:72])
        assert np.array_equal(s['coords'],f['coords'][:,24:72,24:72])
        assert np.array_equal((s['queries']//48+24)*96+s['queries']%48+24,f['queries'])
        assert np.array_equal(s['t'],f['t']) and np.array_equal(s['labels'],f['labels'])
        assert np.array_equal(f['noisy'],n['noisy']) and np.array_equal(f['queries'],n['queries'])
        assert np.all((s['noisy']!=2).sum((1,2))==k) and np.all((f['noisy']!=2).sum((1,2))==k)
    # Ordinary empirical CE: separate masked fractions, identical parents and
    # masks across all 21 old EMAs; no local Ising oracle on marginalized sites.
    fields,chain=mc();ids=select(chain,256)
    for w in (48,96):
        offset=(144-w)//2;clean=(fields[ids,offset:offset+w,offset:offset+w]>0).astype(np.int8)
        for pct in (10,50,90):
            rng=np.random.default_rng(np.random.SeedSequence([SEED,w,pct]))
            mask=rng.random(clean.shape)<pct/100
            q=np.stack([rng.choice(np.flatnonzero(v),64,replace=False) for v in mask.reshape(256,-1)])
            name=f'ordinary_W{w}_t{pct}'
            a.npz(banks/(name+'.npz'),noisy=np.where(mask,2,clean),queries=q,
                  labels=np.take_along_axis(clean.reshape(256,-1),q,1),
                  t=np.full(256,pct/100,dtype=np.float32),coords=np.broadcast_to(a.coords(w,0,'cpu').numpy(),(256,w,w,2)),
                  parent=ids,chain=chain[ids],origin=np.full((256,2),440+offset),source_rows=ids,
                  source_sha=np.array(a.file_sha(ROOT/'inputs/mc_W144_centered.npz')))
            provenance.append(dict(name=name,source='inputs/mc_W144_centered.npz',n=256,
                                   seed=[SEED,w,pct],mask='Bernoulli; 64 hidden queries uniformly without replacement'))
    # Two separated, preregistered cases per MC chain, with full W16 exterior.
    chosen=np.array([np.flatnonzero(chain==c)[j] for c in np.unique(chain) for j in (0,64)])
    labels=['all_plus']+[f'mc_chain{int(chain[i])}_ord{int(np.flatnonzero(np.flatnonzero(chain==chain[i])==i)[0])}' for i in chosen]
    a.npz(out/'a2_exteriors.npz',fields=np.concatenate([np.ones((1,16,16),np.int8),fields[chosen,64:80,64:80]]),
          labels=labels,parent=np.r_[-1,chosen],chain=np.r_[-1,chain[chosen]],origin=np.full((len(chosen)+1,2),504))
    input_hashes={p.relative_to(out).as_posix():a.file_sha(p) for p in banks.glob('*.npz')}
    input_hashes['a2_exteriors.npz']=a.file_sha(out/'a2_exteriors.npz')
    a.write_json(out/'inputs.json',dict(protocol='RG-2 v1.2',seed=SEED,files=input_hashes,provenance=provenance,
                                    a2_selection='all plus plus ordinal 0 and 64 in each of 16 archived MC chains; descriptive'))
    print('PREPARED',len(provenance),'paired banks and',len(labels),'A2 exteriors',flush=True)

def autocorrelation(v):
    v=v-v.mean(1,keepdims=True);length=v.shape[1]
    z=np.fft.rfft(v,n=2*length,axis=1)
    ac=np.fft.irfft(z*z.conj(),n=2*length,axis=1)[:,:length]
    cov=ac.sum(0)/(len(v)*(length-np.arange(length)))
    rho=cov/cov[0] if cov[0]>0 else np.r_[1.,np.zeros(length-1)]
    total=0.
    for i in range(1,length-1,2):
        pair=rho[i]+rho[i+1]
        if pair<=0:break
        total+=pair
    return rho,max(1.,1+2*total)

def physics(out):
    x,chains=mc();ids=np.stack([np.flatnonzero(chains==c) for c in np.unique(chains)])
    stats={};features=[];cuts={};series={};per={'chain':chains,'parent':np.arange(len(chains))}
    for w in (96,144):
        start=(144-w)//2;fine=x[:,start:start+w,start:start+w]
        d=fine[:,1::3,1::3];m=a.majority(fine)
        stats[str(w)]={'disagreement':float((d!=m).mean())}
        for kind,v in [('fine',fine),('D',d),('M',m)]:
            tag=f'{kind}{w}';g,mag,en=a.observables(v)
            full_g,count=uniform_axis_fft(v[:4]);assert np.allclose(full_g,direct_axis(v[:4]),atol=2e-13)
            count=count[:g.shape[1]]
            products=np.rint(g*count[None]).astype(np.int64)
            assert np.max(abs(products/count[None]-g))<1e-13
            per.update({tag+'_g':g,tag+'_m':mag,tag+'_e':en,tag+'_products':products,tag+'_counts':count})
            low=v.shape[1]//4+1;segment=g[:,low:];lo=sum(f.shape[1] for f in features)
            features.append(segment);cuts[tag]=(lo,lo+segment.shape[1])
            for suffix,values in [('m',mag),('abs_m',abs(mag)),('G1',g[:,1]),('longG',segment.mean(1))]:
                series[tag+'_'+suffix]=values[ids]
            stats[str(w)][kind]={'G1':float(g[:,1].mean()),'mean_m':float(mag.mean()),'mean_abs_m':float(abs(mag).mean())}
        for k in range(1,w//6+1):
            left=fine[:,1::3,1::3]
            aligned=.5*((fine[:,1+3*k::3,1::3]*left[:,:-k]).mean((1,2))+
                        (fine[:,1::3,1+3*k::3]*left[:,:,:-k]).mean((1,2)))
            assert np.allclose(aligned,per[f'D{w}_g'][:,k],atol=2e-13)
    a.npz(out/'A3_per_field.npz',**per)
    diagnostic={};acf={}
    for tag,v in series.items():
        rho,tau=autocorrelation(v);diagnostic[tag]=dict(tau_int=tau,pooled_ESS=float(v.size/tau));acf[tag]=rho
    worst=max(r['tau_int'] for r in diagnostic.values())
    # Conservative powers of two, chosen solely from MC dependence, before A1.
    recommended=next((b for b in (4,8,16,32,64) if b>=2*worst),128)
    block=min(64,max(8,recommended))
    a.npz(out/'A3_autocorrelation.npz',**acf)
    a.write_json(out/'A3_dependence.json',dict(series=diagnostic,worst_tau=worst,block=block,
                rule='smallest power of two >= 2*max pooled within-chain IPS tau, at least 8, capped at 64',
                capped=recommended>64,chain_length=128,reference_chains=list(range(1,16,2)),
                pseudo_chains=list(range(0,16,2)),reference_n=256,reference_note='independent chains; block resampling preserves chronological parent order'))
    values=np.concatenate(features,axis=1)[ids]
    sums=np.pad(np.cumsum(np.concatenate([values,values],axis=1),axis=1),((0,0),(1,0),(0,0)))
    def sample(pool,n,b,reps,rng):
        per_chain=n//len(pool);ch=rng.choice(pool,size=(reps,len(pool)))
        answer=np.zeros((reps,values.shape[2]))
        for off in range(0,per_chain,b):
            length=min(b,per_chain-off);start=rng.integers(128,size=ch.shape)
            answer+=(sums[ch,start+length]-sums[ch,start]).sum(1)
        return answer/n
    rows=[];draws={}
    for b in sorted({4,8,16,block}):
        for n in (16,32,64,128,256,512):
            rng=np.random.default_rng(np.random.SeedSequence([SEED,30,b,n]))
            sim=sample(np.arange(0,16,2),n,b,2000,rng);ref=sample(np.arange(1,16,2),256,b,2000,rng)
            for key,(lo,hi) in cuts.items():
                err=np.linalg.norm(sim[:,lo:hi]-ref[:,lo:hi],axis=1)/np.linalg.norm(ref[:,lo:hi],axis=1)
                draws[f'block{b}_n{n}_{key}']=err
                q=np.quantile(err,[.05,.5,.95]);rows.append(dict(block=b,n=n,measure=key,q05=float(q[0]),q50=float(q[1]),q95=float(q[2])))
    a.npz(out/'A3N_draws.npz',**draws);csvout(out/'A3N_quantiles.csv',rows)
    a.write_json(out/'A3.json',dict(physics=stats,block=block,draws=2000,reference_n=256,
                 chosen_A1_n=32,reason='descriptive old-lineage screen; C sample size remains unfrozen',
                 MC_sha256=a.file_sha(ROOT/'inputs/mc_W144_centered.npz'),source='archived 2048 fields from 16 independent MC chains'))
    print('A3/A3N COMPLETE',json.dumps(dict(block=block,worst_tau=worst,physics=stats)),flush=True)

def evaluate(out,device):
    manifest=json.loads((out/'inputs.json').read_text())
    for name,h in manifest['files'].items():assert a.file_sha(out/name)==h
    entries=json.loads((ROOT/'inputs/manifest.json').read_text())['models']
    for e in entries:
        model=a.load_model(e,device)
        for bp in sorted((out/'banks').glob('*.npz')):
            dest=out/'A0_retention'/e['key']/bp.name
            if dest.exists():
                old=load(dest);assert str(old['ema_sha'])==e['sha256'] and str(old['bank_sha'])==a.file_sha(bp)
                continue
            z=load(bp);ps=[];n=len(z['noisy']);started=time.monotonic()
            for i in range(0,n,8):
                sl=slice(i,i+8)
                p=a.predict(model,z['noisy'][sl],z['coords'][sl],z['t'][sl],0,device).cpu().numpy().reshape(len(z['noisy'][sl]),-1)
                ps.append(np.take_along_axis(p,z['queries'][sl],1))
            p=np.clip(np.concatenate(ps).astype(float),1e-12,1-1e-12);y=z['labels']
            ce=-(y*np.log(p)+(1-y)*np.log1p(-p)).mean(1)
            a.npz(dest,p=p,CE=ce,chain=z['chain'],parent=z['parent'],labels=y,
                  ema_sha=e['sha256'],bank_sha=a.file_sha(bp),seconds=time.monotonic()-started)
            print('RETENTION',e['key'],bp.stem,float(ce.mean()),flush=True)

def a2(out,device):
    # phase_a's unchanged conditional enumeration reads this dedicated bank.
    entries=json.loads((ROOT/'inputs/manifest.json').read_text())['models']
    original=np.load
    target=ROOT/'inputs/a2_exteriors.npz'
    def routed(path,*args,**kwargs):
        if Path(path)==target:path=out/'a2_exteriors.npz'
        return original(path,*args,**kwargs)
    try:
        np.load=routed
        a.block_diagnostics([e for e in entries if e['stage']!='midpoint'],device,out,False,32)
    finally:np.load=original
    rows=[]
    ex=load(out/'a2_exteriors.npz');meta={str(label):(int(p),int(c)) for label,p,c in zip(ex['labels'],ex['parent'],ex['chain'])}
    for f in sorted((out/'A2').rglob('*.npz')):
        z=load(f);parent,chain=meta[f.stem]
        for plan in ('serial','adjacent','checker','nine'):
            r=json.loads(str(z[plan+'_result']));rows.append(dict(model=f.parent.name,boundary=f.stem,parent=parent,chain=chain,plan=plan,**{k:v for k,v in r.items() if k!='clock'}))
    csvout(out/'A2_boundaries.csv',rows)
    groups={}
    for r in rows:
        if r['chain']<0:continue
        key=(r['model'],r['chain'],r['plan']);groups.setdefault(key,[]).append(r)
    csvout(out/'A2_chains.csv',[dict(model=key[0],chain=key[1],plan=key[2],n=len(v),
           **{metric:float(np.mean([r[metric] for r in v])) for metric in ('learned_kl','single_conditional_kl','parallel_factorization_kl','TV','p_majority','q_majority')}) for key,v in groups.items()])

def random_audit(out):
    hashes={};arrays={};totals={}
    import hashlib
    for seed in (92601,92602,92603):
        for w in (32,96):
            for i in range(32):
                u,v=a.uniform_fields(seed,w,i);key=f's{seed}_W{w}_i{i}'
                hashes[key]=hashlib.sha256(u.tobytes()+v.tobytes()).hexdigest()
                ev64,ev256=[a.reveal_at(u,k) for k in a.STEPS]
                assert np.array_equal(ev64,ev256//4)
                for k,ev in zip(a.STEPS,(ev64,ev256)):
                    assert ev.min()>=0 and ev.max()<k
                    count=np.bincount(ev.ravel(),minlength=k);assert count.sum()==w*w
                    arrays[key+f'_K{k}_reveal_count']=count
                    totals.setdefault((w,k),np.zeros(k,dtype=np.int64));totals[w,k]+=count
    diagnostics=[]
    for (w,k),count in totals.items():
        prob=a.schedule(k)[:-1]-a.schedule(k)[1:];n=96*w*w
        z=(count-n*prob)/np.sqrt(n*prob*(1-prob))
        diagnostics.append(dict(width=w,steps=k,max_abs_binomial_z=float(abs(z).max()),positions=n))
        arrays[f'K{k}_schedule']=a.schedule(k)
    a.npz(out/'A1_randomness.npz',**arrays)
    a.write_json(out/'A1_randomness.json',dict(root=a.RNG_ROOT,algorithm='PCG64, SeedSequence([root,seed,width,image,role]); integer midpoints / 2**32',
                hashes=hashes,marginal_reveal_checks=diagnostics,positions_revealed_exactly_once=True,nested_grids=True,
                arm_and_steps_excluded_from_key=True,batch_independent=True,tasks_separated_by_width=True))
    print('RNG AUDIT PASSED',flush=True)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','physics','retention','a2','rng'])
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out.absolute();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with a.run_lock(out/(args.action+'.lock')):
        if args.action=='prepare':prepare(out)
        if args.action=='physics':physics(out)
        if args.action=='retention':evaluate(out,torch.device('cuda'))
        if args.action=='a2':a2(out,torch.device('cuda'))
        if args.action=='rng':random_audit(out)

if __name__=='__main__':main()
