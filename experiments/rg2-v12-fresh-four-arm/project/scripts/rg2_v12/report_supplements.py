"""CPU-only uncertainty summaries and explicit saved A2 input-state tables."""
import csv
import hashlib
import json
import numpy as np


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def a3_intervals(project,runs,out):
    data=load(runs/'A_supp/A3_per_field.npz');meta=load(project/'inputs/mc_W144_centered.npz')
    x=2*np.unpackbits(meta['fine_bits'],axis=-1,bitorder='little').astype(np.int8)-1
    chains=np.unique(data['chain']);assert len(chains)==16
    assert all(np.sum(data['chain']==c)==128 for c in chains)
    rng=np.random.default_rng(2026112510);chosen=rng.integers(0,16,size=(10000,16))
    weights=np.zeros((10000,16));np.add.at(weights,(np.repeat(np.arange(10000),16),chosen.ravel()),1/16)
    grid=np.linspace(-1,1,129);draws={'chain_draw_indices':chosen,'chain_ids':chains};bands={};rows=[]
    for w in (96,144):
        lo=(144-w)//2;fine=x[:,lo:lo+w,lo:lo+w]
        majority=np.sign(fine.reshape(len(fine),w//3,3,w//3,3).sum((2,4)))
        disagreement=(fine[:,1::3,1::3]!=majority).mean((1,2))
        for kind in ('fine','D','M'):
            tag=f'{kind}{w}';g=data[tag+'_g'];m=data[tag+'_m'];e=data[tag+'_e']
            values=np.column_stack((g,m,abs(m),e))
            means=np.stack([values[data['chain']==c].mean(0) for c in chains])
            sampled=weights@means;point=values.mean(0);lower,upper=np.quantile(sampled,[.025,.975],axis=0)
            draws[tag]=sampled
            bands[tag]={'g_lower':lower[:g.shape[1]],'g_upper':upper[:g.shape[1]]}
            labels=[f'G_{r}' for r in range(g.shape[1])]+['mean_m','mean_abs_m','mean_energy']
            for label,p,l,u in zip(labels,point,lower,upper):
                rows.append(dict(width=w,operator=kind,metric=label,estimate=float(p),lower95=float(l),upper95=float(u)))
            if kind in ('D','M'):
                cdf=np.stack([(m[data['chain']==c,None]<=grid[None,:]).mean(0) for c in chains])
                sampled_cdf=weights@cdf;lower_cdf,upper_cdf=np.quantile(sampled_cdf,[.025,.975],axis=0)
                draws[tag+'_cdf']=sampled_cdf
                bands[tag].update(cdf_grid=grid,cdf_lower=lower_cdf,cdf_upper=upper_cdf)
        per_chain=np.array([disagreement[data['chain']==c].mean() for c in chains])
        sample=weights@per_chain;lower,upper=np.quantile(sample,[.025,.975]);draws[f'disagreement{w}']=sample
        rows.append(dict(width=w,operator='D_vs_M',metric='disagreement',estimate=float(disagreement.mean()),lower95=float(lower),upper95=float(upper)))
        delta=draws[f'M{w}'][:,:data[f'M{w}_g'].shape[1]]-draws[f'D{w}'][:,:data[f'D{w}_g'].shape[1]]
        lower,upper=np.quantile(delta,[.025,.975],axis=0)
        point=data[f'M{w}_g'].mean(0)-data[f'D{w}_g'].mean(0)
        for r,(p,l,u) in enumerate(zip(point,lower,upper)):
            rows.append(dict(width=w,operator='M_minus_D',metric=f'G_{r}',estimate=float(p),lower95=float(l),upper95=float(u)))
    np.savez_compressed(out/'A3_chain_bootstrap_draws.npz',**draws)
    with (out/'A3_chain_intervals.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(out/'A3_interval_method.json',dict(rng=2026112510,repeats=10000,independent_chains=16,fields_per_chain=128,
          resampling='whole MC chains with replacement, preserving all within-chain chronological fields and paired transforms',
          interval='pointwise percentile 95%; not simultaneous bands',scope='descriptive historical MC',
          data_sha256=sha(runs/'A_supp/A3_per_field.npz')))
    return bands


def a2_states(runs,out):
    source=runs/'A_supp/a2_exteriors.npz'
    if not source.exists():return
    ex=load(source);sites=np.array([(6+i,6+j) for i in range(3) for j in range(3)])
    records=[];state_rows=[];groups=[];known_masks=[]
    plans={'serial':[0],'adjacent':[0,1,2],'checker':[0,2,4],'nine':list(range(9))}
    for pi,(name,first) in enumerate(plans.items()):
        plan=[first]+[[j] for j in range(9) if j not in first];prior=[]
        for round_id,group in enumerate(plan):
            known=sorted(prior);n=2**len(known)
            states=np.full((n,9),2,np.int8)
            if known:states[:,known]=(np.arange(n)[:,None]>>np.arange(len(known)))&1
            state_rows.extend(states);groups.extend([sum(1<<i for i in group)]*n);known_masks.extend([sum(1<<i for i in known)]*n)
            records.extend([(pi,round_id,9-len(known),max((9-len(known))/256,.01))]*n)
            prior.extend(group)
    state_rows=np.array(state_rows);records=np.array(records)
    containers=np.broadcast_to((ex['fields']>0).astype(np.int8)[:,None],(len(ex['fields']),len(state_rows),16,16)).copy()
    containers[:,:,sites[:,0],sites[:,1]]=state_rows[None]
    assert np.array_equal((containers==2).sum((2,3)),np.broadcast_to(records[:,2],containers.shape[:2]))
    np.savez_compressed(out/'A2_reveal_states.npz',boundary_labels=ex['labels'],parent=ex['parent'],chain=ex['chain'],
          plan_names=np.array(list(plans)),sites=sites,plan_index=records[:,0].astype(np.int8),round_index=records[:,1].astype(np.int8),
          hidden_count=records[:,2].astype(np.int8),valid_count=256,state_clock_float64=records[:,3],model_clock_fp32=records[:,3].astype(np.float32),
          known_mask=np.array(known_masks),reveal_group_mask=np.array(groups),input_codes=containers)
    write(out/'A2_reveal_states.json',dict(boundaries=len(ex['fields']),conditional_states_per_boundary=len(state_rows),
          codes={'minus':0,'plus':1,'MASK':2},coordinates='unit physical coordinates on W16; task level 0',
          convention='all possible pre-reveal input states for the frozen exogenous groups; terminal states are the 512 bits in each original result',
          source_sha256=sha(source)))
