"""Frozen 600 predictions and 120 paired controls, unsealed after all finals."""
from __future__ import annotations
import time
import numpy as np
import torch
import sm6_common as c
import sm6_data as d
import sm6_model as m

def jobs():
    specs = list(d.evaluation_specs())
    retention = ('K1_holdout', 'K2_holdout', 'K4_N4_center')
    for seed in c.CFG['seed_labels']:
        for arm in c.CFG['arms']:
            for spec in specs:
                yield dict(seed=seed, arm=arm, step=8000, weight='raw', spec=spec, kind='final')
            for spec in specs:
                if spec['key'] in retention + ('G8_test_N20_center',):
                    yield dict(seed=seed, arm=arm, step=8000, weight='ema', spec=spec, kind='secondary')
            for step in (4000, 6000):
                for spec in specs:
                    eligible = spec['key'] in retention or (spec['family']=='G8' and
                        spec['split']=='validation' and spec['placement']=='center' and
                        spec['side'] in c.CFG['arms'][arm]['valid_side_cycle'])
                    if eligible:
                        yield dict(seed=seed, arm=arm, step=step, weight='raw', spec=spec, kind='learning')

def job_key(job):
    return f"s{job['seed']}_{job['arm']}/{job['step']}_{job['weight']}_{job['kind']}_{job['spec']['key']}"

def input_bank(job):
    return d.bank_for_spec(job['spec'], job['arm'])

def save_bank(root, key, bank):
    path = root / 'banks' / (key+'.npz')
    native_hash = d.native_digest([bank])
    if path.exists():
        assert d.native_digest([c.load(path)]) == native_hash
    else:
        c.save(path, **bank)
    return path.relative_to(root).as_posix(), native_hash

def emit_prediction(root, job, bank, logits, source_sha):
    key = job_key(job)
    bank_path, input_hash = save_bank(root, job['spec']['key']+'_canonical', bank)
    metrics = c.metrics(logits, bank['target'])
    relative = 'evaluation/'+key+'.npz'
    c.save(root / relative, logits=logits, **metrics, target=bank['target'], row_id=bank['row_id'],
           identity=np.array(key), study=np.array(c.CFG['study']),
           common_input_hash=np.array(c.digest(bank['row_id'], bank['tokens'], bank['coordinates'], bank['valid'], bank['query'], bank['target'])),
           native_input_hash=np.array(input_hash), source_sha256=np.array(source_sha), bank_path=np.array(bank_path))
    row = dict(**job, key=key, prediction=relative, bank=bank_path, native_hash=input_hash,
               source_sha256=source_sha, **c.summary(logits, bank['target']))
    c.write(root / ('evaluation/'+key+'.json'), row)
    return row

def control_banks(seed, arm):
    for family, split in (('K4','train'), ('G8','test')):
        for side, allocated in c.CFG['evaluation']['control_PAD_side_allocated_pairs']:
            spec=dict(family=family,split=split,side=side,placement='center',key=f'{family}_{split}_N{side}_center')
            base=d.bank_for_spec(spec,arm)
            n=len(base['query']); extra=allocated-side*side
            pad={k:v.copy() for k,v in base.items()}
            pad['tokens']=np.concatenate([base['tokens'],np.full((n,1,extra),3,np.int64)],2)
            pad['valid']=np.concatenate([base['valid'],np.zeros((n,1,extra),bool)],2)
            pad['coordinates']=np.concatenate([base['coordinates'],np.zeros((n,1,extra,2),np.float32)],2)
            yield dict(seed=seed,arm=arm,family=family,side=side,kind='invalid_PAD',allocated=allocated),base,pad
            if side==20:
                shifted={k:v.copy() for k,v in base.items()}
                shifted['coordinates']+=np.array([17,-11],np.float32)
                yield dict(seed=seed,arm=arm,family=family,side=side,kind='translation'),base,shifted
                perm=c.rng(seed,20,'joint_permutation_'+family).permutation(side*side)
                inv=np.argsort(perm); reordered={k:v.copy() for k,v in base.items()}
                for field in ('tokens','coordinates','valid'):
                    reordered[field]=base[field][:,:,perm].copy()
                reordered['query']=inv[base['query']]
                yield dict(seed=seed,arm=arm,family=family,side=side,kind='joint_permutation'),base,reordered

def emit_control(root, meta, base, altered, base_logits, logits, source_sha):
    key=f"s{meta['seed']}_{meta['arm']}_{meta['family']}_N{meta['side']}_{meta['kind']}"
    base_path,base_hash=save_bank(root,'control_'+key+'_base',base)
    altered_path,altered_hash=save_bank(root,'control_'+key+'_altered',altered)
    bp=c.metrics(base_logits,base['target'])['probability']
    p=c.metrics(logits,altered['target'])['probability']
    delta=float(np.max(np.abs(bp-p)))
    path='controls/'+key+'.npz'
    c.save(root/path,logits=logits,base_logits=base_logits,target=base['target'],row_id=base['row_id'],
           base_bank=np.array(base_path),altered_bank=np.array(altered_path),base_hash=np.array(base_hash),
           altered_hash=np.array(altered_hash),source_sha256=np.array(source_sha),
           identity=np.array(key),study=np.array(c.CFG['study']))
    row=dict(**meta,key=key,path=path,base_bank=base_path,altered_bank=altered_path,
             base_hash=base_hash,altered_hash=altered_hash,source_sha256=source_sha,
             rows=len(p),max_probability_change=delta,passed=delta<=2e-5)
    c.write(root/'controls'/(key+'.json'),row)
    return row

def negative_controls(model,seed,arm,root,source_sha):
    return [emit_control(root,meta,base,changed,m.predict(model,base),m.predict(model,changed),source_sha)
            for meta,base,changed in control_banks(seed,arm)]

def run(device='cuda'):
    lock=c.read(c.OUT/'finals_locked.json'); assert len(lock['finals'])==12
    for rel,expected in lock['finals'].items(): assert c.sha(c.OUT/rel)==expected
    plan=list(jobs()); assert len(plan)==600
    c.write(c.OUT/'evaluation_manifest.json',dict(jobs=plan,jobs_count=600,primary='G8_test_N20_center',unlocked=time.time()))
    rows=[]; controls=[]
    for seed in c.CFG['seed_labels']:
        for arm in c.CFG['arms']:
            folder=c.OUT/'training'/f's{seed}_{arm}';model=m.new_model(seed,device)
            model_jobs=[j for j in plan if j['seed']==seed and j['arm']==arm];loaded=None
            for job in model_jobs:
                c.deadline(); path=folder/('final.pt' if job['step']==8000 else f"step_{job['step']}.pt")
                identity=(job['step'],job['weight'])
                if identity!=loaded:
                    payload=torch.load(path,map_location='cpu',weights_only=False)
                    assert payload['study']==c.CFG['study'] and payload['step']==job['step']
                    assert payload['seed']==seed and payload['arm']==arm
                    model.load_state_dict(payload[job['weight']]);source_sha=c.sha(path);loaded=identity
                bank=input_bank(job);logits=m.predict(model,bank)
                rows.append(emit_prediction(c.OUT,job,bank,logits,source_sha))
                if len(rows)%25==0:c.status('evaluation',predictions=len(rows),expected=600,current=job_key(job))
            payload=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(payload['raw'])
            controls.extend(negative_controls(model,seed,arm,c.OUT,c.sha(folder/'final.pt')))
            c.write(c.OUT/'evaluation'/f's{seed}_{arm}'/'complete.json',
                    dict(time=time.time(),predictions=len(model_jobs),controls=10,final_sha=c.sha(folder/'final.pt')))
            del model,payload
    assert len(rows)==600 and len(controls)==120
    c.write(c.OUT/'evaluation_complete.json',dict(time=time.time(),rows=rows,controls=controls,count=len(rows)))
    return rows,controls
