"""Independent CPU rechecks for the historical RG pilot and stage-A evidence."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
PILOT = REPO / 'experiments/rg-coarse-data-training-transfer/evidence'
RUNS = ROOT / 'project/runs'
OUT = ROOT / 'integrated/data'
SOURCES = {}


def record(path):
    SOURCES[path.relative_to(REPO).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    record(path)
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def read(path):
    record(path)
    return json.loads(path.read_text(encoding='utf8'))


def csvout(name, rows):
    path = OUT / name
    with path.open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def oracle(bank):
    noisy, queries = bank['noisy'], bank['queries']
    width = noisy.shape[1]
    rows = np.arange(len(noisy))[:, None]
    x, y = queries // width, queries % width
    assert ((x > 0) & (y > 0) & (x < width-1) & (y < width-1)).all()
    neighbours = np.stack([noisy[rows,x-1,y],noisy[rows,x+1,y],noisy[rows,x,y-1],noisy[rows,x,y+1]])
    assert np.isin(neighbours,[0,1]).all()
    return 1 / (1 + np.exp(-np.log1p(np.sqrt(2)) * (2*neighbours-1).sum(0)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    beta = np.log1p(np.sqrt(2))/2
    max_ce = max_kl = max_summary = 0.
    banks = {f'fine_W{w}_M{m}': load(ROOT / f'project/inputs/banks/fine_W{w}_M{m}.npz')
             for w in (48,96) for m in (1,8,32)}
    targets = {name:oracle(bank) for name,bank in banks.items()}
    for name, target in targets.items():
        np.testing.assert_allclose(target,banks[name]['target'],rtol=0,atol=1e-15)
    pilot_rows=[]; vectors={}; counts={'pilot_predictions':0,'pilot_exact_predictions':0}
    for path in sorted((PILOT/'predictions').rglob('*.npz')):
        data=load(path);p=data['probabilities'];y=data['labels']
        ce=-(y*np.log(p)+(1-y)*np.log1p(-p)).mean(1)
        max_ce=max(max_ce,float(abs(ce-data['empirical_ce']).max()))
        counts['pilot_predictions']+=1
        key,stage,split=path.parts[-4:-1]
        values={'CE':ce}
        if 'exact_kl' in data:
            q=targets[path.stem]
            assert np.array_equal(y,banks[path.stem]['labels'])
            kl=(q*np.log(q/p)+(1-q)*np.log((1-q)/(1-p))).mean(1)
            max_kl=max(max_kl,float(abs(kl-data['exact_kl']).max()))
            values['KL']=kl;counts['pilot_exact_predictions']+=1
        for metric,value in values.items():
            vectors[key,stage,split,path.stem,metric]=value
    assert counts=={'pilot_predictions':297,'pilot_exact_predictions':72}
    groups={
        'primary_fine_W96_KL':([f'fine_W96_M{m}' for m in (1,8,32)],'KL'),
        'secondary_fine_W48_KL':([f'fine_W48_M{m}' for m in (1,8,32)],'KL'),
        'secondary_coarse_CE':([f'coarse_W32_t{t}' for t in (10,50,90)],'CE'),
        'fine_retention_CE':([f'retention_W48_t{t}' for t in (10,50,90)],'CE'),
        'coarse_level0_CE':([f'coarse_W32_t{t}_level0' for t in (10,50,90)],'CE'),
    }
    published=read(PILOT/'analysis/summary.json')
    matrices={}
    for group,(names,metric) in groups.items():
        for arm in ('initial','fine_only','fine_plus_rg','fine_replay'):
            if arm=='initial' and group=='coarse_level0_CE':continue
            stage='initial' if arm=='initial' else 'final'
            mat=np.stack([np.stack([vectors[f's{seed}_{arm}',stage,'test',name,metric] for name in names]).mean(0)
                          for seed in (92601,92602,92603)])
            matrices[group,arm]=mat
            if arm!='initial':
                expected=published['summaries'][group]['arms'][arm]['seed_means']
                max_summary=max(max_summary,float(abs(mat.mean(1)-expected).max()))
            for seed,value in zip((92601,92602,92603),mat.mean(1)):
                pilot_rows.append(dict(seed=seed,arm=arm,stage=stage,group=group,value=float(value)))
    csvout('history_pilot_metrics.csv',pilot_rows)
    # Independently reconstruct the primary paired t interval and all 2000
    # jointly resampled seed / reference-chain / circular-block replicates.
    delta=matrices['primary_fine_W96_KL','fine_plus_rg']-matrices['primary_fine_W96_KL','fine_replay']
    per_seed=delta.mean(1);radius=student_t.ppf(.975,2)*per_seed.std(ddof=1)/np.sqrt(3)
    t95=[float(per_seed.mean()-radius),float(per_seed.mean()+radius)]
    tag='primary_fine_W96_KLfine_plus_rgfine_replay'
    words=np.frombuffer(hashlib.sha256(tag.encode()).digest()[:16],dtype='<u4').tolist()
    rng=np.random.default_rng(np.random.SeedSequence([2026100123,0,0,*words]))
    chains=banks['fine_W96_M1']['chain_ids'];chain_indices=[np.flatnonzero(chains==c) for c in np.unique(chains)]
    bootstrap=[]
    for _ in range(2000):
        seeds=rng.integers(3,size=3);selected=[]
        for ch in rng.integers(16,size=16):
            ids=chain_indices[ch];start=rng.integers(16,size=4)
            selected.extend(ids[((start[:,None]+np.arange(4))%16).ravel()])
        bootstrap.append(delta[seeds][:,selected].mean())
    joint=np.quantile(bootstrap,[.025,.975])
    expected=published['summaries']['primary_fine_W96_KL']['comparisons']['fine_plus_rg_minus_fine_replay']
    # Student-t inverse-CDF implementations differ slightly by SciPy version.
    t_interval_difference=float(np.max(abs(np.asarray(t95)-expected['t95'])))
    np.testing.assert_allclose(t95,expected['t95'],atol=1e-10,rtol=0)
    np.testing.assert_allclose(joint,expected['joint_seed_chain_block4_bootstrap95'],atol=1e-14,rtol=0)
    # Recompute all 126 stage-A exact scores, including previously unscored midpoint weights.
    exact=[]
    for path in sorted((RUNS/'A_v2/A0_exact').rglob('*.npz')):
        data=load(path);p=data['p'];q=targets[path.stem]
        kl=(q*np.log(q/p)+(1-q)*np.log((1-q)/(1-p))).mean(1)
        max_kl=max(max_kl,float(abs(kl-data['KL']).max()))
        key=path.parent.name;seed=int(key[1:6]);stage=key.rsplit('_',1)[1];arm=key[7:-(len(stage)+1)]
        width=int(path.stem.split('_')[1][1:]);masked=int(path.stem.split('_')[2][1:])
        exact.append(dict(seed=seed,arm=arm,stage=stage,width=width,n_mask=masked,KL=float(kl.mean())))
    assert len(exact)==126
    csvout('history_exact_cells.csv',exact)
    trajectory=[]
    for seed in (92601,92602,92603):
        for arm in ('initial','fine_only','fine_plus_rg','fine_replay'):
            for stage in (['initial'] if arm=='initial' else ['midpoint','final']):
                for w in (48,96):
                    value=np.mean([x['KL'] for x in exact if (x['seed'],x['arm'],x['stage'],x['width'])==(seed,arm,stage,w)])
                    trajectory.append(dict(seed=seed,arm=arm,stage=stage,width=w,KL=float(value)))
    csvout('history_exact_trajectory.csv',trajectory)
    # Recompute all retention CE from probabilities; split the 20 geometry tasks
    # into 7 continuous and 13 nonuniform tasks before interpreting forgetting.
    retention=[]
    for path in sorted((RUNS/'A_supp/A0_retention').rglob('*.npz')):
        d=load(path);p=d['p'];y=d['labels'];ce=-(y*np.log(p)+(1-y)*np.log1p(-p)).mean(1)
        max_ce=max(max_ce,float(abs(ce-d['CE']).max()))
        key=path.parent.name;seed=int(key[1:6]);stage=key.rsplit('_',1)[1];arm=key[7:-(len(stage)+1)]
        task=path.stem
        family=('continuous_geometry' if task.startswith(('geometry_C48','geometry_C96')) else
                'nonuniform_geometry' if task.startswith('geometry_') else task.split('_')[0])
        retention.append(dict(seed=seed,arm=arm,stage=stage,task=task,family=family,CE=float(ce.mean())))
    assert len(retention)==735
    csvout('history_retention_cells.csv',retention)
    families=[]
    for seed in (92601,92602,92603):
        for arm in ('initial','fine_only','fine_plus_rg','fine_replay'):
            for stage in (['initial'] if arm=='initial' else ['midpoint','final']):
                for family in ('continuous_geometry','nonuniform_geometry','container','ordinary'):
                    rows=[r for r in retention if (r['seed'],r['arm'],r['stage'],r['family'])==(seed,arm,stage,family)]
                    initial=[r['CE'] for r in retention if (r['seed'],r['arm'],r['stage'],r['family'])==(seed,'initial','initial',family)]
                    assert len(rows)==len(initial)==dict(continuous_geometry=7,nonuniform_geometry=13,container=9,ordinary=6)[family]
                    mean=np.mean([r['CE'] for r in rows]);base=np.mean(initial)
                    families.append(dict(seed=seed,arm=arm,stage=stage,family=family,n_tasks=len(rows),CE=float(mean),initial_CE=float(base),change=float(mean-base)))
    csvout('history_retention_families.csv',families)
    # Independent full 512-state Hamiltonian and conditional-marginal factorization.
    joint_rows=[];max_truth=max_joint=max_identity=0.
    plans={'serial':[0],'adjacent':[0,1,2],'checker':[0,2,4],'nine':list(range(9))}
    for path in sorted((RUNS/'A_supp/A2').rglob('*.npz')):
        d=load(path);bits=d['bits'];s=(2*bits.astype(np.int64)-1).reshape(512,3,3);ex=d['exterior']
        bond=(s[:,1:]*s[:,:-1]).sum((1,2))+(s[:,:,1:]*s[:,:,:-1]).sum((1,2))
        bond+=(s[:,0,:]*ex[5,6:9]).sum(1)+(s[:,-1,:]*ex[9,6:9]).sum(1)
        bond+=(s[:,:,0]*ex[6:9,5]).sum(1)+(s[:,:,-1]*ex[6:9,9]).sum(1)
        p=np.exp(beta*(bond-bond.max()));p/=p.sum();max_truth=max(max_truth,float(abs(p-d['truth']).max()))
        for name,first in plans.items():
            groups=[first]+[[j] for j in range(9) if j not in first];prior=[];oracle_joint=np.ones(512);conditional=0.;q=d[name+'_Q']
            for group in groups:
                code=bits[:,prior]@(2**np.arange(len(prior))) if prior else np.zeros(512,dtype=int)
                pmass=np.bincount(code,weights=p,minlength=2**len(prior));qmass=np.bincount(code,weights=q,minlength=len(pmass))
                for j in group:
                    pc=np.bincount(code,weights=p*bits[:,j],minlength=len(pmass))/pmass
                    qc=np.bincount(code,weights=q*bits[:,j],minlength=len(qmass))/qmass
                    oracle_joint*=np.where(bits[:,j],pc[code],1-pc[code])
                    conditional+=np.sum(pmass*(pc*np.log(pc/qc)+(1-pc)*np.log((1-pc)/(1-qc))))
                prior+=group
            saved=json.loads(str(d[name+'_result']));factor=np.sum(p*np.log(p/oracle_joint));learned=np.sum(p*np.log(p/q))
            max_joint=max(max_joint,float(abs(oracle_joint-d[name+'_oracle']).max()))
            max_identity=max(max_identity,float(abs(learned-factor-conditional)))
            for actual,key in [(factor,'parallel_factorization_kl'),(learned,'learned_kl'),(conditional,'single_conditional_kl')]:
                assert abs(actual-saved[key])<1e-10,(path,name,key)
            assert abs(q.sum()-1)<1e-8 and abs(oracle_joint.sum()-1)<1e-12
            joint_rows.append(dict(model=path.parent.name,boundary=path.stem,plan=name,learned_KL=float(learned),factorization_KL=float(factor),conditional_KL=float(conditional)))
    assert len(joint_rows)==396*4
    csvout('history_joint_decomposition.csv',joint_rows)
    # Raw historical optimization records, not the summarized clipping table.
    oldlogs=ROOT/'project/artifacts/rg_data_training_pilot_20261001_remote/extracted_rgpilot_v1/artifacts/rg_data_training_pilot_20261001/training'
    logrows=[]
    for path in sorted(oldlogs.glob('*/log.jsonl')):
        record(path);records=[json.loads(x) for x in path.read_text().splitlines()]
        logrows.extend(records)
    assert len(logrows)==30720
    largest_grad=max(x['grad'] for x in logrows);clipped=sum(x['grad']>1 for x in logrows)
    audit=dict(status='passed',source_commit='5516370b3ae6e98160d9468668acdcbf851dc902',
               counts={**counts,'A_exact':len(exact),'A_retention':len(retention),'A_joint_cases':len(joint_rows)//4,'historical_updates':len(logrows)},
               maximum_CE_difference=max_ce,maximum_exact_KL_difference=max_kl,maximum_pilot_summary_difference=max_summary,
               maximum_Hamiltonian_truth_difference=max_truth,maximum_oracle_joint_difference=max_joint,
               maximum_forward_KL_decomposition_residual=max_identity,clip_events=clipped,maximum_unclipped_gradient=largest_grad,
               pilot_primary=dict(paired_seed_values=per_seed.tolist(),t95=t95,joint_2000_bootstrap95=joint.tolist(),t_interval_cross_SciPy_difference=t_interval_difference),
               sources=SOURCES,scope='Independent probabilities/labels, nearest-neighbor local oracle, Hamiltonian enumeration and P-prefix KL decomposition; no GPU inference or new generation')
    assert max_ce<1e-12 and max_kl<1e-12 and max_summary<1e-12 and max_truth<1e-12 and max_joint<1e-12 and max_identity<1e-10 and clipped==0
    dest=ROOT/'integrated/provenance/history_audit.json'
    dest.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in audit.items() if k!='sources'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
