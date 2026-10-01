"""Frozen final/midpoint evaluation, paired pilot summaries, no generation."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t as student_t
import common as c
import model_data as m

def evaluate(model,seed,arm,stage,split,bank_paths,source_sha,level0_ablation=False):
    start=time.time();folder=c.OUT/f'predictions/s{seed}_{arm}/{stage}/{split}'
    for path in bank_paths:
        c.deadline()
        bank=c.old.load(path);result=m.predict(model,bank)
        result.update(bank_sha256=np.array(c.sha(path)),source_sha256=np.array(source_sha),
            level=np.array(int(bank['level'])),chain_ids=bank['chain_ids'],labels=bank['labels'])
        c.save(folder/(path.stem+'.npz'),**result)
        if level0_ablation and path.name.startswith('coarse_'):
            ablated=m.predict(model,bank,level_override=0)
            ablated.update(bank_sha256=result['bank_sha256'],source_sha256=result['source_sha256'],
                level=np.array(0),chain_ids=bank['chain_ids'],labels=bank['labels'])
            c.save(folder/(path.stem+'_level0.npz'),**ablated)
    c.write(folder/'complete.json',dict(seed=seed,arm=arm,stage=stage,split=split,banks=len(bank_paths),
        coarse_level0_ablation=level0_ablation,seconds=time.time()-start,time=time.time()))
    c.log('prediction_set_complete',seed=seed,arm=arm,stage=stage,split=split,banks=len(bank_paths))

def evaluate_finals(states,proto,test_paths,learning_paths):
    for state in states:
        seed,arm=state['seed'],state['arm'];folder=c.OUT/f'training/s{seed}_{arm}'
        fsha=c.sha(folder/'final.pt')
        evaluate(state['ema'],seed,arm,'final','test',test_paths,fsha,level0_ablation=True)
        evaluate(state['ema'],seed,arm,'final','learning',learning_paths,fsha)
        payload=torch.load(folder/'midpoint_ema.pt',map_location='cpu',weights_only=False)
        model=m.ScaleDenoiser().cuda().eval();model.load_state_dict(payload['ema'],strict=True)
        evaluate(model,seed,arm,'midpoint','learning',learning_paths,c.sha(folder/'midpoint_ema.pt'))
        del model

def metric_vector(seed,arm,names,metric='empirical_ce',stage='final',split='test'):
    prefix=c.OUT/f'predictions/s{seed}_{arm}/{stage}/{split}'
    return np.stack([c.old.load(prefix/(name+'.npz'))[metric] for name in names]).mean(0)

def paired_summary(array,chains,tag):
    # Each row is one paired model lineage; columns are shared MC parent samples.
    a=np.asarray(array,dtype=float);means=a.mean(1);mean=float(means.mean())
    se=float(means.std(ddof=1)/np.sqrt(len(means)))
    radius=float(student_t.ppf(.975,len(means)-1)*se)
    r=c.rng(0,0,tag,root=c.CFG['bootstrap_seed']);unique=np.unique(chains)
    groups=[np.flatnonzero(chains==k) for k in unique];values=[]
    for _ in range(c.CFG['bootstrap_replicates']):
        sampled_seeds=r.integers(len(means),size=len(means))
        selected=[]
        for which in r.integers(len(groups),size=len(groups)):
            group=groups[int(which)];n=len(group);block=4
            starts=r.integers(n,size=(n+block-1)//block)
            selected.extend(group[((starts[:,None]+np.arange(block))%n).ravel()[:n]])
        values.append(float(a[sampled_seeds][:,selected].mean()))
    values=np.asarray(values)
    q=np.quantile(values,[.025,.975]);batchq=np.quantile(values.reshape(5,-1),[.025,.975],axis=1)
    signs=np.array([[1 if bits&(1<<j) else -1 for j in range(3)] for bits in range(8)])
    permutation=float(np.mean(np.abs((signs*means).mean(1))>=abs(mean)-1e-15))
    return dict(mean=mean,paired_seed_values=means.tolist(),seed_standard_error=se,t95=[mean-radius,mean+radius],
        joint_seed_chain_block4_bootstrap95=q.tolist(),bootstrap_replicates=len(values),
        bootstrap_quantile_batch_sd=batchq.std(axis=1,ddof=1).tolist(),
        sign_flip_two_sided_p=permutation,leave_one_seed_out_means=[float(np.delete(means,i).mean()) for i in range(3)],
        inferential_status='descriptive_underpowered_pilot_not_confirmatory')

def analyze():
    fine96=[f'fine_W96_M{m}' for m in (1,8,32)]
    fine48=[f'fine_W48_M{m}' for m in (1,8,32)]
    coarse=[f'coarse_W32_t{v}' for v in (10,50,90)]
    retention=[f'retention_W48_t{v}' for v in (10,50,90)]
    groups=dict(primary_fine_W96_KL=(fine96,'exact_kl'),secondary_fine_W48_KL=(fine48,'exact_kl'),
        secondary_coarse_CE=(coarse,'empirical_ce'),fine_retention_CE=(retention,'empirical_ce'),
        coarse_level0_CE=([name+'_level0' for name in coarse],'empirical_ce'))
    chains=c.old.load(c.OUT/'banks/test'/f'{fine96[0]}.npz')['chain_ids'];summaries={};cells=[]
    for name,(names,metric) in groups.items():
        matrices={arm:np.stack([metric_vector(seed,arm,names,metric) for seed in c.SEEDS]) for arm in c.ARMS}
        comparisons={}
        for a,b in [('fine_plus_rg','fine_replay'),('fine_plus_rg','fine_only'),('fine_replay','fine_only')]:
            comparisons[f'{a}_minus_{b}']=paired_summary(matrices[a]-matrices[b],chains,name+a+b)
        summaries[name]=dict(metric=metric,banks=names,arms={arm:dict(seed_means=a.mean(1).tolist(),mean=float(a.mean())) for arm,a in matrices.items()},comparisons=comparisons)
        for arm in c.ARMS:
            for i,seed in enumerate(c.SEEDS):cells.append(dict(summary=name,arm=arm,seed=seed,value=float(matrices[arm][i].mean())))
    retention_changes={}
    initial=np.stack([metric_vector(seed,'initial',retention,stage='initial') for seed in c.SEEDS])
    for arm in c.ARMS:
        end=np.stack([metric_vector(seed,arm,retention) for seed in c.SEEDS])
        change=paired_summary(end-initial,chains,'retention_'+arm)
        change['mean_above_warning_margin']=change['mean']>c.CFG['fine_retention_warning_margin_CE']
        change['any_seed_above_warning_margin']=any(x>c.CFG['fine_retention_warning_margin_CE'] for x in change['paired_seed_values'])
        retention_changes[arm]=change
    learning={}
    for arm in c.ARMS:
        for group,names in [('fine',retention),('coarse',coarse)]:
            before=np.array([metric_vector(seed,'initial',names,stage='initial',split='learning').mean() for seed in c.SEEDS])
            mid=np.array([metric_vector(seed,arm,names,stage='midpoint',split='learning').mean() for seed in c.SEEDS])
            end=np.array([metric_vector(seed,arm,names,stage='final',split='learning').mean() for seed in c.SEEDS])
            learning[arm+'_'+group]=dict(initial=before.tolist(),midpoint=mid.tolist(),final=end.tolist(),mid_minus_final=(mid-end).tolist())
    p=summaries['primary_fine_W96_KL']['comparisons']['fine_plus_rg_minus_fine_replay']
    result=dict(status='analysis_complete_pilot_only',study=c.CFG['study'],time=time.time(),summaries=summaries,
        retention_changes=retention_changes,learning=learning,primary_direction='improvement' if p['mean']<0 else 'no_mean_improvement',
        primary_both_descriptive_upper_bounds_below_zero=p['t95'][1]<0 and p['joint_seed_chain_block4_bootstrap95'][1]<0,
        confirmatory_success_claim=False,no_inverse_rg=True,no_generation=True,
        limitations=['three paired old lineages only','short fixed fine-tuning','reference data previously inspected',
        'coarse CE is empirical, not exact joint likelihood','conditional metrics do not establish generation or RG fixed-point correctness',
        'RG intervention includes larger physical field of view','descriptive intervals are not a calibrated coverage study'])
    c.write(c.OUT/'analysis/summary.json',result)
    c.write(c.OUT/'analysis/all_seed_metrics.json',cells)
    return result

def figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':1.3,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    colors=['#767676','#0F4D92','#B64342'];markers=['o','s','^'];labels=['Fine only','Fine + RG','Fine replay']
    target=c.OUT/'analysis';target.mkdir(exist_ok=True)
    groupnames=['primary_fine_W96_KL','secondary_coarse_CE','fine_retention_CE']
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.2))
    for ax,name,label in zip(axes,groupnames,['Fine W96 local KL','Coarse W32 masked CE','Fine W48 retention CE']):
        for i,arm in enumerate(c.ARMS):
            values=np.array(result['summaries'][name]['arms'][arm]['seed_means'])
            ax.scatter(i+np.array([-.08,0,.08]),values,c=colors[i],marker=markers[i],s=38)
            ax.errorbar(i,values.mean(),yerr=student_t.ppf(.975,2)*values.std(ddof=1)/np.sqrt(3),color=colors[i],fmt='_',capsize=4,markersize=15)
        ax.set_xticks(range(3),['Fine','Fine + RG','Replay'],rotation=18);ax.set_ylabel(label)
    fig.tight_layout(pad=1.8)
    for ext in ['png','pdf']:fig.savefig(target/f'01_all_arms.{ext}',dpi=300)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.2))
    for ax,name,label in zip(axes,groupnames,['Fine W96 local KL difference','Coarse W32 CE difference','Fine W48 CE difference']):
        row=result['summaries'][name]['comparisons']['fine_plus_rg_minus_fine_replay']
        vals=np.array(row['paired_seed_values']);ax.scatter([0,1,2],vals,c='#0F4D92',s=44)
        mean=row['mean'];lo,hi=row['t95'];blo,bhi=row['joint_seed_chain_block4_bootstrap95']
        ax.errorbar(3,mean,yerr=[[mean-lo],[hi-mean]],fmt='s',color='#272727',capsize=5)
        ax.plot([3.25,3.25],[blo,bhi],color='#B64342',lw=4)
        ax.axhline(0,color='#999999',ls='--');ax.set_xticks([0,1,2,3],['92601','92602','92603','Mean'],rotation=25)
        ax.set_ylabel(label)
    fig.tight_layout(pad=1.8)
    for ext in ['png','pdf']:fig.savefig(target/f'02_paired_differences.{ext}',dpi=300)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.3))
    for ax,group,label in zip(axes,['fine','coarse'],['Validation fine CE','Validation coarse CE']):
        for i,arm in enumerate(c.ARMS):
            row=result['learning'][arm+'_'+group]
            values=np.array([row[k] for k in ['initial','midpoint','final']])
            ax.errorbar([0,1024,2048],values.mean(1),yerr=student_t.ppf(.975,2)*values.std(axis=1,ddof=1)/np.sqrt(3),
                color=colors[i],marker=markers[i],label=labels[i],capsize=3)
        ax.set_xlabel('Common fine-update cycles');ax.set_ylabel(label)
    axes[1].legend(fontsize=10);fig.tight_layout(pad=1.8)
    for ext in ['png','pdf']:fig.savefig(target/f'03_fixed_learning.{ext}',dpi=300)
    plt.close(fig)
    c.write(target/'figure_captions.json',dict(
        figure01='Three paired lineages (points), arithmetic mean and descriptive paired-lineage t 95% intervals. Entropies differ by scale: compare arms within each panel, not CE values between scales.',
        figure02='Fine + RG minus compute-matched fine replay. Negative favours RG. Individual lineage differences; black t 95% interval; red joint seed/MC-chain/circular-block4 bootstrap 95% interval (2000 draws). Three seeds remain underpowered; no confirmatory claim.',
        figure03='Fixed initial, midpoint and final EMA validation scores. Fine-only has one update per cycle; other arms have two. Means and t 95% intervals across the same three lineages. No checkpoint is selected using these curves.'))

def audit_banks(test_parent,learning_parent):
    count=0
    for split,parent in [('test',test_parent),('learning',learning_parent)]:
        for path in (c.OUT/'banks'/split).glob('*.npz'):
            actual=c.old.load(path);name=path.stem;n=len(parent.spins)
            if name.startswith('fine_'):
                _,width,mask=name.split('_');w=int(width[1:]);mm=int(mask[1:])
                expected=c.data.local_bank(parent,np.arange(n),w,mm,f'rgpilot_test_fine_W{w}_M{mm}')
                expected['level']=np.array(0);expected['chain_ids']=parent.chain_ids
                expected['target_kind']=expected['target_type']
            else:
                kind,width,clock=name.split('_');w=int(width[1:]);t=int(clock[1:])/100
                expected=m.empirical_bank(parent,n,w,t,kind=='coarse',f'rgpilot_{split}_{name}')
            assert set(actual)==set(expected)
            for key in actual:assert np.array_equal(actual[key],expected[key]),(path.name,key)
            count+=1
    assert count==18
    c.write(c.OUT/'audit/banks.json',dict(status='passed',banks=count,all_native_fields_rebuilt=True,time=time.time()))

def audit_predictions():
    count=0
    for path in (c.OUT/'predictions').rglob('*.npz'):
        a=c.old.load(path);split=path.parent.name;name=path.stem.removesuffix('_level0')
        bp=c.OUT/'banks'/split/(name+'.npz');bank=c.old.load(bp)
        assert c.sha(bp)==str(a['bank_sha256'])
        seed_arm=path.parents[2].name;stage=path.parents[1].name
        seed=int(seed_arm.split('_')[0][1:]);arm=seed_arm.split('_',1)[1]
        source=(c.OUT/f'initial/s{seed}_ema.pt') if stage=='initial' else c.OUT/'training'/seed_arm/('final.pt' if stage=='final' else 'midpoint_ema.pt')
        assert c.sha(source)==str(a['source_sha256'])
        assert np.array_equal(a['labels'],bank['labels']) and np.array_equal(a['chain_ids'],bank['chain_ids'])
        p=a['probabilities'];assert np.isfinite(p).all() and ((p>0)&(p<1)).all()
        labels=bank['labels'];ce=-(labels*np.log(p)+(1-labels)*np.log1p(-p)).mean(1)
        assert np.allclose(ce,a['empirical_ce'],atol=1e-12,rtol=0)
        rows=np.arange(len(p))[:,None]
        assert (bank['noisy'].reshape(len(p),-1)[rows,bank['queries']]==2).all()
        assert np.array_equal(bank['clean'].reshape(len(p),-1)[rows,bank['queries']],labels)
        if 'exact_kl' in a:
            q=c.data.oracle_probability(bank['noisy'],bank['queries'])
            assert np.array_equal(q,bank['target'])
            kl=(q*(np.log(q)-np.log(p))+(1-q)*(np.log1p(-q)-np.log1p(-p))).mean(1)
            assert np.allclose(kl,a['exact_kl'],atol=1e-12,rtol=0)
        count+=1
    # Initial: 3*(12 test + 6 learning). Final: 9*(15 test + 6 learning + 6 midpoint learning).
    expected=3*18+9*27
    assert count==expected,(count,expected)
    result=dict(status='passed',prediction_files=count,all_input_digests_matched=True,
        all_parent_metrics_recomputed=True,coarse_oracle_never_used=True,time=time.time())
    c.write(c.OUT/'audit/predictions.json',result);return result
