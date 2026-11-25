"""Reproducible stage-A tables and publication figures from saved raw results."""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,TwoSlopeNorm,LinearSegmentedColormap
from report_supplements import a3_intervals,a2_states

SEEDS=(92601,92602,92603)
ARMS=('initial','fine_only','fine_plus_rg','fine_replay')
LABELS={'initial':'Initial I-F','fine_only':'Fine only','fine_plus_rg':'Fine + RG','fine_replay':'Fine replay','MC':'MC'}
COLORS={'initial':'#263e41','fine_only':'#657895','fine_plus_rg':'#b65322','fine_replay':'#89568e','MC':'#263e41','D':'#1877a3','M':'#b65322'}
STYLES={'initial':':','fine_only':'--','fine_plus_rg':'-','fine_replay':'-.'}
SPIN=ListedColormap(['#f3ecda','#193f41'])
CAPTIONS={}
plt.rcParams.update({'font.family':['DejaVu Sans','sans-serif'],'font.size':11,'axes.linewidth':1.2,
                     'axes.spines.right':False,'axes.spines.top':False,'legend.frameon':False,
                     'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')

def csvout(p,rows):
    if not rows:return
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):h.update(data)
    return h.hexdigest()

def save(fig,out,name,caption):
    (out/'figures').mkdir(parents=True,exist_ok=True)
    fig.tight_layout(pad=1.5)
    for ext in ('png','pdf'):fig.savefig(out/'figures'/(name+'.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig);CAPTIONS[name]=caption

def majority(x):
    w=x.shape[-1]
    return np.sign(x.reshape(*x.shape[:-2],w//3,3,w//3,3).sum((-3,-1))).astype(np.int8)

def ecdf(ax,x,**kwargs):
    v=np.sort(x);ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',**kwargs)

def mapax(ax,x):
    ax.imshow(x,cmap=SPIN,vmin=-1,vmax=1,interpolation='nearest',origin='upper');ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_visible(False)

def a3(project,runs,out):
    supp=runs/'A_supp';data=load(supp/'A3_per_field.npz');z=load(project/'inputs/mc_W144_centered.npz')
    bands=a3_intervals(project,runs,out)
    x=2*np.unpackbits(z['fine_bits'],axis=-1,bitorder='little').astype(np.int8)-1
    ids=[int(np.flatnonzero(z['chain']==c)[0]) for c in range(4)]
    for w in (96,144):
        start=(144-w)//2
        for j,i in enumerate(ids):
            v=x[i,start:start+w,start:start+w];d=v[1::3,1::3];m=majority(v)
            fig,ax=plt.subplots(2,3,figsize=(10,6.4))
            for axis,img,label in zip(ax.flat[:4],(v,d,m,m!=d),('a  Fine MC','b  Decimation','c  Majority','d  Disagreement')):
                if label.startswith('d'):
                    axis.imshow(img,cmap=ListedColormap(['#f3ecda','#b65322']),vmin=0,vmax=1,interpolation='nearest');axis.set_xticks([]);axis.set_yticks([])
                else:mapax(axis,img)
                axis.set_xlabel(label)
            for tag,label,style in [('D','Decimation','--'),('M','Majority','-')]:
                g=data[f'{tag}{w}_g'].mean(0);ax[1,1].plot(np.arange(1,len(g)),g[1:],color=COLORS[tag],ls=style,label=label)
                ecdf(ax[1,2],data[f'{tag}{w}_m'],color=COLORS[tag],ls=style,label=label)
                b=bands[f'{tag}{w}']
                ax[1,1].fill_between(np.arange(1,len(g)),b['g_lower'][1:],b['g_upper'][1:],color=COLORS[tag],alpha=.15,lw=0)
                ax[1,2].fill_between(b['cdf_grid'],b['cdf_lower'],b['cdf_upper'],color=COLORS[tag],alpha=.10,lw=0,step='post')
            ax[1,1].set(xlabel='Coarse displacement k',ylabel='Raw G(k)',ylim=(0,.82),xlim=(1,w//6))
            sec=ax[1,1].secondary_xaxis('top',functions=(lambda k:3*k,lambda r:r/3));sec.set_xlabel('Physical displacement 3k')
            ax[1,2].set(xlabel='Signed magnetization m',ylabel='ECDF',xlim=(-1,1),ylim=(0,1));ax[1,2].legend(fontsize=9)
            save(fig,out,f'A3_W{w}_chain{j}',f'历史 MC：chain={j}, ordinal=0, parent={i}, global crop origin={440+start}. 前四格来自同一父场；曲线和 ECDF 使用全部 2048 场、16 链，开窗口边界。阴影为10000次完整MC链重抽样的逐点95%分位区间，保留同链相关性及D/M配对，不是同时置信带；分歧率及配对曲线差的区间另见A3_chain_intervals.csv。')
    dep=json.loads((supp/'A3_dependence.json').read_text());b=dep['block'];draws=load(supp/'A3N_draws.npz')
    fig,ax=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
    for axis,tag in zip(ax,('fine96','M96')):
        for j,n in enumerate((16,32,64,128,256,512)):
            v=draws[f'block{b}_n{n}_{tag}'];q=np.quantile(v,[.05,.5,.95])
            jitter=np.random.default_rng(2026112505+j).uniform(-.16,.16,len(v))
            axis.scatter(j+jitter,v,s=1,color='#657895',alpha=.08,rasterized=True)
            axis.plot([j,j],[q[0],q[2]],color='#263e41',lw=2);axis.scatter(j,q[1],color='#b65322',s=28,zorder=5)
        axis.set(xlabel='Pseudo-generated field count',xticks=range(6),xticklabels=[16,32,64,128,256,512],ylim=(0,.45))
    ax[0].set_ylabel('Relative raw-G error')
    save(fig,out,'A3N_noise',f'左：W96 r=25…48；右：majority W32 k=9…16。每格 2000 次；参考 256 场；偶数链为伪生成、奇数链为独立参考；链与链内循环块长 {b} 重抽。点为全部原始重复，竖线为 5%–95%，橙点为中位数。')

def a0(project,runs,out):
    manifest=json.loads((project/'inputs/manifest.json').read_text());rows=[];lookup={}
    for e in manifest['models']:
        for w in (48,96):
            vals=[]
            for m in (1,8,32):
                p=runs/'A_v2/A0_exact'/e['key']/f'fine_W{w}_M{m}.npz'
                if not p.exists():continue
                z=load(p);vals.append(z['KL'].mean())
                rows.append(dict(model=e['key'],seed=e['seed'],arm=e['arm'],stage=e['stage'],cycle=e['cycle'],updates=e['updates'],width=w,masked=m,KL=float(z['KL'].mean()),CE=float(z['CE'].mean()),n=len(z['KL'])))
            if len(vals)==3:lookup[e['seed'],e['arm'],e['stage'],w]=float(np.mean(vals))
    csvout(out/'A0_exact.csv',rows)
    fig,axs=plt.subplots(1,3,figsize=(11,3.6),sharey=True)
    for ax,seed in zip(axs,SEEDS):
        initial=lookup.get((seed,'initial','initial',96))
        for arm in ARMS[1:]:
            vals=[initial,lookup.get((seed,arm,'midpoint',96)),lookup.get((seed,arm,'final',96))]
            ax.plot([0,1024,2048],[np.nan if v is None else v for v in vals],marker='o',color=COLORS[arm],ls=STYLES[arm],label=LABELS[arm])
        ax.set(xlabel=f'Common fine updates · seed {seed}',xticks=[0,1024,2048],ylim=(.012,.05))
    axs[0].set_ylabel('W96 exact local KL (nats/query)');axs[-1].legend(fontsize=9)
    save(fig,out,'A0_trajectory','同一 256 父场、M=1/8/32 等权精确局部 KL。横轴共同细更新为 0/1024/2048；Fine-only 总更新同横轴，Fine+RG/Fine replay 总更新为 0/2048/4096；初始是已训练 I-F EMA。本图三条旧谱系分别呈现，不是确认试验。')
    retained=[]
    for e in manifest['models']:
        for p in sorted((runs/'A_supp/A0_retention'/e['key']).glob('*.npz')):
            z=load(p);retained.append(dict(seed=e['seed'],arm=e['arm'],stage=e['stage'],cycle=e['cycle'],task=p.stem,CE=float(z['CE'].mean()),n=len(z['CE'])))
    csvout(out/'A0_retention.csv',retained)
    if len(retained)==735:
        tasks=sorted({v['task'] for v in retained});differences=[];matrices=[];paired={}
        for seed in SEEDS:
            matrix=np.empty((len(tasks),3))
            for ti,task in enumerate(tasks):
                base=load(runs/'A_supp/A0_retention'/f's{seed}_initial_initial'/(task+'.npz'))
                for ai,arm in enumerate(ARMS[1:]):
                    z=load(runs/'A_supp/A0_retention'/f's{seed}_{arm}_final'/(task+'.npz'))
                    assert np.array_equal(z['parent'],base['parent']) and np.array_equal(z['chain'],base['chain'])
                    assert str(z['bank_sha'])==str(base['bank_sha'])
                    diff=z['CE']-base['CE'];matrix[ti,ai]=diff.mean()
                    paired[f's{seed}_{arm}_{task}']=diff
                    differences.append(dict(seed=seed,arm=arm,task=task,CE_final_minus_initial=float(diff.mean()),n=len(diff),chains=len(np.unique(z['chain']))))
            matrices.append(matrix)
        np.savez_compressed(out/'A0_retention_paired_differences.npz',**paired)
        csvout(out/'A0_retention_changes.csv',differences)
        limit=max(abs(np.array(matrices)).max(),1e-8)
        cmap=LinearSegmentedColormap.from_list('ce_difference',['#1877a3','#f7f7f2','#b65322'])
        fig,axes=plt.subplots(1,4,figsize=(13,11.5),gridspec_kw={'width_ratios':[1,1,1,.09]})
        for axis,seed,values in zip(axes[:3],SEEDS,matrices):
            im=axis.imshow(values,cmap=cmap,vmin=-limit,vmax=limit,aspect='auto')
            axis.set(xticks=range(3),xticklabels=['Fine only','Fine + RG','Replay'],yticks=range(len(tasks)),
                     yticklabels=tasks if axis is axes[0] else [],xlabel=f'Seed {seed}')
            axis.tick_params(axis='x',rotation=25);axis.tick_params(axis='y',labelsize=8)
        fig.colorbar(im,cax=axes[3],label='CE(final) - CE(initial), nats/query')
        save(fig,out,'A0_retention_changes','21个旧EMA、每格256父场配对复测。颜色为各终点相对同seed初始的CE变化，蓝色改善、橙色退化；所有seed共用色轴。C=连续格，U=训练间距，H=未见间距，B=聚簇未见间距，k=可见点数。容器v0=W48、v1=W96固定clock、v2=W96自然clock。ordinary为普通Bernoulli遮盖，t10/50/90分别为10%/50%/90%。这是逐任务描述，不作多重显著性宣称。')
    return dict(exact_cells=len(rows),retention_cells=len(retained),W96=[r for r in rows if r['width']==96])

def historical_logs(project,out):
    root=Path(__file__).absolute().parents[2]
    old=root/'artifacts/rg_data_training_pilot_20261001_remote/extracted_rgpilot_v1/artifacts/rg_data_training_pilot_20261001'
    if not old.exists():return
    manifest=json.loads((old/'science_manifest.json').read_text());expected={x['path']:x['sha256'] for x in manifest['files']}
    summary=[];raw=[];sources=[]
    for p in sorted((old/'training').glob('*/log.jsonl')):
        digest=sha(p);rel='artifacts/rg_data_training_pilot_20261001/'+p.relative_to(old).as_posix();assert digest==expected[rel]
        rows=[json.loads(line) for line in p.read_text().splitlines()]
        sources.append(dict(file=str(p),sha256=digest))
        for r in rows:raw.append(dict(model=p.parent.name,**r))
        for role in ('core','aux'):
            rr=[v for v in rows if v['role']==role]
            if not rr:continue
            summary.append(dict(model=p.parent.name,role=role,updates=len(rr),mean_loss=float(np.mean([v['loss'] for v in rr])),
                mean_unclipped_norm=float(np.mean([v['grad'] for v in rr])),clip_fraction=float(np.mean([v['grad']>1 for v in rr])),
                first128_loss=float(np.mean([v['loss'] for v in rr[:128]])),last128_loss=float(np.mean([v['loss'] for v in rr[-128:]]))))
    csvout(out/'A0_historical_logs.csv',raw);csvout(out/'A0_historical_training_summary.csv',summary);write(out/'historical_log_sources.json',sources)
    for seed in (92601,92602,92603):
        fig,ax=plt.subplots(2,2,figsize=(10.5,6.5))
        clipping=[]
        for arm in ('fine_only','fine_plus_rg','fine_replay'):
            for j,role in enumerate(('core','aux')):
                values=[v for v in raw if v['model']==f's{seed}_{arm}' and v['role']==role]
                if not values:continue
                groups=[values[i:i+128] for i in range(0,len(values),128)]
                x=[np.mean([v['cycle'] for v in group]) for group in groups]
                for i,metric in enumerate(('loss','grad')):
                    avg=[np.mean([v[metric] for v in group]) for group in groups]
                    ax[i,j].plot(x,avg,color=COLORS[arm],ls=STYLES[arm],label=LABELS[arm])
                maxima=[max(v['grad'] for v in group) for group in groups]
                ax[1,j].plot(x,maxima,color=COLORS[arm],ls=STYLES[arm],alpha=.35,lw=.8)
                clipping.append(sum(v['grad']>1 for v in values))
        for axis in ax.flat:axis.set(xlabel=f'Common fine updates · seed {seed}',xlim=(0,2048))
        ax[0,0].set(ylabel='Common-stream training loss',ylim=(.27,.36));ax[0,0].legend(fontsize=9)
        ax[0,1].set(ylabel='Auxiliary-stream training loss',ylim=(.27,.36))
        ax[1,0].set(ylabel='Common gradient norm, before clipping',ylim=(0,1.05))
        ax[1,1].set(ylabel='Auxiliary gradient norm, before clipping',ylim=(0,1.05))
        for axis in ax[1]:axis.axhline(1,color='#767676',ls=':',lw=1)
        save(fig,out,f'A0_training_s{seed}',f'旧谱系{seed}，原始训练日志逐文件SHA校验。实线粗度的曲线为互不重叠128周期块均值，梯度浅色细线为块内最大值；不是置信区间。实际超过裁剪阈值1的事件共{sum(clipping)}次。逐次损失、原始范数和更新身份完整保存在A0_historical_logs.csv；旧日志未记录本轮B所用的共同/辅助梯度余弦。')

def a2(runs,out):
    a2_states(runs,out)
    root=runs/'A_supp/A2';count=len(list(root.rglob('*.npz')))
    plans=('serial','adjacent','checker','nine')
    allrows=[]
    for f in sorted(root.rglob('*.npz')):
        z=load(f)
        assert abs(z['truth'].sum()-1)<1e-12
        assert np.max(abs(z['control_P']-z['control_oracle']))<1e-12
        for plan in plans:
            assert abs(z[plan+'_Q'].sum()-1)<1e-8
            r=json.loads(str(z[plan+'_result']));assert abs(r['identity_residual'])<1e-8
            allrows.append(dict(model=f.parent.name,boundary=f.stem,plan=plan,**{k:v for k,v in r.items() if k!='clock'}))
    csvout(out/'A2_all_boundaries.csv',allrows)
    colors=('#263e41','#b65322','#1877a3','#89568e')
    for seed in SEEDS:
        model=f's{seed}_fine_plus_rg_final'
        for boundary in ('all_plus','mc_chain0_ord0'):
            p=root/model/(boundary+'.npz')
            if not p.exists():continue
            z=load(p);fig,ax=plt.subplots(2,4,figsize=(13,6.3),sharex=True,sharey=True)
            lim=(min(float(z[k].min()) for k in ['truth']+[plan+s for plan in plans for s in ('_oracle','_Q')])/2,1.)
            for j,(plan,col) in enumerate(zip(plans,colors)):
                for i,suffix in enumerate(('_oracle','_Q')):
                    axis=ax[i,j];axis.plot(lim,lim,'--',color='#767676',lw=1)
                    axis.scatter(z['truth'],z[plan+suffix],s=9,facecolors='none',edgecolors=col,alpha=.7,rasterized=True)
                    axis.set(xscale='log',yscale='log',xlim=lim,ylim=lim,xlabel=f'P(x) · {plan}')
            ax[0,0].set_ylabel('Oracle Q(x)');ax[1,0].set_ylabel('Transformer Q(x)')
            save(fig,out,f'A2_joint_s{seed}_{boundary}',f'{model}, {boundary}：512 态精确 P 与各揭示方案 Q；上行精确单点 oracle，下行 Transformer；同图统一坐标。块位于 W16 第6–8行/列，模型 clock=max(剩余隐藏/256,0.01)。固定边界枚举无抽样置信区间。')
            fig,ax=plt.subplots(1,3,figsize=(12,3.7));nplus=z['bits'].sum(1)
            ax[0].plot(range(10),np.bincount(nplus,weights=z['truth'],minlength=10),'--',color='#263e41',lw=2,label='Exact P')
            for j,(plan,col) in enumerate(zip(plans,colors)):
                q=z[plan+'_Q'];r=json.loads(str(z[plan+'_result']))
                ax[0].plot(range(10),np.bincount(nplus,weights=q,minlength=10),color=col,label=plan)
                ax[1].scatter(j,r['q_majority'],color=col);ax[2].bar(j,r['parallel_factorization_kl'],color='#657895')
                ax[2].bar(j,r['single_conditional_kl'],bottom=r['parallel_factorization_kl'],color='#b65322')
            ax[1].axhline(z['truth'][nplus>=5].sum(),ls='--',color='#263e41')
            ax[0].set(xlabel='Number of + spins',ylabel='Probability',xticks=range(10));ax[0].legend(fontsize=8)
            ax[1].set(ylabel='P(majority +)',xticks=range(4),xticklabels=plans,ylim=(0,1))
            ax[2].set(ylabel='KL(P || Q), nats',xticks=range(4),xticklabels=plans)
            for axis in ax[1:]:axis.tick_params(axis='x',rotation=25)
            save(fig,out,f'A2_decomposition_s{seed}_{boundary}',f'{model}, {boundary}。左：正自旋数分布；中：块多数为正概率，虚线为真值；右：灰蓝为因子化 KL，赭橙为单点条件 KL，其和逐边界核对联合 KL。')
    return dict(cases=count,expected=396,normalization_and_KL_identity='passed for available cases')

def connected(g,m,chains=None):
    avg=g.mean(0);naive=avg-m.mean()**2
    means=m if chains is None else np.array([m[chains==c].mean() for c in np.unique(chains)])
    cross=(means.sum()**2-(means*means).sum())/(len(means)*(len(means)-1))
    return naive,avg-cross

def a1(project,runs,out,base_name='A_v2',cohort='旧谱系',expected=960):
    base=runs/base_name
    if not (base/'A1.json').exists():return dict(samples=len(list((base/'samples').rglob('*.npz'))),expected=expected,status='incomplete')
    summary=json.loads((base/'A1.json').read_text());csvout(out/'A1_metrics.csv',summary['arms']);csvout(out/'A1_BC_distance.csv',summary['paths_BC'])
    metadata=load(project/'inputs/mc_W144_centered.npz');all_mc=2*np.unpackbits(metadata['fine_bits'],axis=-1,bitorder='little').astype(np.int8)-1
    refs={w:load(base/f'MC_reference_W{w}.npz') for w in (96,32)}
    refchain=metadata['chain'][refs[96]['parent_ids']]
    curves={};corrections=[]
    for w,z in refs.items():
        gc,corrected=connected(z['g'],z['m'],refchain)
        curves[f'MC_W{w}_Gc']=gc;curves[f'MC_W{w}_Gc_independent_chain_corrected']=corrected
        counts=2*w*(w-np.arange(z['g'].shape[1]));products=np.rint(z['g']*counts).astype(np.int64)
        assert np.max(abs(products/counts-z['g']))<1e-13
        dest=out/'sufficient_statistics';dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/f'MC_W{w}.npz',products=products,counts=counts,g=z['g'],m=z['m'],e=z['e'],
                            Gc_ensemble=gc,Gc_independent_chains_corrected=corrected,parent_ids=z['parent_ids'],chain=refchain)
        corrections.append(dict(model='MC',steps=0,path=f'W{w}',n=len(z['m']),independent_units=len(np.unique(refchain)),
                                naive_mbar_squared=float(z['m'].mean()**2),Gc_correction=float(corrected[0]-gc[0])))
    for seed in SEEDS:
        for k in (64,256):
            results={}
            for arm in ARMS:
                stage='initial' if arm=='initial' else 'final';model=f's{seed}_{arm}_{stage}'
                results[arm]={path:load(base/'metrics'/model/f'{path}_K{k}.npz') for path in ('fine','B')}
                for path,z in results[arm].items():
                    w=96 if path=='fine' else 32;counts=2*w*(w-np.arange(z['g'].shape[1]));products=np.rint(z['g']*counts).astype(np.int64)
                    assert np.max(abs(products/counts-z['g']))<1e-13
                    gc,corrected=connected(z['g'],z['m'])
                    corrections.append(dict(model=model,steps=k,path=path,n=len(z['m']),independent_units=len(z['m']),
                                            naive_mbar_squared=float(z['m'].mean()**2),Gc_correction=float(corrected[0]-gc[0])))
                    dest=out/'sufficient_statistics'/model;dest.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(dest/f'{path}_K{k}.npz',products=products,counts=counts,g=z['g'],m=z['m'],e=z['e'],
                                        Gc_ensemble=gc,Gc_independent_images_corrected=corrected,sample_ids=np.arange(len(z['m'])))
            fig,ax=plt.subplots(2,2,figsize=(10,7))
            for arm,data in [('MC',{'fine':refs[96],'B':refs[32]}),*results.items()]:
                style='-' if arm=='MC' else STYLES[arm];label=LABELS[arm]
                z=data['fine'];g=z['g'].mean(0);gc,_=connected(z['g'],z['m'],refchain if arm=='MC' else None)
                ax[0,0].plot(np.arange(1,49),g[1:],color=COLORS[arm],ls=style,label=label)
                ax[0,1].plot(np.arange(1,49),gc[1:],color=COLORS[arm],ls=style,label=label)
                coarse=data['B']['g'].mean(0)
                ax[1,0].plot(np.arange(1,17),coarse[1:],color=COLORS[arm],ls=style)
                ax[1,1].plot(3*np.arange(1,17),(coarse-refs[32]['g'].mean(0))[1:],color=COLORS[arm],ls=style)
            for axis,lo,hi in zip(ax.flat,(25,25,9,27),(48,48,16,48)):axis.axvspan(lo,hi,color='#edf0e8',zorder=-1)
            ax[0,0].set(xlabel='Fine displacement r',ylabel='Raw G(r)',ylim=(0,1));ax[0,0].legend(fontsize=9)
            ax[0,1].set(xlabel='Fine displacement r',ylabel='Ensemble-connected Gc(r)',ylim=(-.1,1))
            ax[1,0].set(xlabel='Coarse displacement k',ylabel='Raw coarse G(k)',ylim=(0,1))
            ax[1,1].set(xlabel='Physical displacement 3k',ylabel='Coarse G - MC',ylim=(-.7,.7))
            save(fig,out,f'A1_curves_s{seed}_K{k}',f'{cohort} {seed}, K={k}, 每模型32张；MC为8条奇数链各32场。B粗场逐图取真实多数投票。Gc=G−系综平均磁化平方；独立图/独立链的有限样本修正另存 sufficient_statistics。阴影是预定主距离区间；本图不作确认性判断。')
            for image_id in (0,1,2,3):
                fig,ax=plt.subplots(2,5,figsize=(10.5,4.5))
                parent_id=int(refs[96]['parent_ids'][image_id]);fine=all_mc[parent_id,24:120,24:120]
                mapax(ax[0,0],fine);mapax(ax[1,0],majority(fine));ax[0,0].set_xlabel('MC')
                for j,arm in enumerate(ARMS,1):
                    stage='initial' if arm=='initial' else 'final';model=f's{seed}_{arm}_{stage}'
                    z=load(base/'samples'/model/f'W96_K{k}'/f'{image_id:05d}.npz');fine=z['spins']
                    mapax(ax[0,j],fine);mapax(ax[1,j],majority(fine));ax[0,j].set_xlabel(LABELS[arm])
                ax[0,0].set_ylabel('W96');ax[1,0].set_ylabel('Majority W32')
                save(fig,out,f'A1_samples_s{seed}_K{k}_i{image_id}',f'预登记生成ID={image_id}，{cohort}={seed}，K={k}；MC parent={parent_id}、global origin=(464,464)。下行均为正上方细图的3×3多数投票。MC为独立参考构型；同一seed的模型列复用配对U/V随机数，列间不计算像素误差。')
            for path,w in (('fine',96),('B',32)):
                fig,ax=plt.subplots(1,4,figsize=(13,3.3))
                for arm,z in [('MC',refs[w]),*[(arm,r[path]) for arm,r in results.items()]]:
                    style='-' if arm=='MC' else STYLES[arm];c=COLORS[arm]
                    ax[0].hist(z['m'],bins=np.linspace(-1,1,17),density=True,histtype='step',color=c,ls=style,label=LABELS[arm])
                    ecdf(ax[1],z['m'],color=c,ls=style)
                    ax[2].hist(abs(z['m']),bins=np.linspace(0,1,17),density=True,histtype='step',color=c,ls=style)
                    ax[3].hist(z['e'],bins=np.linspace(-1,1,81),density=True,histtype='step',color=c,ls=style)
                for axis in ax:axis.set_ylabel('Density')
                ax[0].set(xlabel='Signed m',xlim=(-1,1),ylim=(0,8));ax[0].legend(fontsize=8)
                ax[1].set(xlabel='Signed m',ylabel='ECDF',xlim=(-1,1),ylim=(0,1))
                ax[2].set(xlabel='|m|',xlim=(0,1),ylim=(0,16));ax[3].set(xlabel='Negative nearest-neighbor product',xlim=(-1,1),ylim=(0,45))
                save(fig,out,f'A1_distributions_{path}_s{seed}_K{k}',f'{cohort} {seed}, K={k}, {path} W{w}。保留生成磁化原符号，所有seed使用相同分箱与全幅坐标。每键能量口径为窗口内邻键乘积负均值；粗格为操作性统计。小样本优先读ECDF。')
            direct_path=base/'metrics'/f's{seed}_fine_plus_rg_final'/f'C_K{k}.npz'
            if not direct_path.exists():continue
            direct=load(direct_path)
            gc,corrected=connected(direct['g'],direct['m']);counts=64*(32-np.arange(direct['g'].shape[1]))
            products=np.rint(direct['g']*counts).astype(np.int64)
            assert np.max(abs(products/counts-direct['g']))<1e-13
            model=f's{seed}_fine_plus_rg_final';dest=out/'sufficient_statistics'/model
            np.savez_compressed(dest/f'C_K{k}.npz',products=products,counts=counts,g=direct['g'],m=direct['m'],e=direct['e'],
                                Gc_ensemble=gc,Gc_independent_images_corrected=corrected,sample_ids=np.arange(len(direct['m'])))
            corrections.append(dict(model=model,steps=k,path='C',n=len(direct['m']),independent_units=len(direct['m']),
                                    naive_mbar_squared=float(direct['m'].mean()**2),Gc_correction=float(corrected[0]-gc[0])))
            fig,ax=plt.subplots(1,3,figsize=(12,3.5))
            for label,z,c,sty in [('MC A',refs[32],'#263e41','--'),('Fine→majority B',results['fine_plus_rg']['B'],'#b65322','-'),('Direct coarse C',direct,'#1877a3',':')]:
                ax[0].plot(np.arange(1,17),z['g'].mean(0)[1:],color=c,ls=sty,label=label)
                ecdf(ax[1],z['m'],color=c,ls=sty)
                ax[2].hist(z['e'],bins=np.linspace(-1,1,81),density=True,histtype='step',color=c,ls=sty)
            ax[0].set(xlabel='Coarse displacement k',ylabel='Raw G(k)',ylim=(0,1));ax[0].legend(fontsize=9)
            ax[1].set(xlabel='Signed m',ylabel='ECDF',xlim=(-1,1),ylim=(0,1));ax[2].set(xlabel='Negative neighbor product',ylabel='Density',xlim=(-1,1),ylim=(0,45))
            save(fig,out,f'A1_paths_s{seed}_K{k}',f'Fine+RG旧谱系{seed}, K={k}。A=真实MC多数投票；B=同模型细生成后的投票；C=level1、块中心物理坐标直接生成。三路径均比较分布，B/C同时对独立MC参照报告误差。')
    np.savez_compressed(out/'MC_connected_corrections.npz',**curves)
    csvout(out/'connected_finite_sample_correction.csv',corrections)
    return dict(samples=len(list((base/'samples').rglob('*.npz'))),expected=expected,status='complete',metrics=summary['arms'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--require-complete',action='store_true');args=p.parse_args()
    project,runs,out=[v.absolute() for v in (args.project,args.runs,args.out)];out.mkdir(parents=True,exist_ok=True)
    write(out/'figure_configuration.json',dict(spin_colors=['#f3ecda','#193f41'],colors=COLORS,
          generated_ids=[0,1,2,3],A3_MC_ids='ordinal 0 in chains 0/1/2/3',A1_MC_ids='first four from frozen odd-chain reference',
          m_edges=np.linspace(-1,1,17).tolist(),abs_m_edges=np.linspace(0,1,17).tolist(),energy_edges=np.linspace(-1,1,81).tolist(),
          no_post_generation_spin_flips=True,source_sha256=sha(Path(__file__)),
          supplement_source_sha256=sha(Path(__file__).with_name('report_supplements.py'))))
    a3(project,runs,out);result={'A0':a0(project,runs,out),'A2':a2(runs,out),'A1':a1(project,runs,out)}
    historical_logs(project,out)
    complete=result['A0']['exact_cells']==126 and result['A0']['retention_cells']==735 and result['A2']['cases']==396 and result['A1']['samples']==960
    result['complete']=complete;result['interpretation']='old-lineage descriptive diagnostics; no stage-C confirmatory claim'
    write(out/'summary.json',result);write(out/'captions.json',CAPTIONS)
    if args.require_complete:assert complete,result
    figures=''.join(f'<figure><img src="figures/{name}.png"><figcaption>{html.escape(caption)}</figcaption></figure>' for name,caption in CAPTIONS.items())
    status='阶段 A 已完成' if complete else '阶段 A 运行中的结果快照'
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>RG-2 v1.2 实验结果</title>
    <style>body{{max-width:1180px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#243a38}}img{{width:100%;height:auto}}figure{{margin:45px 0}}figcaption{{color:#415c58}}table{{border-collapse:collapse}}td,th{{border:1px solid #cdd8d5;padding:8px 18px}}</style>
    <h1>RG-2 v1.2 · {status}</h1><p>本轮范围为 A、B；C 待开发结果和资源评估后决定。下列图均来自保存的原始数值，可由配套报告脚本和 CSV/NPZ 重画。</p>
    <table><tr><th>项目</th><th>完成数</th><th>预定数</th></tr><tr><td>A0 精确局部测试</td><td>{result['A0']['exact_cells']}</td><td>126</td></tr>
    <tr><td>A0 条件与几何保留</td><td>{result['A0']['retention_cells']}</td><td>735</td></tr>
    <tr><td>A2 模型–边界组合</td><td>{result['A2']['cases']}</td><td>396</td></tr><tr><td>A1 直接生成样本</td><td>{result['A1']['samples']}</td><td>960</td></tr></table>
    <p>A 阶段沿用三条历史训练谱系及历史 MC，只用于诊断。精确局部 KL、普通条件 CE、自由生成和块联合分布分别呈现。B/C 的确认性主张不能由这些旧模型结果替代。</p>{figures}</html>'''
    (out/'report.html').write_text(text,encoding='utf8')
    files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
    write(out/'manifest.json',files);print(json.dumps({k:v for k,v in result.items() if k not in ('A0','A1')}),flush=True)

if __name__=='__main__':main()
