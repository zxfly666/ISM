"""Development report; two paired seeds are never labelled confirmation."""
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import report as r

def readlines(p):return [json.loads(line) for line in p.read_text().splitlines()]

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();project,runs,out=[v.absolute() for v in (args.project,args.runs,args.out)];out.mkdir(parents=True,exist_ok=True)
    config=json.loads((runs/'B_v2/identity.json').read_text())['config'];r.SEEDS=config['seeds'];r.ARMS=tuple(config['arms'])
    r.LABELS.update(F='F · Fine replay',P='P · RoPE only',D='D · Decimation',M='M · Majority')
    r.COLORS.update(F='#657895',P='#89568e',D='#1877a3',M='#b65322');r.STYLES.update(F='--',P='-.',D=':',M='-')
    r.write(out/'figure_configuration.json',dict(colors=r.COLORS,styles=r.STYLES,generated_ids=[0,1,2,3],
            no_post_generation_spin_flips=True,source_sha256=r.sha(Path(__file__)),
            common_report_source_sha256=r.sha(Path(r.__file__)),
            supplement_source_sha256=r.sha(Path(r.__file__).with_name('report_supplements.py'))))
    result=r.a1(project,runs,out,'B_eval','开发 seed',512)
    allupdates=[];allval=[];allgrad=[];training_blocks=[]
    for seed in r.SEEDS:
        fig,ax=plt.subplots(2,2,figsize=(11,7))
        for arm in r.ARMS:
            folder=runs/'B_v2'/f's{seed}_{arm}';updates=readlines(folder/'updates.jsonl');valid=readlines(folder/'validation.jsonl');grads=readlines(folder/'gradients.jsonl')
            assert len(updates)==12000 and updates[-1]['step']==12000
            assert [v['step'] for v in valid]==list(range(0,12001,1000))
            for row in updates:allupdates.append(dict(seed=seed,arm=arm,**row))
            for row in grads:allgrad.append(dict(seed=seed,arm=arm,**row))
            for row in valid:
                for value in row['values']:allval.append(dict(seed=seed,arm=arm,step=row['step'],**value))
            ax[0,0].plot([v['step'] for v in valid],[np.mean([z['loss'] for z in v['values']]) for v in valid],color=r.COLORS[arm],ls=r.STYLES[arm],label=r.LABELS[arm])
            ax[0,1].plot([v['step'] for v in grads],[v['cosine'] for v in grads],color=r.COLORS[arm],ls=r.STYLES[arm])
            ax[1,0].plot([v['step'] for v in grads],[v['aux_to_common_norm'] for v in grads],color=r.COLORS[arm],ls=r.STYLES[arm])
            ax[1,1].plot(np.arange(500,12001,500),[np.mean([v['clipped'] for v in updates[j:j+500]]) for j in range(0,12000,500)],color=r.COLORS[arm],ls=r.STYLES[arm])
        for axis in ax.flat:axis.set_xlabel(f'Optimizer updates · seed {seed}')
        ax[0,0].set(ylabel='Common-fine validation loss',ylim=(0,.75));ax[0,0].legend(fontsize=9)
        ax[0,1].set(ylabel='Common/aux gradient cosine',ylim=(-1,1));ax[0,1].axhline(0,color='#aaaaaa',lw=.8)
        ax[1,0].set_ylabel('Aux / common gradient norm');ax[1,1].set(ylabel='Clip fraction, 500-update blocks',ylim=(0,1))
        r.save(fig,out,f'B_learning_s{seed}',f'开发seed={seed}，四臂各12000次更新。验证使用EMA，输入固定且来自独立验证链；损失为方案规定的inverse-t、按全部token归一的masked CE。梯度在同一raw参数状态下分别反传，排除任务嵌入；范数与实际裁剪率按原始梯度记录。')
        fig,ax=plt.subplots(1,2,figsize=(10,3.8))
        for arm in r.ARMS:
            updates=[v for v in allupdates if v['seed']==seed and v['arm']==arm]
            for level,axis in enumerate(ax):
                values=[v for v in updates if v['level']==level]
                chunks=[values[i:i+256] for i in range(0,len(values),256)]
                x=[np.mean([v['step'] for v in group]) for group in chunks]
                y=[np.mean([v['loss'] for v in group]) for group in chunks]
                axis.plot(x,y,color=r.COLORS[arm],ls=r.STYLES[arm],label=r.LABELS[arm])
                for group,loss in zip(chunks,y):
                    training_blocks.append(dict(seed=seed,arm=arm,level=level,first_update=group[0]['step'],
                           last_update=group[-1]['step'],n_updates=len(group),mean_loss=float(loss)))
        for axis in ax:axis.set(xlabel=f'Optimizer updates · seed {seed}',ylim=(0,1),xlim=(0,12000))
        ax[0].set_ylabel('Common-stream training loss');ax[1].set_ylabel('Auxiliary-stream training loss');ax[1].legend(fontsize=9)
        r.save(fig,out,f'B_training_s{seed}',f'开发seed={seed}。每条流按互不重叠的256次流内更新取均值，末块保留剩余更新；横轴为实际优化器更新。辅助流的目标数据随实验臂不同，其损失用于学习状态诊断，跨臂同任务优劣由共同流验证与独立生成评价判断。原始逐次数据及这些均值均保存。')
    r.csvout(out/'B_updates.csv',allupdates);r.csvout(out/'B_validation.csv',allval);r.csvout(out/'B_gradients.csv',allgrad)
    r.csvout(out/'B_training_blocks.csv',training_blocks)
    stats=json.loads((runs/'B_eval/development_statistics.json').read_text());r.csvout(out/'B_per_seed_metrics.csv',stats['per_seed'])
    local=[]
    for f in sorted((runs/'B_eval/A0_exact').rglob('*.npz')):
        z=r.load(f);local.append(dict(model=f.parent.name,bank=f.stem,KL=float(z['KL'].mean()),CE=float(z['CE'].mean()),n=len(z['KL'])))
    r.csvout(out/'B_local_retention.csv',local);r.write(out/'development_statistics.json',stats)
    fig,ax=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for axis,k in zip(ax,(64,256)):
        rows=[v for v in stats['comparisons'] if v['path']=='fine' and v['steps']==k]
        for j,row in enumerate(rows):
            lo,hi=row['descriptive_joint_bootstrap95'];axis.plot([j,j],[lo,hi],color='#263e41',lw=2)
            axis.scatter([j-.08,j+.08],row['paired_seed_values'],c=['#657895','#b65322'],s=35,zorder=5)
            axis.scatter(j,row['estimate'],color='#263e41',marker='D',s=24,zorder=6)
        axis.axhline(0,color='#767676',lw=1);axis.axhline(-.03,color='#b65322',lw=1,ls=':')
        axis.set(xticks=range(3),xticklabels=[v['comparison'] for v in rows],xlabel=f'W96, K={k}')
    ax[0].set_ylabel('Paired difference\nin relative raw-G error')
    r.save(fig,out,'B_development_effects','灰蓝点为开发seed 112501，赭橙点为112502，黑菱形为等权平均。竖线是10000次联合seed→配对生成图及独立MC链/块重抽样的描述性95%分位区间；两seed的区间不作确认性解释。虚线−0.03仅是原方案候选阈值，尚未据此开展C。')
    result.update(stage='B',models=8,updates=96000,confirmatory=False,C_launched=False)
    r.write(out/'summary.json',result);r.write(out/'captions.json',r.CAPTIONS)
    figures=''.join(f'<figure><img src="figures/{name}.png"><figcaption>{html.escape(caption)}</figcaption></figure>' for name,caption in r.CAPTIONS.items())
    rows=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in (x['steps'],x['comparison'],round(x['estimate'],5),[round(v,5) for v in x['paired_seed_values']],[round(v,5) for v in x['descriptive_joint_bootstrap95']]))+'</tr>' for x in stats['comparisons'] if x['path']=='fine')
    (out/'report.html').write_text(f'<!doctype html><meta charset="utf-8"><title>RG-2 阶段 B</title><style>body{{max-width:1180px;margin:40px auto;font:16px/1.7 system-ui;color:#243a38}}img{{width:100%}}figure{{margin:45px 0}}td,th{{padding:8px;border:1px solid #ccd}}</style><h1>RG-2 v1.2 · 阶段 B 开发结果</h1><p>2个独立配对seed × F/P/D/M；8模型，各12000次更新；W96细生成64/256步各32张，合计512张。阶段C尚未启动。</p><p>以下为开发诊断。两个训练seed的区间不能替代新seed、新MC链上的确认试验。</p><table><tr><th>步数</th><th>对比</th><th>平均差</th><th>逐seed差</th><th>描述性95%区间</th></tr>{rows}</table>{figures}',encoding='utf8')
    r.write(out/'manifest.json',{f.relative_to(out).as_posix():r.sha(f) for f in out.rglob('*') if f.is_file() and f.name!='manifest.json'})
    print('B REPORT COMPLETE',flush=True)

if __name__=='__main__':main()
