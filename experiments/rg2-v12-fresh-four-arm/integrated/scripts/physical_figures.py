"""Publication figures from frozen RG-2 arrays; no training or new generation.

Run from the experiment directory:
    python integrated/scripts/physical_figures.py
All displayed sample identities are fixed before inspecting their appearance.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'integrated'
RUNS = ROOT / 'project/runs'
COLORS = dict(MC='#263e41', F='#657895', P='#89568e', D='#1877a3', M='#b65322')
STYLES = dict(MC='-', F='--', P='-.', D=':', M='-')
SPIN = ListedColormap(['#f3ecda', '#193f41'])
OLD = dict(initial='#263e41', fine_only='#657895', fine_replay='#89568e', fine_plus_rg='#b65322')
OLDLABEL = dict(initial='Initial I-F', fine_only='Fine-only', fine_replay='Fine replay', fine_plus_rg='Fine+RG')
SEEDS = (112501,112502)
OLDSEEDS = (92601,92602,92603)
ARMS = ('F','P','D','M')
SOURCES = {}
FIGURES = {}
CHECKS = []
plt.rcParams.update({'font.family':['DejaVu Sans','sans-serif'], 'font.size':14,
    'axes.labelsize':15,'axes.titlesize':15,'xtick.labelsize':12,'ytick.labelsize':12,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.4,
    'legend.frameon':False,'legend.fontsize':12,'lines.linewidth':2.2,'pdf.fonttype':42,
    'savefig.facecolor':'white','figure.facecolor':'white'})


def source(path):
    path = path.resolve()
    key = path.relative_to(ROOT).as_posix()
    if key not in SOURCES:
        SOURCES[key] = {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    return path


def load(path):
    with np.load(source(path),allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def js(path):
    return json.loads(source(path).read_text(encoding='utf8'))


def lines(path):
    return [json.loads(x) for x in source(path).read_text(encoding='utf8').splitlines()]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def csvout(name,rows):
    with (OUT/'data'/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def close(a,b,label):
    error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    assert error < 2e-12,(label,error)
    CHECKS.append({'check':label,'maximum_absolute_difference':error})


def save(fig,name,caption,setting):
    if fig.legends:
        # Figure-level legends are deliberately outside the subplot rectangle.
        fig.set_layout_engine('constrained',rect=(0,0,1,.92))
    fig.savefig(OUT/'figures'/f'{name}.png',dpi=260,bbox_inches='tight',pad_inches=.12)
    fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    FIGURES[name]={'caption_zh':caption,'setting':setting,'input_files':sorted(SOURCES)}
    print(name,flush=True)


def majority(x):
    w=x.shape[-1]
    return np.sign(x.reshape(*x.shape[:-2],w//3,3,w//3,3).sum((-3,-1))).astype(np.int8)


def physics(x,rs=None):
    x=np.asarray(x,dtype=np.int8)
    if x.ndim==2:x=x[None]
    n,w,_=x.shape
    if rs is None:rs=np.arange(w//2+1)
    g=[]
    for r in rs:
        if r==0:g.append(np.ones(n));continue
        s=(x[:,r:,:]*x[:,:-r,:]).sum((1,2),dtype=np.int64)
        s+=(x[:,:,r:]*x[:,:,:-r]).sum((1,2),dtype=np.int64)
        g.append(s/(2*w*(w-r)))
    return np.array(g).T,x.mean((1,2))


def ecdf(ax,values,**kwargs):
    values=np.sort(values)
    # Include support endpoints so all plots show the full empirical CDF.
    xx=np.r_[-1.,values,1.];yy=np.r_[0.,np.arange(1,len(values)+1)/len(values),1.]
    ax.step(xx,yy,where='post',**kwargs)


def spin(ax,x,label=None):
    ax.imshow(x,cmap=SPIN,vmin=-1,vmax=1,interpolation='nearest',origin='upper')
    ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_visible(False)
    if label:ax.set_title(label,pad=9)


def legend(fig,arms=('MC','F','P','D','M'),ncol=5):
    handles=[Line2D([],[],color=COLORS[a],ls=STYLES[a],label=a) for a in arms]
    fig.legend(handles=handles,loc='upper center',ncol=ncol,bbox_to_anchor=(.5,1.015),fontsize=14)


def core_data():
    refs={w:load(RUNS/f'B_eval/MC_reference_W{w}.npz') for w in (96,32)}
    expected=js(RUNS/'B_eval/development_statistics.json')['per_seed']
    expected={(r['seed'],r['steps'],r['arm'],r['path']):r for r in expected}
    data={};rows=[];pairrows=[]
    for seed in SEEDS:
        for s in (64,256):
            for arm in ARMS:
                model=f's{seed}_{arm}_final'
                samplefiles=sorted((RUNS/f'B_eval/samples/{model}/W96_K{s}').glob('*.npz'))
                assert len(samplefiles)==32
                fields=np.stack([load(p)['spins'] for p in samplefiles])
                for path,w,xx in (('fine',96,fields),('B',32,majority(fields))):
                    z=load(RUNS/f'B_eval/metrics/{model}/{path}_K{s}.npz')
                    g,m=physics(xx);e=-g[:,1]
                    for key,computed in (('g',g),('m',m),('e',e)):close(computed,z[key],f'B/{model}/{path}/S{s}/{key}')
                    data[seed,s,arm,path]=z
                    ref=refs[w];lo=25 if w==96 else 9
                    error=float(np.linalg.norm(g.mean(0)[lo:]-ref['g'].mean(0)[lo:])/np.linalg.norm(ref['g'].mean(0)[lo:]))
                    row=dict(seed=seed,S=s,arm=arm,path='fine' if path=='fine' else 'majority_of_fine',W=w,n_gen=32,
                        G_error=error,G1=float(g[:,1].mean()),G_far=float(g[:,-1].mean()),
                        signed_long_bias=float((g.mean(0)-ref['g'].mean(0))[lo:].mean()),
                        mean_m=float(m.mean()),mean_abs_m=float(abs(m).mean()),mean_energy=float(e.mean()),
                        wall_fraction=float(((1-g[:,1])/2).mean()),positive_m_count=int((m>0).sum()),
                        W1_signed_m=float(wasserstein_distance(m,ref['m'])),
                        W1_abs_m=float(wasserstein_distance(abs(m),abs(ref['m']))),
                        W1_energy=float(wasserstein_distance(e,ref['e'])))
                    e0=expected[seed,s,arm,path]
                    for key in ('G_error','W1_signed_m','W1_abs_m','W1_energy'):close(row[key],e0[key],f'summary/{seed}/{s}/{arm}/{path}/{key}')
                    rows.append(row)
            for path in ('fine','majority_of_fine'):
                rr={r['arm']:r for r in rows if r['seed']==seed and r['S']==s and r['path']==path}
                for a,b in [('P','F'),('D','P'),('M','D'),('M','F'),('M','P')]:
                    pairrows.append(dict(seed=seed,S=s,path=path,comparison=a+'-'+b,
                        G_error_difference=rr[a]['G_error']-rr[b]['G_error'],
                        error_percentage_points=100*(rr[a]['G_error']-rr[b]['G_error']),
                        relative_error_change_percent=100*(rr[a]['G_error']/rr[b]['G_error']-1),
                        energy_W1_difference=rr[a]['W1_energy']-rr[b]['W1_energy']))
    csvout('physical_fresh_metrics.csv',rows);csvout('physical_paired_comparisons.csv',pairrows)
    rr=[]
    for w,z in refs.items():
        rr.append(dict(W=w,n=256,G1=float(z['g'][:,1].mean()),G_far=float(z['g'][:,-1].mean()),
                       mean_m=float(z['m'].mean()),mean_abs_m=float(abs(z['m']).mean()),mean_energy=float(z['e'].mean()),
                       wall_fraction=float((1-z['g'][:,1]).mean()/2),positive_m_count=int((z['m']>0).sum())))
    csvout('physical_reference_metrics.csv',rr)
    return refs,data,rows


def fresh_figures(refs,data,rows,meta,fine_mc):
    id0=int(refs[96]['parent_ids'][0]);mc=fine_mc[id0,24:120,24:120]
    for seed in SEEDS:
        for s in (64,256):
            fig,ax=plt.subplots(2,5,figsize=(15,6.5),layout='constrained')
            fields=[mc]+[load(RUNS/f'B_eval/samples/s{seed}_{arm}_final/W96_K{s}/00000.npz')['spins'] for arm in ARMS]
            for j,(label,x) in enumerate(zip(('MC',)+ARMS,fields)):
                for i,xx in enumerate((x,majority(x))):
                    g,m=physics(xx,[1]);energy=-g[0,0]
                    spin(ax[i,j],xx,label if i==0 else None)
                    ax[i,j].set_xlabel(f'm={m[0]:+.3f}\ne={energy:.3f}',fontsize=13)
            ax[0,0].set_ylabel('Fine W96',fontsize=15);ax[1,0].set_ylabel('Majority W32',fontsize=15)
            save(fig,f'F06_samples_s{seed}_S{s}',f'训练 seed={seed}，反向采样 S={s}，预登记生成 ID=0。上排是 W96 细场，下排是同一图的 3×3 多数投票 W32；MC 固定为参考池第一张 parent={id0}。每格注明实际有符号磁化 m 和每键负近邻乘积 e，粗格 e 仅为操作性统计。浅色 −1、深色 +1，未事后翻转。',dict(seed=seed,S=s,generation_id=0,MC_parent=id0,n_gen_per_model=32))
    for path,w,name,ylabel in [('fine',96,'F07_fine_correlations','Raw fine G(r)'),('B',32,'F08_coarse_correlations','Raw majority G(k)'),('fine',96,'F15_connected_correlations','Ensemble-connected Gc(r)')]:
        connected='connected' in name
        fig,axs=plt.subplots(2,2,figsize=(13,9),sharex=True,sharey=True,layout='constrained')
        for i,seed in enumerate(SEEDS):
            for j,s in enumerate((64,256)):
                ax=axs[i,j]
                for arm,z in [('MC',refs[w])]+[(a,data[seed,s,a,path]) for a in ARMS]:
                    g=z['g'].mean(0)
                    if connected:g=g-z['m'].mean()**2
                    ax.plot(np.arange(1,len(g)),g[1:],color=COLORS[arm],ls=STYLES[arm],label=arm)
                ax.axvspan(25 if w==96 else 9,w//2,color='#edf0e8',zorder=-1)
                ax.set_title(f'Seed {seed}  |  S={s}',loc='left')
                ax.set(xlim=(1,w//2),ylim=(-.12,1) if connected else (0,1),xlabel='Fine displacement r' if w==96 else 'Coarse displacement k')
                if j==0:ax.set_ylabel(ylabel)
        legend(fig)
        save(fig,name,('系综连通关联按各方法的全体样本平均磁化平方扣除，并非逐图扣除。' if connected else '完整原始关联保留绝对幅度；')+f'两行分别为两个训练 seed，两列为 S=64/256。每个模型 n_gen=32；MC 为固定的 8 条奇数链各 32 场，共 256 场。阴影为长程评价范围 '+('r=25–48。' if w==96 else 'k=9–16；对应细格物理距离 3k=27–48。')+'均采用开窗口内的轴向有效点对。',dict(W=w,n_gen=32,n_MC=256,connected=connected))
    for path,w in (('fine',96),('B',32)):
        pathlabel='fine' if path=='fine' else 'coarse'
        for metric in ('m','abs_m','energy'):
            fig,axs=plt.subplots(2,2,figsize=(13,9),sharex=True,sharey=True,layout='constrained')
            for i,seed in enumerate(SEEDS):
                for j,s in enumerate((64,256)):
                    ax=axs[i,j]
                    for arm,z in [('MC',refs[w])]+[(a,data[seed,s,a,path]) for a in ARMS]:
                        values=z['e'] if metric=='energy' else abs(z['m']) if metric=='abs_m' else z['m']
                        ecdf(ax,values,color=COLORS[arm],ls=STYLES[arm])
                    ax.set_title(f'Seed {seed}  |  S={s}',loc='left')
                    ax.set(xlim=(0,1) if metric=='abs_m' else (-1,1),ylim=(0,1),xlabel={'m':'Signed magnetization m','abs_m':'Absolute magnetization |m|','energy':'Negative nearest-neighbor product e'}[metric])
                    if j==0:ax.set_ylabel('Empirical cumulative probability')
            legend(fig)
            save(fig,f'F09_{pathlabel}_{metric}',f'W{w} '+('细生成场' if path=='fine' else '同一细生成场经多数投票得到的粗场')+'的完整经验累积分布。每臂每 seed 每 S 有 32 张图，MC 参考 256 场；各面板使用相同完整支持区间，没有丢弃极端值。ECDF 逐点显示所有样本，避免小样本直方图分箱造成假象。'+('粗格近邻乘积不等于原 Ising Hamiltonian 能量。' if path=='B' and metric=='energy' else ''),dict(W=w,metric=metric,full_support=True,n_gen=32,n_MC=256))
    fig,axs=plt.subplots(1,2,figsize=(13,5.5),sharex=True,sharey=True,layout='constrained')
    for ax,s in zip(axs,(64,256)):
        for arm in ARMS:
            rr=[r for r in rows if r['path']=='fine' and r['S']==s and r['arm']==arm]
            for seed,marker in zip(SEEDS,('o','^')):
                r=next(v for v in rr if v['seed']==seed)
                ax.scatter(r['W1_energy'],r['G_error'],c=COLORS[arm],marker=marker,s=100,edgecolor='white',linewidth=.7)
                ax.annotate(arm,(r['W1_energy'],r['G_error']),xytext=(6,4),textcoords='offset points',color=COLORS[arm],fontsize=14)
        ax.set(title=f'S={s}',xlabel='Energy-distribution Wasserstein distance',xlim=(0,.12),ylim=(0,.62))
    axs[0].set_ylabel('Long-range relative G error')
    handles=[Line2D([],[],ls='',marker=m,color='#263e41',label=f'Seed {seed}',markersize=8) for seed,m in zip(SEEDS,('o','^'))]
    fig.legend(handles=handles,loc='upper center',ncol=2,bbox_to_anchor=(.5,1.05))
    save(fig,'F10_energy_longrange_tradeoff','每个点代表一个训练 seed 下的一个实验臂，左右分别为 S=64/256，全部为 W96 细生成。横轴为每键能量分布相对独立 MC 的 Wasserstein 距离，纵轴为 r=25–48 的原始关联相对 L2 误差；越靠近左下越好。圆点和三角分别表示两组训练重复，不将 32 张生成图误当作训练重复；没有合成综合评分。',dict(n_gen=32,axes=['W1_energy','G_error']))
    fig,axs=plt.subplots(1,2,figsize=(13,5.2),sharey=True,layout='constrained')
    for ax,s in zip(axs,(64,256)):
        for i,seed in enumerate(SEEDS):
            rr=[next(r for r in rows if r['seed']==seed and r['S']==s and r['arm']==a and r['path']=='fine') for a in ARMS]
            ax.plot(np.arange(4)+(.07 if i else -.07),[r['wall_fraction'] for r in rr],ls='--' if i else '-',color='#888888',lw=1)
            for j,(a,r) in enumerate(zip(ARMS,rr)):
                ax.scatter(j+(.07 if i else -.07),r['wall_fraction'],color=COLORS[a],marker='^' if i else 'o',s=90,zorder=3)
        ax.axhline((1-refs[96]['g'][:,1].mean())/2,color=COLORS['MC'],ls=':',label='MC')
        ax.set(title=f'S={s}',xticks=range(4),xticklabels=ARMS,ylim=(.13,.22),xlabel='Training arm')
    axs[0].set_ylabel('Opposite nearest-neighbor bond fraction')
    sec=axs[1].secondary_yaxis('right',functions=(lambda p:1-2*p,lambda g:(1-g)/2));sec.set_ylabel('Equivalent G(1) = 1 - 2 p_wall')
    fig.legend(handles=handles+[Line2D([],[],color=COLORS['MC'],ls=':',label='MC reference')],loc='upper center',ncol=3,bbox_to_anchor=(.5,1.06))
    save(fig,'F15_local_bonds','直接测量的最近邻异号键比例 p_wall=(1−G(1))/2；右轴为完全等价的 G(1)。纵轴为清楚标出的局部范围，所有 16 个模型／采样设置的均值都在图中。此统计说明局部异号键增多与否，不直接识别团簇破碎机制。',dict(W=96,n_gen=32,displayed_quantity='mean opposite-bond fraction',local_axis_range=[.13,.22]))


def mc_figures(meta,fine_mc):
    z=load(RUNS/'A_supp/A3_per_field.npz');rows=[]
    assert len(fine_mc)==2048 and len(np.unique(meta['chain']))==16
    fixed=int(np.flatnonzero(meta['chain']==0)[0])
    fig,axs=plt.subplots(1,4,figsize=(14,4.2),layout='constrained')
    x=fine_mc[fixed,24:120,24:120];d=x[1::3,1::3];m=majority(x)
    for ax,v,label in zip(axs[:3],(x,d,m),('Fine W96','Decimation W32','Majority W32')):spin(ax,v,label)
    axs[3].imshow(d!=m,cmap=ListedColormap(['#f3ecda','#b65322']),vmin=0,vmax=1,interpolation='nearest')
    axs[3].set(title='D / M disagreement',xticks=[],yticks=[])
    for sp in axs[3].spines.values():sp.set_visible(False)
    save(fig,'F01_mc_fields','同一真实 MC 构型：chain=0、ordinal=0、parent=0，W96 裁剪的全局原点 (464,464)。浅色 −1、深色 +1；右图橙色表示该块的中心抽点与多数投票不同。选择身份预先固定；这张构型不是 2048 场的总体统计。',dict(MC_parent=fixed,chain=0,ordinal=0,W=96,crop_origin=[464,464]))
    for w in (96,144):
        lo=(144-w)//2;xx=fine_mc[:,lo:lo+w,lo:lo+w];dd=xx[:,1::3,1::3];mm=majority(xx)
        disagreement=(dd!=mm).mean((1,2))
        for kind,fields in [('fine',xx),('D',dd),('M',mm)]:
            key=f'{kind}{w}';gg=z[key+'_g'];mag=z[key+'_m'];en=z[key+'_e'];ww=fields.shape[-1]
            selected=np.unique([1,3,min(9,ww//2),ww//2]);direct,dm=physics(fields,selected)
            close(direct,gg[:,selected],key+'/selected integer pair sums');close(dm,mag,key+'/magnetization')
            close(gg,z[key+'_products']/z[key+'_counts'],key+'/stored integer sufficient statistics')
            close(en,-gg[:,1],key+'/energy')
            rows.append(dict(parent_W=w,operator=kind,output_W=ww,n=2048,chains=16,G1=float(gg[:,1].mean()),
                G_far=float(gg[:,-1].mean()),mean_m=float(mag.mean()),mean_abs_m=float(abs(mag).mean()),
                mean_energy=float(en.mean()),wall_fraction=float((1-gg[:,1]).mean()/2),
                disagreement_D_M=float(disagreement.mean())))
        fig,axs=plt.subplots(2,2,figsize=(12.5,8.8),layout='constrained')
        for kind,color,style in [('fine',COLORS['MC'],'-'),('D',COLORS['D'],':'),('M',COLORS['M'],'-')]:
            g=z[f'{kind}{w}_g'].mean(0);mag=z[f'{kind}{w}_m'];r=np.arange(1,len(g))
            axs[0,0].plot(r,g[1:],color=color,ls=style,label='Fine' if kind=='fine' else kind)
            axs[0,1].plot(r if kind=='fine' else 3*r,g[1:],color=color,ls=style)
            ecdf(axs[1,0],mag,color=color,ls=style)
            ecdf(axs[1,1],abs(mag),color=color,ls=style)
        axs[0,0].set(xlabel='Displacement in each output lattice',ylabel='Raw G',xlim=(1,w//2),ylim=(0,1))
        axs[0,1].set(xlabel='Fine-lattice physical displacement',ylabel='Raw G',xlim=(1,w//2),ylim=(0,1))
        axs[1,0].set(xlabel='Signed magnetization m',ylabel='ECDF',xlim=(-1,1),ylim=(0,1))
        axs[1,1].set(xlabel='Absolute magnetization |m|',ylabel='ECDF',xlim=(0,1),ylim=(0,1))
        fig.legend(handles=axs[0,0].get_legend_handles_labels()[0],labels=['Fine','D','M'],loc='upper center',ncol=3,bbox_to_anchor=(.5,1.03),fontsize=14)
        save(fig,f'F02_mc_statistics_W{w}',f'真实 W{w} MC 的细场、中心抽点 D 和多数投票 M，使用全部 2048 场、16 条独立链。左上按各输出格距作图，右上把粗格距离 k 换为细格物理距离 3k；下排为有符号与绝对磁化的完整 ECDF。曲线无误差条；链间不确定性保留于统计附录。D 与 M 的总体块分歧率为 {100*disagreement.mean():.4f}%。',dict(parent_W=w,n_MC=2048,chains=16))
    csvout('physical_mc_summary.csv',rows)
    return rows


def distribution_details(refs,data):
    """Keep full-support ECDFs and add clearly labelled data-range detail."""
    low,high=-.85,-.45
    fig,axs=plt.subplots(2,2,figsize=(13,9),sharex=True,sharey=True,layout='constrained')
    for i,seed in enumerate(SEEDS):
        for j,s in enumerate((64,256)):
            ax=axs[i,j]
            for arm,z in [('MC',refs[96])]+[(a,data[seed,s,a,'fine']) for a in ARMS]:
                assert z['e'].min()>=low and z['e'].max()<=high
                ecdf(ax,z['e'],color=COLORS[arm],ls=STYLES[arm])
            ax.set(title=f'Seed {seed}  |  S={s}',xlim=(low,high),ylim=(0,1),xlabel='Fine energy e — observed-range magnification')
            if j==0:ax.set_ylabel('Empirical cumulative probability')
    legend(fig)
    save(fig,'F09_fine_energy_zoom','W96 细场能量 ECDF 的明确局部放大：横轴 −0.85 至 −0.45。已逐值核验，该范围覆盖全部 512 张生成图和 256 张 MC 参考的实际能量，没有排除极端样本。完整理论支持区间 [−1,1] 的版本另保留为 F09_fine_energy；本图只为看清分布位移。',dict(W=96,zoom_range=[low,high],all_observed_values_inside=True,full_support_figure='F09_fine_energy'))
    fig,axs=plt.subplots(2,2,figsize=(13,9),sharex=True,sharey=True,layout='constrained')
    bins=np.linspace(-1,1,17)
    for i,seed in enumerate(SEEDS):
        for j,s in enumerate((64,256)):
            ax=axs[i,j]
            for arm,z in [('MC',refs[96])]+[(a,data[seed,s,a,'fine']) for a in ARMS]:
                ax.hist(z['m'],bins=bins,density=True,histtype='step',color=COLORS[arm],ls=STYLES[arm],lw=2.1)
            ax.set(title=f'Seed {seed}  |  S={s}',xlim=(-1,1),ylim=(0,4.5),xlabel='Signed magnetization m')
            if j==0:ax.set_ylabel('Probability density (16 equal bins)')
    legend(fig)
    save(fig,'F09_fine_m_histogram','W96 有符号磁化直方图，用固定 16 个等宽分箱覆盖 [−1,1]，不同 seed 与 S 使用相同横纵范围。每臂仅 32 张图，因此局部峰形依赖分箱；完整 ECDF 另见 F09_fine_m，不能仅据本图确认模式缺失或自旋对称性破缺。',dict(W=96,bins=bins.tolist(),n_gen=32,n_MC=256))


def old_local():
    models=js(ROOT/'project/inputs/manifest.json')['models'];rows=[];lookup={}
    for e in models:
        for w in (48,96):
            vals=[]
            for nm in (1,8,32):
                z=load(RUNS/f"A_v2/A0_exact/{e['key']}/fine_W{w}_M{nm}.npz")
                vals.append(float(z['KL'].mean()))
            lookup[e['seed'],e['arm'],e['stage'],w]=np.mean(vals)
            rows.append(dict(seed=e['seed'],arm=e['arm'],stage=e['stage'],common_updates=e['cycle'],total_updates=e['updates'],W=w,KL=float(np.mean(vals))))
    fig,axs=plt.subplots(2,3,figsize=(14,8.5),sharex=True,sharey='row',layout='constrained')
    for j,seed in enumerate(OLDSEEDS):
        for i,w in enumerate((48,96)):
            ax=axs[i,j]
            for arm in ('fine_only','fine_replay','fine_plus_rg'):
                yy=[lookup[seed,'initial','initial',w]]+[lookup[seed,arm,t,w] for t in ('midpoint','final')]
                ax.plot([0,1024,2048],yy,color=OLD[arm],marker='o',ls='-' if arm=='fine_plus_rg' else '--',label=OLDLABEL[arm])
            ax.set(title=f'Seed {seed}  |  W{w}',xticks=[0,1024,2048],xlabel='Common-fine updates')
            if j==0:ax.set_ylabel('Exact local KL (nats/query)')
            ax.set_ylim((0,.0045) if w==48 else (0,.05))
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,bbox_to_anchor=(.5,1.04))
    save(fig,'F05_old_local_trajectories','旧 I-F EMA 暖启动的三个 seed 分别展示。每点为 n_mask=1/8/32 三个条件任务的等权平均，每任务使用相同 256 个父场；上排 W48，下排 W96。横轴为共同细任务更新 0/1024/2048；Fine-only 总更新相同，另外两臂总更新为其两倍。因此图中三臂并非等总更新比较，Fine+RG 对 Fine replay 才是等预算辅助任务对照。',dict(seeds=list(OLDSEEDS),widths=[48,96],n_per_task=256))
    csvout('physical_old_local_trajectory.csv',rows)


def old_paths():
    ref=load(RUNS/'A_v2/MC_reference_W32.npz');rows=[]
    for s in (64,256):
        fig,axs=plt.subplots(3,3,figsize=(14,12),sharex='col',sharey='col',layout='constrained')
        for i,seed in enumerate(OLDSEEDS):
            b=load(RUNS/f'A_v2/metrics/s{seed}_fine_plus_rg_final/B_K{s}.npz')
            c=load(RUNS/f'A_v2/metrics/s{seed}_fine_plus_rg_final/C_K{s}.npz')
            for name,z,color,style in [('MC → majority',ref,COLORS['MC'],'-'),('Fine generation → majority',b,COLORS['M'],'-'),('Direct coarse generation',c,COLORS['D'],':')]:
                axs[i,0].plot(np.arange(1,17),z['g'].mean(0)[1:],color=color,ls=style,label=name)
                ecdf(axs[i,1],z['m'],color=color,ls=style)
                ecdf(axs[i,2],z['e'],color=color,ls=style)
                rows.append(dict(seed=seed,S=s,path=name,G_error=float(np.linalg.norm((z['g'].mean(0)-ref['g'].mean(0))[9:])/np.linalg.norm(ref['g'].mean(0)[9:])),
                                 mean_m=float(z['m'].mean()),mean_abs_m=float(abs(z['m']).mean()),mean_e=float(z['e'].mean()),W1_m=float(wasserstein_distance(z['m'],ref['m'])),W1_e=float(wasserstein_distance(z['e'],ref['e']))))
            axs[i,0].set(title=f'Seed {seed}',ylabel='Raw coarse G(k)',xlabel='Coarse displacement k',xlim=(1,16),ylim=(0,1))
            axs[i,0].axvspan(9,16,color='#edf0e8',zorder=-1)
            axs[i,1].set(xlabel='Signed magnetization m',ylabel='ECDF',xlim=(-1,1),ylim=(0,1))
            axs[i,2].set(xlabel='Coarse negative neighbor product',ylabel='ECDF',xlim=(-1,1),ylim=(0,1))
        fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,bbox_to_anchor=(.5,1.025),fontsize=12)
        save(fig,f'F11_old_three_paths_S{s}',f'仅使用旧 Fine+RG 终点 EMA，S={s}。三行分别为三个旧训练 seed；三列是 W32 原始关联、有符号磁化和负近邻乘积 ECDF。真实路径从 256 张 MC W96 场粗化；模型的细生成粗化和直接粗生成各 32 张。直接粗生成采用任务 level 1 与间距 3 的块中心坐标。本图不属于 Fresh M，不证明逆 RG；两个模型路径相近也不等于均符合真实分布。',dict(S=s,weights='old Fine+RG final EMA',n_model_path=32,n_MC=256))
    csvout('physical_old_three_paths.csv',rows)


def joint_figures():
    plans=('serial','adjacent','checker','nine');labels=('Serial','Adjacent 3','Checker 3','Parallel 9')
    pc=['#263e41','#657895','#1877a3','#b65322']
    for seed in OLDSEEDS:
        p=RUNS/f'A_supp/A2/s{seed}_fine_plus_rg_final/mc_chain0_ord0.npz';z=load(p)
        truth=z['truth'];nplus=z['bits'].sum(1);true_mass=np.bincount(nplus,weights=truth,minlength=10)
        fig,axs=plt.subplots(2,2,figsize=(12.5,9),layout='constrained')
        for plan,label,color in zip(plans,labels,pc):
            q=z[plan+'_Q'];r=json.loads(str(z[plan+'_result']))
            axs[0,0].scatter(truth,q,s=9,color=color,alpha=.35,label=label,rasterized=True)
            axs[0,1].plot(np.arange(10),np.bincount(nplus,weights=q,minlength=10),color=color,marker='o',ms=3,label=label)
        lim=(1e-10,1)
        axs[0,0].plot(lim,lim,color='#aaaaaa',lw=1)
        axs[0,0].set(xscale='log',yscale='log',xlim=lim,ylim=lim,xlabel='Exact P(x)',ylabel='Transformer Q(x)')
        axs[0,1].plot(np.arange(10),true_mass,color='black',ls='--',lw=2.5,label='Exact truth')
        axs[0,1].set(xlabel='Number of positive spins in the 3×3 block',ylabel='Probability',xticks=range(10),ylim=(0,1))
        truemajor=float(truth[nplus>=5].sum())
        for j,(plan,label,color) in enumerate(zip(plans,labels,pc)):
            r=json.loads(str(z[plan+'_result']))
            axs[1,0].scatter(j,r['q_majority'],s=85,color=color)
            axs[1,1].bar(j,r['parallel_factorization_kl'],color='#657895')
            axs[1,1].bar(j,r['single_conditional_kl'],bottom=r['parallel_factorization_kl'],color='#b65322')
        axs[1,0].axhline(truemajor,color='black',ls='--')
        axs[1,0].set(xticks=range(4),xticklabels=labels,ylabel='Probability of positive block majority',ylim=(0,1.03))
        axs[1,1].set(xticks=range(4),xticklabels=labels,ylabel='Forward KL(P || Q), nats',ylim=(0,.7))
        for ax in axs[1]:ax.tick_params(axis='x',rotation=20)
        axs[0,1].legend(fontsize=10)
        axs[1,1].legend(handles=[plt.Rectangle((0,0),1,1,color='#657895',label='Oracle factorization'),plt.Rectangle((0,0),1,1,color='#b65322',label='Conditional prediction')],fontsize=10)
        save(fig,f'F12_joint_s{seed}',f'旧 Fine+RG seed={seed}，固定 MC 边界 chain 0、ordinal 0；W16 容器内只隐藏中心 3×3。512 状态的精确 P 与四种揭示分组的模型 Q 均归一。上排展示联合概率散点与正自旋数量分布；下排展示块多数为正概率和 forward KL 的两部分。每次计划仅首组并行，其余自旋逐点揭示；同色组三点在其余内部变量仍隐藏时并不保证条件独立。该固定边界为预先固定案例，不作为总体均值。',dict(seed=seed,boundary='mc_chain0_ord0',internal_states=512,weights='old Fine+RG final EMA'))
    rows=[]
    for p in sorted((RUNS/'A_supp/A2').rglob('*.npz')):
        z=load(p)
        if 'truth' not in z:continue
        model=p.parent.name
        seed=int(model.split('_')[0][1:]);arm=model[len(f's{seed}_'):-len('_final')] if model.endswith('_final') else 'initial'
        for plan in plans:
            r=json.loads(str(z[plan+'_result']));q=z[plan+'_Q'];oracle=z[plan+'_oracle'];truth=z['truth']
            kl=float(np.sum(truth*np.log(truth/q)));factor=float(np.sum(truth*np.log(truth/oracle)))
            close(kl,r['learned_kl'],f'A2/{model}/{p.stem}/{plan}/KL')
            close(factor,r['parallel_factorization_kl'],f'A2/{model}/{p.stem}/{plan}/factor')
            close(r['learned_kl'],r['parallel_factorization_kl']+r['single_conditional_kl'],f'A2/{model}/{p.stem}/{plan}/decomposition')
            rows.append(dict(seed=seed,arm=arm,boundary=p.stem,plan=plan,learned_KL=kl,factorization_KL=factor,prediction_KL=r['single_conditional_kl'],p_majority=r['p_majority'],q_majority=r['q_majority']))
    csvout('physical_joint_all_boundaries.csv',rows)
    fig,axs=plt.subplots(1,3,figsize=(14,4.9),sharey=True,layout='constrained')
    for ax,seed in zip(axs,OLDSEEDS):
        for j,plan in enumerate(plans):
            rr=[r for r in rows if r['seed']==seed and r['arm']=='fine_plus_rg' and r['plan']==plan and r['boundary']!='all_plus']
            assert len(rr)==32
            factor=np.mean([r['factorization_KL'] for r in rr]);pred=np.mean([r['prediction_KL'] for r in rr])
            ax.bar(j,factor,color='#657895');ax.bar(j,pred,bottom=factor,color='#b65322')
            yy=np.array([r['learned_KL'] for r in rr]);offset=np.linspace(-.16,.16,len(yy))
            ax.scatter(j+offset,yy,s=11,c='#263e41',alpha=.45,zorder=3)
        ax.set(title=f'Seed {seed}',xticks=range(4),xticklabels=labels,ylim=(0,.95));ax.tick_params(axis='x',rotation=25)
    axs[0].set_ylabel('Forward KL(P || Q), nats')
    fig.legend(handles=[plt.Rectangle((0,0),1,1,color='#657895',label='Mean factorization'),plt.Rectangle((0,0),1,1,color='#b65322',label='Mean prediction'),Line2D([],[],marker='o',ls='',color='#263e41',label='Each MC boundary',markersize=5)],loc='upper center',ncol=3,bbox_to_anchor=(.5,1.07))
    save(fig,'F12_decomposition_all_boundaries','旧 Fine+RG 三个 seed 在全部 32 个预定 MC 边界上的小块 forward KL。柱高为均值，灰蓝与赭橙分别是 oracle 因子化项和条件预测项；每个黑点为一个边界的总 KL，不是误差条。全正边界单独保留在数据表中，未混入 MC 均值。九点并行的因子化项对模型无关，不能直接作为 W96 全图误差的定量解释。',dict(boundaries=32,excluded_from_mean='all_plus',weights='old Fine+RG final EMA'))


def precision_figure():
    z=load(RUNS/'A_supp/A3N_draws.npz');dep=js(RUNS/'A_supp/A3_dependence.json');block=dep['block'];rows=[]
    fig,axs=plt.subplots(1,2,figsize=(12.5,4.9),sharey=True,layout='constrained')
    ns=np.array([16,32,64,128,256,512])
    for ax,tag,label in zip(axs,('fine96','M96'),('Fine W96: r=25–48','Majority W32: k=9–16')):
        quantiles=[]
        for n in ns:
            v=z[f'block{block}_n{n}_{tag}'];q=np.quantile(v,[.05,.5,.95]);quantiles.append(q)
            rows.append(dict(path=tag,n=int(n),block=block,repeats=len(v),q05=float(q[0]),median=float(q[1]),q95=float(q[2])))
        q=np.array(quantiles)
        ax.fill_between(ns,q[:,0],q[:,2],color='#657895',alpha=.18,label='5th–95th percentile')
        ax.plot(ns,q[:,1],color=COLORS['M'],marker='o',label='Median')
        ax.plot(ns,q[:,2],color=COLORS['F'],ls='--',marker='.',label='95th percentile')
        ax.axhline(.03,color='#999999',ls=':',label='0.03 candidate effect scale')
        ax.set(title=label,xscale='log',xticks=ns,xticklabels=ns,xlabel='Pseudo-generated MC sample count',ylim=(0,.32))
    axs[0].set_ylabel('MC–MC relative raw-G error');axs[1].legend(fontsize=10,loc='upper right')
    save(fig,'F13_mc_precision','真实 MC 对真实 MC 的有限样本校准：偶数链充当伪生成，奇数链提供独立 256 场参考，按冻结链／循环块长 8 方案，每个样本量有 2000 次重抽样。阴影是该校准误差的 5%–95% 分位范围，橙线为中位数；0.03 仅为候选效应尺度。此图不是严格误差下界，也不是模型配对差的显著性门槛。',dict(block=block,repeats=2000,n_reference=256))
    csvout('physical_mc_precision.csv',rows)


def training_figures():
    summary=[];probe={}
    for seed in SEEDS:
        fig,axs=plt.subplots(2,2,figsize=(12.5,8.7),layout='constrained')
        for arm in ARMS:
            folder=RUNS/f'B_v2/s{seed}_{arm}'
            updates=lines(folder/'updates.jsonl');valid=lines(folder/'validation.jsonl');grads=lines(folder/'gradients.jsonl')
            probe[seed,arm]=grads
            for lev in (0,1):
                selected=[r for r in updates if r['level']==lev]
                chunks=[selected[i:i+256] for i in range(0,len(selected),256)]
                axs[0,lev].plot([np.mean([r['step'] for r in c]) for c in chunks],[np.mean([r['loss'] for r in c]) for c in chunks],color=COLORS[arm],ls=STYLES[arm],lw=2.2)
            axs[1,0].plot([r['step'] for r in valid],[np.mean([v['loss'] for v in r['values']]) for r in valid],color=COLORS[arm],ls=STYLES[arm])
            axs[1,1].plot(np.arange(500,12001,500),[np.mean([r['clipped'] for r in updates[i:i+500]]) for i in range(0,len(updates),500)],color=COLORS[arm],ls=STYLES[arm])
            summary.append(dict(seed=seed,arm=arm,initial_validation=float(np.mean([v['loss'] for v in valid[0]['values']])),final_validation=float(np.mean([v['loss'] for v in valid[-1]['values']])),
                clip_fraction=float(np.mean([r['clipped'] for r in updates])),probe_cosine_mean=float(np.mean([r['cosine'] for r in grads])),probe_cosine_min=float(min(r['cosine'] for r in grads)),probe_cosine_max=float(max(r['cosine'] for r in grads)),probe_aux_common_norm_mean=float(np.mean([r['aux_to_common_norm'] for r in grads]))))
        axs[0,0].set(ylabel='Common-stream training loss',ylim=(0,.8));axs[0,1].set(ylabel='Auxiliary-stream training loss',ylim=(0,.8))
        axs[1,0].set(ylabel='Common-fine EMA validation loss',ylim=(0,.75))
        axs[1,1].set(ylabel='Gradient-clipping fraction',ylim=(0,1))
        for ax in axs.flat:ax.set(xlabel='Optimizer updates',xlim=(0,12000))
        legend(fig,arms=ARMS,ncol=4)
        save(fig,f'F14_training_s{seed}',f'Fresh seed={seed}。上排分别为共同／辅助训练流按 256 次流内更新分块的损失均值，末块保留剩余更新。下排是四个宽度等权共同细任务固定验证，以及 500 次总更新分块的实际裁剪比例。线型和颜色在全部面板均对应实验臂。辅助目标随实验臂不同，辅助损失不能直接用于跨臂同任务优劣判断；训练与验证均为 inverse-t masked CE、按全部 token 归一。',dict(seed=seed,updates=12000,loss_normalization='inverse-t masked CE over all tokens'))
    fig,axs=plt.subplots(2,2,figsize=(12.5,8.7),sharex=True,sharey='col',layout='constrained')
    for i,seed in enumerate(SEEDS):
        for arm in ARMS:
            grads=probe[seed,arm]
            axs[i,0].plot([r['step'] for r in grads],[r['cosine'] for r in grads],color=COLORS[arm],ls=STYLES[arm])
            axs[i,1].plot([r['step'] for r in grads],[r['aux_to_common_norm'] for r in grads],color=COLORS[arm],ls=STYLES[arm])
        axs[i,0].set(title=f'Seed {seed}',ylabel='Fixed-probe gradient cosine',ylim=(-1,1));axs[i,0].axhline(0,color='#bbbbbb',lw=1)
        axs[i,1].set(title=f'Seed {seed}',ylabel='Fixed-probe auxiliary/common norm',ylim=(0,1.8))
        for ax in axs[i]:ax.set(xlabel='Optimizer updates',xlim=(0,12000))
    legend(fig,arms=ARMS,ncol=4)
    save(fig,'F14_fixed_probe_gradients','每个训练时点只使用一对固定 W48 验证 probe，分别计算共同流和辅助流在同一 raw 参数状态上的梯度；两次反传之间没有更新，参数集合排除任务嵌入。左列为余弦，右列为辅助／共同梯度范数比；两行分别为两个训练 seed。本图不能直接代表全训练分布上的梯度冲突；这里的 probe 诊断不等于实际训练梯度裁剪。',dict(seeds=list(SEEDS),gradient_probe='one fixed W48 pair',excluded_parameter='scale_embedding.weight'))
    csvout('physical_training_summary.csv',summary)


def main():
    for folder in ('figures','data','provenance'):(OUT/folder).mkdir(parents=True,exist_ok=True)
    meta=load(ROOT/'project/inputs/mc_W144_centered.npz')
    fine_mc=2*np.unpackbits(meta['fine_bits'],axis=-1,bitorder='little').astype(np.int8)-1
    refs,data,rows=core_data()
    fresh_figures(refs,data,rows,meta,fine_mc)
    distribution_details(refs,data)
    mcrows=mc_figures(meta,fine_mc)
    old_local();old_paths();joint_figures();precision_figure();training_figures()
    write(OUT/'data/physical_numerical_checks.json',dict(status='passed',checks=len(CHECKS),maximum_absolute_difference=max(v['maximum_absolute_difference'] for v in CHECKS),details=CHECKS,
            scope='All fresh saved samples and fine/coarse metrics, SciPy distances, MC magnetization and selected direct integer displacements plus all stored sufficient-statistic identities, all 396 small-block KL decompositions',new_training=False,new_generation=False))
    write(OUT/'provenance/physical_figure_sources.json',dict(source_commit='5516370b3ae6e98160d9468668acdcbf851dc902',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            palette=COLORS,line_styles=STYLES,spin_colors=['#f3ecda','#193f41'],figures=FIGURES,inputs=SOURCES,
            fixed_example_rule='Fresh generation ID 0 for every seed/S; MC reference first parent; A3 chain0 ordinal0; A2 mc_chain0_ord0',
            no_post_generation_sign_flips=True,no_new_model_sampling=True))
    print(json.dumps(dict(figures=len(FIGURES),inputs=len(SOURCES),checks=len(CHECKS),max_difference=max(v['maximum_absolute_difference'] for v in CHECKS))),flush=True)


if __name__=='__main__':main()
