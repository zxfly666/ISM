"""Frozen factorial estimands, paired uncertainty and report figures."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import core_geometry_factorial_design as d
import core_geometry_factorial_statistics as st
import evaluate_study as ev
from ism_diffusion import geometry_study as gs
from ism_diffusion.scale_evaluation import open_energy_density

def save(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True);ev.atomic_npz(path,**arrays)

def generation_values(g,ref,rw=None,sw=None):
    """g[seed][arm], paired per-seed image weights, ref all MC parents."""
    rw=np.ones(len(ref['chain'])) if rw is None else rw
    rg=(rw@ref['pair_sum'])/np.maximum(rw@ref['pair_count'],1e-100)
    values=np.empty((6,5,7))
    for s in range(6):
        for a in range(5):
            v=g[s][a];ww=np.ones(len(v['m'])) if sw is None else sw[s]
            curve=(ww@v['pair_sum'])/np.maximum(ww@v['pair_count'],1e-100)
            for j,(lo,hi) in enumerate([(1,8),(9,24),(25,48)]):
                values[s,a,j]=np.sqrt(np.mean((curve[lo:hi+1]-rg[lo:hi+1])**2))/max(np.sqrt(np.mean(rg[lo:hi+1]**2)),1e-12)
            for j,k in enumerate(['m','m2','abs_m','energy'],3):
                values[s,a,j]=np.average(v[k],weights=ww)-np.average(ref[k],weights=rw)
    return values

def generation_bootstrap(g,ref,reps,block,seed,check):
    random=d.rng(seed,block,'generation_bootstrap');draw=np.empty((reps,5,7))
    for i in range(reps):
        if i%50==0:check()
        rw=st.cluster_block_weights(ref['chain'],random,block,8)
        sw=np.array([np.repeat(np.bincount(random.integers(0,4,4),minlength=4),16) for _ in range(6)])
        ss=random.integers(0,6,6)
        draw[i]=generation_values(g,ref,rw,sw)[ss].mean(0)
    return draw

def figures(out,per_seed,boot,summary,gen_point,gen_boot,learning):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':['DejaVu Sans','sans-serif'],'font.size':13,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.5,
        'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    colors=['#767676','#B64342','#42949E','#0F4D92','#9A4D8E']
    def export(fig,name):
        fig.tight_layout(pad=1.5)
        for ext in ['png','pdf']:fig.savefig(out/(name+'.'+ext),dpi=300,facecolor='white',bbox_inches='tight')
        plt.close(fig)
    labels=['W48 held gaps','W96 continuous','W48 held gaps']
    fig,axs=plt.subplots(1,3,figsize=(15,4.5))
    for h,ax in enumerate(axs):
        g=st.PRIMARY_GEOMETRY[h];yy=per_seed[:,:,g,:3,0].mean(2)
        ax.plot(np.arange(5),yy.T,color='#999999',lw=1,alpha=.7)
        for a in range(5):ax.scatter(np.full(6,a),yy[:,a],color=colors[a],s=20,zorder=3)
        ax.set_xticks(range(5),d.ARMS);ax.set_ylabel('CE (nats; lower better)');ax.set_xlabel(labels[h])
    export(fig,'primary_paired_CE')
    fig,ax=plt.subplots(figsize=(8,4.5))
    for h,x in enumerate(summary):
        lo,hi=x['ci98_333'];ax.plot([lo,hi],[h,h],color='#0F4D92',lw=2.5)
        ax.scatter(x['mean'],h,color='#0F4D92',s=35)
    ax.axvline(0,color='black',ls='--',lw=1);ax.axvline(-.002,color='#999999',ls=':',lw=1)
    ax.set_yticks(range(3),['H1: G11 - G10','H2: G11 - G01','H3: G11 - G11S'])
    ax.set_xlabel('Paired CE difference (98.333% crossed interval)')
    export(fig,'primary_factorial_contrasts')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for ax,g in zip(axs,[2,3]):
        for a in [3,4]:
            yy=per_seed[:,a,g,:3,0].mean(0);dd=boot['arm_means'][:,a,g,:3,0]
            lo,hi=np.quantile(dd,[.025,.975],axis=0)
            ax.plot([2,32,512],yy,'o-',color=colors[a],label=d.ARMS[a])
            ax.fill_between([2,32,512],lo,hi,color=colors[a],alpha=.12)
        ax.set_xscale('log');ax.set_xlabel(d.GEOMETRIES[g][0]+' / K');ax.set_ylabel('CE (pointwise 95%)');ax.legend()
    export(fig,'native_coordinate_control')
    # Secondary full 2x2 interaction on every geometry, all three primary Ks.
    cc,_,_=st.conditional_primary(boot['arm_means'])
    point,_,_=st.conditional_primary(per_seed.mean(0))
    fig,ax=plt.subplots(figsize=(12,4.5))
    for g in range(6):
        lo,hi=np.quantile(cc[:,6,g,:3,0].mean(1),[.025,.975]);p=point[6,g,:3,0].mean()
        ax.plot([g,g],[lo,hi],color='#0F4D92',lw=2);ax.scatter(g,p,color='#0F4D92')
    ax.axhline(0,color='black',ls='--',lw=1)
    ax.set_xticks(range(6),[x[0] for x in d.GEOMETRIES],rotation=20,ha='right')
    ax.set_ylabel('Interaction CE difference\n(95%; secondary)')
    export(fig,'secondary_interaction')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for ki,ax in enumerate(axs):
        for a in range(5):
            vv=learning[:,a,:,:,0][:,:,ki]
            ax.plot([4000,8000,12000],vv.mean(0),'o-',label=d.ARMS[a],color=colors[a])
            ax.fill_between([4000,8000,12000],vv.min(0),vv.max(0),color=colors[a],alpha=.10)
        ax.axhline(np.log(2),color='black',ls=':',lw=1)
        ax.set_xlabel('Update (band = min/max of six seeds)');ax.set_ylabel(['Validation CE: t~.5','Validation CE: t~.95'][ki]);ax.legend()
    export(fig,'learning_curves')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    axs[0].plot(range(5),gen_point[:,:,2].T,color='#999999',lw=1)
    for a in range(5):axs[0].scatter(np.full(6,a),gen_point[:,a,2],color=colors[a],s=25)
    axs[0].set_xticks(range(5),d.ARMS);axs[0].set_ylabel('W96 G25-48 NRMSE (secondary)')
    for j,(u,v) in enumerate([(3,1),(3,2),(3,4)]):
        dd=gen_boot[:,u,2]-gen_boot[:,v,2];lo,hi=np.quantile(dd,[.025,.975])
        axs[1].plot([lo,hi],[j,j],color='#0F4D92',lw=2)
        axs[1].scatter((gen_point[:,u,2]-gen_point[:,v,2]).mean(),j,color='#0F4D92')
    axs[1].axvline(0,color='black',ls='--',lw=1)
    axs[1].set_yticks(range(3),['G11 - G10','G11 - G01','G11 - G11S']);axs[1].set_xlabel('NRMSE difference (95%, secondary)')
    export(fig,'secondary_generation')

def analyze_arrays(out,values,chain,g,ref,learning,check=lambda:None,scratch=False):
    out.mkdir(parents=True,exist_ok=False)
    assert values.shape==(6,5,6,5,128,2) and learning.shape==(6,5,3,2,2)
    per_seed=values.mean(4);point=per_seed.mean(0)
    save(out/'conditional_point.npz',values=values,per_seed=per_seed,chain=chain,arms=np.array(d.ARMS))
    primary_boot=None
    conditional_started=time.perf_counter()
    for block,reps in [(2,20000),(1,10000),(4,10000)]:
        check();boot=st.conditional_bootstrap(values,chain,20 if scratch else reps,block,2026092531,check=check)
        save(out/f'conditional_bootstrap_block{block}.npz',**boot)
        if block==2: primary_boot=boot
    for mode in ['seed_only','mc_only']:
        boot=st.conditional_bootstrap(values,chain,20 if scratch else 10000,2,2026092532,mode,check)
        save(out/f'conditional_{mode}.npz',**boot)
    summary=st.primary_summary(per_seed,primary_boot)
    conditional_seconds=time.perf_counter()-conditional_started
    retention=[]
    _,_,ret=st.conditional_primary(point)
    for k in range(2):
        ci=st.interval(primary_boot['retention'][:,k],.975)
        retention.append(dict(condition=k,estimate=float(ret[k]),ci95=st.interval(primary_boot['retention'][:,k],.95),
            ci97_5=ci,retained=bool(ci[1]<=.005)))
    records=[]
    for s,seed in enumerate(d.SEEDS):
        for a,arm in enumerate(d.ARMS):
            for geo,(name,w,_) in enumerate(d.GEOMETRIES):
                for k,K in enumerate(d.evaluation_ks(w)):
                    records.append(dict(seed=seed,arm=arm,geometry=name,K=K,t=1-K/w**2,
                        ce=float(per_seed[s,a,geo,k,0]),brier=float(per_seed[s,a,geo,k,1])))
    ev.write_csv(out/'per_seed_metrics.csv',records)
    contrast,_,_=st.conditional_primary(point)
    cb,_,_=st.conditional_primary(primary_boot['arm_means'])
    records=[]
    for c,name in enumerate(st.CONTRAST_NAMES):
        for geo,(gname,w,_) in enumerate(d.GEOMETRIES):
            for k,K in enumerate(d.evaluation_ks(w)):
                lo,hi=np.quantile(cb[:,c,geo,k,0],[.025,.975])
                records.append(dict(contrast=name,geometry=gname,K=K,ce=float(contrast[c,geo,k,0]),
                    ci95_low=float(lo),ci95_high=float(hi),role='secondary_pointwise'))
    ev.write_csv(out/'paired_contrasts.csv',records)
    gen_point=generation_values(g,ref);save(out/'generation_point.npz',values=gen_point)
    gen_first=None;gen_summaries={}
    generation_started=time.perf_counter()
    for block,reps in [(8,10000),(4,5000),(16,5000)]:
        gb=generation_bootstrap(g,ref,20 if scratch else reps,block,2026092541,check)
        save(out/f'generation_bootstrap_block{block}.npz',values=gb)
        if block==8:gen_first=gb
        gen_summaries[str(block)]=[dict(u=d.ARMS[u],v=d.ARMS[v],
            estimate=(gen_point[:,u]-gen_point[:,v]).mean(0).tolist(),
            ci95=st.interval(gb[:,u]-gb[:,v],.95)) for u,v in [(3,1),(3,2),(3,4)]]
    generation_seconds=time.perf_counter()-generation_started
    learning_flags=[]
    for s,seed in enumerate(d.SEEDS):
        for a,arm in enumerate(d.ARMS):
            final=learning[s,a,2,:,0];late=learning[s,a,1,:,0]-final
            learning_flags.append(dict(seed=seed,arm=arm,final_ce=final.tolist(),late_improvement=late.tolist(),
                insufficient=bool(np.any(final>=np.log(2)) or np.any(late>.005))))
    save(out/'plot_data.npz',per_seed=per_seed,learning=learning,generation=gen_point)
    render_started=time.perf_counter()
    figures(out,per_seed,primary_boot,summary,gen_point,gen_first,learning)
    render_seconds=time.perf_counter()-render_started
    projected=1.2*(conditional_seconds*(600 if scratch else 1)+generation_seconds*(20000/60 if scratch else 1)+render_seconds)+120
    gs.atomic_json(out/'timing.json',dict(conditional_seconds=conditional_seconds,generation_seconds=generation_seconds,
        render_seconds=render_seconds,projected_full_statistics_seconds=projected,
        scratch_fixture=scratch,includes_20pct_safety_and_120sec_table_reserve=True))
    result=dict(primary=summary,retention=retention,learning_flags=learning_flags,
        generation_secondary=gen_summaries,scientific_scratch=bool(scratch),
        query_is_not_independent_seed=True,signed_bias_target_is_zero=True)
    gs.atomic_json(out/'crossed_uncertainty.json',result)
    return result

def fixture(out,check=lambda:None):
    chain=np.repeat(np.arange(8),16);random=np.random.default_rng(100)
    values=random.random((6,5,6,5,128,2))*.01+.5
    learning=np.full((6,5,3,2,2),.5)
    ref=dict(pair_sum=np.ones((1024,96)),pair_count=np.ones((1024,96)),
        chain=np.repeat(np.arange(8),128),m=np.zeros(1024),m2=np.zeros(1024),abs_m=np.zeros(1024),energy=np.zeros(1024))
    one={k:np.array(v[:64],copy=True) for k,v in ref.items() if k!='chain'}
    g=[[one for _ in range(5)] for _ in range(6)]
    assert np.allclose(generation_values(g,ref),0)
    result=analyze_arrays(out,values,chain,g,ref,learning,check,True)
    assert all(np.allclose(x['estimate'],0) for x in result['generation_secondary']['8'])
    return result
