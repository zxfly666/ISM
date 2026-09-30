"""All preregistered I analyses; immutable raw data, explicit uncertainty axes."""
import csv,time
from pathlib import Path
import numpy as np
import identification_common as c
import identification_statistics as st
import identification_data as d

def csvwrite(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def collect(root,check=c.check,generation_images=128):
    main=[];secondary=[];gen=[];pad=[];low=[];learning=[];names=[]
    geometry=c.CONFIG['conditional_geometries']
    for s,seed in enumerate(c.SEEDS):
        mm=[];se=[];gg=[];pp=[];ll=[];vv=[]
        for a,arm in enumerate(c.ARMS):
            out=root/'evaluation'/f's{seed}_{arm}';assert c.read(out/'complete.json')['status']=='complete'
            mr=[];sr=[]
            for gi,g in enumerate(geometry):
                for k in g['ks']:
                    check();b=c.load(root/'banks'/f'{g["name"]}_k{k}.npz');reps=4 if arm=='I-R' and g['kind']!='continuous' else 1
                    risks=[]
                    for rep in range(reps):
                        z=c.load(out/'conditional'/f'{g["name"]}_k{k}_r{rep}.npz')
                        assert str(z['common_data_hash'])==d.common_hash(b)
                        assert str(z['actual_input_hash'])==c.ah(b['noisy'],d.native_coordinates(b,arm,rep),b['t'],b['queries'])
                        assert np.array_equal(z['parent'],b['parent']) and np.array_equal(z['chain'],b['chain'])
                        p=np.clip(z['probability'],1e-12,1-1e-12);y=b['labels'];ce=-(y*np.log(p)+(1-y)*np.log1p(-p));br=(z['probability']-y)**2
                        assert np.allclose(ce,z['ce'],atol=1e-12,rtol=0) and np.allclose(br,z['brier'],atol=1e-12,rtol=0)
                        risks.append(np.stack([ce.mean(1),br.mean(1)],-1))
                    risk=np.mean(risks,axis=0)
                    if g['name']=='H48':mr.append(risk)
                    else:
                        sr.append(risk)
                        if s==0 and a==0:names.append(f'{g["name"]}_k{k}')
            mm.append(np.array(mr));se.append(np.array(sr));gg.append(c.load(out/'generation/statistics.npz'))
            one=[]
            for k in c.CONFIG['padding']['ks']:
                z=[c.load(out/'padding'/f'k{k}_v{v}.npz') for v in range(3)]
                assert all(np.array_equal(z[0]['parent'],v['parent']) and np.array_equal(z[0]['chain'],v['chain']) for v in z[1:])
                one.append(np.stack([np.stack([(z[j]['ce']-z[i]['ce']).mean(1),(z[j]['brier']-z[i]['brier']).mean(1),np.abs(z[j]['probability']-z[i]['probability']).mean(1)],-1) for i,j in [(0,1),(1,2)]]))
            pp.append(np.array(one))
            lowdata=c.load(out/'low_k.npz')['probability']
            if lowdata.shape[-1]==1:lowdata=np.repeat(lowdata,4,axis=-1)
            assert lowdata.shape==(80,2,6,4);ll.append(lowdata)
            vv.append(np.array([c.load(root/'training'/f's{seed}_{arm}'/f'validation_{step}.npz')['metrics'] for step in [4000,8000,12000]]))
        main.append(mm);secondary.append(se);gen.append(gg);pad.append(pp);low.append(ll);learning.append(vv)
    # Independently keyed image identities must match across all arms.
    for seed in c.SEEDS:
        for start in range(0,generation_images,16):
            z=[c.load(root/'evaluation'/f's{seed}_{arm}'/'generation'/f'shard_{start:05d}.npz') for arm in c.ARMS]
            for k in ['image_ids','rng_seeds']:assert all(np.array_equal(z[0][k],v[k]) for v in z[1:])
    chain=c.load(root/'banks/H48_k512.npz')['chain'];subchain=c.load(root/'banks/C48_k2.npz')['chain']
    return dict(main=np.array(main),secondary=np.array(secondary),secondary_names=np.array(names),generation=gen,padding=np.array(pad),
                low_model=np.array(low),learning=np.array(learning),chain=chain,subchain=subchain,
                low_reference=c.load(root/'reference/low_joint_counts.npz'),reference=c.load(root/'reference/generation_reference.npz'))

def figures(out,main,main_boot,secondary,secondary_boot,names,primary,pad,pad_boot,low_point,low_boot,gen_point,gen_boot,learning):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':['DejaVu Sans','sans-serif'],'font.size':12,'axes.linewidth':1.5,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42})
    colors=['#0F4D92','#767676','#B64342'];markers=['o','s','^'];captions={}
    def export(fig,name,caption):
        fig.tight_layout(pad=1.5);fig.savefig(out/(name+'.png'),dpi=300,facecolor='white');fig.savefig(out/(name+'.pdf'),facecolor='white');plt.close(fig);captions[name]=caption
    def ci_points(ax,x,point,ci,**kw):
        ax.errorbar(x,point,yerr=np.maximum(np.stack([point-ci[0],ci[1]-point]),0),capsize=3,**kw)
    fig,ax=plt.subplots(figsize=(8,3.8))
    for i,row in enumerate(primary):
        lo,hi=row['primary_ci'];m=row['estimate'];ax.errorbar(m,i,xerr=[[max(0,m-lo)],[max(0,hi-m)]],fmt=markers[i],color=colors[i*2],capsize=5)
    ax.axvline(0,color='black',lw=1);ax.axvline(-.002,color='#767676',ls='--');ax.set_yticks([0,1],['P1: Fine - Span','P2: Fine - Random']);ax.set_xlabel('Paired CE difference (nat; negative favors Fine)')
    export(fig,'primary_contrasts','H48 K512; preregistered paired seed x MC-chain/block 97.5% CIs. Dashed line: -0.002 practical threshold.')
    fig,ax=plt.subplots(figsize=(8,5));ks=np.array([2,8,32,128,512,1152])
    for a in [1,2]:
        point=(main[:,0,:,:,0]-main[:,a,:,:,0]).mean((0,2));draw=main_boot[:,0,:,0]-main_boot[:,a,:,0]
        ax.plot(ks,point,marker=markers[a],color=colors[a],label='Fine - '+['Fine','Span','Random'][a]);ci=st.interval(draw);ax.fill_between(ks,ci[0],ci[1],color=colors[a],alpha=.17)
    ax.set_xscale('log',base=2);ax.set_xticks(ks,ks.astype(str));ax.axhline(0,color='black',lw=1);ax.set_xlabel('Visible sites K (natural clock changes with K)');ax.set_ylabel('Paired CE difference (nat)');ax.legend()
    export(fig,'observation_regimes','H48; pointwise exploratory95% paired intervals. K512 alone is the preregistered primary cell; density and natural clock are not separated.')
    fig,ax=plt.subplots(figsize=(9,5));labels=['C48','U48','H48','B48','C96','H96']
    for a in [1,2]:
        pts=[];cis=[]
        for name in labels:
            if name=='H48':v=main[:,0,4,:,0]-main[:,a,4,:,0];draw=main_boot[:,0,4,0]-main_boot[:,a,4,0]
            else:
                j=list(names).index(name+'_k512');v=secondary[:,0,j,:,0]-secondary[:,a,j,:,0];draw=secondary_boot[:,0,j,0]-secondary_boot[:,a,j,0]
            pts.append(v.mean());cis.append(st.interval(draw))
        ci_points(ax,np.arange(6)+(a-1.5)*.15,np.array(pts),np.array(cis).T,fmt=markers[a],color=colors[a],label='Fine - '+['Fine','Span','Random'][a])
    ax.set_xticks(range(6),labels);ax.axhline(0,color='black',lw=1);ax.set_ylabel('K512 paired CE difference (nat)');ax.legend()
    export(fig,'geometry_stress','All geometries, pointwise95% CIs. B48 is clustered-layout stress, not a marginally matched coordinate-distribution control.')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8));groups=[slice(0,32),slice(32,64),slice(64,72),slice(72,80)]
    for ki,ax in enumerate(axs):
        pts=np.array([low_point[g,ki,0].mean() for g in groups]);draw=np.stack([low_boot[:,g,ki,0].mean(1) for g in groups],-1)
        ci_points(ax,np.arange(4),pts,st.interval(draw),fmt='o',color=colors[0]);ax.set_xticks(range(4),['Train','Held','Train null','Held null'],rotation=15);ax.set_ylabel(f'K={ki+1}: Bayes geometry information (nat)');ax.axhline(0,color='black',lw=1)
    export(fig,'low_k_information','Constructed paired challenges, not representative natural H48 cases.95% intervals include finite MC reference uncertainty; null pairs preserve actual query/visible positions.')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    for ki,ax in enumerate(axs):
        # Correct native-information Bayes excess differs for Fine vs Span/Random.
        for a in range(3):
            pts=np.array([low_point[g,ki,5+a].mean() for g in groups[:2]])
            draw=np.stack([low_boot[:,g,ki,5+a].mean(1) for g in groups[:2]],-1)
            ci_points(ax,np.arange(2)+(a-1)*.12,pts,st.interval(draw),fmt=markers[a],color=colors[a],label=c.ARMS[a])
        ax.set_xticks([0,1],['Train challenges','Held challenges']);ax.set_ylabel(f'K={ki+1}: native-information excess CE');ax.legend()
    export(fig,'low_k_excess_risk','Fine excess relative to fine Bayes; Span/Random excess relative to coarse Bayes. Different information baselines must not be mistaken for common absolute-risk ranking.')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    for contrast,ax in enumerate(axs):
        for a in range(3):
            p=pad[:,a,:,contrast,:,0].mean((0,2));draw=pad_boot[:,a,:,contrast,0]
            ci_points(ax,np.arange(3)+(a-1)*.1,p,st.interval(draw),fmt=markers[a],color=colors[a],label=c.ARMS[a])
        ax.set_xticks(range(3),['K2','K32','K512']);ax.axhline(0,color='black',lw=1);ax.set_ylabel(['Large fixed clock - small','Large natural - large fixed'][contrast]+' CE');ax.legend()
    export(fig,'masked_context_extension','Same observed spins and queries; extra sites MASK but attended. Left: representation expansion at fixed clock; right: clock intervention response. Pointwise95% intervals, no unique attention-mechanism attribution.')
    fig,axs=plt.subplots(2,4,figsize=(15,7.5));metric=['G1-8 NRMSE','G9-24 NRMSE','G25-48 NRMSE','m bias','m2 bias','abs(m) bias','Energy bias']
    for j,ax in enumerate(axs.flat):
        if j==7:ax.axis('off');continue
        for a in range(3):
            ci=st.interval(gen_boot[:,a,j]);p=gen_point[:,a,j].mean();ci_points(ax,[a],np.array([p]),ci[:,None],fmt=markers[a],color=colors[a])
            ax.scatter(np.full(6,a)+np.linspace(-.07,.07,6),gen_point[:,a,j],s=12,color=colors[a],alpha=.6)
        ax.set_xticks(range(3),['F','S','R']);ax.set_ylabel(metric[j]);ax.axhline(0,color='black',lw=.8)
    export(fig,'secondary_generation','W96 generated128 images/model; secondary95% intervals, paired training seeds/images and MC chain blocks. Signed biases target zero, not lower values; dots show each training seed.')
    fig,axs=plt.subplots(2,4,figsize=(16,7));random=c.rng(2026092651,0,'learning_seed_only');inds=random.integers(6,size=(10000,6))
    for j,ax in enumerate(axs.flat):
        for a in range(3):
            vals=learning[:,a,:,j,0];p=vals.mean(0);ci=st.interval(vals[inds].mean(1))
            ax.plot([4,8,12],p,marker=markers[a],color=colors[a],label=c.ARMS[a]);ax.fill_between([4,8,12],ci[0],ci[1],color=colors[a],alpha=.15)
        ax.set_xlabel('Training step (thousands)');ax.set_ylabel('Validation CE '+str(j+1));ax.legend(fontsize=9)
    export(fig,'learning_curves','Eight fixed old-val conditions. Bands are training-seed-only95% bootstrap, conditional on fixed validation data; not unseen-geometry convergence guarantees.')
    c.write(out/'figure_captions.json',captions)

def analyze(root,arrays=None,scratch=False,check=c.check):
    started=time.perf_counter();root=Path(root);out=root/'analysis';out.mkdir(parents=True,exist_ok=False)
    z=collect(root,check) if arrays is None else arrays
    main=z['main'];sec=z['secondary'];names=z['secondary_names'];pad=z['padding'];learning=z['learning']
    c.save(out/'conditional_point.npz',main=main,secondary=sec,secondary_names=names,chain=z['chain'],subset_chain=z['subchain'])
    boot={};sens={}
    for block,reps in [(8,20000),(4,10000),(16,10000)]:
        draw=st.bootstrap(main.transpose(0,1,2,4,3),z['chain'],16 if scratch else reps,block,'main',check=check)
        c.save(out/f'main_bootstrap_block{block}.npz',arm_means=draw);boot[block]=draw
    for mode in ['seed_only','mc_only']:
        sens[mode]=st.bootstrap(main.transpose(0,1,2,4,3),z['chain'],16 if scratch else 10000,8,'main',mode,check)
        c.save(out/f'main_bootstrap_{mode}.npz',arm_means=sens[mode])
    sb={}
    for block in [2,1,4]:
        sb[block]=st.bootstrap(sec.transpose(0,1,2,4,3),z['subchain'],16 if scratch else 10000,block,'secondary',check=check)
        c.save(out/f'secondary_bootstrap_block{block}.npz',arm_means=sb[block])
    point=main.mean(3);primary=[]
    for a in [1,2]:
        ps=point[:,0,4,0]-point[:,a,4,0];draw=boot[8][:,0,4,0]-boot[8][:,a,4,0]
        r=st.compare_summary(ps,draw,'P'+str(a));r['blocks']={str(k):st.interval(v[:,0,4,0]-v[:,a,4,0],.975).tolist() for k,v in boot.items()}
        r['components']={k:st.interval(v[:,0,4,0]-v[:,a,4,0],.975).tolist() for k,v in sens.items()}
        delta=main[:,0,4,:,0]-main[:,a,4,:,0];r['leave_one_chain_out']=[float(delta[:,z['chain']!=ch].mean()) for ch in np.unique(z['chain'])];primary.append(r)
    # Keep original three-K mean and the targeted condition interaction secondary.
    delta=point[:,0,:,0]-point[:,1,:,0];bd=boot[8][:,0,:,0]-boot[8][:,1,:,0]
    secondary_summaries=dict(original_three_K_mean=dict(estimate=float(delta[:,[0,2,4]].mean()),ci95=st.interval(bd[:,[0,2,4]].mean(1)).tolist()),
                            high_low_interaction=dict(estimate=float((delta[:,4]-delta[:,[0,2]].mean(1)).mean()),ci95=st.interval(bd[:,4]-bd[:,[0,2]].mean(1)).tolist()))
    retention=[]
    for k in [115,1152]:
        j=list(names).index('C48_k'+str(k))
        for a in [1,2]:
            x=sb[2][:,0,j,0]-sb[2][:,a,j,0];ci=st.interval(x,.9875)
            retention.append(dict(k=k,contrast='Fine-'+c.ARMS[a],estimate=float((sec[:,0,j,:,0]-sec[:,a,j,:,0]).mean()),ci98_75=ci.tolist(),ci95=st.interval(x).tolist(),retained=bool(ci[1]<=.005)))
    flags=[]
    for s,seed in enumerate(c.SEEDS):
        for a,arm in enumerate(c.ARMS):
            final=learning[s,a,2,:,0];late=learning[s,a,1,:,0]-final
            flags.append(dict(seed=seed,arm=arm,final_ce=final.tolist(),late_improvement=late.tolist(),insufficient=bool(np.any(final>=np.log(2)) or np.any(late>.005))))
    pb=st.bootstrap(pad.transpose(0,1,2,3,5,4),z['subchain'],16 if scratch else 10000,2,'padding',check=check)
    c.save(out/'padding_point_bootstrap.npz',values=pad,bootstrap=pb)
    # Save alternative subset block sensitivities, not a favorable-method choice.
    for block in [1,4]:c.save(out/f'padding_bootstrap_block{block}.npz',bootstrap=st.bootstrap(pad.transpose(0,1,2,3,5,4),z['subchain'],16 if scratch else 10000,block,'padding',check=check))
    counts=z['low_reference']['counts'];lp=st.low_metrics(counts.mean(0),z['low_model'])
    low_all=st.low_bootstrap(counts,z['low_reference']['chain'],z['low_model'],16 if scratch else 2000,check,complete=True);lb=low_all['ce']
    p,pv,coarse=st.low_distribution(counts.mean(0));rare=pv.mean(-2)<.01
    c.save(out/'low_k_point_bootstrap.npz',point=lp,bootstrap=lb,conditional=p,visible_probability=pv,coarse_conditional=coarse,rare=rare,model_probability=z['low_model'])
    c.save(out/'low_k_brier_response.npz',brier_point=st.low_metrics(counts.mean(0),z['low_model'],'brier'),brier_bootstrap=low_all['brier'],
           model_response_point=(z['low_model'][:,:,:,1]-z['low_model'][:,:,:,0]).mean((0,-1)),model_response_bootstrap=low_all['model_response'],
           reference_response_point=p[:,1]-p[:,0],reference_response_bootstrap=low_all['reference_response'],
           undefined_conditional=~np.isfinite(p),undefined_coarse=~np.isfinite(coarse))
    gp=st.gen_values(z['generation'],z['reference']);gb={}
    for block,unit,reps in [(8,'image',10000),(4,'image',5000),(16,'image',5000),(8,'shard',5000)]:
        gb[(block,unit)]=st.gen_bootstrap(z['generation'],z['reference'],16 if scratch else reps,block,unit,check)
        c.save(out/f'generation_bootstrap_block{block}_{unit}.npz',arm_means=gb[(block,unit)])
    c.save(out/'generation_point.npz',values=gp,per_seed_absolute_bias=np.abs(gp[:,:,3:]))
    gen_summary=[]
    for a in [1,2]:gen_summary.append(dict(contrast='Fine-'+c.ARMS[a],estimate=(gp[:,0]-gp[:,a]).mean(0).tolist(),ci95=st.interval(gb[(8,'image')][:,0]-gb[(8,'image')][:,a]).tolist(),secondary=True))
    rows=[]
    for si,seed in enumerate(c.SEEDS):
        for a,arm in enumerate(c.ARMS):
            for j,k in enumerate([2,8,32,128,512,1152]):rows.append(dict(seed=seed,arm=arm,geometry='H48',k=k,ce=float(point[si,a,j,0]),brier=float(point[si,a,j,1])))
    csvwrite(out/'main_per_seed_metrics.csv',rows)
    c.save(out/'plot_data.npz',main=point,secondary=sec.mean(3),secondary_names=names,padding=pad.mean(4),low_k=lp,generation=gp,learning=learning)
    figures(out,main,boot[8],sec,sb[2],names,primary,pad,pb,lp,lb,gp,gb[(8,'image')],learning)
    result=dict(primary=primary,retention=retention,secondary=secondary_summaries,learning_flags=flags,generation_secondary=gen_summary,
                low_k_rare_conditions=int(rare.sum()),low_k_undefined_conditionals=int((~np.isfinite(p)).sum()),new_primary_not_old_H3_replacement=True)
    c.write(out/'summary.json',result)
    c.write(out/'timing.json',dict(elapsed_seconds=time.perf_counter()-started,scratch=scratch,
        note='Full statistics reserve is2700s; scratch timing is not a coverage or speed guarantee'))
    return result

def fixture_arrays():
    random=c.rng(2026092651,0,'analysis_fixture');ns=6;np_=256
    #16 MC chains x16 parents supports all main block sizes without pretending to be formal128/chain.
    chain=np.repeat(np.arange(16),16);subchain=chain.copy()
    base=random.uniform(.3,.6,(ns,1,6,np_,2));main=np.repeat(base,3,axis=1)+np.array([0,.004,.006])[None,:,None,None,None]
    names=[f'{g["name"]}_k{k}' for g in c.CONFIG['conditional_geometries'] if g['name']!='H48' for k in g['ks']]
    sec=random.uniform(.3,.6,(6,3,len(names),np_,2));pad=random.normal(0,.001,(6,3,3,2,np_,3));pad[...,2]=np.abs(pad[...,2])
    lowmodel=random.uniform(.35,.65,(6,3,80,2,6,4));counts=np.full((np_,80,2,8),32,dtype=np.int64)
    def physical(n):
        r=np.arange(96);curve=np.exp(-r/70)
        return dict(pair_sum=np.tile(curve,(n,1))+random.normal(0,.01,(n,96)),pair_count=np.ones((n,96)),m=random.normal(0,.2,n),m2=random.uniform(.1,.2,n),abs_m=random.uniform(.2,.3,n),energy=random.normal(-.7,.01,n))
    ref=physical(np_);ref['chain']=chain;gen=[[physical(128) for _ in range(3)] for _ in range(6)]
    return dict(main=main,secondary=sec,secondary_names=np.array(names),chain=chain,subchain=subchain,
                generation=gen,reference=ref,padding=pad,low_model=lowmodel,
                low_reference=dict(counts=counts,chain=chain),learning=random.uniform(.3,.5,(6,3,3,8,2)))
