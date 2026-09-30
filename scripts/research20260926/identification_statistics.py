"""Paired crossed resampling and information diagnostics, no torch or files."""
import numpy as np
from scipy.stats import t as student_t
import identification_common as c

def block_weights(chain,random,block):
    groups=[np.flatnonzero(chain==x) for x in np.unique(chain)];weights=np.zeros(len(chain))
    for which in random.integers(len(groups),size=len(groups)):
        g=groups[which];n=len(g);assert n>=block
        starts=random.integers(n,size=(n+block-1)//block);local=((starts[:,None]+np.arange(block))%n).ravel()[:n]
        np.add.at(weights,g[local],1/(len(groups)*n))
    return weights

def bootstrap(values,chain,reps,block,role,mode='joint',check=c.check):
    """[seed, arbitrary outcome axes..., parent]; shared weights across outcomes."""
    v=np.asarray(values,dtype=np.float64);assert v.shape[-1]==len(chain) and np.isfinite(v).all()
    ns=v.shape[0];shape=v.shape[1:-1];flat=v.reshape(ns,-1,len(chain));n=len(chain)
    random=c.rng(c.CONFIG['role_seeds']['bootstrap'],block,role+'_'+mode);output=np.empty((reps,flat.shape[1]))
    for start in range(0,reps,128):
        check();b=min(128,reps-start);sw=[];pw=[]
        for _ in range(b):
            sw.append(np.bincount(random.integers(ns,size=ns),minlength=ns)/ns if mode!='mc_only' else np.ones(ns)/ns)
            pw.append(block_weights(chain,random,block) if mode!='seed_only' else np.ones(n)/n)
        sw=np.array(sw);pw=np.array(pw);result=np.zeros((b,flat.shape[1]))
        for s in range(ns):result+=(pw@flat[s].T)*sw[:,s,None]
        output[start:start+b]=result
    return output.reshape(reps,*shape)

def interval(x,level=.95):return np.quantile(x,[(1-level)/2,(1+level)/2],axis=0)

def compare_summary(per_seed,draw,name,level=.975,margin=-.002):
    per_seed=np.asarray(per_seed);mean=float(per_seed.mean());se=float(per_seed.std(ddof=1)/np.sqrt(len(per_seed)))
    ci=interval(draw,level);half=float(student_t.ppf((1+level)/2,len(per_seed)-1)*se)
    return dict(name=name,estimate=mean,per_seed=per_seed.tolist(),ci95=interval(draw).tolist(),primary_ci_level=level,primary_ci=ci.tolist(),
                practical_margin=margin,directional=bool(ci[1]<0),practical=bool(ci[1]<margin),
                bootstrap_sd=float(draw.std(ddof=1)),precision_target_met=bool(draw.std(ddof=1)<=.0005),
                paired_t_ci=[mean-half,mean+half],method_sensitive=bool((ci[1]<0)!=(mean+half<0)),
                leave_one_seed_out=[float(np.delete(per_seed,i).mean()) for i in range(len(per_seed))],
                half_endpoint_difference=(interval(draw[:len(draw)//2],level)-interval(draw[len(draw)//2:],level)).tolist())

def low_distribution(counts):
    """counts[..., side=2, cell=8]; valid symmetry enforced jointly, never per-C clipping."""
    table=(counts+counts[...,::-1]).astype(np.float64);table/=table.sum(-1,keepdims=True)
    # code=y*4+v1*2+v2; conditions [+],[-],[++],[+-],[-+],[--].
    den=np.stack([table[...,2:4].sum(-1)+table[...,6:8].sum(-1),table[...,0:2].sum(-1)+table[...,4:6].sum(-1),
                  table[...,3]+table[...,7],table[...,2]+table[...,6],table[...,1]+table[...,5],table[...,0]+table[...,4]],-1)
    num=np.stack([table[...,6:8].sum(-1),table[...,4:6].sum(-1),table[...,7],table[...,6],table[...,5],table[...,4]],-1)
    # Undefined individual events remain NaN in saved conditional arrays.
    # Their zero empirical mass contributes zero to the weighted overall risk.
    p=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
    # Side prior1/2, posterior given the visible pattern is proportional to den.
    ds=den.sum(-2);coarse=np.divide(num.sum(-2),ds,out=np.full_like(ds,np.nan),where=ds>0)
    return p,den,coarse

def entropy(p):
    pc=np.clip(p,1e-15,1-1e-15);return -(p*np.log(pc)+(1-p)*np.log1p(-pc))

def low_metrics(counts,model,metric='ce'):
    """model[seed,arm,pair,side,condition,rep]; average losses, NOT probabilities.

    Returns [pair, K=1/2, metric=(information, F/S/R risk, fine entropy,
    F excess fine, S/R excess coarse)] with model-seed mean.
    """
    p,pv,coarse=low_distribution(counts)
    fine_h=entropy(p) if metric=='ce' else p*(1-p)
    coarse_h=entropy(coarse) if metric=='ce' else coarse*(1-coarse)
    pred=np.clip(model,1e-12,1-1e-12)
    truth=p[None,None,...,None]
    risk=-(truth*np.log(pred)+(1-truth)*np.log1p(-pred)) if metric=='ce' else truth*(1-pred)**2+(1-truth)*pred**2
    ce=risk.mean(-1).mean(0)
    vals=[]
    for sl in [slice(0,2),slice(2,6)]:
        fh=np.where(pv[...,sl]>0,.5*pv[...,sl]*fine_h[...,sl],0).sum((-2,-1))
        weights=pv[...,sl].mean(-2)
        ch=np.where(weights>0,weights*coarse_h[...,sl],0).sum(-1)
        risk=np.where(pv[None,...,sl]>0,.5*pv[None,...,sl]*ce[...,sl],0).sum((-2,-1)).T
        vals.append(np.column_stack([ch-fh,risk,fh,risk[:,0]-fh,risk[:,1]-ch,risk[:,2]-ch]))
    return np.stack(vals,axis=1)

def low_bootstrap(counts,chain,model,reps=2000,check=c.check,complete=False):
    random=c.rng(c.CONFIG['role_seeds']['bootstrap'],8,'low_k_joint');out=[];brier=[];responses=[];oracle=[]
    for i in range(reps):
        if i%25==0:check()
        ids=random.integers(model.shape[0],size=model.shape[0]);pw=block_weights(chain,random,8)
        table=np.tensordot(pw,counts,axes=(0,0));out.append(low_metrics(table,model[ids]))
        if complete:
            brier.append(low_metrics(table,model[ids],'brier'))
            prob=low_distribution(table)[0];oracle.append(prob[:,1]-prob[:,0])
            responses.append((model[ids,:,:,1]-model[ids,:,:,0]).mean((0,-1)))
    return dict(ce=np.array(out),brier=np.array(brier),model_response=np.array(responses),reference_response=np.array(oracle)) if complete else np.array(out)

def gen_values(g,ref,pw=None,iw=None):
    ns=len(g);na=len(g[0]);pw=np.ones(len(ref['chain']))/len(ref['chain']) if pw is None else pw
    rg=pw@ref['pair_sum']/np.maximum(pw@ref['pair_count'],1e-100);result=np.empty((ns,na,7))
    rm=np.array([pw@ref[k] for k in ['m','m2','abs_m','energy']])
    for s in range(ns):
        for a in range(na):
            z=g[s][a];w=np.ones(len(z['m']))/len(z['m']) if iw is None else iw[s]/iw[s].sum()
            curve=w@z['pair_sum']/np.maximum(w@z['pair_count'],1e-100)
            for j,(lo,hi) in enumerate([(1,8),(9,24),(25,48)]):result[s,a,j]=np.sqrt(np.mean((curve[lo:hi+1]-rg[lo:hi+1])**2))/max(np.sqrt(np.mean(rg[lo:hi+1]**2)),1e-12)
            result[s,a,3:]=np.array([w@z[k] for k in ['m','m2','abs_m','energy']])-rm
    return result

def gen_bootstrap(g,ref,reps,block=8,unit='image',check=c.check):
    ns=len(g);n=len(g[0][0]['m']);random=c.rng(c.CONFIG['role_seeds']['bootstrap'],block,'generation_'+unit);out=[]
    for i in range(reps):
        if i%50==0:check()
        pw=block_weights(ref['chain'],random,block)
        iw=np.array([np.bincount(random.integers(n,size=n),minlength=n) if unit=='image' else np.repeat(np.bincount(random.integers(n//16,size=n//16),minlength=n//16),16) for _ in range(ns)])
        ids=random.integers(ns,size=ns);out.append(gen_values(g,ref,pw,iw)[ids].mean(0))
    return np.array(out)
