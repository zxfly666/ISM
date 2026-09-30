"""CPU paired crossed bootstrap; no new claims or adaptive sample sizes."""
from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t

import core_geometry_factorial_design as design

CONTRAST_NAMES=('H1_distance','H2_context','H3_fine_geometry','size_at_fixed_distance',
                'distance_at_fixed_size','combined','interaction')
CONTRAST=np.array([[0,-1,0,1,0],[0,0,-1,1,0],[0,0,0,1,-1],
                   [-1,1,0,0,0],[-1,0,1,0,0],[-1,0,0,1,0],[1,-1,-1,1,0]],dtype=float)
PRIMARY_GEOMETRY=(2,3,2)


def cluster_block_weights(chain,random,block,expected_groups=None):
    """Resample chains, then circular time blocks in each sampled chain.

    Input order within a chain must be chronological. Returned weights sum to one;
    the same vector is reused across all arms, training seeds and conditions.
    """
    chain=np.asarray(chain)
    groups=[np.flatnonzero(chain==c) for c in np.unique(chain)]
    if not groups or block<1 or any(len(g)<block for g in groups):
        raise ValueError('Invalid bootstrap groups/block')
    if expected_groups is not None and len(groups)!=expected_groups:
        raise ValueError('Unexpected MC chain count')
    weights=np.zeros(len(chain),dtype=float)
    for which in random.integers(0,len(groups),len(groups)):
        g=groups[int(which)];n=len(g)
        starts=random.integers(0,n,int(np.ceil(n/block)))
        local=((starts[:,None]+np.arange(block))%n).ravel()[:n]
        np.add.at(weights,g[local],1./(len(groups)*n))
    assert np.isclose(weights.sum(),1.)
    return weights


def conditional_primary(arm_means):
    """Input [..., arm=5, geometry=6, condition=5, metric=2]."""
    contrasts=np.einsum('ca,...agkm->...cgkm',CONTRAST,arm_means)
    primary=np.stack([contrasts[...,c,g,:3,0].mean(-1)
                      for c,g in enumerate(PRIMARY_GEOMETRY)],axis=-1)
    retention=contrasts[...,5,0,3:5,0]
    return contrasts,primary,retention


def conditional_bootstrap(values,chain,reps,block,seed,mode='joint',check=lambda:None):
    """values[training_seed, arm, geometry, condition, parent, CE/Brier]."""
    values=np.asarray(values,dtype=float)
    expected=(6,5,6,5,128,2)
    if values.shape!=expected or not np.isfinite(values).all():
        raise ValueError(f'Expected finite conditional array {expected}, got {values.shape}')
    if mode not in ('joint','seed_only','mc_only'):
        raise ValueError(mode)
    chain=np.asarray(chain)
    if chain.shape!=(128,) or len(np.unique(chain))!=8:
        raise ValueError('Expected 128 parents across eight chains')
    random=design.rng(seed,block,'conditional_bootstrap_'+mode)
    means=np.empty((reps,5,6,5,2));primary=np.empty((reps,3));retention=np.empty((reps,2))
    for i in range(reps):
        if i%100==0: check()
        sw=np.bincount(random.integers(0,6,6),minlength=6)/6 if mode!='mc_only' else np.full(6,1/6)
        pw=cluster_block_weights(chain,random,block,8) if mode!='seed_only' else np.full(128,1/128)
        means[i]=np.einsum('s,sagkpm,p->agkm',sw,values,pw,optimize=True)
        _,primary[i],retention[i]=conditional_primary(means[i])
    return dict(arm_means=means,primary=primary,retention=retention,
                block=np.array(block),mode=np.array(mode),reps=np.array(reps))


def interval(values,level,axis=0):
    return np.quantile(values,[(1-level)/2,(1+level)/2],axis=axis).tolist()


def primary_summary(per_seed_means,bootstrap):
    _,seed_effects,_=conditional_primary(per_seed_means)
    result=[]
    for i,name in enumerate(CONTRAST_NAMES[:3]):
        x=seed_effects[:,i];mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(6))
        alpha=.05/3
        half=float(student_t.ppf(1-alpha/2,5)*se)
        ci=interval(bootstrap['primary'][:,i],1-alpha)
        result.append(dict(name=name,mean=mean,per_seed=x.tolist(),paired_seed_mcse=se,
            ci95=interval(bootstrap['primary'][:,i],.95),ci98_333=ci,
            paired_t_ci98_333=[mean-half,mean+half],
            bootstrap_directional=bool(ci[1]<0),practical=bool(ci[1]<-.002),
            statistical_method_sensitive=bool((ci[1]<0)!=((mean+half)<0)),
            leave_one_seed_out=[float(np.delete(x,j).mean()) for j in range(6)],
            bootstrap_half_endpoint_difference=(np.quantile(bootstrap['primary'][:len(bootstrap['primary'])//2,i],
                [alpha/2,1-alpha/2])-np.quantile(bootstrap['primary'][len(bootstrap['primary'])//2:,i],
                [alpha/2,1-alpha/2])).tolist()))
    return result


def self_test():
    chain=np.repeat(np.arange(8),16)
    weight=cluster_block_weights(chain,np.random.default_rng(1),2,8)
    assert np.isclose(weight.sum(),1.) and np.all(weight>=0)
    # Parent and seed variation is common across all arms: paired differences
    # must remain constant under every crossed resampling realization.
    v=np.empty((6,5,6,5,128,2))
    for s in range(6):
        for a in range(5):
            v[s,a]=s/100+a/1000+np.arange(128)[None,None,:,None]/10000
    boot=conditional_bootstrap(v,chain,16,2,123)
    replay=conditional_bootstrap(v,chain,16,2,123)
    assert np.array_equal(boot['primary'],replay['primary'])
    assert np.allclose(boot['primary'],[.002,.001,-.001],atol=1e-14)
    contrasts,primary,retention=conditional_primary(v.mean(4))
    assert np.allclose(contrasts[:,6],0,atol=1e-14)
    assert np.allclose(retention,.003,atol=1e-14)
    fixture=np.array([0.,2.,3.,10.,8.])
    assert np.array_equal(CONTRAST@fixture,[8,7,2,2,3,10,5])
    summary=primary_summary(v.mean(4),boot)
    assert np.allclose([x['mean'] for x in summary],[.002,.001,-.001])
    return dict(status='passed_conditional_bootstrap_fixture_only',
        pairing_preserves_constant_differences=True,deterministic_replay=True,
        contrast_order_checked=True,three_K_average_checked=True,
        caveat='Algebra and replay tests, not a coverage guarantee with six training seeds')


if __name__=='__main__':
    import json
    print(json.dumps(self_test()))
