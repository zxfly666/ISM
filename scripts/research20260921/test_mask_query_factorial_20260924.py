"""Reference-free prelaunch integration checks for four-arm statistics and banks."""
import numpy as np
from types import SimpleNamespace
from mask_query_factorial_design import ARMS,GEN_DEFS,COND_DEFS,make_cases,auxiliary_views
from evaluate_mask_query_factorial_20260924 import condition_bank
from finalize_mask_query_factorial_20260924 import scores,CONTRASTS


def test_scores_and_banks():
    cases=make_cases();pred=np.ones((4,6,116,2,20,3))*.5
    chain=np.repeat(np.arange(8),128)
    ref=dict(counts=np.ones((116,1024,8),np.int32)*32,chain=chain,reference_p=np.ones((116,8))/8)
    gr=[]
    for d in GEN_DEFS:
        width=d['width'] if d['kind']=='continuous' else 471
        count=np.ones((1024,width))*10
        gr.append(dict(pair_count=count,pair_sum=count*.3,parent=np.arange(1024),chain=chain,
                       m=np.zeros(1024),m2=np.ones(1024)*.1,abs_m=np.ones(1024)*.2,energy=np.ones(1024)*-.6))
    cond=[];gen=[]
    for a in range(4):
        cond.append([[dict(ce=np.ones(128)*(.6-.1*a),parent=np.arange(128)) for _ in COND_DEFS] for _ in range(6)])
        gen.append([[{k:v[:d['samples']].copy() for k,v in gr[gi].items() if k not in ('parent','chain')}
                     for gi,d in enumerate(GEN_DEFS)] for _ in range(6)])
    point,keys=scores(cases,pred,cond,gen,ref,gr)
    boot,_=scores(cases,pred,cond,gen,ref,gr,np.ones(1024),np.random.default_rng(123))
    assert point.shape==(4,6,83) and np.max(abs(point-boot))<1e-10
    assert np.max(abs(point[:,:,:56]))<1e-10
    for label,coeff in CONTRASTS.items():
        assert coeff.sum()==0
        if label=='interaction':assert abs((coeff@point.mean(1))[keys.index('w48_t05/CE')])<1e-10
    pattern=np.where((np.indices((1024,1024)).sum(0)%3)==0,1,-1).astype(np.int8)
    synthetic=SimpleNamespace(spins=np.broadcast_to(pattern,(1024,1024,1024)),chain_ids=chain,lattice_size=1024)
    for d in COND_DEFS:
        b=condition_bank(synthetic,d)
        assert b['noisy'].shape==(128,1,d['width']**2)
        assert (b['noisy'][np.arange(128)[:,None],0,b['queries']]==2).all()
        assert len(np.unique(b['parent']))==128 and len(np.unique(b['chain']))==8
    cc=np.stack(np.meshgrid(np.arange(24),np.arange(24),indexing='ij'),-1)[None]
    clean=np.zeros((1,24,24),dtype=np.int64)
    choices,retries=auxiliary_views(dict(clean=clean,noisy=clean.copy(),coords={'A':cc}),19,1)
    assert retries>0 and (choices['R00']['noisy']==2).any()
    choices,_=auxiliary_views(dict(clean=clean,noisy=np.full_like(clean,2),coords={'A':cc}),19,2)
    assert np.array_equal(choices['R00']['queries'],choices['R01']['queries'])
    assert all((v['labels']==0).all() for v in choices.values())
    return dict(statistical_columns=len(keys),four_arm_scores_bootstrap=True,
        synthetic_heldout_banks=True,zero_hidden_retry=True,zero_evidence_identity=True)
