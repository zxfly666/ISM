import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location("fgm",Path(__file__).resolve().parents[1]/"scripts/research20260921/fixed_geometry_math.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_spin_flip_identity_and_reference_positive():
    p=m.symmetric_distribution(np.arange(1,9,dtype=float)**2)
    g=(p[:,None]*(m.SIGNS[:,[0,0,1]]*m.SIGNS[:,[1,2,2]])).sum(0)
    for a in (0,1):
        for b in (0,1):
            expected=.5*(1+((2*a-1)*g[0]+(2*b-1)*g[1])/(1+(2*a-1)*(2*b-1)*g[2]))
            np.testing.assert_allclose(m.conditional_q(p,a,b),expected,atol=1e-14)
    np.testing.assert_allclose(m.two_query_reference(p).sum((-1,-2)),1)


def test_oracle_reconstruction_and_kl_decompositions():
    p=m.symmetric_distribution(np.arange(1,9,dtype=float)**3)
    specs=m.query_specs(2304)
    pred=np.full((20,3),.5)
    for i,s in enumerate(specs):
        a=s["a"]
        if s["swap"]:
            pred[i,0]=m.conditional_q(p,1-a,a)
            continue
        mask=m.BITS[:,1]==a
        if s["reveal"]>=0: mask &= m.BITS[:,s["reveal"]]==s["bit"]
        pred[i]=(p[mask,None]*m.BITS[mask]).sum(0)/p[mask].sum()
    out=m.metrics(pred,p,specs)
    assert out["sequential_kl"]<1e-12
    assert out["response_squared_error"]<1e-25
    assert out["order_tv"]<1e-12
    assert out["parallel_kl"]>0
    for vals in (pred,np.clip(pred*.8+.05,1e-5,1-1e-5)):
        out=m.metrics(vals,p,specs)
        assert out["parallel_identity_residual"]<1e-12
        assert out["sequential_identity_residual"]<1e-12


def test_fixed_background_and_rank_control():
    cases=m.make_cases()
    assert len(cases)==112
    for case in cases:
        a,b,c=np.array(case["physical_sites"])
        assert len({tuple(x) for x in (a,b,c)})==3
        for rep in ("A","B","C"):
            tokens,coords,t,specs=m.case_inputs(case,rep)
            for a in (0,1):
                i=m.index_of(specs,"conditional",a,2,1-a)
                j=m.index_of(specs,"swap",a,2,1-a)
                np.testing.assert_array_equal(tokens[i],tokens[j])
                np.testing.assert_array_equal(coords[i,0,3:],coords[j,0,3:])
                np.testing.assert_array_equal(coords[i,0,0],coords[j,0,0])
                np.testing.assert_array_equal(coords[i,0,[1,2]],coords[j,0,[2,1]])
                assert t[i]==t[j]
        if case["primary"]:
            q,e,f=np.array(case["rank_sites"])
            assert abs(q-e).sum()==abs(q-f).sum()
            pq,pe,pf=np.array(case["physical_sites"])
            ratio=abs(pq-pe).sum()/abs(pq-pf).sum()
            assert ratio in (.1,10)
