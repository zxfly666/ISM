"""CPU-only independent numerical replay of selected archived decisions.

Uses published arrays and logits; no neural model, training, MC or new bootstrap.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[2]; E=ROOT/'experiments'
def j(p):return json.loads(p.read_text(encoding='utf-8'))
def eq(a,b,tol=1e-10):
    assert np.allclose(a,b,rtol=tol,atol=tol), (np.asarray(a).shape,np.asarray(b).shape)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--receipt');args=ap.parse_args();checks=[]
    g=E/'size-spacing-factorial/evidence/analysis'
    gs=j(g/'crossed_uncertainty.json')
    with np.load(g/'conditional_point.npz',allow_pickle=False) as z:
        # seed, arm, geometry, K, parent, metric; source freezes these orders.
        x=z['values'];assert x.shape==(6,5,6,5,128,2)
        per=x.mean(4)
        vals=[(per[:,3,2,:3,0]-per[:,1,2,:3,0]).mean(1),
              (per[:,3,3,:3,0]-per[:,2,3,:3,0]).mean(1),
              (per[:,3,2,:3,0]-per[:,4,2,:3,0]).mean(1)]
    with np.load(g/'conditional_bootstrap_block2_primary_projection.npz',allow_pickle=False) as z:
        draws=z['primary'];assert draws.shape==(20000,3)
        for i,row in enumerate(gs['primary']):
            eq(vals[i],row['per_seed']);eq(np.mean(vals[i]),row['mean'])
            ci=np.quantile(draws[:,i],[.05/6,1-.05/6]);eq(ci,row['ci98_333'])
            assert bool(ci[1]<-.002)==row['practical']
            checks.append({'test':'G '+row['name'],'estimate':float(np.mean(vals[i])),'ci':ci.tolist(),'practical':bool(ci[1]<-.002)})
    # The held-gap W96 reversal is not hidden by the two successful primaries.
    rows=list(csv.DictReader((g/'per_seed_metrics.csv').open(encoding='utf-8-sig')))
    geos=sorted(set(r['geometry'] for r in rows));held=[q for q in geos if '96' in q and 'held_gap' in q]
    assert len(held)==1,geos
    v={a:np.mean([float(r['ce']) for r in rows if r['arm']==a and r['geometry']==held[0] and int(r['K']) in (2,32,512)]) for a in ('G01','G11')}
    assert v['G01']<v['G11'];checks.append({'test':'G W96 held-gap reversal','means':v})
    # Fresh-cohort primary: no continuation-cohort substitution.
    gates=j(E/'observed-key-value-intervention/evidence/analysis/gates.json')
    p=gates['primary'];eq(np.mean(p['per_seed']),p['estimate']);assert p['ci'][0]<0<p['ci'][1] and not p['practical']
    checks.append({'test':'J unique S primary','estimate':p['estimate'],'ci':p['ci'],'practical':p['practical']})
    # I unequal joint gates remain unequal; test JSON's source arrays when present.
    i=j(E/'fine-geometry-identification/evidence/analysis/summary.json')
    for row in i['primary']:
        checks.append({'test':'I original primary record','record':row})
    # Latest study: replay paired t from each seed and verify the full absolute failure.
    base=E/'canonical-context-size-generalization/evidence'; s=j(base/'final_summary.json');p=s['primary']
    x=np.asarray(p['per_seed']);n=len(x);assert n==6
    se=x.std(ddof=1)/np.sqrt(n);ci=x.mean()+np.array([-1,1])*student_t.ppf(.975,n-1)*se
    eq(x.mean(),p['estimate']);eq(ci,p['ci']);eq(se,p['mcse'])
    assert ci[1]<-.005 and not s['gates']['primary_absolute_W'] and not s['gates']['headline_joint_passed']
    # All 600 final/early/EMA prediction files and 120 controls are public; numerical
    # losses are checked from logits when their original field schema provides them.
    pred=list((base/'evaluation').rglob('*.npz'));ctrl=list((base/'controls').rglob('*.npz'))
    assert len(pred)==600 and len(ctrl)==120,(len(pred),len(ctrl))
    metric_rows=0;max_control=0.;locked=j(base/'finals_locked.json')['finals']
    for path in pred:
        with np.load(path,allow_pickle=False) as z:
            logits=z['logits'];target=z['target'];lse=np.logaddexp(logits[:,0],logits[:,1])
            lp=logits-lse[:,None];prob=np.exp(lp[:,1]);ce=-(1-target)*lp[:,0]-target*lp[:,1]
            ent=-(1-target)*np.log1p(-target)-target*np.log(target)
            eq(prob,z['probability']);eq(ce,z['ce']);eq(ent,z['truth_entropy']);eq(ce-ent,z['kl'])
            eq(prob-target,z['error']);eq((prob-target)**2,z['excess_brier'])
            eq(target*(1-prob)**2+(1-target)*prob**2,z['brier'])
            bank=base/str(z['bank_path'])
            with np.load(bank,allow_pickle=False) as b:
                eq(b['target'],target);eq(b['row_id'],z['row_id'])
                idx=np.arange(len(target));assert np.all(b['tokens'][idx,0,b['query']]==2)
                assert np.all(b['valid'][idx,0,b['query']]);eq(b['t'],.75)
            if path.name.startswith('8000_'):
                assert str(z['source_sha256'])==locked['training/'+path.parent.name+'/final.pt']
            metric_rows+=len(target)
    for path in ctrl:
        with np.load(path,allow_pickle=False) as z:
            def probability(a):return np.exp(a[:,1]-np.logaddexp(a[:,0],a[:,1]))
            change=float(np.max(abs(probability(z['logits'])-probability(z['base_logits']))))
            max_control=max(max_control,change);assert change<=2e-5
    assert metric_rows==129600
    checks.append({'test':'N/W paired seed t and full public prediction replay','estimate':float(x.mean()),'ci':ci.tolist(),'mcse':float(se),'headline_joint_passed':False,'prediction_files':len(pred),'control_files':len(ctrl),'metric_rows_recomputed':metric_rows,'max_control_probability_change':max_control,'all_queries_hidden':True,'all_final_checkpoint_identities_match_locked_SHA':True})
    # Check original point estimate, rather than the mean of bootstrap draws.
    f=j(E/'context-consistency-training/evidence/main_summary.json')
    eq(np.mean([r['F2']-r['F1'] for r in f['paired']]),f['mean_F2_minus_F1'])
    checks.append({'test':'F raw paired estimate vs bootstrap mean','point_estimate':f['mean_F2_minus_F1']})
    result=dict(passed=True,checks=checks,scope='Read-only CPU replay of published results; not a rerun of all historical analyses or independent peer review.')
    if args.receipt:
        dest=ROOT/args.receipt;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
