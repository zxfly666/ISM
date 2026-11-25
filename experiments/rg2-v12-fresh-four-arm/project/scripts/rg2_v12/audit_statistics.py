"""Independently recompute all saved B bootstrap draws and distribution distances."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import wasserstein_distance


def load(path):
    with np.load(path,allow_pickle=False) as data:
        return {key:data[key] for key in data.files}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    root=args.evaluation;start=time.monotonic();stats=json.loads((root/'development_statistics.json').read_text())
    draws=load(root/'development_bootstrap_draws.npz');parents=draws['reference_parent_ids']
    assert parents.shape==(256,) and np.all(np.diff(parents)>0)
    mcindex=np.searchsorted(parents,draws['mc_parent_draw'])
    np.testing.assert_array_equal(parents[mcindex],draws['mc_parent_draw'])
    assert draws['seed_draw'].shape==(10000,2) and draws['sample_draw'].shape==(10000,2,32)
    seeds=(112501,112502);arms=('F','P','D','M');worst=0.;distances=0
    for steps in (64,256):
        for path,width,lo in [('fine',96,25),('B',32,9)]:
            reference=load(root/f'MC_reference_W{width}.npz')
            np.testing.assert_array_equal(reference['parent_ids'],parents)
            curves=[]
            for seed in seeds:
                group=[]
                for arm in arms:
                    z=load(root/'metrics'/f's{seed}_{arm}_final'/f'{path}_K{steps}.npz');group.append(z['g'][:,lo:])
                    row=next(v for v in stats['per_seed'] if (v['seed'],v['arm'],v['steps'],v['path'])==(seed,arm,steps,path))
                    ref=reference['g'].mean(0)[lo:];error=np.sqrt(np.sum((z['g'].mean(0)[lo:]-ref)**2)/np.sum(ref**2))
                    assert abs(error-row['G_error'])<1e-12
                    for key,a,b in [('W1_signed_m',z['m'],reference['m']),('W1_abs_m',abs(z['m']),abs(reference['m'])),('W1_energy',z['e'],reference['e'])]:
                        assert abs(wasserstein_distance(a,b)-row[key])<1e-12;distances+=1
                curves.append(group)
            curves=np.asarray(curves)
            recomputed=np.empty((10000,4))
            for replicate in range(10000):
                ref=reference['g'][mcindex[replicate],lo:].mean(0)
                denominator=np.sqrt(np.sum(ref**2));values=[]
                for slot,seed_index in enumerate(draws['seed_draw'][replicate]):
                    ids=draws['sample_draw'][replicate,slot]
                    values.append([np.sqrt(np.sum((curves[seed_index,arm][ids].mean(0)-ref)**2))/denominator for arm in range(4)])
                recomputed[replicate]=np.mean(values,axis=0)
            for control in ('D','F','P'):
                delta=recomputed[:,3]-recomputed[:,arms.index(control)]
                saved=draws[f'{path}_K{steps}_M_minus_{control}']
                difference=float(abs(delta-saved).max());worst=max(worst,difference);assert difference<1e-12
                row=next(v for v in stats['comparisons'] if (v['path'],v['steps'],v['comparison'])==(path,steps,'M-'+control))
                np.testing.assert_allclose(np.quantile(delta,[.025,.975]),row['descriptive_joint_bootstrap95'],rtol=0,atol=1e-12)
    result=dict(status='passed',all_120000_saved_contrast_draws_recomputed=True,distribution_distances_checked=distances,
                actual_reference_parents=256,maximum_bootstrap_difference=worst,seconds=time.monotonic()-start,
                implementation='Independent scalar replicate/seed-slot loops; SciPy Wasserstein distances',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),confirmatory=False)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
