"""Derive the integrated report's tables from frozen arrays; no GPU computation."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / 'project/runs/B_eval'
OUT = ROOT / 'integrated/data'


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def dump_csv(path, rows):
    with path.open('w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def table(headers, rows):
    def cell(x): return str(x).replace('|', r'\|')
    return '\n'.join(['| ' + ' | '.join(cell(x) for x in headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
                     + ['| ' + ' | '.join(cell(x) for x in row) + ' |' for row in rows])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads((EVAL / 'development_statistics.json').read_text())
    source_files = [EVAL / 'development_statistics.json']
    refs = {p: load(EVAL / f'MC_reference_W{w}.npz') for p, w in [('fine', 96), ('coarse', 32)]}
    source_files.extend(EVAL / f'MC_reference_W{w}.npz' for w in (96, 32))
    rows, mcrows, differences = [], [], []
    max_delta = 0.
    for path, w, lower in [('fine', 96, 25), ('coarse', 32, 9)]:
        ref = refs[path]; refg = ref['g'].mean(0)
        mcrows.append(dict(path=path, n=len(ref['m']), G1=float(refg[1]), mean_m=float(ref['m'].mean()),
            mean_abs_m=float(abs(ref['m']).mean()), mean_e=float(ref['e'].mean()), p_wall=float((1-refg[1])/2),
            positive_fraction=float(np.mean(ref['m'] > 0)), G_far_first=float(refg[lower]), G_far_last=float(refg[-1])))
        for seed in (112501, 112502):
            for steps in (64, 256):
                for arm in ('F', 'P', 'D', 'M'):
                    file = EVAL / 'metrics' / f's{seed}_{arm}_final' / f'{"fine" if path == "fine" else "B"}_K{steps}.npz'
                    data = load(file); source_files.append(file)
                    g = data['g'].mean(0)
                    error = float(np.sqrt(np.sum((g[lower:]-refg[lower:])**2) / np.sum(refg[lower:]**2)))
                    row = dict(seed=seed, S=steps, arm=arm, path=path, W=w, n_gen=len(data['m']),
                        E_G=error, G1=float(g[1]), mean_m=float(data['m'].mean()), mean_abs_m=float(abs(data['m']).mean()),
                        mean_e=float(data['e'].mean()), p_wall=float((1-g[1])/2),
                        positive_fraction=float(np.mean(data['m'] > 0)),
                        W1_m=float(wasserstein_distance(data['m'], ref['m'])),
                        W1_abs_m=float(wasserstein_distance(abs(data['m']), abs(ref['m']))),
                        W1_energy=float(wasserstein_distance(data['e'], ref['e'])),
                        long_residual_min=float((g-refg)[lower:].min()), long_residual_max=float((g-refg)[lower:].max()))
                    assert abs(row['G1'] + row['mean_e']) < 2e-15
                    assert all(np.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
                    previous = next(x for x in old['per_seed'] if x['seed']==seed and x['arm']==arm and x['steps']==steps
                                    and x['path']==('fine' if path=='fine' else 'B'))
                    for new, prior in [('E_G','G_error'), ('W1_m','W1_signed_m'), ('W1_abs_m','W1_abs_m'), ('W1_energy','W1_energy')]:
                        delta=abs(row[new]-previous[prior]); max_delta=max(max_delta,delta); assert delta<2e-14
                    rows.append(row)
    for path in ('fine', 'coarse'):
        for seed in (112501, 112502):
            for steps in (64, 256):
                values={r['arm']:r for r in rows if r['path']==path and r['seed']==seed and r['S']==steps}
                for left,right in [('P','F'),('D','P'),('M','D'),('M','F'),('M','P')]:
                    change=values[left]['E_G']-values[right]['E_G']
                    differences.append(dict(path=path,seed=seed,S=steps,comparison=left+'-'+right,
                        difference=change,percentage_point_change=100*change,
                        relative_change_percent=100*change/values[right]['E_G']))
    dump_csv(OUT/'report_per_seed.csv',rows);dump_csv(OUT/'report_mc_reference.csv',mcrows)
    dump_csv(OUT/'report_paired_differences.csv',differences)
    tables={}
    for path in ('fine','coarse'):
        body=[]
        for seed in (112501,112502):
            for steps in (64,256):
                values={r['arm']:r for r in rows if r['seed']==seed and r['S']==steps and r['path']==path}
                body.append([seed,steps]+[f"{values[a]['E_G']:.5f}" for a in ('F','P','D','M')])
        tables[path+'_errors']=table(['训练 seed','S','F','P','D','M'],body)
        body=[]
        for r in (r for r in rows if r['path']==path):
            body.append([r['seed'],r['S'],r['arm']]+[f"{r[k]:.5f}" for k in ['G1','mean_m','mean_abs_m','mean_e','p_wall','W1_m','W1_abs_m','W1_energy']])
        tables[path+'_full']=table(['seed','S','臂','G(1)','平均 m','平均 |m|','平均 e','异号键比例','W1(m)','W1(|m|)','W1(e)'],body)
    body=[]
    for seed in (112501,112502):
        for steps in (64,256):
            values={r['comparison']:r for r in differences if r['seed']==seed and r['S']==steps and r['path']=='fine'}
            body.append([seed,steps]+[f"{values[c]['difference']:+.5f}" for c in ['P-F','D-P','M-D','M-F','M-P']])
    tables['fine_pairs']=table(['seed','S','P−F','D−P','M−D','M−F','M−P'],body)
    tables['bootstrap']=table(['尺度','S','比较','seed 112501','seed 112502','等权均值','描述性95%区间'],
       [['细 W96' if x['path']=='fine' else '粗 W32',x['steps'],x['comparison'],*[f'{v:+.5f}' for v in x['paired_seed_values']],
         f"{x['estimate']:+.5f}",'['+', '.join(f'{v:+.5f}' for v in x['descriptive_joint_bootstrap95'])+']'] for x in old['comparisons']])
    local=[]
    for seed in (112501,112502):
        for arm in ('F','P','D','M'):
            result=[]
            for w in (48,96):
                values=[]
                for nmask in (1,8,32):
                    p=EVAL/'A0_exact'/f's{seed}_{arm}_final'/f'fine_W{w}_M{nmask}.npz';z=load(p);source_files.append(p)
                    values.append(float(z['KL'].mean()))
                result.append(float(np.mean(values)))
            local.append([seed,arm,*[f'{x:.6f}' for x in result]])
    tables['local']=table(['seed','臂','W48 局部 KL','W96 局部 KL'],local)
    (OUT/'report_tables.json').write_text(json.dumps(tables,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    audit=dict(status='passed',source_commit='5516370b3ae6e98160d9468668acdcbf851dc902',
       independent_relative_L2_and_Scipy_Wasserstein=True,metric_cells=len(rows),Wasserstein_checks=3*len(rows),
       energy_identity_checks=len(rows),max_difference_from_frozen_statistics=max_delta,
       formulas='E_G from each ensemble mean; W1 via scipy; e=-G(1); p_wall=(1-G(1))/2',
       sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(source_files))})
    (ROOT/'integrated/provenance/report_values_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(status=audit['status'],cells=len(rows),max_delta=max_delta,MC=mcrows),ensure_ascii=False))
    print(tables['fine_pairs'])
    print('long_range_residuals',json.dumps([{k:r[k] for k in ['seed','S','arm','long_residual_min','long_residual_max']} for r in rows if r['path']=='fine']))


if __name__=='__main__':main()
