"""Auditable aggregates, six figures and a conservative Chinese result report."""
from __future__ import annotations

import json
import time
import numpy as np
import sg_common as c
import sg_data as d
import sg_statistics as stats


def functional_diagnostics(root, rows):
    result = []
    for row in rows:
        if row['step'] != 24000 or row['weight'] != 'raw' or row['kind'] != 'final' or row['spec']['family'] != 'G8' or row['spec']['split'] != 'test':
            continue
        z = c.load(root / row['prediction'])
        lookup = {int(v): i for i, v in enumerate(z['row_id'])}
        p = z['probability']
        flip = [abs(p[i]+p[lookup[256*(rid//256)+255-rid%256]]-1) for rid, i in lookup.items()]
        response = float(p[lookup[85]]-p[lookup[240]])
        result.append(dict(seed=row['seed'], arm=row['arm'], cell=row['spec']['key'],
                           predicted_response=response, exact_response=.628539361054709,
                           response_error=abs(response-.628539361054709), max_spin_flip_error=float(max(flip))))
    return result


def figures(root, summary, timing):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from scipy.stats import t as student_t
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': 1.2, 'legend.frameon': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    colors = {'A': '#767676', 'B': '#B64342', 'C': '#42949E', 'D': '#0F4D92'}
    markers = dict(zip('ABCD', ('o', 's', '^', 'D')))
    sides = c.CFG['evaluation']['center_sides']
    rows = summary['rows']
    folder = root / 'analysis/figures'; folder.mkdir(parents=True, exist_ok=True)
    data = root / 'analysis/plot_data'; data.mkdir(parents=True, exist_ok=True)
    captions = []
    def export(fig, name, arrays, caption):
        fig.tight_layout(pad=1.3)
        fig.savefig(folder / (name+'.png'), dpi=300)
        fig.savefig(folder / (name+'.pdf'))
        plt.close(fig)
        c.save(data / (name+'.npz'), **arrays)
        captions.append(dict(name=name, caption=caption))
    def collect(arm, side, field, split='test', step=24000, kind='final'):
        return np.array([next(r[field] for r in rows if r['seed'] == seed and r['arm'] == arm and
                   r['step'] == step and r['weight'] == 'raw' and r['kind'] == kind and
                   r['spec']['key'] == f'G8_{split}_N{side}_center') for seed in c.CFG['seed_labels']])
    fig, ax = plt.subplots(figsize=(7.2, 4.6)); arrays = {'sides': np.array(sides)}
    for arm in 'ABCD':
        v = np.stack([collect(arm, side, 'mean_kl') for side in sides])
        arrays[arm] = v
        mean, se = v.mean(1), v.std(1, ddof=1)/np.sqrt(6)
        ax.plot(sides, mean, marker=markers[arm], color=colors[arm], label=arm)
        ax.fill_between(sides, np.maximum(0, mean-se), mean+se, color=colors[arm], alpha=.15)
    ax.set(xlabel='Valid container side', ylabel='Exact conditional KL (nats/query)', xticks=sides)
    ax.axvline(20, ls=':', color='black', lw=1); ax.legend(ncol=4, loc='upper left', bbox_to_anchor=(0, 1.17))
    export(fig, '01_size_kl', arrays, 'G8 held-out pattern groups; final raw; per-arm deployment clock. Bands are ±1 MCSE across six paired training seeds, not uncertainty from enumerated rows. Side20 is primary.')
    fig, ax = plt.subplots(figsize=(7.2, 4.6)); arrays = {'sides': np.array(sides)}
    for arm in 'ABCD':
        v = np.stack([collect(arm, side, 'max_probability_error') for side in sides])
        arrays[arm] = v
        ax.errorbar(sides, v.mean(1), yerr=v.std(1, ddof=1)/np.sqrt(6), color=colors[arm], marker=markers[arm], label=arm, capsize=3)
        for i, side in enumerate(sides):
            ax.scatter(np.full(6, side), v[i], color=colors[arm], s=12, alpha=.35)
    ax.axhline(.05, ls='--', color='black', lw=1); ax.set(xlabel='Valid container side', ylabel='Worst probability error within a seed', xticks=sides)
    ax.legend(ncol=4, loc='upper left', bbox_to_anchor=(0, 1.17))
    export(fig, '02_absolute_ability', arrays, 'Each faint point is one seed maximum over the exact test rows. Error bars summarize seed means ±MCSE; passing requires each of the six maxima below0.05 and each max KL below0.01.')
    fig, ax = plt.subplots(figsize=(7.2, 4.3)); p = summary['primary']; v = np.array(p['per_seed'])
    ax.scatter(np.arange(1, 7), v, color=colors['D'], marker='D', label='Paired seed effect')
    ax.errorbar([7.5], [p['estimate']], yerr=[[p['estimate']-p['ci'][0]], [p['ci'][1]-p['estimate']]], color='black', fmt='o', capsize=5)
    ax.axhline(0, color='gray', ls=':'); ax.axhline(-.005, color='black', ls='--')
    ax.set(xticks=list(range(1, 7))+[7.5], xticklabels=[str(s) for s in c.CFG['seed_labels']]+['Mean'], ylabel='KL(D) - KL(A), side20')
    export(fig, '03_primary_paired', {'per_seed': v, 'ci95': np.array(p['ci'])}, 'Prespecified main contrast, G8 test side20 center; final raw. Black interval is the paired t95% interval across six training seeds. Dashed line is the−0.005 material margin.')
    fig, ax = plt.subplots(figsize=(7.5, 4.5)); names = list(summary['mechanisms']); means=[]; ci=[]
    for name in names:
        x = summary['mechanisms'][name]; means.append(x['estimate']); ci.append(x['ci'])
    means=np.array(means); ci=np.array(ci)
    ax.errorbar(means, np.arange(5), xerr=np.stack([means-ci[:, 0], ci[:, 1]-means]), fmt='o', color=colors['D'], capsize=4)
    ax.axvline(0, color='gray', ls=':'); ax.set(yticks=np.arange(5), yticklabels=names, xlabel='Paired conditional KL contrast')
    export(fig, '04_factorial_mechanisms', {'means': means, 'ci99': ci, 'labels': np.array(names)}, 'Five secondary factor contrasts at the same primary test; each interval is two-sided99% paired t (Bonferroni family). These do not replace the primary test.')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4)); arrays={}
    for arm in 'ABCD':
        vals = np.array([r['predicted_response'] for r in summary['functional'] if r['arm']==arm and r['cell']=='G8_test_N20_center'])
        arrays[arm] = vals
        i='ABCD'.index(arm)
        axes[0].errorbar(i, vals.mean(), yerr=vals.std(ddof=1)/np.sqrt(6), fmt=markers[arm], color=colors[arm], capsize=4)
        axes[0].scatter(np.full(6, i), vals, color=colors[arm], alpha=.4, s=13)
    axes[0].axhline(.628539361054709, color='black', ls='--'); axes[0].set(xticks=range(4), xticklabels=list('ABCD'), ylabel='Response: pattern85 minus pattern240')
    axes[1].bar(['240', '85'], [.1857303194726456, .8142696805273546], color=['#767676', '#0F4D92'], edgecolor='black')
    axes[1].set(ylabel='Exact probability', xlabel='Boundary pattern (both4 positive)', ylim=(0,1))
    export(fig, '05_noncounting_response', arrays, 'Equal-positive-count witness at the same query. Left: six model responses with ±MCSE. Right: exact oracle probabilities, not sampled estimates; no MC uncertainty. Arrangement matters despite equal counts.')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5)); arrays={}
    for arm in 'ABCD':
        ss=sorted(set(c.CFG['arms'][arm]['valid_side_cycle']))
        values=[]
        for step in (8000,16000,24000):
            v=np.mean(np.stack([collect(arm,side,'mean_kl','validation',step,'final' if step==24000 else 'learning') for side in ss]),0)
            values.append(v)
        values=np.array(values); arrays[arm]=values
        axes[0].errorbar([8,16,24],values.mean(1),yerr=values.std(1,ddof=1)/np.sqrt(6),marker=markers[arm],color=colors[arm],label=arm,capsize=3)
    axes[0].set(xlabel='Fixed update count (thousands)',ylabel='G8 validation KL at trained sizes',xticks=[8,16,24])
    axes[0].legend(ncol=4,loc='upper left',bbox_to_anchor=(0,1.18))
    labels=list(timing); secs=np.array(list(timing.values()),float); arrays['timing_seconds']=secs; arrays['timing_labels']=np.array(labels)
    axes[1].barh(labels,secs/3600,color='#767676',edgecolor='black'); axes[1].set(xlabel='Measured hours (not performance CI)')
    axes[1].xaxis.set_major_locator(MaxNLocator(3))
    axes[1].ticklabel_format(axis='x',style='sci',scilimits=(-2,3))
    export(fig, '06_learning_timing', arrays, 'Early raw snapshots evaluated only after all finals locked. Left bands/bars are seed MCSE, not checkpoint-selection uncertainty. Right is the single-run measured timing; local backup completion is reported separately by its receipt.')
    c.write(root/'analysis/figure_captions.json', captions)
    return captions


def report(root, summary, timing, fixture=False):
    p=summary['primary']; g=summary['gates']
    lines=['# Dense尺寸×时间输入与局部几何：实验结果', '',
           '**软件合成fixture，非神经网络实验结果。**' if fixture else '本报告使用冻结24k final raw；EMA和早期快照均为次要，未按结果选优。', '',
           '## 冻结主结论', '',
           f"20×20中心G8测试：D−A平均条件KL={p['estimate']:.9g}，95%配对t区间[{p['ci'][0]:.9g},{p['ci'][1]:.9g}]，MCSE={p['mcse']:.9g}。", '',
           '| 判定 | 结果 |', '|---|---|']
    lines += [f'| {k} | {"通过" if v else "未通过"} |' for k,v in g.items()]
    if g['headline_joint_passed']:
        meaning='组合配方通过本轮所测局部能力与保留门，并达到事前实质改善门；仍不是任意物理距离、长程条件或完整生成分布的验证。'
    elif not g['D_trained_patterns']:
        meaning='候选组仍未在所有训练模式/尺寸上建立必要能力；不能把新尺寸失败唯一归因于外推，也不能否定老师完整idea。'
    elif not g['primary_absolute_D']:
        meaning='已训练模式的学习与新尺寸绝对能力需要分开：20×20仍有seed未过，不能将相对KL下降写成问题解决。'
    elif not g['retention']:
        meaning='新测试的局部能力不等于无代价改进：保留门未全部通过。'
    else:
        meaning='主绝对能力与相对实质改善不同：本轮没有同时满足所有预先指定门槛；不能补挑seed、clock或checkpoint。'
    lines += ['', meaning, '', '## 方法与老师idea的对应', '',
        '六个fresh配对seed×四臂。A/B只训练4/6容器，C/D训练4/6/8/12；A/C使用1−K/N_valid，B/D始终0.75。dense结构不变：1,976,706参数、宽128、4头、7块、MLP4、RoPE10000、dropout0、FP32。每模型24000更新、batch128，K1/K2/K4/G8各32。AdamW(0.9,0.95)、weight decay0.05、clip1、EMA0.999；512步warmup至3e−4，余弦至3e−5。', '',
        '老师idea强调先在dense中进行多block size及训练阶段的真实距离覆盖，而不是只在推理缩放RoPE。本轮仅检验前者和位置敏感局部条件能力；时间输入是现有诊断引出的配方对照。没有真实物理stride训练、长程边缘条件、MC或生成，不能把有限局部通过推广成整个idea已证实。', '',
        'G8是2×2隐藏腔体、8点完整可见边界。真值来自16状态求和，并与完整4×4的65536状态枚举核对。1024条模式按D4与全局翻转72组分为696训练/160验证/168测试，全部容器/平移共用拆分。K1/K2维持原4×4有限物理系统；旧留出属于保留检查。', '',
        '模式240和85在相同query均含4正4负，却有精确概率0.1857303195与0.8142696805；只会计数者最坏误差至少0.3142696805。响应与翻转完整记录在functional_diagnostics.json。', '',
        '本轮KL是精确局部Bernoulli KL，CE还包含真值熵；不把这一解释套给旧MC条件CE。独立统计单位是六个训练seed，不是168模式、方向或平移。六seed t区间依赖小样本近似，bootstrap/LOO只作敏感性；非显著不是等效。', '',
        '## 机制、保留与学习限制', '', '| 比较 | 均值 | 区间 |', '|---|---:|---|']
    for k,v in summary['mechanisms'].items():
        lines.append(f"| {k} (99%) | {v['estimate']:.9g} | [{v['ci'][0]:.9g}, {v['ci'][1]:.9g}] |")
    for k,v in summary['retention'].items():
        lines.append(f"| 保留 {k} (单侧98.3333%上界) | {v['estimate']:.9g} | 上界 {v['upper']:.9g} |")
    flagged=[f"{v['seed']}_{v['arm']}" for v in summary['learning_flags'] if v['learning_incomplete']]
    lines += ['', '16k→24k仍明显改善的learning_incomplete分支：'+(', '.join(flagged) if flagged else '无')+'。该标记仅限制负面解释，不触发续训。', '',
        '## 运行和交付范围', '', *[f'- {k}: {v:.3f}秒（实测）。' for k,v in timing.items()], '',
        '远端完成不等于完整交付。24个恢复final、48个raw+EMA快照、全训练日志/输入配对证据、全部预测/统计、六图及其原始数组、源码和科学manifest必须联合备份核验；本地完成时刻、空间和真实范围由后续delivery_receipt记录。当前报告不预称本地SHA核验或实际看图完成。', '',
        '所有逐格结果含不理想的结果保存在final_summary.json和evaluation；图注位于analysis/figure_captions.json。软件失败/预算失败会另写failure.json，不能将不完整六seed研究称为主门通过。结束后不自动追加新轮。', '']
    (root/'RESULTS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')


def analyze(root, rows, controls, timing, fixture=False):
    result=stats.summarize(rows,controls)
    result['functional']=functional_diagnostics(root,rows)
    result.update(time=time.time(),fixture=fixture,timing=timing)
    c.write(root/'final_summary.json',result)
    c.write(root/'analysis/functional_diagnostics.json',result['functional'])
    figures(root,result,timing)
    report(root,result,timing,fixture)
    return result
