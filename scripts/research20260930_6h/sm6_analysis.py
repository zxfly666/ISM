"""Auditable aggregates, five figures and a conservative Chinese result report."""
from __future__ import annotations

import json
import time
import numpy as np
import sm6_common as c
import sm6_data as d
import sm6_statistics as stats


def functional_diagnostics(root, rows):
    result = []
    for row in rows:
        if row['step'] != 8000 or row['weight'] != 'raw' or row['kind'] != 'final' or row['spec']['family'] != 'G8' or row['spec']['split'] != 'test':
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
    colors = {'N': '#767676', 'W': '#0F4D92'}
    markers = dict(zip('NW', ('o', 'D')))
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
    def collect(arm, side, field, split='test', step=8000, kind='final'):
        return np.array([next(r[field] for r in rows if r['seed'] == seed and r['arm'] == arm and
                   r['step'] == step and r['weight'] == 'raw' and r['kind'] == kind and
                   r['spec']['key'] == f'G8_{split}_N{side}_center') for seed in c.CFG['seed_labels']])
    fig, ax = plt.subplots(figsize=(7.2, 4.6)); arrays = {'sides': np.array(sides)}
    for arm in 'NW':
        v = np.stack([collect(arm, side, 'mean_kl') for side in sides])
        arrays[arm] = v
        mean, se = v.mean(1), v.std(1, ddof=1)/np.sqrt(6)
        ax.plot(sides, mean, marker=markers[arm], color=colors[arm], label=arm)
        ax.fill_between(sides, np.maximum(0, mean-se), mean+se, color=colors[arm], alpha=.15)
    ax.set(xlabel='Valid container side', ylabel='Exact conditional KL (nats/query)', xticks=sides)
    ax.axvline(20, ls=':', color='black', lw=1); ax.legend(ncol=2, loc='upper left', bbox_to_anchor=(0, 1.17))
    export(fig, '01_size_kl', arrays, 'G8 held-out pattern groups; final raw; common canonical clock0.75. Bands are ±1 MCSE across six paired training seeds, not uncertainty from enumerated rows. Side20 is primary.')
    fig, ax = plt.subplots(figsize=(7.2, 4.6)); arrays = {'sides': np.array(sides)}
    for arm in 'NW':
        v = np.stack([collect(arm, side, 'max_probability_error') for side in sides])
        arrays[arm] = v
        ax.errorbar(sides, v.mean(1), yerr=v.std(1, ddof=1)/np.sqrt(6), color=colors[arm], marker=markers[arm], label=arm, capsize=3)
        for i, side in enumerate(sides):
            ax.scatter(np.full(6, side), v[i], color=colors[arm], s=12, alpha=.35)
    ax.axhline(.05, ls='--', color='black', lw=1); ax.set(xlabel='Valid container side', ylabel='Worst probability error within a seed', xticks=sides)
    ax.legend(ncol=2, loc='upper left', bbox_to_anchor=(0, 1.17))
    export(fig, '02_absolute_ability', arrays, 'Each faint point is one seed maximum over the exact test rows. Error bars summarize seed means ±MCSE; passing requires each of the six maxima below0.05 and each max KL below0.01.')
    fig, ax = plt.subplots(figsize=(7.2, 4.3)); p = summary['primary']; v = np.array(p['per_seed'])
    ax.scatter(np.arange(1, 7), v, color=colors['W'], marker='D', label='Paired seed effect')
    ax.errorbar([7.5], [p['estimate']], yerr=[[p['estimate']-p['ci'][0]], [p['ci'][1]-p['estimate']]], color='black', fmt='o', capsize=5)
    ax.axhline(0, color='gray', ls=':'); ax.axhline(-.005, color='black', ls='--')
    ax.set(xticks=list(range(1, 7))+[7.5], xticklabels=[str(s) for s in c.CFG['seed_labels']]+['Mean'], ylabel='KL(W) - KL(N), side20')
    export(fig, '03_primary_paired', {'per_seed': v, 'ci95': np.array(p['ci'])}, 'Prespecified main contrast, G8 test side20 center; final raw. Black interval is the paired t95% interval across six training seeds. Dashed line is the−0.005 material margin.')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4)); arrays={}
    for arm in 'NW':
        vals = np.array([r['predicted_response'] for r in summary['functional'] if r['arm']==arm and r['cell']=='G8_test_N20_center'])
        arrays[arm] = vals
        i='NW'.index(arm)
        axes[0].errorbar(i, vals.mean(), yerr=vals.std(ddof=1)/np.sqrt(6), fmt=markers[arm], color=colors[arm], capsize=4)
        axes[0].scatter(np.full(6, i), vals, color=colors[arm], alpha=.4, s=13)
    axes[0].axhline(.628539361054709, color='black', ls='--'); axes[0].set(xticks=range(2), xticklabels=list('NW'), ylabel='Response: pattern85 minus pattern240')
    axes[1].bar(['240', '85'], [.1857303194726456, .8142696805273546], color=['#767676', '#0F4D92'], edgecolor='black')
    axes[1].set(ylabel='Exact probability', xlabel='Boundary pattern (both4 positive)', ylim=(0,1))
    export(fig, '04_noncounting_response', arrays, 'Equal-positive-count witness at the same query. Left: six model responses with ±MCSE. Right: exact oracle probabilities, not sampled estimates; no MC uncertainty. Arrangement matters despite equal counts.')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5)); arrays={}
    for arm in 'NW':
        ss=sorted(set(c.CFG['arms'][arm]['valid_side_cycle']))
        values=[]
        for step in (4000,6000,8000):
            v=np.mean(np.stack([collect(arm,side,'mean_kl','validation',step,'final' if step==8000 else 'learning') for side in ss]),0)
            values.append(v)
        values=np.array(values); arrays[arm]=values
        axes[0].errorbar([4,6,8],values.mean(1),yerr=values.std(1,ddof=1)/np.sqrt(6),marker=markers[arm],color=colors[arm],label=arm,capsize=3)
    axes[0].set(xlabel='Fixed update count (thousands)',ylabel='G8 validation KL at trained sizes',xticks=[4,6,8])
    axes[0].legend(ncol=2,loc='upper left',bbox_to_anchor=(0,1.18))
    labels=list(summary['retention']); vals=[summary['retention'][k] for k in labels]
    mean=np.array([v['estimate'] for v in vals]); upper=np.array([v['upper'] for v in vals])
    arrays['retention_estimate']=mean; arrays['retention_upper']=upper; arrays['retention_labels']=np.array(labels)
    axes[1].errorbar(np.arange(3),mean,yerr=np.stack([np.zeros(3),upper-mean]),fmt='o',color=colors['W'],capsize=4)
    axes[1].axhline(.002,color='black',ls='--'); axes[1].axhline(0,color='gray',ls=':')
    axes[1].set(xticks=np.arange(3),xticklabels=['K1 holdout','K2 holdout','K4 side4'],ylabel='Retention KL(W) - KL(N)')
    export(fig, '05_learning_retention', arrays, 'Early raw snapshots evaluated only after all twelve finals locked. Left: G8 validation at arm-trained sides, seed mean +/- MCSE. Right: three retention paired means with one-sided98.3333% upper bounds; dashed line +0.002. Absolute W gates are additionally required.')
    c.write(root/'analysis/figure_captions.json', captions)
    return captions


def report(root, summary, timing, fixture=False):
    p=summary['primary'];g=summary['gates']
    lines=['# 固定clock Dense多尺寸训练：6小时实验报告','',
        '**仅软件人工fixture，不是网络结果。**' if fixture else '固定8k final raw主分析；EMA与4k/6k诊断均为次要，没有按结果选checkpoint。',
        '',f"主终点：20×20中心、168条G8未见模式。W−N平均精确KL={p['estimate']:.9g}，95%配对t区间[{p['ci'][0]:.9g},{p['ci'][1]:.9g}]，MCSE={p['mcse']:.9g}。",
        '', '| 预定判定 | 结果 |','|---|---|']
    lines += [f'| {k} | {"通过" if v else "未通过"} |' for k,v in g.items()]
    if g['headline_joint_passed']:
        meaning='通过本轮有限局部任务的主实质、绝对、保留与结构门；不能推广为老师整体idea或完整生成分布已验证。'
    elif not g['W_trained_patterns']:
        meaning='W对训练模式的绝对能力仍未全部建立；不能把新容器错误唯一归因为外推，也不能否定老师完整idea。'
    elif not g['primary_absolute_W']:
        meaning='主测试绝对能力未全部通过。即使相对KL下降，也不能称问题已解决。'
    elif not g['retention']:
        meaning='保留门未全部通过，不能称无代价改进。'
    else:
        meaning='没有同时满足全部预设门。没有达到实质差异不等于等效；两组均准确也不代表此轮无价值。'
    lines += ['',meaning,'','## 科学范围和参数','',
        '六对新seed93061–93066，root2026093061；N有效边长4/6/4/6，W为4/6/8/12；分配边长两组均4/6/8/12，t固定0.75。物理样本/目标/初始化配对；N多余槽位invalid PAD不参加attention，真实MASK参加attention。有效token与FLOPs并不相同。',
        '原dense结构1,976,706参数，宽128/4头/7块/MLP4/RoPE10000，FP32，dropout0，无AMP/TF32。每模型8000更新，batch128，K1/K2/K4/G8各32；两个64样本微批合成一步。AdamW(.9,.95)，wd.05，clip1，EMA.999，512步warmup至3e−4、余弦至3e−5；总96000更新。',
        'K1/K2维持原4×4物理系统；K4四邻居全可见；G8隐藏2×2腔体由8点边界屏蔽。G8真值对16隐藏状态精确求和并用65536状态完整4×4枚举交叉核对。D4与全局翻转72组以冻结SHA划分696训练/160验证/168测试，不按预测改变。',
        '老师建议在dense结构中增加训练block size及真实距离覆盖。本轮只覆盖“多容器尺寸训练”在固定clock下的作用；没有native-clock对照、真实物理stride变化、长程边缘条件、MC或生成。t=.75是控制条件，不是本轮证明的最优策略。',
        '## 统计、诊断与限制','',
        '独立统计单位为六个配对训练seed，不是模式行、平移或容器数。固定测试模式等权平均不代表Ising自然系综加权风险。精确局部Bernoulli KL与CE/真值熵分列，期望Brier与超额Brier分列；不将此精确KL解释套给旧MC条件CE。',
        f"主实质上界<−.005；方向上界<0。20k配对seed bootstrap区间{p['seed_bootstrap_ci95']}；删一seed/符号翻转完整保留。方向稳定={p['sensitivity_direction_stable']}，精度MCSE≤.002={p['precision_target_met']}。t小样本近似、bootstrap不增加独立重复；符号翻转需要对称/可交换性，不能无条件当作精确总体检验。",
        'W主绝对门要求六个seed各自maxKL≤.01且max概率误差≤.05。三个保留任务还要求W绝对门与W−N单侧98.333333%上界≤.002。',
        '', '| 保留任务 | W−N均值 | 单侧上界 | 绝对W全通过 |','|---|---:|---:|---|']
    for k,v in summary['retention'].items():
        lines.append(f"| {k} | {v['estimate']:.9g} | {v['upper']:.9g} | {v['all_W_absolute_passed']} |")
    flagged=[f"{v['seed']}_{v['arm']}" for v in summary['learning_flags'] if v['learning_incomplete']]
    lines += ['', '6k→8k训练尺寸G8验证仍改善>.002的分支：'+(', '.join(flagged) if flagged else '无')+'。该标记只限制负面解释，不触发加步；无标记也不是收敛证明。',
        '同query、同4正4负的patterns240/85精确概率为.1857303195/.8142696805。保存模型响应与全测试翻转互补误差，检查是否学会非计数排列信息。不能用该单对替代全部168条误差。',
        '', '## 计算与交付','', *[f'- {k}: {v:.3f}秒（实测）。' for k,v in timing.items()], '',
        '完整科学范围为12可恢复final、24份raw+EMA快照、96000步日志/配对身份、600正式预测、120控制对与全部原始logits/输入/指标、五图PNG/PDF/plot_data、代码协议和manifest。失败不删除、不换seed、不延长；原12h轮不重启。',
        '本报告生成仅表示远端分析完成。全部本地路径/字节/SHA联合覆盖、实际查看最终五图以及真实备份范围由delivery_receipt单独确认；同D盘目录不是异盘灾备。GitHub上传权限待单独解决，本轮不自动推送。',
        '所有逐格结果及不理想结果在final_summary.json、evaluation、controls、analysis；不自动启动下一轮。','']
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
