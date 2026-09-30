"""Ten prespecified J figures with saved plot data and vector/raster exports."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
import intervention_common as c
import intervention_statistics as st

COLORS = ["#767676", "#D55E00", "#0072B2", "#CC79A7", "#009E73"]
MARKERS = ["x", "s", "o", "D", "^"]
GROUPS = ["B0", "C-D", "C-O", "S-D", "S-O"]


def make_figures(root, core, mech, pad, oracle, low, learning, scratch):
    out = Path(root)/"analysis"
    figures = out/"figures"; figures.mkdir()
    data = out/"plot_data"; data.mkdir()
    captions = {}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.2,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})

    def axes(n=2, h=3.6):
        fig, ax = plt.subplots(1, n, figsize=(5.5*n, h), squeeze=False)
        for j, a in enumerate(ax[0]):
            a.text(.01, 1.03, f"({chr(97+j)})", transform=a.transAxes, fontweight="bold")
        return fig, ax[0]

    def finish(fig, name, caption, **arrays):
        if scratch:
            fig.text(.5, .99, "SOFTWARE SCRATCH — NOT EXPERIMENTAL RESULTS", ha="center", va="top", color="#B64342", fontsize=10)
        legend = {}
        for axis in fig.axes:
            handles, labels = axis.get_legend_handles_labels()
            legend.update(zip(labels, handles))
            if axis.get_legend() is not None:
                axis.get_legend().remove()
        if legend:
            fig.legend(list(legend.values()), list(legend), loc='lower center',
                       bbox_to_anchor=(.5, .015), ncol=min(5, len(legend)), fontsize=9)
        fig.tight_layout(pad=2, rect=(0, .16 if legend else .01, 1, .95 if scratch else .99))
        fig.savefig(figures/(name+".png"), dpi=300)
        fig.savefig(figures/(name+".pdf"))
        plt.close(fig)
        c.save(data/(name+".npz"), **arrays)
        captions[name] = caption+(" Software-only synthetic/small checks; not science." if scratch else "")

    def errorline(a, x, point, draw, label, group, level=.95):
        ci = st.interval(draw, level)
        # Percentile intervals can exclude a point estimate; use explicit segments.
        a.vlines(x, ci[0], ci[1], color=COLORS[group], linewidth=1.4)
        a.plot(x, point, marker=MARKERS[group], linestyle="-", color=COLORS[group], label=label, markersize=5)
        return ci

    x = core["H96_k512"]; point, draw = x["point"], x["draws"]
    fig, ax = axes()
    for g in range(5):
        errorline(ax[0], [g], [point[:, g, 0].mean()], draw[:, g:g+1, 0], GROUPS[g], g)
    ax[0].set(xticks=range(5), xticklabels=GROUPS, ylabel="H96 K512 CE (nats)")
    pairs = [(4, 3, "S O−D"), (2, 1, "C O−D"), (2, 0, "C O−B0")]
    deltas = np.array([point[:, a, 0].mean()-point[:, b, 0].mean() for a, b, _ in pairs])
    ds = np.stack([draw[:, a, 0]-draw[:, b, 0] for a, b, _ in pairs], -1)
    errorline(ax[1], np.arange(3), deltas, ds, "paired contrast", 2)
    ax[1].axhline(0, color="black", linestyle="--", linewidth=1)
    ax[1].axhline(-.005, color="#767676", linestyle=":", linewidth=1)
    ax[1].set(xticks=range(3), xticklabels=[p[2] for p in pairs], ylabel="Paired ΔCE (nats)")
    finish(fig, "01_primary_and_progress", "H96 K512: absolute final EMA CE and separately labelled S primary/C secondary contrasts. 95% joint seed×chain×block8 intervals; negative contrast favors O. B0 and S absolute levels are descriptive, not paired architecture effects.",
           per_seed=point, draws=draw, contrast=deltas, contrast_draws=ds)

    fig, ax = axes()
    banks = ["H48_k512", "C48_k115", "C48_k1152"]
    retained = []; retained_draws = []
    for panel, (group, ref, level) in enumerate(((4, 3, 1-.05/3), (2, 1, 1-.05/6))):
        points = np.array([(core[b]["point"][:, group, 0]-core[b]["point"][:, ref, 0]).mean() for b in banks])
        boot = np.stack([core[b]["draws"][:, group, 0]-core[b]["draws"][:, ref, 0] for b in banks], -1)
        errorline(ax[panel], np.arange(3), points, boot, GROUPS[group]+"−"+GROUPS[ref], group, level)
        if panel == 1:
            pb = np.array([(core[b]["point"][:, 2, 0]-core[b]["point"][:, 0, 0]).mean() for b in banks])
            db = np.stack([core[b]["draws"][:, 2, 0]-core[b]["draws"][:, 0, 0] for b in banks], -1)
            errorline(ax[panel], np.arange(3)+.1, pb, db, "C-O−B0", 0, level)
        ax[panel].plot(np.arange(3), [.002, .005, .005], "k_", markersize=20, label="retention limit")
        ax[panel].axhline(0, color="#767676", linewidth=.8)
        ax[panel].set(xticks=range(3), xticklabels=["H48 K512", "C48 K115", "C48 K1152"], ylabel="Paired ΔCE (nats)")
        ax[panel].legend(fontsize=9, loc="upper left", bbox_to_anchor=(0, -.2), ncol=2)
        retained.append(points)
        retained_draws.append(boot)
    finish(fig, "02_retention", "Prespecified retention families: S three 98.3333% intervals; C six 99.1667% intervals (versus C-D and B0). H48 margin .002; C48 margins .005. Families and estimates are not pooled.", estimates=np.array(retained), paired_draws=np.array(retained_draws), continuation_vs_base=pb, continuation_vs_base_draws=db)

    fig, ax = axes(3, h=4.5)
    mp, md = mech["point"], mech["draws"]
    for g in range(5):
        errorline(ax[0], np.arange(8), mp[:, g, :, 0].mean(0), md[:, g, :, 0], GROUPS[g], g)
    labels = ["576/48", "2304/48", "576/96", "2304/96"]*2
    ax[0].set(xticks=range(8), xticklabels=labels, ylabel="CE (nats)", xlabel="N / D; t576 then t2304")
    ax[0].tick_params(axis="x", labelrotation=45)
    drift = c.load(out/"mechanism_probability_drift.npz")
    for g in range(5):
        errorline(ax[1], np.arange(8), drift["per_seed_parent"][:, g].mean((0, -1)), drift["draws"][:, g], GROUPS[g], g)
    ax[1].set(xticks=range(8), xticklabels=labels, ylabel="Mean |Δ probability|", xlabel="Relative to N576 / D48 / t576")
    ax[1].tick_params(axis="x", labelrotation=45)
    for g in range(5):
        errorline(ax[2], np.arange(8), mp[:,g,:,1].mean(0), md[:,g,:,1], GROUPS[g], g)
    ax[2].set(xticks=range(8), xticklabels=labels, ylabel='Brier score', xlabel='N / D; t576 then t2304')
    ax[2].tick_params(axis='x', labelrotation=45)
    ax[0].legend(ncol=5, fontsize=9, loc="upper left", bbox_to_anchor=(0, -.43))
    finish(fig, "03_factorial", "Fixed physical evidence/query factorial. Eight CE, probability-drift and Brier views with pointwise 95% joint block2 intervals. Changing the clock is separate from background changes; structural O invariance at fixed clock is not an accuracy result.", per_seed=mp, draws=md, drift_per_seed=drift["per_seed_parent"], drift_draws=drift["draws"], max_probability_drift=drift["max_probability_drift"])

    fig, ax = axes()
    pp, pd = pad["point"], pad["draws"]
    for j, k in enumerate((32, 512)):
        for g in range(5):
            errorline(ax[j], np.arange(3), pp[:, g, j*3:j*3+3, 0].mean(0), pd[:, g, j*3:j*3+3, 0], GROUPS[g], g)
        ax[j].set(xticks=range(3), xticklabels=["W48 natural", "W96 fixed t", "W96 natural"], ylabel=f"K{k} CE (nats)")
        ax[j].tick_params(axis="x", labelrotation=15)
    ax[0].legend(ncol=5, fontsize=9, loc="upper left", bbox_to_anchor=(0, -.25))
    finish(fig, "04_padding", "Identical center observations and hidden queries under I-style extension; outside positions are MASK, never PAD. K32 and K512, pointwise 95% joint block2 intervals.", per_seed=pp, draws=pd)

    fig, ax = axes(3)
    op, od = oracle["point"], oracle["draws"]
    for j, view in enumerate(("C48 natural", "C96 fixed t48", "C96 natural")):
        for g in range(5):
            take = [j, j+3, j+6]
            errorline(ax[j], np.arange(3), op[:, g, take, 0].mean(0), od[:, g, take, 0], GROUPS[g], g,
                      1-.05/3 if j == 0 and g in (2, 4) else .95)
        ax[j].axhline(.01, color="black", linestyle="--", linewidth=1)
        ax[j].set(xticks=range(3), xticklabels=["4", "32", "512"], xlabel=f"K; {view}", ylabel="Exact Bernoulli KL (nats)")
    ax[0].legend(ncol=5, fontsize=9, loc="upper left", bbox_to_anchor=(0, -.25))
    finish(fig, "05_gibbs", "Exact four-unit-neighbor soft targets; 16 patterns averaged within each background. C-O/S-O C48 ability uses three 98.3333% intervals each; other bars are diagnostic 95%. K4 backgrounds are identical and add no MC information.", per_seed=op, draws=od)

    fig, aa = plt.subplots(2, 2, figsize=(12, 8))
    ax=aa.ravel()
    for j,a in enumerate(ax): a.text(.01,1.03,f'({chr(97+j)})',transform=a.transAxes,fontweight='bold')
    types = np.unique(low["types"])
    risk, ld = low["point"], low["draws"]["model_risk"]
    plot_risk = np.stack([risk[:, low["types"] == t].mean(1) for t in types], 1)
    plot_draw = np.stack([ld[:, :, low["types"] == t].mean(2) for t in types], 2)
    for j, k in enumerate((1, 2)):
        for g in range(5):
            errorline(ax[j], np.arange(len(types)), plot_risk[g, :, j, 0], plot_draw[:, g, :, j, 0], GROUPS[g], g)
        ax[j].set(xticks=range(len(types)), xticklabels=[str(t).replace("_gap", "").replace("negative", "neg") for t in types], xlabel=f"Fixed layout groups; K{k}", ylabel="Expected CE (nats)")
        ax[j].tick_params(axis="x", labelrotation=20)
    ax[0].legend(ncol=5, fontsize=9, loc="upper left", bbox_to_anchor=(0, -.3))
    response=c.load(out/'low_response.npz')
    negative=np.char.endswith(low['types'].astype(str),'negative');positive=~negative
    error=np.abs(response['response_error'][:,positive])
    bd=np.abs(response['bootstrap_model_response'][:,:,positive]-response['bootstrap_reference_response'][:,None,positive])
    err=np.stack([np.nanmean(error[...,sl],axis=(1,2)) for sl in (slice(0,2),slice(2,6))],1)
    err_draw=np.stack([np.nanmean(bd[...,sl],axis=(2,3)) for sl in (slice(0,2),slice(2,6))],2)
    neg=np.stack([np.abs(response['model_response'][:,negative,sl]).max((1,2)) for sl in (slice(0,2),slice(2,6))],1)
    neg_draw=np.stack([np.abs(response['bootstrap_model_response'][:,:,negative,sl]).max((2,3)) for sl in (slice(0,2),slice(2,6))],2)
    for g in range(5):
        errorline(ax[2],np.arange(2),err[g],err_draw[:,g],GROUPS[g],g)
        errorline(ax[3],np.arange(2),neg[g],neg_draw[:,g],GROUPS[g],g)
    ax[2].set(xticks=[0,1],xticklabels=['K1','K2'],ylabel='Mean |response error|',xlabel='64 fixed positive pairs')
    ax[3].set(xticks=[0,1],xticklabels=['K1','K2'],ylabel='Max |negative-pair response|',xlabel='16 fixed background-negative pairs')
    ax[3].axhline(0,color='black',linestyle='--',linewidth=1)
    finish(fig, "06_low_k", "All80 fixed low-K pairs. Expected CE, positive-pair absolute response error against the finite-MC pair reference, and background-negative response magnitude. 2000 joint seed/chain/block8 resamples; layouts fixed. Response summaries use finite entries only, with finite counts saved; undefined and rare entries remain in full output. Pair/pooled risks and Brier are saved separately; no information-utilization percentage.", estimates=plot_risk, draws=plot_draw, types=types,
           response_error=err,response_error_draws=err_draw,negative_response=neg,negative_response_draws=neg_draw,
           response_finite_counts=np.isfinite(error).sum((1,2)))

    fig, ax = axes()
    for j, cohort in enumerate(("continuation", "fresh")):
        z = learning[cohort]
        for arm in range(2):
            # This curve is a clearly labelled descriptive average over val banks.
            a = z["values"][:, arm, :, :, 0].mean(-1)
            mean = a.mean(0); half = student_t.ppf(.975, 5)*a.std(0, ddof=1)/np.sqrt(6)
            g = (1 if cohort == "continuation" else 3)+arm
            ax[j].plot(z["steps"], mean, marker=MARKERS[g], color=COLORS[g], label=GROUPS[g])
            ax[j].fill_between(z["steps"], mean-half, mean+half, color=COLORS[g], alpha=.15)
        ax[j].set(xlabel=f"{cohort.capitalize()} updates", ylabel="Mean old-validation CE (nats)")
        ax[j].legend()
    finish(fig, "07_learning", "Descriptive old-validation trajectory, averaged over eight banks only for display. Shading: six-seed t intervals conditional on this validation set. Actual learning flags are per seed and per bank and are never averaged away. C last interval2k–4k; S16k–24k; no checkpoint selection.", continuation=learning["continuation"]["values"], fresh=learning["fresh"]["values"])

    fig, ax = axes()
    names = ["block8_joint", "block4_joint", "block16_joint", "block8_seed_only", "block8_mc_only"]
    means, cis = [], []
    for label in names:
        z = c.load(out/("H96_k512_"+label+".npz"))["draws"]
        delta = z[:, 4, 0]-z[:, 3, 0]
        means.append(deltas[0]); cis.append(st.interval(delta))
    cis = np.array(cis)
    ax[0].vlines(range(5), cis[:, 0], cis[:, 1], color=COLORS[4]); ax[0].plot(range(5), means, "o", color=COLORS[4])
    ax[0].axhline(-.005, color="black", linestyle="--", linewidth=1)
    ax[0].set(xticks=range(5), xticklabels=["joint b8", "joint b4", "joint b16", "seed only", "MC only"], ylabel="S O−D H96 ΔCE (nats)")
    ax[0].tick_params(axis="x", labelrotation=25)
    timing = c.read(Path(root)/"runtime.json") if (Path(root)/"runtime.json").exists() else {"software_scratch_seconds": 0.}
    numeric = {k: v for k, v in timing.items() if k.endswith("_seconds") and isinstance(v, (float, int))}
    ax[1].barh(range(len(numeric)), list(numeric.values()), color=COLORS[0], edgecolor="black")
    ax[1].set(yticks=range(len(numeric)), yticklabels=[k.replace("_seconds", "").replace("_", " ") for k in numeric], xlabel="Recorded seconds (not replicate CIs)")
    finish(fig, "08_sensitivity_and_timing", "Primary95% sensitivity intervals using prescribed blocks and seed-only/MC-only resampling. Runtime panel records actual stage/benchmark durations, not independent scientific replications; memory/resource records accompany runtime JSON. Full per-seed t/LOO and endpoint stability are in numerical summaries.", sensitivity_ci=cis, sensitivity_labels=np.array(names))

    fig, ax = axes()
    for j, b in enumerate(("H96_k512", "H48_k512")):
        z = core[b]; pp, dd = z["point"], z["draws"]
        for xvalue, (a, ref, g, label) in enumerate(((2, 1, 2, "continuation"), (4, 3, 4, "fresh"))):
            errorline(ax[j], [xvalue], [(pp[:, a, 0]-pp[:, ref, 0]).mean()], (dd[:, a:a+1, 0]-dd[:, ref:ref+1, 0]), label, g)
        ax[j].axhline(0, color="black", linestyle="--", linewidth=1)
        ax[j].set(xticks=[0, 1], xticklabels=["C: +4k", "S: fresh24k"], ylabel=b.replace("_", " ")+" O−D ΔCE")
    finish(fig, "09_cohort_comparison", "Separate C and S O−D estimates (95%); these are different complete training regimens. The difference-of-effects bootstrap shares MC draws but resamples cohort seeds independently. Different significance statuses alone do not establish an interaction.", H96=core["H96_k512"]["point"], H48=core["H48_k512"]["point"])

    fig, ax = axes()
    for j, b in enumerate(("H96_k512", "H48_k512")):
        z = core[b]
        for group, indices in ((3, [5, 3]), (4, [6, 4])):
            errorline(ax[j], np.array([12000, 24000]), z["point"][:, indices, 0].mean(0), z["draws"][:, indices, 0], GROUPS[group], group)
        ax[j].set(xlabel="Snapshot along the same 24k schedule", ylabel=b.replace("_", " ")+" CE (nats)", xticks=[12000, 24000])
        ax[j].legend()
    finish(fig, "10_fresh_trajectory", "Post-training evaluation of fixed S12k/24k EMA snapshots on common core banks;95% paired joint intervals. The12k point is the midpoint of this24k learning-rate schedule, not an independently trained I-style12k model. Final24k remains primary even if worse.", H96=core["H96_k512"]["point"], H48=core["H48_k512"]["point"])
    c.write(out/"figure_captions.json", captions)
    if len(captions) != 10:
        raise RuntimeError("Missing prespecified figures")
