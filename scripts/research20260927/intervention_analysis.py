"""Fixed J analysis of saved predictions; no forward passes or selection.

All final identities are required. Bootstrap counts only shrink in a separately
labelled software_scratch directory; the formal runner never sets that flag.
"""
from pathlib import Path
import time
import numpy as np
import intervention_common as c
import intervention_statistics as st

METRICS = ["ce", "brier"]
FINAL_GROUPS = ["B0", "C-D", "C-O", "S-D", "S-O"]
MECHANISM = [f"N{n}_D{d}_t{t}" for t in (576, 2304) for d in (48, 96) for n in (576, 2304)]
PADDING = [f"k{k}_{v}" for k in (32, 512) for v in ("small_natural", "large_fixed_clock", "large_natural_clock")]
ORACLE = [f"k{k}_{v}" for k in (4, 32, 512) for v in ("C48_natural", "C96_fixed_t48", "C96_natural")]


def model_cell(group, index, snapshot=None):
    seed = c.CONFIG["base_seeds"][index] if group in ("B0", "C-D", "C-O") else c.COHORTS["fresh"]["initialization_seeds"][index]
    name = f"s{seed}_{group}"
    return name+("_12000" if snapshot == 12000 and group.startswith("S") else "")


def read_prediction(root, cell, category, name):
    return c.load(Path(root)/"evaluation"/cell/category/(name+".npz"))


def risk_arrays(root, category, names, *, oracle=False, intermediate=False):
    groups = FINAL_GROUPS+(["S-D_12000", "S-O_12000"] if intermediate else [])
    result, probabilities, hashes = [], [], {}
    parent_ids = chain_ids = None
    for index in range(6):
        by_group, by_prob = [], []
        for group in groups:
            model = model_cell(group.split("_")[0], index, 12000 if "_12000" in group else None)
            by_name, name_prob = [], []
            for name in names:
                p = read_prediction(root, model, category, name)
                key = name
                h = str(p["common_data_hash"])
                if key in hashes and hashes[key] != h:
                    raise RuntimeError("Cross-model common input mismatch: "+category+"/"+key)
                hashes[key] = h
                q = p["probability"]
                if oracle:
                    if len(q) % 16:
                        raise ValueError("Incomplete Gibbs patterns")
                    risks = np.stack([p[k].reshape(-1, 16).mean(1) for k in ("kl", "brier")])
                    chain, parent = p["chain"][::16], p["parent"][::16]
                else:
                    risks = np.stack([p[k].mean(1) for k in METRICS])
                    chain, parent = p["chain"], p["parent"]
                if parent_ids is None:
                    parent_ids, chain_ids = parent, chain
                if not np.array_equal(parent, parent_ids) or not np.array_equal(chain, chain_ids):
                    raise RuntimeError("Parent order mismatch within an analysis family")
                by_name.append(risks); name_prob.append(q)
            by_group.append(by_name); by_prob.append(name_prob)
        result.append(by_group); probabilities.append(by_prob)
    return np.array(result), np.array(probabilities), parent_ids, chain_ids, hashes


def grouped_resample(values, chain, reps, block, role, mode, check):
    # Axis1 first five groups; if present, the final two are paired S12k.
    si = [3, 4]+([5, 6] if values.shape[1] == 7 else [])
    C, S = st.resample([values[:, :3], values[:, si]], chain, reps, block, role, mode, check)
    return np.concatenate([C, S], axis=1)


def contrasts(point, draws):
    # point [seed,7,...]; draws [bootstrap,7,...]; all subtraction is paired.
    pairs = {"S_O_minus_D": (4, 3), "C_O_minus_D": (2, 1), "C_O_minus_B0": (2, 0)}
    if point.shape[1] == 7:
        pairs.update(S12_O_minus_D=(6, 5), S_D_24_minus_12=(3, 5), S_O_24_minus_12=(4, 6))
    return {name: (point[:, i]-point[:, j], draws[:, i]-draws[:, j]) for name, (i, j) in pairs.items()}


def core(root, out, check, reps):
    summaries, values_by_bank = {}, {}
    for spec in c.CONFIG["core_banks"]:
        name = spec["name"]
        x, _, parent, chain, hashes = risk_arrays(root, "core", [name], intermediate=True)
        x = x[:, :, 0]  # seed, group, metric, parent
        block = 8 if spec["parents"] == 4096 else 2
        sensitivities = [4, 16] if block == 8 else [1, 4]
        c.save(out/(name+"_per_seed_parent.npz"), risks=x, parent=parent, chain=chain,
               groups=np.array(FINAL_GROUPS+["S-D_12000", "S-O_12000"]), metrics=np.array(METRICS))
        draws_by_mode = {}
        for length, mode, n in [(block, "joint", 20000)]+[(b, "joint", 10000) for b in sensitivities]+[(block, m, 10000) for m in ("seed_only", "mc_only")]:
            check()
            draws = grouped_resample(x, chain, reps(n), length, "core_full" if block == 8 else "core_subset", mode, check)
            key = f"block{length}_{mode}"
            draws_by_mode[key] = draws
            c.save(out/(name+"_"+key+".npz"), draws=draws)
        main = draws_by_mode[f"block{block}_joint"]
        point = x.mean(-1)
        rows = {}
        for contrast, (per_seed, sample) in contrasts(point, main).items():
            level = .95
            bound = -.005 if contrast in ("S_O_minus_D", "C_O_minus_D") else 0.
            if name != "H96_k512" and contrast in ("S_O_minus_D", "C_O_minus_D", "C_O_minus_B0"):
                level = 1-.05/(3 if contrast.startswith("S") else 6)
                bound = .002 if name == "H48_k512" else .005
            rows[contrast] = {metric: st.summary(per_seed[:, j], sample[:, j], level if j == 0 else .95, bound)
                              for j, metric in enumerate(METRICS)}
            rows[contrast]["sensitivity"] = {}
            for key, draw in draws_by_mode.items():
                _, dd = contrasts(point, draw)[contrast]
                rows[contrast]["sensitivity"][key] = dict(ci=st.interval(dd[:, 0], level).tolist(), sd=float(dd[:, 0].std(ddof=1)))
        # C/S seeds are distinct; do NOT form a six-seed paired t interval here.
        diff = (main[:, 4]-main[:, 3])-(main[:, 2]-main[:, 1])
        estimate = (point[:, 4]-point[:, 3]).mean(0)-(point[:, 2]-point[:, 1]).mean(0)
        rows["cohort_difference_of_effects"] = dict(estimate=estimate.tolist(), ci95=st.interval(diff).tolist(),
            independent_seed_groups_shared_MC=True, not_pure_initialization_effect=True)
        c.save(out/(name+"_cohort_difference.npz"), draws=diff, estimate=estimate)
        summaries[name] = rows
        values_by_bank[name] = dict(point=point, draws=main)
    c.write(out/"core_summary.json", summaries)
    return summaries, values_by_bank


def diagnostic(root, out, category, names, check, reps, oracle=False):
    x, probs, parent, chain, hashes = risk_arrays(root, category, names, oracle=oracle)
    block, sensitivities = (1, [2]) if oracle else (2, [1, 4])
    c.save(out/(category+"_point.npz"), risks=x, probabilities=probs, parent=parent, chain=chain,
           names=np.array(names), groups=np.array(FINAL_GROUPS), metrics=np.array(["kl", "brier"] if oracle else METRICS))
    main = None
    for length in [block]+sensitivities:
        draw = grouped_resample(x, chain, reps(10000), length, category, "joint", check)
        c.save(out/f"{category}_bootstrap_block{length}.npz", draws=draw)
        if length == block:
            main = draw
    summary = {"names": names, "point": x.mean((0, -1)).tolist(), "ci95": st.interval(main).tolist(), "invariance": {}}
    point = x.mean(-1)
    # Risk contrasts share the same seed/parent weights as absolute outcomes.
    delta = main-main[:, :, :1]
    c.save(out/(category+"_vs_first.npz"), estimate=point.mean(0)-point.mean(0)[:, :1],
           ci95=st.interval(delta), draws=delta)
    if category == "mechanism":
        # The view order is [clock,D,N]. Separate N, D and their interaction.
        cube = x.reshape(6, 5, 2, 2, 2, 2, len(parent))
        dcube = main.reshape(len(main), 5, 2, 2, 2, 2)
        def effects(v):
            return np.stack([v[:, :, :, 0, 1]-v[:, :, :, 0, 0],
                             v[:, :, :, 1, 0]-v[:, :, :, 0, 0],
                             v[:, :, :, 1, 1]-v[:, :, :, 1, 0]-v[:, :, :, 0, 1]+v[:, :, :, 0, 0]], 3)
        c.save(out/"mechanism_factorial_effects.npz", per_seed_parent=effects(cube), draws=effects(dcube),
               names=np.array(["N_effect_at_D48", "D_effect_at_N576", "N_by_D_interaction"]))
        for g in (2, 4):
            for clock in range(2):
                group = probs[:, g, clock*4:clock*4+4]
                delta_p = group-group[:, :1]
                maximum = float(np.max(np.abs(delta_p)))
                summary["invariance"][f"{FINAL_GROUPS[g]}_clock{clock}"] = maximum
                if maximum > 2e-5:
                    raise RuntimeError("Observed-key fixed-clock factorial invariance failed")
        drift = np.abs(probs-probs[:, :, :1]).mean(-1)  # seed,group,view,parent
        drift_draw = grouped_resample(drift, chain, reps(10000), 2, "mechanism_probability_drift", "joint", check)
        c.save(out/"mechanism_probability_drift.npz", per_seed_parent=drift, draws=drift_draw,
               max_probability_drift=np.abs(probs-probs[:, :, :1]).max((0, -1, -2)))
    elif category == "padding":
        for g in (2, 4):
            for start, k in ((0, 32), (3, 512)):
                maximum = float(np.max(np.abs(probs[:, g, start+1]-probs[:, g, start])))
                summary["invariance"][f"{FINAL_GROUPS[g]}_k{k}"] = maximum
                if maximum > 2e-5:
                    raise RuntimeError("Observed-key fixed-clock padding invariance failed")
        diff = np.stack([main[:, :, i+1]-main[:, :, i] for i in (0, 1, 3, 4)], 2)
        c.save(out/"padding_changes.npz", draws=diff,
               estimate=np.stack([point.mean(0)[:, i+1]-point.mean(0)[:, i] for i in (0, 1, 3, 4)], 1))
    else:
        summary["ability_gates"] = {}
        for g in range(5):
            summary["ability_gates"][FINAL_GROUPS[g]] = {}
            for view, k in ((0, 4), (3, 32), (6, 512)):
                row = st.summary(point[:, g, view, 0], main[:, g, view, 0], 1-.05/3, .01)
                summary["ability_gates"][FINAL_GROUPS[g]][str(k)] = row
        # Exactly repeated K4 backgrounds must not contribute new MC variability.
        for view in range(3):
            arr = probs[:, :, view].reshape(6, 5, len(parent), 16, 1)
            if not np.array_equal(arr, np.broadcast_to(arr[:, :, :1], arr.shape)):
                raise RuntimeError("K4 inputs/predictions unexpectedly depend on background identity")
        c.save(out/"oracle_pattern_errors.npz", probability_error=probs-np.tile(
            c.load(Path(root)/"diagnostic_banks/oracle/k4_C48_natural.npz")["labels"][:, 0], (6, 5, 9, 1))[..., None])
    c.write(out/(category+"_summary.json"), summary)
    return dict(point=point, draws=main, summary=summary)


def low(root, out, check, reps):
    layouts = c.load(Path(root)/"design/low_layouts.npz")
    raw = c.load(Path(root)/"reference/low_joint_counts.npz")
    grouping, group_keys = st.geometry_groups(layouts)
    p = np.stack([np.stack([read_prediction(root, model_cell(g, i), "low", "low_k")["probability"].reshape(80, 2, 6)
                           for g in FINAL_GROUPS]) for i in range(6)])
    table = raw["counts"].mean(0)
    values = [st.low_values(table, grouping, x) for x in p]
    risk = np.stack([x[0] for x in values])
    refs, ref = values[0][1:]
    negative = np.char.endswith(layouts["types"].astype(str), "negative")
    for g in (2, 4):
        if np.abs(p[:, g, negative, 1]-p[:, g, negative, 0]).max() > 2e-5:
            raise RuntimeError("Low-K O background-negative control failed")
    c.save(out/"low_point.npz", probability=p, model_risk=risk, reference_risk=refs, grouping=grouping,
           group_keys=np.array(group_keys), types=layouts["types"], **ref)
    draws = st.low_bootstrap(raw["counts"], raw["chain"], grouping, p, reps(2000), check)
    c.save(out/"low_bootstrap.npz", **draws)
    response = (p[:, :, :, 1]-p[:, :, :, 0]).mean(0)
    c.save(out/"low_response.npz", model_response=response,
           reference_response=ref["pair_probability"][:, 1]-ref["pair_probability"][:, 0],
           response_error=response-(ref["pair_probability"][:, 1]-ref["pair_probability"][:, 0])[None],
           bootstrap_model_response=draws["model_response"], bootstrap_reference_response=draws["reference_response"])
    return dict(point=risk.mean(0), reference=refs, draws=draws, types=layouts["types"])


def learning(root, out):
    result, flags = {}, {}
    for cohort, spec in c.COHORTS.items():
        names = sorted(p.stem for p in (Path(root)/"validation_banks").glob("*.npz"))
        arrays = []
        for i in range(6):
            by_arm = []
            for mode in ("D", "O"):
                ident = c.identity(cohort, i, mode)
                by_step = []
                for step in spec["validation_steps"]:
                    by_step.append([np.array([c.load(Path(root)/"training"/ident["cell"]/"validation"/f"step_{step}"/(name+".npz"))[k].mean()
                                               for k in METRICS]) for name in names])
                by_arm.append(by_step)
            arrays.append(by_arm)
        a = np.array(arrays)  # seed,arm,step,bank,metric
        start, end = [spec["validation_steps"].index(s) for s in spec["learning_flag_last_interval"]]
        f = dict(final_ce_at_least_ln2=(a[:, :, end, :, 0] >= np.log(2)).tolist(),
                 last_interval_improvement_gt_005=(a[:, :, start, :, 0]-a[:, :, end, :, 0] > .005).tolist(),
                 bank_names=names, steps=spec["validation_steps"], no_flag_is_not_proof_of_convergence=True)
        flags[cohort] = f
        c.save(out/("learning_"+cohort+".npz"), values=a, steps=np.array(spec["validation_steps"]), names=np.array(names))
        result[cohort] = dict(values=a, steps=spec["validation_steps"], names=names)
    c.write(out/"learning_flags.json", flags)
    return result, flags


def run(root, check, *, software_scratch=False):
    root = Path(root)
    if software_scratch and "scratch" not in str(root):
        raise ValueError("Reduced bootstrap forbidden in formal output")
    out = root/"analysis"
    out.mkdir(exist_ok=False)
    began = time.time()
    reps = (lambda n: 20) if software_scratch else (lambda n: n)
    cs, cv = core(root, out, check, reps)
    mech = diagnostic(root, out, "mechanism", MECHANISM, check, reps)
    pad = diagnostic(root, out, "padding", PADDING, check, reps)
    oracle = diagnostic(root, out, "oracle", ORACLE, check, reps, True)
    low_data = low(root, out, check, reps)
    lv, flags = learning(root, out)
    ability = oracle["summary"]["ability_gates"]
    s_ret = all(cs[name]["S_O_minus_D"]["ce"]["retained"] for name in ("H48_k512", "C48_k115", "C48_k1152"))
    c_ret = all(cs[name][contrast]["ce"]["retained"] for name in ("H48_k512", "C48_k115", "C48_k1152") for contrast in ("C_O_minus_D", "C_O_minus_B0"))
    gates = dict(primary=cs["H96_k512"]["S_O_minus_D"]["ce"], fresh_retention=s_ret,
        fresh_ability=all(x["retained"] for x in ability["S-O"].values()),
        continuation_retention=c_ret, continuation_ability=all(x["retained"] for x in ability["C-O"].values()),
        continuation_secondary=cs["H96_k512"]["C_O_minus_D"]["ce"],
        continuation_progress=cs["H96_k512"]["C_O_minus_B0"]["ce"],
        structural_checks_passed=True, software_scratch=software_scratch)
    gates["fresh_qualified"] = bool(gates["primary"]["practical"] and s_ret and gates["fresh_ability"])
    gates["continuation_qualified"] = bool(gates["continuation_secondary"]["practical"] and
        gates["continuation_progress"]["directional"] and c_ret and gates["continuation_ability"])
    c.write(out/"gates.json", gates)
    from intervention_figures import make_figures
    make_figures(root, cv, mech, pad, oracle, low_data, lv, software_scratch)
    c.write(out/"complete.json", dict(status="software_scratch_complete" if software_scratch else "analysis_complete_needs_visual_review",
                                     seconds=time.time()-began, figures=10, bootstrap_replication_mode="scratch20" if software_scratch else "preregistered_full"))
    return gates
