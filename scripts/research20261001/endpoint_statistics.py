"""Fixed estimands and paired seed/image/MC-chain-block uncertainty; numpy only."""
from __future__ import annotations

import itertools
import numpy as np
from scipy.stats import t as student_t
import endpoint_common as c

METRICS = ("G_short_NRMSE", "G_mid_NRMSE", "G_long_NRMSE", "m_bias", "m2_bias", "abs_m_bias", "energy_bias")


def features(stats, average_origins=False):
    sums, counts = np.asarray(stats["pair_sum"]), np.asarray(stats["pair_count"])
    if average_origins:
        sums, counts = sums.mean(1), counts.mean(1)
    # At a fixed width and region, pair counts are fixed for every image.
    # Thus averaging per-image curves exactly equals ratio of pooled pair sums.
    assert np.all(counts == counts[0]) and np.all(counts > 0)
    curve = sums/counts
    moments = [np.asarray(stats[k]).mean(1) if average_origins else np.asarray(stats[k]) for k in ["m", "m2", "abs_m", "energy"]]
    return np.column_stack([curve, *moments]).astype(np.float64)


def bands_for(view, width):
    if view in ("majority32", "decimation32"):
        return [(1, 2), (3, 8), (9, 16)]
    if view == "majority30_origins":
        return [(1, 2), (3, 8), (9, 15)]
    if width == 48 or view == "center48":
        return [(1, 4), (5, 12), (13, 24)]
    return [(1, 8), (9, 24), (25, 48)]


def metric_values(generated_mean, reference_mean, bands):
    """Broadcast-compatible: [..., curve+four moments] -> [..., seven metrics]."""
    g, r = np.asarray(generated_mean), np.asarray(reference_mean)
    result = []
    for lo, hi in bands:
        num = np.sqrt(np.mean((g[..., lo:hi+1]-r[..., lo:hi+1])**2, axis=-1))
        den = np.maximum(np.sqrt(np.mean(r[..., lo:hi+1]**2, axis=-1)), 1e-12)
        result.append(num/den)
    result.extend([g[..., -4+i]-r[..., -4+i] for i in range(4)])
    return np.stack(result, axis=-1)


def block_weights(chain, random, block):
    groups = [np.flatnonzero(chain == x) for x in np.unique(chain)]
    result = np.zeros(len(chain), dtype=np.float64)
    for gi in random.integers(len(groups), size=len(groups)):
        group = groups[gi]
        assert len(group) >= block
        starts = random.integers(len(group), size=(len(group)+block-1)//block)
        local = ((starts[:, None]+np.arange(block)) % len(group)).ravel()[:len(group)]
        np.add.at(result, group[local], 1/(len(groups)*len(group)))
    return result


def interval(draws, level=.95):
    return np.quantile(draws, [(1-level)/2, (1+level)/2], axis=0)


def generation_bootstrap(g, reference, chain, bands, reps, block=8, mode="joint", role="generation"):
    """g[seed,arm,sampler,image,features]. Duplicate outer draws get own image draws."""
    ns, na, nm, ni, nf = g.shape
    r = c.rng(c.CFG["role_seeds"]["bootstrap"], block, role+"_"+mode)
    result = np.empty((reps, na, nm, 7), dtype=np.float64)
    physical = np.empty((reps, na, nm, 3), dtype=np.float64)
    for start in range(0, reps, 64):
        c.check(); size = min(64, reps-start)
        ids = r.integers(ns, size=(size, ns)) if mode in ("joint", "seed_only") else np.broadcast_to(np.arange(ns), (size, ns))
        if mode in ("joint", "mc_only"):
            pw = np.stack([block_weights(chain, r, block) for _ in range(size)])
            rm = pw@reference
        else:
            rm = np.broadcast_to(reference.mean(0), (size, nf))
        per_draw = np.zeros((size, na, nm, 7))
        per_physical = np.zeros((size, na, nm, 3))
        for slot in range(ns):
            if mode in ("joint", "image_only"):
                weights = np.stack([np.bincount(r.integers(ni, size=ni), minlength=ni)/ni for _ in range(size)])
            else:
                weights = np.full((size, ni), 1/ni)
            gm = np.einsum("bi,bamif->bamf", weights, g[ids[:, slot]], optimize=True)
            values = metric_values(gm, rm[:, None, None, :], bands)
            per_draw += values/ns
            per_physical += np.stack([values[..., 0], np.abs(values[..., 6]), np.abs(values[..., 5])], -1)/ns
        result[start:start+size] = per_draw
        physical[start:start+size] = per_physical
    return result, physical


def linear_bootstrap(values, chain, reps, block, role, mode="joint"):
    """values[seed,...,parent]. Shared parent weights across every outcome/arm."""
    ns = len(values)
    shape = values.shape[1:-1]
    flat = values.reshape(ns, -1, len(chain))
    r = c.rng(c.CFG["role_seeds"]["bootstrap"], block, role+"_"+mode)
    result = np.zeros((reps, flat.shape[1]), dtype=np.float64)
    for start in range(0, reps, 128):
        c.check(); n = min(128, reps-start)
        sw = np.stack([np.bincount(r.integers(ns, size=ns), minlength=ns)/ns for _ in range(n)]) if mode != "mc_only" else np.full((n, ns), 1/ns)
        pw = np.stack([block_weights(chain, r, block) for _ in range(n)]) if mode != "seed_only" else np.full((n, len(chain)), 1/len(chain))
        for s in range(ns):
            result[start:start+n] += (pw@flat[s].T)*sw[:, s, None]
    return result.reshape(reps, *shape)


def contrast_summary(per_seed, draws, level=.975, margin=-.05):
    x = np.asarray(per_seed)
    mean = float(x.mean())
    se = float(x.std(ddof=1)/np.sqrt(len(x)))
    half = float(student_t.ppf((1+level)/2, len(x)-1)*se)
    ci = interval(draws, level)
    tci = [mean-half, mean+half]
    signs = np.array(list(itertools.product([-1, 1], repeat=len(x))))
    permuted = (signs*x).mean(1)
    # Numerical Monte Carlo uncertainty of percentile endpoints, not scientific SE.
    br = c.rng(c.CFG["role_seeds"]["bootstrap"], len(draws), "percentile_endpoint_mcse")
    endpoint_replicates = np.stack([interval(np.asarray(draws)[br.integers(len(draws), size=len(draws))], level) for _ in range(200)])
    return dict(estimate=mean, per_seed=x.tolist(), bootstrap_ci=ci.tolist(), ci_level=level,
                paired_t_ci=tci, supplementary_bootstrap_ci95=interval(draws).tolist(),
                practical_threshold=margin, directional=bool(ci[1] < 0 and tci[1] < 0),
                practical=bool(ci[1] < margin and tci[1] < margin),
                paired_seed_sd=float(x.std(ddof=1)), bootstrap_scientific_sd=float(np.std(draws, ddof=1)),
                bootstrap_endpoint_mcse=endpoint_replicates.std(0, ddof=1).tolist(), endpoint_mcse_resamples=200,
                supplementary_practical_thresholds={str(v): bool(ci[1] < v and tci[1] < v) for v in [-.02, -.10]},
                split_half_endpoint_difference=(interval(draws[:len(draws)//2], level)-interval(draws[len(draws)//2:], level)).tolist(),
                leave_one_lineage_out=[float(np.delete(x, i).mean()) for i in range(len(x))],
                sign_flip_two_sided_p=float(np.mean(np.abs(permuted) >= abs(mean)-1e-14)),
                sign_flip_assignments=len(permuted))


def upper_gate(draws, estimate, level, threshold):
    upper = float(np.quantile(draws, level))
    return dict(estimate=float(estimate), one_sided_level=level, upper=upper, threshold=threshold, passed=bool(upper <= threshold))


def gather_generation(root, width, view="full", image_type="final"):
    data = []
    for seed in c.SEEDS:
        arms = []
        for arm in c.ARMS:
            samplers = []
            for sampler in ["monotone-256", "reveal192-repair64"]:
                z = c.load(root/"evaluation"/f"s{seed}_{arm}"/"generation"/f"w{width}_{sampler}"/f"stats_{image_type}_{view}.npz")
                samplers.append(features(z, view == "majority30_origins"))
            arms.append(samplers)
        data.append(arms)
    rp = root/"reference"/("stats_w48.npz" if width == 48 else f"stats_{view}.npz")
    rz = c.load(rp)
    return np.asarray(data), features(rz, view == "majority30_origins"), rz["chain"]


def repair_diagnostics(root, out, reference, chain, reps, fixture=False):
    rows, kl, changed, energy = [], [], [], []
    for seed in c.SEEDS:
        arms, arm_kl, arm_changed, arm_energy = [], [], [], []
        for arm in c.ARMS:
            folder = root/"evaluation"/f"s{seed}_{arm}"/"generation/w96_reveal192-repair64"
            arms.append([features(c.load(folder/f"stats_{typ}_full.npz")) for typ in ["prefix", "final", "oracle"]])
            kk, cc, en = [], [], []
            for path in sorted(folder.glob("shard_*.npz")):
                z = c.load(path)
                p = np.clip(z["repair_probability"].astype(float), 1e-12, 1-1e-12)
                y = z["repair_target"].astype(float)
                yc = np.clip(y, 1e-15, 1-1e-15)
                kk.append((y*np.log(yc/p)+(1-y)*np.log((1-yc)/(1-p))).mean(-1))
                cc.append((z["repair_old"] != z["repair_new"]).mean(-1))
                import endpoint_data as data
                initial=data.physical_stats(z["prefix_spins"])["energy"]
                trace=[]
                for old,new,neighbors in [("repair_old","repair_new","repair_neighbors"),("oracle_old","oracle_new","oracle_neighbors")]:
                    delta=-(2*(z[new].astype(float)-z[old]))*(2*z[neighbors].astype(float)-1).sum(-1)
                    delta=delta.sum(-1)/(2*96*95)
                    trace.append(np.column_stack([initial,initial[:,None]+delta.cumsum(1)]))
                en.append(np.stack(trace,1))
            if fixture and not kk:
                arm_kl.append(np.linspace(.02, .01, 64)); arm_changed.append(np.full(64, .3))
                arm_energy.append(np.full((2,65),-.70))
            else:
                assert sum(len(x) for x in kk) == 128
                arm_kl.append(np.concatenate(kk).mean(0)); arm_changed.append(np.concatenate(cc).mean(0))
                arm_energy.append(np.concatenate(en).mean(0))
        rows.append(arms); kl.append(arm_kl); changed.append(arm_changed);energy.append(arm_energy)
    values = np.asarray(rows)
    bands = bands_for("full", 96)
    point = metric_values(values.mean(-2), reference.mean(0), bands)
    draws, _ = generation_bootstrap(values, reference, chain, bands, reps, role="prefix_model_oracle_paired")
    c.save(out/"repair_diagnostics.npz", per_seed=point, draws=draws, on_policy_local_kl=np.asarray(kl),
           changed_fraction_of_selected=np.asarray(changed), type_names=np.array(["prefix192", "model_repair", "oracle_repair"]))
    c.save(out/"repair_energy_trajectory.npz", per_seed=np.asarray(energy), steps=np.arange(65), methods=np.array(["model","oracle"]))
    return dict(model_minus_prefix_mean=(point[:, :, 1]-point[:, :, 0]).mean(0).tolist(),
                model_minus_prefix_ci95=interval(draws[:, :, 1]-draws[:, :, 0]).tolist(),
                oracle_minus_prefix_mean=(point[:, :, 2]-point[:, :, 0]).mean(0).tolist(),
                oracle_minus_prefix_ci95=interval(draws[:, :, 2]-draws[:, :, 0]).tolist(),
                oracle_is_not_a_neural_result=True, finite_correction_not_a_stationary_or_mixing_guarantee=True)


def phase0_diagnostics(root, out, reference, chain, reps, fixture=False):
    rows = []
    for seed in c.SEEDS:
        rows.append([[features(c.load(root/"phase0"/f"s{seed}"/f"floor{floor}"/"stats_final_full.npz")) for floor in [.01, .002]]])
    g = np.asarray(rows)
    point = metric_values(g.mean(-2), reference.mean(0), bands_for("full", 96))
    draws, _ = generation_bootstrap(g, reference, chain, bands_for("full", 96), reps, role="phase0_clock")
    local = np.empty((6, 2, 4, 2))
    for si, seed in enumerate(c.SEEDS):
        for wi, width in enumerate([48, 96]):
            for mi, masks in enumerate([1, 2, 8, 32]):
                for fi, floor in enumerate([.01, .002]):
                    z = c.load(root/"phase0"/f"s{seed}"/"conditional"/f"phase0_w{width}_m{masks}_floor{floor}.npz")
                    local[si, wi, mi, fi] = z["kl"].mean()
    c.save(out/"phase0_clock.npz", per_seed=point, draws=draws, floors=np.array([.01, .002]), exact_local_kl=local)
    return dict(frozen_old_models=True, images_per_model_floor=32, mean=(point[:, 0, 1]-point[:, 0, 0]).mean(0).tolist(),
                ci95=interval(draws[:, 0, 1]-draws[:, 0, 0]).tolist(), primary=False)


def analyze(root, out=None, fixture=False, full_reps=False):
    root = __import__("pathlib").Path(root)
    out = root/"analysis" if out is None else __import__("pathlib").Path(out)
    out.mkdir(parents=True, exist_ok=False)
    reps = 32 if fixture and not full_reps else 20000
    sensitivity_reps = 16 if fixture and not full_reps else 10000
    g, ref, chain = gather_generation(root, 96)
    bands = bands_for("full", 96)
    point = metric_values(g.mean(-2), ref.mean(0), bands)
    draw, physical_draw = generation_bootstrap(g, ref, chain, bands, reps)
    c.save(out/"generation_primary_draws_block8.npz", metrics=draw, physical=physical_draw)
    c.save(out/"generation_point.npz", per_seed=point, metric_names=np.array(METRICS), generated_curves=g.mean(-2), reference_mean=ref.mean(0))
    primary = {
        "P1_training_coverage": contrast_summary(point[:, 2, 0, 2]-point[:, 0, 0, 2], draw[:, 2, 0, 2]-draw[:, 0, 0, 2]),
        "P2_committed_repair": contrast_summary(point[:, 2, 1, 2]-point[:, 2, 0, 2], draw[:, 2, 1, 2]-draw[:, 2, 0, 2]),
    }
    sensitivities = {}
    for block, mode in [(4, "joint"), (16, "joint"), (8, "seed_only"), (8, "image_only"), (8, "mc_only")]:
        actual_block = min(block, min(np.bincount(chain.astype(int)))) if fixture else block
        bd, _ = generation_bootstrap(g, ref, chain, bands, sensitivity_reps, actual_block, mode, f"sensitivity_block{block}")
        c.save(out/f"generation_sensitivity_block{block}_{mode}.npz", metrics=bd)
        contrasts = [bd[:, 2, 0, 2]-bd[:, 0, 0, 2], bd[:, 2, 1, 2]-bd[:, 2, 0, 2]]
        sensitivities[f"block{block}_{mode}"] = {name: interval(v, .975).tolist() for name, v in zip(primary, contrasts)}
    for name, record in primary.items():
        record["sensitivity_direction_reversal"] = any(v[name][0] > 0 for v in sensitivities.values())
        record["robust_direction_in_all_sensitivities"] = all(v[name][1] < 0 for v in sensitivities.values())
    secondary = {}
    for name, a, b in [("L_minus_A_under_S256", (1, 0), (0, 0)), ("E_minus_L_under_S256", (2, 0), (1, 0)),
                       ("repair_under_A", (0, 1), (0, 0)), ("repair_under_L", (1, 1), (1, 0))]:
        values = draw[:, a[0], a[1]]-draw[:, b[0], b[1]]
        secondary[name] = dict(estimate=(point[:, a[0], a[1]]-point[:, b[0], b[1]]).mean(0).tolist(), ci95=interval(values).tolist())
    inter = draw[:, 2, 1]-draw[:, 2, 0]-draw[:, 0, 1]+draw[:, 0, 0]
    secondary["training_by_repair_interaction"] = dict(estimate=(point[:, 2, 1]-point[:, 2, 0]-point[:, 0, 1]+point[:, 0, 0]).mean(0).tolist(), ci95=interval(inter).tolist())
    physical_point = np.stack([point[..., 0], np.abs(point[..., 6]), np.abs(point[..., 5])], -1).mean(0)
    retention = {}
    for label, left, right in [("P1", (2, 0), (0, 0)), ("P2", (2, 1), (2, 0))]:
        retention[label] = {}
        for j, (name, margin) in enumerate([("short_NRMSE", .02), ("energy_absolute_bias", .01), ("abs_m_absolute_bias", .02)]):
            differences = physical_draw[:, left[0], left[1], j]-physical_draw[:, right[0], right[1], j]
            estimate = physical_point[left[0], left[1], j]-physical_point[right[0], right[1], j]
            retention[label][name] = dict(single_claim=upper_gate(differences, estimate, 1-.05/3, margin),
                                         both_claims=upper_gate(differences, estimate, 1-.05/6, margin))
    # Predefined additional views. Their targets and distance bands remain separate.
    views = {}
    for width, view in [(48, "full"), (96, "center48"), (96, "edge8"), (96, "majority32"), (96, "decimation32"), (96, "majority30_origins")]:
        vg, vr, vc = gather_generation(root, width, view)
        vb = bands_for(view, width)
        vp = metric_values(vg.mean(-2), vr.mean(0), vb)
        vd, _ = generation_bootstrap(vg, vr, vc, vb, sensitivity_reps, role=f"w{width}_{view}")
        name = f"w{width}_{view}"
        c.save(out/f"secondary_{name}.npz", point=vp, draws=vd, generated_mean=vg.mean(-2), reference_mean=vr.mean(0))
        views[name] = dict(bands=vb, mean=vp.mean(0).tolist(), ci95=interval(vd).tolist(),
                           E_S_minus_A_S=interval(vd[:, 2, 0]-vd[:, 0, 0]).tolist(),
                           E_R_minus_E_S=interval(vd[:, 2, 1]-vd[:, 2, 0]).tolist())
        if width == 48:
            local_bands = [(1, 8)]*3
            lp = metric_values(vg.mean(-2), vr.mean(0), local_bands)[..., 0]
            ld, _ = generation_bootstrap(vg, vr, vc, local_bands, sensitivity_reps, role="w48_additional_G1_8")
            c.save(out/"secondary_w48_additional_G1_8.npz", point=lp, draws=ld[..., 0])
            views["w48_additional_G1_8"] = dict(band=[1, 8], mean=lp.mean(0).tolist(), ci95=interval(ld[..., 0]).tolist())
    local_names = [f"exact_w{w}_m{m}" for w in [48, 96] for m in [1, 2, 8, 32]]
    ret_names = ["C48_K115", "C48_K1152", "H48_K512"]
    exact, retained, stress = [], [], []
    for seed in c.SEEDS:
        er, rr, ss = [], [], []
        for arm in c.ARMS:
            cell = root/"evaluation"/f"s{seed}_{arm}"
            er.append(np.stack([c.load(cell/"conditional"/(name+".npz"))["kl"].mean(1) for name in local_names]))
            rr.append(np.stack([c.load(cell/"conditional"/(name+".npz"))["ce"].mean(1) for name in ret_names]))
            ss.append([float(np.abs(c.load(cell/"conditional"/f"stress_w{w}.npz")["probability_error"]).max()) for w in [48, 96]])
        exact.append(er); retained.append(rr); stress.append(ss)
    exact, retained, stress = np.asarray(exact), np.asarray(retained), np.asarray(stress)
    local_chain = c.load(root/"banks/exact_w48_m1.npz")["chain"]
    edraw = linear_bootstrap(exact, local_chain, sensitivity_reps, 2, "local_exact")
    rdraw = linear_bootstrap(retained, local_chain, sensitivity_reps, 2, "retention")
    c.save(out/"conditional_draws.npz", exact_per_parent=exact, retained_per_parent=retained, exact_draws=edraw, retained_draws=rdraw, stress_max_probability_error=stress)
    capability = {name: upper_gate(edraw[:, 2, j], exact[:, 2, j].mean(), .99375, .01) for j, name in enumerate(local_names)}
    old_retention = {name: upper_gate(rdraw[:, 2, j]-rdraw[:, 0, j], (retained[:, 2, j]-retained[:, 0, j]).mean(), 1-.05/3, .005) for j, name in enumerate(ret_names)}
    conditional_sensitivity = {}
    for block in [1, 4]:
        ex = linear_bootstrap(exact, local_chain, sensitivity_reps, block, "local_exact_sensitivity")
        rt = linear_bootstrap(retained, local_chain, sensitivity_reps, block, "retention_sensitivity")
        c.save(out/f"conditional_sensitivity_block{block}.npz", exact_draws=ex, retention_draws=rt)
        conditional_sensitivity[str(block)] = dict(exact_E_ci95=interval(ex[:, 2]).tolist(), retention_E_minus_A_ci95=interval(rt[:, 2]-rt[:, 0]).tolist())
    learning = np.empty((6, 3, 3, 4))
    for si, seed in enumerate(c.SEEDS):
        for ai, arm in enumerate(c.ARMS):
            for ti, step in enumerate([4000, 6000, 8000]):
                for mi, m in enumerate([1, 2, 8, 32]):
                    z = c.load(root/"evaluation"/f"s{seed}_{arm}"/"learning"/f"step{step}_exact_w48_m{m}.npz")
                    learning[si, ai, ti, mi] = z["kl"].mean()
    flags = (learning[:, :, 1].mean(-1)-learning[:, :, 2].mean(-1)) > .002
    c.save(out/"learning.npz", kl=learning, learning_incomplete=flags)
    repair = repair_diagnostics(root, out, ref, chain, sensitivity_reps, fixture)
    phase0 = phase0_diagnostics(root, out, ref, chain, sensitivity_reps, fixture)
    summary = dict(status="analysis_complete", fixture=fixture, primary=primary, primary_sensitivity=sensitivities,
                   secondary_contrasts=secondary, generation_secondary_views=views, physical_retention=retention,
                   local_capability=capability, stress_max_probability_error=stress.tolist(), old_conditional_retention=old_retention,
                   learning_incomplete=flags.tolist(), conditional_sensitivity=conditional_sensitivity,
                   repair_diagnostics=repair, phase0_clock_diagnostic=phase0,
                   independent_training_lineages=6, continuation_branches=18,
                   scientific_success=bool(all(v["practical"] and not v["sensitivity_direction_reversal"] for v in primary.values())
                       and all(v["passed"] for v in capability.values()) and np.all(stress[:, 2] <= .05)
                       and all(v["passed"] for v in old_retention.values())
                       and all(v["both_claims"]["passed"] for section in retention.values() for v in section.values())),
                   interpretive_limits=["conditional ability is not correct joint generation", "oracle is not a neural result",
                                        "RG diagnostics are not RG training", "six inherited lineages, not18 fresh replications"])
    c.write(out/"summary.json", summary)
    return summary
