"""Recompute stored J oracle predictions; never retrain or alter old results."""
from __future__ import annotations

import hashlib
import io
import tarfile

import numpy as np

import capability_common as c


def review(out):
    backup = c.Path(c.CONFIG["J_backup_dir"])
    final = c.ROOT/c.CONFIG["J_final_local"]
    original = c.read(final/"analysis/oracle_summary.json")
    bank_manifest = c.read(backup/"banks_complete_v1.manifest.json")
    bank_archive = backup/bank_manifest["archive"]
    if c.sha(bank_archive) != bank_manifest["sha256"]:
        raise RuntimeError("Old bank archive SHA mismatch")
    bank_hashes = {row["path"]: row for row in bank_manifest["files"]}
    inputs = {}
    with tarfile.open(bank_archive, "r:gz") as tar:
        for k in (4, 32, 512):
            name = "artifacts/observed_context_intervention_18h_20260927/diagnostic_banks/oracle/k%d_C48_natural.npz" % k
            raw = tar.extractfile(name).read()
            if hashlib.sha256(raw).hexdigest() != bank_hashes[name]["sha256"]:
                raise RuntimeError("Old oracle input SHA mismatch")
            with np.load(io.BytesIO(raw), allow_pickle=False) as f:
                inputs[k] = {key: f[key] for key in f.files}
            c.save(out/("J_oracle_input_k%d.npz" % k), **inputs[k])
    rows, sources = [], []
    for group in ("S-D", "S-O"):
        for seed in range(92711, 92717):
            prefix = "evaluation_s%d_%s_v1" % (seed, group)
            manifest = c.read(backup/(prefix+".manifest.json"))
            archive = backup/manifest["archive"]
            if c.sha(archive) != manifest["sha256"]:
                raise RuntimeError("Old evaluation archive SHA mismatch")
            members = {row["path"]: row for row in manifest["files"]}
            source = dict(archive=str(archive), sha256=manifest["sha256"], verified_members=[])
            with tarfile.open(archive, "r:gz") as tar:
                for k in (4, 32, 512):
                    path = "artifacts/observed_context_intervention_18h_20260927/evaluation/s%d_%s/oracle/k%d_C48_natural.npz" % (seed, group, k)
                    raw = tar.extractfile(path).read()
                    if len(raw) != members[path]["bytes"] or hashlib.sha256(raw).hexdigest() != members[path]["sha256"]:
                        raise RuntimeError("Old prediction member mismatch")
                    with np.load(io.BytesIO(raw), allow_pickle=False) as f:
                        a = {key: f[key] for key in f.files}
                    b = inputs[k]
                    if not np.array_equal(a["labels"], b["labels"]):
                        raise RuntimeError("Prediction and bank soft targets disagree")
                    actual = c.array_hash(b["noisy"], b["input_coordinates"], b["t"], b["queries"])
                    if str(a["actual_input_hash"]) != actual:
                        raise RuntimeError("Old prediction native-input hash mismatch")
                    if not np.all(b["noisy"].reshape(len(b["t"]), -1)[np.arange(len(b["t"]))[:, None], b["queries"]] == 2):
                        raise RuntimeError("Old oracle query is not hidden")
                    # Analytic oracle is defined by the four true unit neighbors, not hard query labels.
                    signs = 2*((np.arange(16)[:, None] >> np.arange(4)) & 1)-1
                    expected = np.tile(1/(1+np.exp(-2*c.BETA*signs.sum(1))), len(a["labels"])//16).reshape(a["labels"].shape)
                    if np.max(np.abs(a["labels"]-expected)) > 1e-12:
                        raise RuntimeError("Stored target differs from analytic Gibbs law")
                    computed = c.metrics(a["probability"], a["labels"])
                    err = max(float(np.max(np.abs(computed[key]-a[key]))) for key in ("ce", "kl", "brier"))
                    if err > 1e-12:
                        raise RuntimeError("Stored losses fail independent recomputation")
                    if k == 4 and not np.array_equal(a["probability"].reshape(-1, 16),
                                                     np.broadcast_to(a["probability"].reshape(-1, 16)[:1], (64, 16))):
                        raise RuntimeError("K4 repeated background counted incorrectly")
                    summary = c.summarize(a["probability"], a["labels"])
                    rows.append(dict(group=group, seed=seed, k=k, recomputation_max_error=err,
                                     summary=summary, all_64_backgrounds_independent=(k != 4)))
                    c.save(out/("J_%s_%s_k%s.npz" % (seed, group, k)), **a)
                    source["verified_members"].append(members[path])
            sources.append(source)
    comparison = {}
    for group in ("S-D", "S-O"):
        comparison[group] = {}
        for k in (4, 32, 512):
            estimates = [row["summary"]["mean_kl"] for row in rows if row["group"] == group and row["k"] == k]
            old = original["ability_gates"][group][str(k)]
            if np.max(np.abs(np.asarray(estimates)-old["per_seed"])) > 1e-12:
                raise RuntimeError("Per-seed estimates disagree with frozen J analysis")
            comparison[group][str(k)] = dict(per_seed=estimates, mean=float(np.mean(estimates)),
                original_ci=old["ci"], original_ci_level=old["ci_level"],
                original_threshold=old["margin"], original_gate_passed=old["retained"])
    result = dict(status="passed_recomputation_not_capability", sources=sources,
                  bank_archive_sha256=bank_manifest["sha256"], groups=comparison, rows=rows,
                  original_summary_sha256=c.sha(final/"analysis/oracle_summary.json"),
                  new_forward_passes=0, new_bootstrap_draws=0,
                  reuse="Previously examined development evidence, not independent confirmation",
                  no_causal_comparison_with_new_finite_4x4_soft_target_training=True)
    c.write(out/"J_mixed_recipe_review.json", result)
    return result
