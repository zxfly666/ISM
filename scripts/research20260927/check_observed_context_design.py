"""Read-only, standard-library checks for the J design. Never runs science.
No torch import, no checkpoint deserialization, no network/GPU/MC or file writes.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/research_reboot_20260921"
CONFIG = DOCS / "OBSERVED_CONTEXT_INTERVENTION_CONFIG_20260927.json"
PLAN = DOCS / "OBSERVED_CONTEXT_INTERVENTION_PLAN_20260927_ZH.md"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    c = read(CONFIG)
    assert c["status"] == "design_only_not_implemented_not_preflighted_not_authorized_to_start"
    assert not any(c[x] for x in ["authorized_to_start", "gpu_or_new_mc_started", "automation_created"])
    assert c["budget_started"] is None and c["deadline"] is None
    assert c["arms"] == ["J-D", "J-O"]
    assert len(set(c["base_seeds"])) == len(set(c["continuation_data_seeds"])) == 6
    assert set(c["base_seeds"]).isdisjoint(c["continuation_data_seeds"])
    assert len(set(c["seeds"].values())) == len(c["seeds"])
    assert c["base_step"] + c["continuation_steps"] == c["final_global_step"] == 16000
    assert not c["fresh_initialization"]
    assert c["checkpoint_restore"] == [
        "raw_model", "ema", "optimizer_including_moments_and_steps", "recorded_rng_states"
    ]
    tr, mc, xp = c["training"], c["mc"], c["expected_counts"]
    nbases, narms = len(c["base_seeds"]), len(c["arms"])
    ntrain = nbases * narms
    nmodel = ntrain + nbases
    nsteps, tokens = c["continuation_steps"], tr["tokens_per_update"]
    assert nsteps % 32 == 0
    cycles = nsteps // 32
    batches = [tokens // w**2 for w in tr["widths"]]
    assert batches == [72, 32, 18, 8]
    assert all(tokens % w**2 == 0 for w in tr["widths"])
    computed = {
        "trained_branches": ntrain, "frozen_baselines": nbases, "formal_evaluation_models": nmodel,
        "training_updates": ntrain * nsteps,
        "fixture_updates": c["exact_gibbs"]["fixture"]["arms"] * c["exact_gibbs"]["fixture"]["steps_per_arm"],
        "ordinary_updates_per_branch": cycles * tr["ordinary_updates_per_cycle"],
        "sparse_updates_per_branch": cycles * tr["sparse_updates_per_cycle"],
        "window_exposures_per_branch": cycles * 4 * len(tr["physical_kinds"]) * sum(batches),
        "training_tokens_per_branch": nsteps * tokens,
        "total_training_tokens": ntrain * nsteps * tokens,
        "sparse_query_slots_per_branch": cycles * tr["sparse_updates_per_cycle"] * tr["sparse_queries_per_update"],
        "mc_parents": mc["chains"] * mc["parents_per_chain"],
        "core_banks": len(c["core_banks"]),
        "core_prediction_files": nmodel * len(c["core_banks"]),
        "core_input_forwards": nmodel * sum(b["parents"] for b in c["core_banks"]),
        "recoverable_final_checkpoints": ntrain,
        "diagnostic_2k_ema_checkpoints": ntrain * len(tr["diagnostic_ema_steps"]),
        "figures": 8,
    }
    for b in c["core_banks"]:
        w = b["width"]
        if b["kind"] == "held_gap":
            q = w // 4 - 1
            inc = [3] * (3*q + 3) + [6] * q
        else:
            inc = [1] * (w - 1)
        assert len(inc) == w - 1 and sum(inc) == b["span"]
        assert b["k"] + c["queries_per_parent"] <= w*w
    me = c["mechanism"]
    assert me["views"] == len(me["valid_token_counts"]) * len(me["domain_sides"]) * len(me["times"])
    assert (me["evidence_query_pool_inclusive"][1] - me["evidence_query_pool_inclusive"][0] + 1)**2 >= me["k"] + me["queries"]
    for n in me["valid_token_counts"]:
        for d in me["domain_sides"]:
            assert me["k"] + me["queries"] + 2 <= n <= d*d
    computed["mechanism_prediction_files"] = nmodel * me["views"]
    computed["mechanism_input_forwards"] = nmodel * me["views"] * me["parents"]
    pad = c["padding"]
    computed["padding_prediction_files"] = nmodel * len(pad["ks"]) * len(pad["views"])
    computed["padding_input_forwards"] = computed["padding_prediction_files"] * pad["parents"]
    oracle = c["exact_gibbs"]["formal"]
    assert mc["chains"] * len(oracle["frames_per_chain"]) == oracle["parents"]
    computed["oracle_prediction_files"] = nmodel * len(oracle["ks"]) * len(oracle["views"])
    computed["oracle_input_forwards"] = computed["oracle_prediction_files"] * oracle["parents"] * oracle["patterns"]
    low = c["low_k"]
    computed["low_k_prediction_files"] = nmodel
    computed["low_k_input_forwards"] = nmodel * low["layout_pairs"] * low["sides"] * low["symbol_conditions"]
    computed["low_joint_count_entries"] = low["parents"] * low["layout_pairs"] * low["sides"] * 8
    computed["validation_prediction_files"] = ntrain * len(tr["validation_steps"]) * tr["validation_banks"]
    computed["validation_input_forwards"] = computed["validation_prediction_files"] * tr["validation_parents_per_bank"]
    modules = ["core", "mechanism", "padding", "oracle", "low_k", "validation"]
    computed["total_prediction_files"] = sum(computed[x + "_prediction_files"] for x in modules)
    computed["total_input_forwards_excluding_preflight"] = sum(computed[x + "_input_forwards"] for x in modules)
    computed["w96_input_forwards"] = (
        nmodel * sum(b["parents"] for b in c["core_banks"] if b["width"] == 96)
        + nmodel * len(pad["ks"]) * 2 * pad["parents"]
        + nmodel * len(oracle["ks"]) * 2 * oracle["parents"] * oracle["patterns"]
    )
    computed["other_input_forwards"] = computed["total_input_forwards_excluding_preflight"] - computed["w96_input_forwards"]
    assert computed == xp, {"computed": computed, "expected": xp}
    assert c["retention"]["contrasts"] == len(c["retention"]["comparators"]) * len(c["retention"]["bounds"]) == 6
    assert math.isclose(c["retention"]["ci"], 1 - 0.05 / 6)
    assert math.isclose(oracle["gate_ci"], 1 - 0.05 / len(oracle["ks"]))
    assert c["primary"]["practical_upper_below"] == -0.005
    assert c["baseline_progress_guard"]["upper_below"] == 0
    assert c["generation"] == {"enabled": False, "images": 0, "shards": 0}

    # Verify old frozen inputs stay unchanged. Only read files and hashes.
    old = read(ROOT / "artifacts/geometry_identification_20260926_remote/budget_gate.json")
    frozen = old["source_sha256"]
    assert len(frozen) == 76
    mismatches = [rel for rel, h in frozen.items() if sha_file(ROOT / rel) != h]
    assert not mismatches, mismatches

    # Verify existing source material, never alter independent-review files.
    review_checks = read(ROOT / "independent_review_20260926/delivery_checks.json")
    review_hashes = []
    for r in review_checks["markdown"]:
        p = Path(r["path"])
        got = sha_file(p)
        assert got == r["sha256"], str(p)
        review_hashes.append({"path": str(p), "sha256": got})

    # Verify all six chosen base final bytes in the existing local archives.
    # Does not extract or torch.load(), and does not represent a restore test.
    backup = Path(c["base_backup_root"])
    bases = []
    for seed, expected in zip(c["base_seeds"], c["base_final_sha256"]):
        stem = f"training_s{seed}_I-F_v1"
        manifest = read(backup / (stem + ".manifest.json"))
        receipt = read(backup / (stem + ".verification.json"))
        assert receipt["status"] == "passed"
        assert manifest["cell"] == f"s{seed}_I-F"
        entry = next(x for x in manifest["files"] if x["path"].endswith("/final.pt"))
        assert entry["sha256"] == expected
        archive = backup / (stem + ".tar.gz")
        with tarfile.open(archive, "r:gz") as t:
            m = t.getmember(entry["path"])
            assert m.isfile() and m.size == entry["bytes"]
            f = t.extractfile(m)
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(4 * 1024**2), b""):
                h.update(chunk)
            assert h.hexdigest() == expected
        bases.append({"seed": seed, "final_sha256": expected, "bytes": entry["bytes"], "source_archive": str(archive)})

    beta = math.log1p(math.sqrt(2)) / 2
    probabilities = {str(s): 1 / (1 + math.exp(-2 * beta * s)) for s in [-4, -2, 0, 2, 4]}
    assert all(math.isclose(probabilities[str(s)] + probabilities[str(-s)], 1) for s in [-4, -2, 0, 2, 4])
    md = PLAN.read_text(encoding="utf-8")
    missing_links = []
    for dest in re.findall(r"\]\((D:/[^)\n]+)\)", md):
        target = re.sub(r":\d+$", "", dest)
        if not Path(target).exists():
            missing_links.append(dest)
    assert not missing_links, missing_links
    assert md.count("349,056") >= 1 and md.count("792") >= 1
    print(json.dumps({
        "status": "passed_static_design_only",
        "scope": "counts, CI families, point-set feasibility, analytic Gibbs probabilities, source/legacy hashes, six archived base final hashes, local links",
        "not_performed": ["new_network_implementation", "GPU_preflight", "checkpoint_deserialization_or_training_restore", "MC", "training", "formal_prediction", "new_independent_peer_review"],
        "config_sha256": sha_file(CONFIG), "plan_sha256": sha_file(PLAN),
        "checker_sha256": sha_file(Path(__file__)),
        "expected_counts": computed, "unchanged_I_frozen_files": len(frozen),
        "unchanged_review_files": review_hashes, "six_base_final_bytes_verified": bases,
        "analytic_gibbs_probability_by_neighbor_sum": probabilities,
        "ema_initial_weight_coefficient_after_4000": 0.999 ** 4000,
        "new_compute_budget_started": False, "authorized_to_start": False
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

