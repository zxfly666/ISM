"""CPU-only plan arithmetic and provenance checks; never imports torch or runs science.

An optional --output creates a NEW budget receipt with exclusive creation.
No SSH, GPU, training, MC sampling, generation or changes to old experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/research_reboot_20260921"
STEM = "ENDPOINT_MASK_SAMPLER_DISENTANGLEMENT"
CONFIG = DOCS / f"{STEM}_CONFIG_20261001.json"
PLAN = DOCS / f"{STEM}_PLAN_20261001_ZH.md"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def project_source(path: Path):
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}


def inspect_archive_json(archive: Path, member: str, manifest: Path):
    # Verify the exact JSON member against its already existing backup manifest.
    # This is not a repeat verification of every member or a restore of a model.
    records = read(manifest)["files"]
    item = next(v for v in records if v["path"] == member)
    with tarfile.open(archive, "r:gz") as tar:
        payload = tar.extractfile(member).read()
    assert len(payload) == item["bytes"]
    assert hashlib.sha256(payload).hexdigest() == item["sha256"]
    return json.loads(payload), {"archive_basename": archive.name, "member": member, "member_sha256_verified": item["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    c = read(CONFIG)
    assert c["execution_authorized"] is False
    assert c["lifecycle"].startswith("design_only")
    seeds = c["base"]["seeds"]
    train = c["training"]
    nseed, narms = len(seeds), len(train["arms"])
    models = nseed * narms
    updates = models * train["additional_updates_per_branch"]
    assert train["additional_updates_per_branch"] % train["schedule_period"] == 0
    assert train["additional_updates_per_branch"] % train["block_updates"] == 0
    for arm in train["arms"].values():
        assert math.isclose(arm["ordinary_fraction"] + arm["sparse_visible_fraction"] + arm["late_mask_fraction"], 1)
        assert (arm["ordinary_fraction"] * 4).is_integer()
    masks = {
        str(w): [m for m in train["late_mask_candidates"] if m / (w * w) <= train["late_mask_max_fraction"]]
        for w in train["widths"]
    }
    assert masks == {"16": [1, 2, 4], "24": [1, 2, 4, 8], "32": [1, 2, 4, 8, 16], "48": [1, 2, 4, 8, 16, 32]}
    assert all(train["tokens_per_update"] % (w * w) == 0 for w in train["widths"])
    for name in ("monotone-256", "reveal192-repair64"):
        assert sum(c["samplers"][name].values()) == 256
    formal = {w: models * 2 * n for w, n in c["generation"]["images_per_lineage_arm_sampler"].items()}
    phase0 = c["phase0"]
    diag = nseed * len(phase0["clocks"]) * phase0["frozen_images_per_lineage_clock"]
    combined = {"96": formal["96"] + diag, "48": formal["48"]}
    prefixes = models * sum(c["generation"]["images_per_lineage_arm_sampler"].values())
    e = c["evaluation"]
    conditional = {
        "exact_local": models * e["exact_local"]["parents"] * len(e["exact_local"]["widths"]) * len(e["exact_local"]["mask_counts"]),
        "neighbor_stress": models * e["neighbor_stress"]["backgrounds"] * e["neighbor_stress"]["patterns"] * len(e["neighbor_stress"]["widths"]),
        "retention": models * e["retention"]["parents"] * len(e["retention"]["tasks"]),
        "learning": models * e["learning"]["parents"] * len(e["learning"]["checkpoints"]) * len(e["learning"]["mask_counts"]),
        "phase0": nseed * phase0["conditional_parents"] * len(phase0["conditional_widths"]) * len(phase0["conditional_mask_counts"]) * len(phase0["clocks"]),
    }
    assert conditional == {"exact_local": 36864, "neighbor_stress": 2304, "retention": 13824, "learning": 27648, "phase0": 1536}
    assert (models, updates, sum(formal.values()), diag, prefixes) == (18, 144000, 6912, 384, 3456)
    assert all(v % 16 == 0 for v in combined.values())
    counts = {
        "independent_training_lineages": nseed,
        "continued_models": models,
        "new_fresh_models": 0,
        "updates": updates,
        "input_tokens": updates * train["tokens_per_update"],
        "early_EMA_snapshots": models * len(train["diagnostic_ema_updates"]),
        "formal_generation": formal,
        "diagnostic_generation_w96": diag,
        "all_generated_final_images": sum(combined.values()),
        "prefix_images_not_independent_replicates": prefixes,
        "oracle_images_not_neural_results": prefixes,
        "shards16_by_width_including_phase0": {w: n // 16 for w, n in combined.items()},
        "image_network_forwards_generation_only": sum(combined.values()) * 256,
        "batch16_network_calls_generation_only": sum(combined.values()) // 16 * 256,
        "conditional_image_inputs": conditional,
        "total_conditional_image_inputs": sum(conditional.values()),
        "new_MC_parents": c["reference"]["chains"] * c["reference"]["parents_per_chain"],
    }
    hist = c["budget"]["historical"]
    gate_path = ROOT / "artifacts/geometry_identification_20260926_remote/budget_gate.json"
    gate = read(gate_path)
    assert gate["raw"]["training"] == hist["training_forecast_seconds"]
    assert gate["shard16_seconds"] == hist["w96_shard16_seconds"]
    scenarios = {}
    for name, s in c["budget"]["scenarios"].items():
        train_s = s.get("training_seconds_per_update", hist["training_forecast_seconds"] / hist["training_updates"] * s.get("training_multiplier_on_historical", 1))
        shard96 = s.get("w96_shard16_seconds", hist["w96_shard16_seconds"] * s.get("generation_multiplier_on_historical", 1))
        shard48 = s.get("w48_shard16_seconds", shard96 * s.get("w48_to_w96_ratio_assumption", 1))
        serial = {
            "preflight": s["preflight_seconds"],
            "training_and_MC_critical_path": max(updates * train_s, s["mc_seconds"]),
            "w96_generation": combined["96"] / 16 * shard96,
            "w48_generation": combined["48"] / 16 * shard48,
            "conditional": s["conditional_seconds"],
            "analysis_and_RG": s["analysis_RG_seconds"],
            "io": s["io_seconds"],
            "backup_and_visual": s["backup_visual_seconds"],
        }
        total = math.fsum(serial.values())
        scenarios[name] = {
            "component_seconds": serial,
            "training_seconds": updates * train_s,
            "parallel_MC_seconds": s["mc_seconds"],
            "total_seconds": total,
            "total_hours": total / 3600,
            "slack_to_proposed_12h_seconds": c["budget"]["proposed_hard_wall_seconds"] - total,
            "would_meet_10_5h_forecast_number_only": total <= c["budget"]["proposed_launch_total_forecast_max_seconds"],
            "actual_launch_approved": False,
        }
    sources = [project_source(p) for p in [CONFIG, PLAN, gate_path,
        ROOT / "scripts/research20260926/identification_data.py",
        ROOT / "scripts/research20260926/identification_training.py",
        ROOT / "scripts/research20260926/identification_statistics.py",
        ROOT / "experiments/fine-geometry-identification/evidence/run_protocol.json",
        Path(__file__).resolve()]]
    receipts, generation_records, base_metadata = [], [], []
    if args.backup_dir:
        for seed in seeds:
            prefix = f"evaluation_s{seed}_I-F_v1"
            member = f"artifacts/geometry_identification_20260926/evaluation/s{seed}_I-F/generation/complete.json"
            record, source = inspect_archive_json(args.backup_dir / f"{prefix}.tar.gz", member, args.backup_dir / f"{prefix}.manifest.json")
            assert record["images"] == 128 and record["status"] == "complete"
            receipts.append(source)
            generation_records.append({"lineage": seed, "images": record["images"], "sampling_seconds": record["sampling_seconds"]})
            manifest = read(args.backup_dir / f"training_s{seed}_I-F_v1.manifest.json")
            final = next(v for v in manifest["files"] if v["path"].endswith("/final.pt"))
            base_metadata.append({"lineage": seed, "manifest_basename": f"training_s{seed}_I-F_v1.manifest.json", **final})
        reference, source = inspect_archive_json(args.backup_dir / "reference_complete_v1.tar.gz", "artifacts/geometry_identification_20260926/reference/complete.json", args.backup_dir / "reference_complete_v1.manifest.json")
        assert reference["elapsed_seconds"] == hist["mc2048_seconds"]
        receipts.append(source)
    # Markdown local link existence only. This does not claim a rendered visual audit.
    links = re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", PLAN.read_text(encoding="utf-8"))
    pending_name = f"{STEM}_BUDGET_20261001.json"
    checked, pending = [], []
    for target in links:
        if "://" in target or target.startswith("#"):
            continue
        path = (PLAN.parent / target.split("#")[0]).resolve()
        if path.name == pending_name and not path.exists():
            pending.append(target)
        else:
            assert path.is_file(), target
            checked.append(target)
    result = {
        "experiment_id": c["experiment_id"],
        "status": "CPU_design_arithmetic_passed_not_a_GPU_preflight_or_scientific_result",
        "new_experiment_launched": False,
        "counts": counts,
        "allowed_late_mask_counts": masks,
        "scenarios": scenarios,
        "historical_sampling_completed_records": generation_records,
        "historical_mean_seconds_per_w96_image": statistics.mean(r["sampling_seconds"] / r["images"] for r in generation_records) if generation_records else None,
        "archive_JSON_members_reverified": receipts,
        "base_final_hashes_from_existing_manifests_not_reloaded_in_this_design_check": base_metadata,
        "source_snapshot": sources,
        "local_links_checked": checked,
        "budget_link_pending_creation": pending,
        "limits": ["Forecasts are scenarios, not measured current-machine speed or bounds.", "Conservative scenario is below12h but above10.5h launch gate: live preflight must resolve this, no automatic launch.", "No fresh model training, GPU evaluation, MC generation, remote access or publishing occurred.", "Plan/config/source checks are not implementation correctness tests."]
    }
    if args.output:
        resolved = args.output.resolve()
        assert resolved.parent == DOCS.resolve(), "Receipt must be inside the established project docs folder"
        with resolved.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"status": result["status"], "counts": counts, "hours": {k: v["total_hours"] for k, v in scenarios.items()}, "links_ok": len(checked), "receipt_created": bool(args.output)}, indent=2))


if __name__ == "__main__":
    main()
