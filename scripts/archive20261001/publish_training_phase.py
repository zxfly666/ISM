"""Curate verified training-stage evidence; no training, GPU or scientific analysis.

Inputs are custodian-held verified archives and small management receipts. Originals
remain immutable. Text privacy edits are logged with original and published SHA.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/archive20260930"))
from curate import public_text

STUDY = "endpoint_mask_sampler_disentanglement_20261001"
SLUG = "late-mask-coverage-and-committed-spin-correction"
LOCAL = ROOT / "artifacts" / (STUDY + "_remote")
PUBLIC = ROOT / "experiments" / SLUG
DOC = ROOT / "docs/research_reboot_20260921"
OUT_PREFIX = "artifacts/" + STUDY + "/"
EXPORT_PREFIX = "artifacts/" + STUDY + "_exports/"


def sanitize(raw, markdown=False):
    out, edits = public_text(raw, markdown)
    text = out.decode("utf-8")
    for pattern, replacement in [(r"(?i)D:(?:/|\\+)ISM_research_backups", "<CUSTODIAN_BACKUPS>"),
                                 (r"(?i)D:(?:/|\\+)ISM", "<PROJECT_ROOT>")]:
        text, count = re.subn(pattern, lambda _: replacement, text)
        if count:
            edits.append(dict(kind="Windows project/backup root redaction", count=count))
    return text.encode("utf-8"), edits


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    union = read(LOCAL / "training_phase_independent_union_v1.json")
    assert union["status"] == "passed" and not union["missing"]
    verification = read(LOCAL / "training_phase_complete_v1.verification.json")
    assert verification["status"] == "passed"
    manifest = read(LOCAL / "training_phase_complete_v1.manifest.json")
    assert sha((LOCAL / "training_phase_complete_v1.tar.gz").read_bytes()) == manifest["archive_sha256"]
    assert sha((LOCAL / "training_phase_complete_v1.manifest.json").read_bytes()) == verification["manifest_sha256"]
    records = []
    artifacts = {}

    def publish(raw, source, relative, edits=None):
        target = PUBLIC / relative
        edited = list(edits or [])
        out, sanitized = sanitize(raw, target.suffix == ".md")
        edited += sanitized
        if target.suffix == ".json":
            json.loads(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(out)
        records.append(dict(source=source, published=target.relative_to(ROOT / "experiments").as_posix(),
                            original_bytes=len(raw), original_sha256=sha(raw), bytes=len(out), sha256=sha(out),
                            transformations=edited))
        return target

    expected = {r["path"]: r for r in manifest["files"]}
    allowed_root = {"budget.json", "effective_config.json", "final_lock.json", "formal_started.json",
                    "run_protocol.json", "training_phase_complete.json", "preflight_failure.json",
                    "preflight_failure_v2.json", "preflight_failure_v3.json", "preflight_blocked_precision.json"}
    allowed_admin = {"training_phase_full_audit_v1.json", "training_checkpoint_metadata_v1.json",
                     "training_phase_science_manifest_v1.json"}
    with tarfile.open(LOCAL / "training_phase_complete_v1.tar.gz", "r:gz") as tf:
        for entry in tf:
            if not entry.isfile():
                raise ValueError(entry.name)
            name = entry.name
            relative = name.removeprefix(OUT_PREFIX)
            selected = name.startswith(OUT_PREFIX) and (relative in allowed_root or
                       relative.startswith("training/") and relative.endswith(("initial.json", "complete.json")))
            admin = name.startswith(EXPORT_PREFIX) and Path(name).name in allowed_admin
            if not (selected or admin):
                continue
            raw = tf.extractfile(entry).read()
            row = expected[name]
            assert len(raw) == row["bytes"] and sha(raw) == row["sha256"]
            if admin:
                dest = "evidence/" + Path(name).name
            elif relative.startswith("preflight"):
                dest = "technical-history/" + relative
            else:
                dest = "evidence/" + relative
            publish(raw, name, dest)
            artifacts[Path(name).name] = json.loads(raw)
    assert artifacts["training_phase_full_audit_v1.json"]["input_reconstruction"] is True
    assert len(artifacts["training_checkpoint_metadata_v1.json"]["checkpoints"]) == 18

    for name in ["precision.json", "restore_diagnosis_v1.json", "restore_diagnosis_v2.json",
                 "restore_diagnosis_v4.json", "fp32_revision_feasibility_v1_result.json",
                 "cpu_prepare_v5_checks.json", "software_environment.json"]:
        publish((LOCAL / name).read_bytes(), "custodian-local-observation/" + name, "technical-history/" + name)
    for name in ["training_phase_complete_v1.verification.json", "training_phase_independent_union_v1.json",
                 "training_phase_remote_archive_v1.json", "training_phase_complete_v1.manifest.json"]:
        publish((LOCAL / name).read_bytes(), "custodian-training-closure/" + name, "evidence/" + name)

    # Scientific design remains full; remove private-discussion source/timestamp
    # mapping, not design variables, outcomes, thresholds, counts or failed gates.
    plan_path = DOC / "ENDPOINT_MASK_SAMPLER_DISENTANGLEMENT_PLAN_20261001_ZH.md"
    original = plan_path.read_bytes()
    text = original.decode("utf-8-sig")
    start = text.index("## 1. 与导师建议的对应，以及必须修正的前提")
    end = text.index("### 已核对的代码事实", start)
    text = text[:start] + "## 1. 科学动机与已核对的实现前提\n\n本公开副本移除私人讨论来源与时间定位表；训练覆盖、采样纠错、局部与长程统计及真实粗粒化的科学检验完整保留。\n\n" + text[end:]
    text = text.replace("[.002,1]我还是", "[.002,1]")
    target = publish(text.encode("utf-8"), plan_path.relative_to(ROOT).as_posix(), "protocol/SCIENTIFIC_PLAN_ZH.md",
                     [{"kind": "private-discussion provenance/timestamp table omitted; scientific design retained"},
                      {"kind": "remove stray non-scientific text in E arm table: 我还是"}])
    records[-1]["original_bytes"] = len(original)
    records[-1]["original_sha256"] = sha(original)
    for name in ["ENDPOINT_MASK_SAMPLER_DISENTANGLEMENT_CONFIG_20261001.json",
                 "ENDPOINT_MASK_SAMPLER_DISENTANGLEMENT_BUDGET_20261001.json"]:
        publish((DOC / name).read_bytes(), "docs/research_reboot_20260921/" + name, "protocol/" + name)

    completion = artifacts["training_phase_complete.json"]
    checkpoints = artifacts["training_checkpoint_metadata_v1.json"]["checkpoints"]
    status = dict(experiment_id=STUDY, public_name="Late-mask coverage and committed-spin correction",
                  historical_alias="A/L/E (training arms)", date="2026-10-01",
                  status="training_complete_evaluation_pending", trained_models=18, fresh_models=0,
                  inherited_training_lineages=6, updates=144000, tokens=2654208000,
                  final_checkpoints=18, diagnostic_ema_snapshots=36,
                  formal_new_MC_parents=0, formal_generated_images=0, formal_prediction_files=0,
                  primary_effects={"P1": None, "P2": None}, primary_decision="not_evaluated",
                  training_started=completion["started"], training_completed=completion["completed"],
                  training_seconds=completion["seconds"], local_backup_verified=union["time"],
                  backup_union_paths=union["science_paths"], source="evidence/training_phase_complete.json",
                  audit="evidence/training_phase_full_audit_v1.json")
    write(PUBLIC / "evidence/phase_status.json", status)
    # This table contains training integrity identities, not efficacy comparisons.
    write(PUBLIC / "evidence/branch_inventory.json", [dict(seed=x["seed"], arm=x["arm"],
          step=x["step"], global_step=x["global_step"], final_sha256=x["final_sha256"],
          snapshots=x["snapshots"]) for x in checkpoints])

    science = artifacts["training_phase_science_manifest_v1.json"]
    package_paths = {r["path"]: "training_phase_complete_v1.tar.gz" for r in manifest["files"]}
    archive_rows = []
    receipt_rows = []
    for filename in ["training_phase_complete_v1.verification.json", "training_phase_initial_v1.verification.json",
                     "preflight_blocked_precision_v1.verification.json"] + [f"base_s{s}_reverified_v1.verification.json" for s in range(92601,92607)]:
        receipt_path = LOCAL / filename
        r = read(receipt_path)
        mp = Path(r["manifest_path"])
        mm = read(mp)
        for row in mm["files"]:
            package_paths.setdefault(row["path"], r["archive"])
        archive_rows.append(dict(archive=r["archive"], archive_bytes=r["archive_bytes"],
                                 archive_sha256=r["archive_sha256"], manifest_sha256=r["manifest_sha256"], members=r["members"]))
        receipt_rows.append(dict(receipt=filename, original_sha256=sha(receipt_path.read_bytes()), result=r))
    pub_by_hash = {r["original_sha256"]: r["published"] for r in records}
    withheld = []
    for row in science["files"]:
        public = pub_by_hash.get(row["sha256"])
        if row["path"].endswith(".py") and (ROOT / row["path"]).is_file():
            public = "../" + row["path"]
        withheld.append(dict(**row, archive=package_paths.get(row["path"]), public_copy=public,
                             availability="public evidence/code copy; see provenance" if public else "custodian-held; not a public download"))
    write(PUBLIC / "withheld-manifest.json", dict(files=withheld, archives=archive_rows,
          excludes=science["exclusions"], note="Manifest-only large weights/data/logs. Request archive/member/SHA through repository issue; no public hosting guarantee."))
    raw, _ = sanitize(json.dumps(receipt_rows, ensure_ascii=False).encode("utf-8"))
    write(PUBLIC / "backup-verification.json", json.loads(raw))
    write(PUBLIC / "provenance.json", dict(policy="Frozen science unchanged. Derived stage status is not an efficacy result.", files=records))
    evidence = ["# 训练阶段证据索引", "", "原始科学文件未改；私人来源信息仅在公开副本中移除。", "",
                "| 公开文件 | 原来源 | SHA（完整值见 provenance） |", "|---|---|---|"]
    for r in records:
        rel = r["published"].removeprefix(SLUG + "/")
        evidence.append(f"| [{rel}]({rel}) | `{r['source']}` | `{r['original_sha256'][:12]}` |")
    evidence += ["", "派生状态：[phase_status](evidence/phase_status.json)；完整分支身份：[branch_inventory](evidence/branch_inventory.json)。",
                 "原科学入口：[training-only driver](../../scripts/research20261001/run_training_phase.py)。",
                 "CPU审计：[audit_training](../../scripts/research20261001/endpoint_management.py)，不要调用要求全轮评价的 audit_all。",
                 "科学源码均保持原路径与冻结SHA；计划的 sampling/evaluation/statistics 代码存在不代表已经正式执行。", ""]
    (PUBLIC / "EVIDENCE.md").write_text("\n".join(evidence), encoding="utf-8")

    registry_path = ROOT / "experiments/registry.json"
    registry = [x for x in read(registry_path) if x["slug"] != SLUG]
    registry.append(dict(slug=SLUG, public_name=status["public_name"], historical_alias=status["historical_alias"],
                         experiment_id=STUDY, date="2026-10-01", training_kind="continued: 18; six I-F lineages",
                         parents=["fine-geometry-identification/I-F"], new_reference=False, new_generation=0,
                         code=["scripts/research20261001/run_training_phase.py", "scripts/research20261001/endpoint_training.py"],
                         formal_completed=False, training_completed=True, status=status["status"],
                         published_files=len(records), published_bytes=sum(r["bytes"] for r in records)))
    write(registry_path, registry)
    source_path = ROOT / "experiments/source-manifest.json"
    sources = {x["path"]: x for x in read(source_path)}
    for p in sorted((ROOT / "scripts/research20261001").glob("*.py")):
        rel = p.relative_to(ROOT).as_posix()
        assert sha(p.read_bytes()) == artifacts["run_protocol.json"]["sources"][rel]
        sources[rel] = dict(path=rel, bytes=p.stat().st_size, sha256=sha(p.read_bytes()))
    write(source_path, list(sources.values()))
    allow_path = ROOT / "experiments/publication-allowlist.json"
    allow = read(allow_path)
    allow["scientific_sources"] = list(sources)
    for name in ["ENDPOINT_MASK_SAMPLER_DISENTANGLEMENT_CONFIG_20261001.json",
                 "ENDPOINT_MASK_SAMPLER_DISENTANGLEMENT_BUDGET_20261001.json"]:
        rel = "docs/research_reboot_20260921/" + name
        if rel not in allow["effective_config_sources"]:
            allow["effective_config_sources"].append(rel)
    write(allow_path, allow)
    print(json.dumps(dict(status="curated", evidence_files=len(records), scientific_paths=len(withheld),
                          training_seconds=completion["seconds"], evaluation_pending=True)))


if __name__ == "__main__":
    main()
