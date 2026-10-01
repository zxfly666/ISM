"""Read-only CPU monitoring and post-exit byte closure for this evaluation stage."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import run_evaluation as run

c, mg, OUT, EXPORT, ROOT = run.c, run.mg, run.OUT, run.EXPORT, run.ROOT


def processes():
    groups = set()
    for name, key in [("formal_started.json", "pgid"), ("reference_process.json", "pgid")]:
        if (OUT / name).exists():
            groups.add(int(c.read(OUT / name)[key]))
    result = []
    if groups:
        lines = subprocess.run(["ps", "-eo", "pid,ppid,pgid,etime,pcpu,comm,args"],
                               capture_output=True, text=True, check=True).stdout.splitlines()[1:]
        for line in lines:
            fields = line.split(None, 6)
            if len(fields) >= 6 and int(fields[2]) in groups:
                result.append(line)
    return result


def monitor():
    now = time.time()
    protocol = c.read(OUT / "run_protocol.json")
    c.check_sources(protocol["sources"], enforce_deadline=False)
    for path, expected in protocol["locked_checkpoints"].items():
        assert c.sha(ROOT / path) == expected, path
    formal = list(OUT.glob("evaluation/*/generation/*/shard_*.npz"))
    phase0 = list(OUT.glob("phase0/*/floor*/shard_*.npz"))
    prediction = list(OUT.glob("evaluation/*/conditional/*.npz"))
    learning = list(OUT.glob("evaluation/*/learning/*.npz"))
    old_pred = list(OUT.glob("phase0/*/conditional/*.npz"))
    states = {name: c.read(OUT / name) for name in ["status.json", "formal_started.json", "reference_process.json",
        "reference/qa.json", "reference/complete.json", "launch_forecast.json", "final_summary.json",
        "remote_export_complete.json", "failure.json", "reference_failure.json"] if (OUT / name).exists()}
    result = dict(time=now, stage=run.STAGE, deadline=run.DEADLINE,
        compute_deadline=run.COMPUTE_DEADLINE, remaining_seconds=run.DEADLINE-now,
        resources=c.resources(), processes=processes(), states=states,
        frozen_sources="passed", frozen_checkpoint_files=len(protocol["locked_checkpoints"]),
        formal_images=len(formal)*16, expected_formal_images=6912,
        phase0_images=len(phase0)*16, expected_phase0_images=384,
        predictions=len(prediction)+len(learning)+len(old_pred), expected_predictions=546,
        complete_model_cells=len(list(OUT.glob("evaluation/*/complete.json"))),
        scientific_result_available=(OUT / "analysis/summary.json").exists(),
        training_already_complete_not_restarted=True)
    rows = []
    path = OUT / "run.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("stage") == "generation_shard":
                rows.append(row)
    if rows:
        result["generation_latest"] = rows[-1]
        result["generation_measured_seconds"] = sum(r["seconds"] for r in rows)
        result["generation_remaining_shards"] = 456-len(rows)
    destination = EXPORT / ("monitor_" + time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + ".json")
    c.write(destination, result)
    return dict(path=str(destination), **result)


def closure():
    assert (OUT / "remote_export_complete.json").exists()
    assert not processes(), "Cannot close still-active scientific process groups"
    science = c.read(OUT / "science_manifest.json")
    c.check_sources(c.read(OUT / "run_protocol.json")["sources"], enforce_deadline=False)
    for row in science["files"]:
        path = ROOT / row["path"]
        assert path.stat().st_size == row["bytes"] and c.sha(path) == row["sha256"], path
    late = [OUT / name for name in ["formal.stdout", "reference.stdout", "status.json", "run.jsonl",
            "remote_export_complete.json", "science_manifest.json"] if (OUT / name).exists()]
    combined = {r["path"]: r for r in science["files"]}
    combined.update({r["path"]: r for r in mg.record_paths(late)})
    destination = EXPORT / "closure_manifest_v1.json"
    c.write(destination, dict(files=list(combined.values()), time=time.time(),
        status="remote_bytes_closed_needs_local_union_visual_review_report",
        inherited_science_manifest_sha256=c.sha(OUT / "science_manifest.json"), no_scientific_recomputation=True))
    return mg.archive(ROOT, EXPORT, "evaluation_administrative_closure_v1", late+[destination])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["monitor", "closure"])
    args = parser.parse_args(); run.configure()
    print(json.dumps(monitor() if args.mode == "monitor" else closure(), ensure_ascii=False))
