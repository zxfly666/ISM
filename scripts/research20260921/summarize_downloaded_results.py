"""Read downloaded completed-run records; emit descriptive, not inferential, summaries."""
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "artifacts/geometry_alignment_20260921_remote"
results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(DATA.glob("s*_complete.json"))]
assert len(results) == 18, len(results)
output = {}
for arm in "ABC":
    chosen = [r for r in results if r["arm"] == arm]
    assert len(chosen) == 6
    out = {}
    out["primary_ce"] = mean(r["conditional"]["mean_ce"] for r in chosen)
    out["per_task_ce"] = {task: mean(r["conditional"]["per_task"][task] for r in chosen)
                          for task in chosen[0]["conditional"]["per_task"]}
    out["generation"] = {}
    for geom in chosen[0]["generation"]:
        gs = [r["generation"][geom] for r in chosen]
        out["generation"][geom] = {
            "mean_seed_nrmse": {k: mean(g["errors"][k]["nrmse"] for g in gs) for k in gs[0]["errors"]},
            "seed_nrmse": {k: [g["errors"][k]["nrmse"] for g in gs] for k in gs[0]["errors"]},
            **{k: mean(g[k] for g in gs) for k in ("model_m2", "mc_m2", "model_abs_m", "mc_abs_m")},
        }
    val_by_step = {}
    geometry_by_family = {"uniform": [], "same_summary": []}
    markov_by_width = {16: [], 64: []}
    for r in chosen:
        stem = f"s{r['seed']}_{arm}"
        records = [json.loads(line) for line in (DATA / (stem + "_train.jsonl")).read_text().splitlines() if line.strip()]
        for row in records:
            if "validation" in row:
                val_by_step.setdefault(row["step"], []).append(row["validation"]["mean_ce"])
        with (DATA / (stem + "_geometry_probe.csv")).open(newline="") as f:
            rows = list(csv.DictReader(f))
        for family in geometry_by_family:
            matched = [row for row in rows if row["layout"].startswith(family)]
            geometry_by_family[family].append(sum(float(row["event_mass"]) * float(row["kl"]) for row in matched) / len(set(row["layout"] for row in matched)))
        with (DATA / (stem + "_natural_markov.csv")).open(newline="") as f:
            rows = list(csv.DictReader(f))
        for width in markov_by_width:
            markov_by_width[width].append(mean(float(row["kl"]) for row in rows if int(row["width"]) == width))
    out["validation_mean"] = {s: mean(v) for s,v in val_by_step.items() if s in (1000,2000,4000,8000,16000,20000,24000)}
    out["geometry_probe_kl"] = {k: {"mean": mean(v), "per_seed": v} for k,v in geometry_by_family.items()}
    out["markov_kl_by_width"] = {k: mean(v) for k,v in markov_by_width.items()}
    output[arm] = out
print(json.dumps(output, ensure_ascii=False, indent=2))
