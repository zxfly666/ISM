#!/usr/bin/env bash
set -euo pipefail

ROOT="${ISM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PYTHON_BIN:-python}"
cd "$ROOT"
mkdir -p artifacts/level1_rapid

declare -a variants=(t0 t3 pphase punit)
declare -a outputs=(T0 T3 Pphase Punit)

echo "[$(date -Is)] Level-1 training driver started"
for index in "${!variants[@]}"; do
  variant="${variants[$index]}"
  output="${outputs[$index]}"
  config="configs/level1_${variant}.json"
  log="artifacts/level1_rapid/train_${variant}.log"
  performance="artifacts/level1_rapid/${output}/performance.json"

  if [[ -s "${performance}" ]]; then
    echo "[$(date -Is)] ${output} already complete; skipping"
    continue
  fi

  echo "[$(date -Is)] starting ${output} with ${config}"
  "$PY" -u train_level1.py --config "${config}" \
    > "${log}" 2>&1
  echo "[$(date -Is)] completed ${output}"
done

"$PY" - <<'PY'
import json
from pathlib import Path
import torch

root = Path("artifacts/level1_rapid")
names = ["T0", "T3", "Pphase", "Punit"]
summary = {}
for name in names:
    run = json.loads((root / name / "run_config.json").read_text())
    history = json.loads((root / name / "history.json").read_text())
    performance = json.loads((root / name / "performance.json").read_text())
    checkpoint = torch.load(
        root / name / "last.pt", map_location="cpu", weights_only=False
    )
    summary[name] = {
        "initialization_hash": checkpoint["initialization_hash"],
        "effective_steps": run["effective_steps"],
        "final_validation": history[-1]["validation"],
        "performance": performance,
    }
(root / "training_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
PY

echo "[$(date -Is)] Level-1 training driver completed"
