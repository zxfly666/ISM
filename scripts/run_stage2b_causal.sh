#!/usr/bin/env bash
set -euo pipefail

ROOT="${ISM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PYTHON_BIN:-python}"
OUT="$ROOT/artifacts/stage2b_causal"
cd "$ROOT"
mkdir -p "$OUT/logs" "$OUT/probe" "$OUT/final"
date -Iseconds > "$OUT/STARTED"

record_exit_status() {
  rc=$?
  {
    printf 'exit_code=%s\n' "$rc"
    printf 'time=%s\n' "$(date -Iseconds)"
  } > "$OUT/EXIT_STATUS"
}
trap record_exit_status EXIT

"$PY" -m unittest discover -s tests -v 2>&1 | tee "$OUT/logs/tests.log"

checkpoint_complete() {
  "$PY" -c "import sys,torch; p=torch.load(sys.argv[1],map_location='cpu',weights_only=False); sys.exit(0 if int(p['step'])>=int(sys.argv[2]) else 1)" "$1" "$2"
}

if ! checkpoint_complete "$OUT/LG_Gap_Unit/last.pt" 8000 2>/dev/null; then
  "$PY" -u train_stage2.py --config configs/stage2b_gap_unit.json --resume 2>&1 \
    | tee "$OUT/logs/gap_unit.log"
fi
if ! checkpoint_complete "$OUT/LG_U_RandPE/last.pt" 8000 2>/dev/null; then
  "$PY" -u train_stage2.py --config configs/stage2b_randpe_control.json --resume 2>&1 \
    | tee "$OUT/logs/randpe.log"
fi
if ! checkpoint_complete "$OUT/LG_Gap_Matched/last.pt" 8000 2>/dev/null; then
  "$PY" -u train_stage2.py --config configs/stage2b_gap_matched.json --resume 2>&1 \
    | tee "$OUT/logs/gap_matched.log"
fi

"$PY" -c "import torch; paths=['$OUT/LG_Gap_Unit/last.pt','$OUT/LG_U_RandPE/last.pt','$OUT/LG_Gap_Matched/last.pt']; payload=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]; hashes=[p['initialization_hash'] for p in payload]; assert len(set(hashes))==1, hashes; assert all(int(p['step'])==8000 for p in payload); print('paired initialization:',hashes[0])" \
  2>&1 | tee "$OUT/logs/pairing_audit.log"

CHECKPOINTS=(
  --checkpoint LG-T3=artifacts/stage2a/LG_T3/last.pt
  --checkpoint LG-Punit=artifacts/stage2a/LG_Punit/last.pt
  --checkpoint Gap-Unit="$OUT/LG_Gap_Unit/last.pt"
  --checkpoint U-RandPE="$OUT/LG_U_RandPE/last.pt"
  --checkpoint Gap-Matched="$OUT/LG_Gap_Matched/last.pt"
)

if [ ! -f "$OUT/probe/summary.json" ] || [ ! -f "$OUT/probe/per_sample.npz" ]; then
  "$PY" -u probe_stage2b_causal.py \
    --parent-data data/level1/parents_l1024.npz \
    "${CHECKPOINTS[@]}" \
    --output-dir "$OUT/probe" \
    --samples 512 --batch-size 16 --precision float32 2>&1 \
    | tee "$OUT/logs/probe.log"
fi

if [ ! -f "$OUT/final/stage2b_causal_decision.json" ]; then
  "$PY" -u analyze_stage2b_causal.py \
    --probe-data "$OUT/probe/per_sample.npz" \
    --probe-summary "$OUT/probe/summary.json" \
    --output-dir "$OUT/final" 2>&1 | tee "$OUT/logs/analysis.log"
fi

nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader > "$OUT/gpu_after.csv"
date -Iseconds > "$OUT/COMPLETE"
