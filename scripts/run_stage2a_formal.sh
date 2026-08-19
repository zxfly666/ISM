#!/usr/bin/env bash
set -euo pipefail

ROOT="${ISM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PYTHON_BIN:-python}"
OUT="$ROOT/artifacts/stage2a"
cd "$ROOT"
mkdir -p "$OUT/logs" "$OUT/formal"
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

if [ ! -f "$OUT/stage0/frozen_sampler.json" ]; then
  "$PY" -u run_stage0_sampler.py \
    --parent-data data/level1/parents_l1024.npz \
    --checkpoint T0=artifacts/level1_rapid/T0/best_val.pt \
    --checkpoint T3=artifacts/level1_rapid/T3/best_val.pt \
    --checkpoint Punit=artifacts/level1_rapid/Punit/best_val.pt \
    --output-dir "$OUT/stage0" \
    --width 64 --samples 64 --reference-samples 1024 --batch-size 16 \
    --methods s0 s1 s2 --steps 32 64 128 256 --seeds 1234 2345 \
    --precision bfloat16 2>&1 | tee "$OUT/logs/stage0.log"
  "$PY" analyze_stage0_sampler.py \
    --manifest "$OUT/stage0/manifest.json" \
    --output "$OUT/stage0/frozen_sampler.json" 2>&1 \
    | tee "$OUT/logs/stage0_analysis.log"
fi

checkpoint_complete() {
  "$PY" -c "import sys,torch; p=torch.load(sys.argv[1],map_location='cpu',weights_only=False); sys.exit(0 if int(p['step'])>=int(sys.argv[2]) else 1)" "$1" "$2"
}

if ! checkpoint_complete "$OUT/Dense_T3_plus/last.pt" 8000 2>/dev/null; then
  "$PY" -u train_level1.py --config configs/stage2a_dense_t3_plus.json 2>&1 \
    | tee "$OUT/logs/dense_t3_plus.log"
fi
if ! checkpoint_complete "$OUT/Dense_Punit_plus/last.pt" 8000 2>/dev/null; then
  "$PY" -u train_level1.py --config configs/stage2a_dense_punit_plus.json 2>&1 \
    | tee "$OUT/logs/dense_punit_plus.log"
fi
if ! checkpoint_complete "$OUT/LG_T3/last.pt" 8000 2>/dev/null; then
  "$PY" -u train_stage2.py --config configs/stage2a_lg_t3.json --resume 2>&1 \
    | tee "$OUT/logs/lg_t3.log"
fi
if ! checkpoint_complete "$OUT/LG_Punit/last.pt" 8000 2>/dev/null; then
  "$PY" -u train_stage2.py --config configs/stage2a_lg_punit.json --resume 2>&1 \
    | tee "$OUT/logs/lg_punit.log"
fi

CHECKPOINTS=(
  --checkpoint T0=artifacts/level1_rapid/T0/best_val.pt
  --checkpoint T3=artifacts/level1_rapid/T3/best_val.pt
  --checkpoint Punit=artifacts/level1_rapid/Punit/best_val.pt
  --checkpoint Dense-T3+="$OUT/Dense_T3_plus/best_val.pt"
  --checkpoint Dense-Punit+="$OUT/Dense_Punit_plus/best_val.pt"
  --checkpoint LG-Punit="$OUT/LG_Punit/best_val.pt"
  --checkpoint LG-T3="$OUT/LG_T3/best_val.pt"
)

if [ ! -f "$OUT/formal/coordinates/summary.json" ] || \
   [ ! -f "$OUT/formal/coordinates/per_sample.npz" ]; then
  "$PY" -u probe_stage2_coordinates.py \
    --parent-data data/level1/parents_l1024.npz \
    "${CHECKPOINTS[@]}" \
    --output-dir "$OUT/formal/coordinates" \
    --samples 2048 --width 24 --physical-strides 3 6 \
    --coordinate-scales 1 1.5 2 2.5 3 4 5 6 8 10 \
    --t-grid 0.2 0.5 0.8 0.95 --batch-size 8 --precision float32 2>&1 \
    | tee "$OUT/logs/coordinates.log"
fi

if [ ! -f "$OUT/formal/markov_curve/COMPLETE" ]; then
  "$PY" -u run_stage2_markov_curve.py \
    --parent-data data/level1/parents_l1024.npz \
    "${CHECKPOINTS[@]}" \
    --output-dir "$OUT/formal/markov_curve" \
    --batch-size 16 --precision float32 2>&1 \
    | tee "$OUT/logs/markov_curve.log"
  date -Iseconds > "$OUT/formal/markov_curve/COMPLETE"
fi

SAMPLER_METHOD=$("$PY" -c "import json; print(json.load(open('$OUT/stage0/frozen_sampler.json'))['frozen_sampler']['method'])")
SAMPLER_STEPS=$("$PY" -c "import json; print(json.load(open('$OUT/stage0/frozen_sampler.json'))['frozen_sampler']['steps'])")
REFINE_SWEEPS=$("$PY" -c "import json; print(json.load(open('$OUT/stage0/frozen_sampler.json'))['frozen_sampler']['refinement_sweeps'])")

if [ ! -f "$OUT/formal/generation/metrics.json" ] || \
   [ ! -f "$OUT/formal/generation/samples_and_correlations.npz" ]; then
  "$PY" -u sample_level1.py \
    --parent-data data/level1/parents_l1024.npz \
    "${CHECKPOINTS[@]}" \
    --output-dir "$OUT/formal/generation" \
    --width 64 --samples 256 --reference-samples 1024 --batch-size 16 \
    --steps "$SAMPLER_STEPS" --sampler-method "$SAMPLER_METHOD" \
    --refinement-sweeps "$REFINE_SWEEPS" --sensitivity-samples 0 2>&1 \
    | tee "$OUT/logs/generation.log"
fi

if [ ! -f "$OUT/formal/final/stage2a_decision.json" ]; then
  "$PY" -u analyze_stage2a.py \
    "${CHECKPOINTS[@]}" \
    --coordinate-summary "$OUT/formal/coordinates/summary.json" \
    --coordinate-data "$OUT/formal/coordinates/per_sample.npz" \
    --markov-root "$OUT/formal/markov_curve" \
    --generation-metrics "$OUT/formal/generation/metrics.json" \
    --generation-data "$OUT/formal/generation/samples_and_correlations.npz" \
    --sampler-selection "$OUT/stage0/frozen_sampler.json" \
    --output-dir "$OUT/formal/final" 2>&1 | tee "$OUT/logs/analysis.log"
fi

nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader > "$OUT/gpu_after.csv"
date -Iseconds > "$OUT/COMPLETE"
