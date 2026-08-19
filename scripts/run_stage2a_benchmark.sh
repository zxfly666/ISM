#!/usr/bin/env bash
set -euo pipefail

ROOT="${ISM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PYTHON_BIN:-python}"
cd "$ROOT"
mkdir -p artifacts/stage2a/benchmark

"$PY" -m unittest discover -s tests -v 2>&1 | tee artifacts/stage2a/benchmark/tests.log
"$PY" -u train_stage2.py \
  --config configs/stage2a_benchmark_lg_t3.json 2>&1 \
  | tee artifacts/stage2a/benchmark/train_lg_t3.log

"$PY" -u run_stage0_sampler.py \
  --parent-data data/level1/parents_l1024.npz \
  --checkpoint T3=artifacts/level1_rapid/T3/best_val.pt \
  --output-dir artifacts/stage2a/benchmark/sampler \
  --width 64 --samples 4 --reference-samples 32 --batch-size 2 \
  --methods s0 s1 s2 --steps 32 --seeds 1234 --precision bfloat16 2>&1 \
  | tee artifacts/stage2a/benchmark/sampler.log

nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader > artifacts/stage2a/benchmark/gpu_after.csv
date -Iseconds > artifacts/stage2a/benchmark/COMPLETE
