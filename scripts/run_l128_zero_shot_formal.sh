#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/ISM}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SAMPLES="${SAMPLES:-1536}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-128}"

cd "$PROJECT_DIR"

if [[ -n "${WAIT_PID:-}" ]]; then
  echo "Waiting for process $WAIT_PID before starting queued seeds."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
fi

if [[ "$#" -eq 0 ]]; then
  set -- 1234 2345 3456
fi

checkpoint="artifacts/pilot_l64/best.pt"
reference="data/ising_l128_reference_10k.npz"

if [[ ! -f "$checkpoint" ]]; then
  echo "Missing checkpoint: $checkpoint" >&2
  exit 1
fi
if [[ ! -f "$reference" ]]; then
  echo "Missing reference ensemble: $reference" >&2
  exit 1
fi

for seed in "$@"; do
  output_dir="artifacts/l128_zero_shot/formal_s${seed}"
  if [[ -f "$output_dir/metrics.json" && -f "$output_dir/samples.npz" ]]; then
    echo "Seed $seed already complete; skipping."
    continue
  fi

  echo "Starting L=128 zero-shot seed $seed"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" -u sample_and_evaluate.py \
    --checkpoint "$checkpoint" \
    --output-dir "$output_dir" \
    --lattice-size 128 \
    --samples "$SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --steps "$STEPS" \
    --temperature 1.0 \
    --sampler ancestral \
    --reference-data "$reference" \
    --reference-split test \
    --device cuda \
    --seed "$seed"
  echo "Completed seed $seed"
done
