#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/ISM

wait_pid="${WAIT_PID:-}"
if [[ -n "${wait_pid}" ]]; then
  echo "Waiting for sampling queue process ${wait_pid}."
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 20
  done
fi

for seed in 1234 2345 3456; do
  test -s "artifacts/l128_zero_shot/formal_s${seed}/samples.npz"
  test -s "artifacts/l128_zero_shot/formal_s${seed}/metrics.json"
done

echo "All three seeds are complete; starting aggregate analysis."
/root/miniconda3/bin/python -u analyze_l128_zero_shot.py \
  --reference data/ising_l128_reference_10k.npz \
  --model artifacts/l128_zero_shot/formal_s1234/samples.npz \
  --model artifacts/l128_zero_shot/formal_s2345/samples.npz \
  --model artifacts/l128_zero_shot/formal_s3456/samples.npz \
  --l64-summary artifacts/final_l64/final_summary.json \
  --output-dir artifacts/final_l128_zero_shot \
  --bootstrap 1000 \
  --seed 20260808

echo "L=128 aggregate analysis complete."
