#!/usr/bin/env bash
set -euo pipefail

ROOT="${ISM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PYTHON_BIN:-python}"
cd "$ROOT"
root="artifacts/level1_rapid"
parent="data/level1/parents_l1024.npz"
mkdir -p "${root}/probes" "${root}/generation"

for checkpoint in \
  "${root}/T0/best_val.pt" \
  "${root}/T3/best_val.pt" \
  "${root}/Pphase/best_val.pt" \
  "${root}/Punit/best_val.pt"; do
  if [[ ! -s "${checkpoint}" ]]; then
    echo "missing checkpoint: ${checkpoint}" >&2
    exit 2
  fi
done

echo "[$(date -Is)] exact Markov-blanket probe started"
"$PY" -u probe_level1.py \
  --parent-data "${parent}" \
  --checkpoint "T0=${root}/T0/best_val.pt" \
  --checkpoint "T3=${root}/T3/best_val.pt" \
  --output-dir "${root}/probes" \
  --precision float32 \
  markov --centers 8192 --large-width 48 --batch-size 8 \
  > "${root}/probe_markov.log" 2>&1
echo "[$(date -Is)] exact Markov-blanket probe completed"

echo "[$(date -Is)] paired coordinate probe started"
"$PY" -u probe_level1.py \
  --parent-data "${parent}" \
  --checkpoint "T3=${root}/T3/best_val.pt" \
  --checkpoint "Punit=${root}/Punit/best_val.pt" \
  --checkpoint "Pphase=${root}/Pphase/best_val.pt" \
  --output-dir "${root}/probes" \
  --precision float32 \
  coordinates --samples 2048 --width 24 --strides 3 6 --batch-size 16 \
  --wrong-scale 0.5 \
  > "${root}/probe_coordinates.log" 2>&1
echo "[$(date -Is)] paired coordinate probe completed"

echo "[$(date -Is)] W=64 generation evaluation started"
"$PY" -u sample_level1.py \
  --parent-data "${parent}" \
  --checkpoint "T0=${root}/T0/best_val.pt" \
  --checkpoint "T3=${root}/T3/best_val.pt" \
  --output-dir "${root}/generation" \
  --width 64 --samples 128 --reference-samples 512 --steps 64 \
  --sensitivity-samples 32 --sensitivity-steps 128 --batch-size 2 \
  > "${root}/generation.log" 2>&1
echo "[$(date -Is)] W=64 generation evaluation completed"

echo "[$(date -Is)] final paired-bootstrap aggregation started"
"$PY" -u analyze_level1.py \
  --root "${root}" --parent-data "${parent}" --bootstrap 5000 \
  > "${root}/final_analysis.log" 2>&1
echo "[$(date -Is)] final paired-bootstrap aggregation completed"
