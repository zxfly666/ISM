# Dense size × clock geometry study (2026-09-30)

This directory implements the preregistered 2×2 experiment. The original dense
model is imported unchanged from `scripts/research20260927/intervention_model.py`.
Six fresh paired training seeds are the independent comparison units. Exact
boundary-pattern records, symmetry copies and translations are not replications.

The design and historical configuration live in
`docs/research_reboot_20260921/DENSE_SIZE_CLOCK_GEOMETRY_12H_*`.
Execution authorization is recorded separately. `design_only` in the historical
configuration does not grant execution permission or override a later explicit
authorization record.

## Controlled execution

1. `python -m unittest discover -s scripts/research20260930 -p test_size_clock.py -v`
2. `python scripts/research20260930/run_size_clock.py --mode prepare`
3. Inspect all six **synthetic fixture** figures. They are not neural results.
4. On the authorized, verified single GPU: `--mode preflight`. This starts the
   sole12-hour budget before correctness, single-attempt capacity and timing.
5. Back up and verify source/protocol/preflight evidence locally, then `--mode lock`.
6. `--mode run` under a process-group timeout bounded by the original deadline.

Modes are explicit. A unique `run.lock` prevents duplicate formal starts. Failures
are immutable and never trigger automatic retries, altered seeds or extra updates.
Training uses FP32, no AMP/TF32, zero dropout, deterministic Torch algorithms and
addressed NumPy PCG64/SeedSequence streams. Same-device/same-stack save/restore
must give a bitwise-identical next raw/EMA/AdamW update. Across software/hardware
versions, bitwise identity is not promised; report versions and compare against
saved numerical tolerances and seed uncertainty.

`sg_evaluation.jobs()` is the immutable prediction-job inventory. Primary evaluation
is G8 held-out pattern groups at side20, centered, using each arm's deployed clock
policy. All finals are hashed and locked before any final/early held-out prediction.
Final raw is primary; EMA and early snapshots never select the headline.

## Auditing and publication

`sg_management.py monitor` is CPU/read-only for science (it only creates a management
snapshot under the sibling export directory). `verify` checks an entire downloaded
archive and every member byte/SHA; `union` checks all scientific manifest paths
against verified local archives. Checkpoint weights and bulk raw data are not
intended for GitHub; publish code, approved protocol, compact numerical evidence,
plots, captions, environment version record and report. Document actual archive
location and excluded duplicate last/scratch files without claiming offsite backup.

Source/data hashes, exact environment versions and BLAS/CUDA information are captured
for each real run. This supplies a publication-oriented reproducibility recipe;
reproducing on a different GPU/software build is not a bitwise claim. Credentials,
SSH sessions, gateway keys, user screenshots and local authentication files must
never enter a publication package.
