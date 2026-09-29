# Dense size × clock geometry: preflight only

Publication status at handoff: committed locally, **not pushed to GitHub**.
Safety review requires explicit confirmation to publish the compact experimental
evidence and 22,806-byte exact-label bank to `zxfly666/ISM`. No alternate upload
path was used to bypass that decision.

**Status: `NOT_LAUNCHED_TIME_GATE_FAILED`.** No formal model or held-out
cross-size result exists for this study. Correctness and the single-attempt
train-only fitting tests passed; the original 12-hour launch gate did not.

The measured mean full update was 0.108157 s. The planned 576,000 updates alone
project to 17.305 h without a safety multiplier. The preregistered conservative
all-in projection was 26.486 h. The runner stopped before creating `run.lock`.
This is a resource-feasibility result, not a negative result for the scientific
hypothesis or a complete test of the teacher's idea.

- [Chinese report](../../docs/research_reboot_20260921/DENSE_SIZE_CLOCK_GEOMETRY_12H_RESULTS_20260930_ZH.md)
- [Design](../../docs/research_reboot_20260921/DENSE_SIZE_CLOCK_GEOMETRY_12H_PLAN_20260930_ZH.md)
- [Code and reproduction entry](../../scripts/research20260930/README.md)
- [Machine-readable summary](summary.json)
- [Explicit public evidence manifest](evidence_manifest.json)
- [Source/data hashes](source_data_sha256.json)
- [Original deployed source bytes (74,536 bytes)](deployed_source_bytes_v1.tar.gz)
- [Isolated source-bundle CPU test](source_bundle_verification.json)
- [Local preflight backup coverage](backup_verification.json)
- [Measured update timings](evidence/update_timing.json)
- [Original launch gate](evidence/launch_gate.json)
- [GPU correctness](evidence/model_checks_gpu.json)
- [Train-only fitting results](evidence/capacity_complete.json)

Use Python 3.9+ for the complete management/packaging CLI. The measured GPU
environment was Python 3.12.3; local full SHA-union verification used Python
3.11.5. Older Python can run some individual tests but is not a supported
environment for the complete workflow (`str.removesuffix` and path checks).
The small source bundle preserves the exact 21 deployed source/data members;
Git can otherwise normalize text line endings. Its hashes were checked against
the original archive and all seven CPU tests passed in an isolated copy.

The two scratch fits use one fixed 64-row training subset, not the held-out
20×20 endpoint or six independent formal seeds. The 3240 synthetic fixture
predictions used for software testing are intentionally not presented as neural
results or published as result figures.

`evidence/` contains verbatim members copied from a fully member-SHA-verified
preflight archive. Its allowlist excludes private transport material and large
temporary weights. A separate same-disk local backup covers all 6754 paths of
the preflight scientific manifest; it is not offsite disaster recovery and does
not contain nonexistent formal outputs.

The historical design contains links to older internal reports/teacher notes
that are not all part of this minimal release. Runtime source dependencies and
the small inherited exact-label bank are included. Do not use the historical
configuration's `design_only` flag to infer missing user authorization: the
separate authorization record and this measured stop status are authoritative.

No new run, reduced sample size, altered precision or extended deadline was
authorized by the failed gate. Any new performance/budget revision needs its
own explicit decision and must preserve the failed preflight evidence.
