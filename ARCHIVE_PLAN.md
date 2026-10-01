# Public research archive plan — 2026-09-30

Status: archive implemented; local evidence/link/numerical review completed. See
`experiments/NEWCOMER_TEST.md` for the exact validation scope; remote publication is
verified separately by Git HEAD, never inferred from this status line.

This is a publication/curation operation, **not a new experiment**. Original scientific
outputs, frozen source files and historical decisions are not overwritten. The scope
is the September research sequence and its relationship to the already published
L64/L128 and scale-aware-context pilots.

## Evidence and decision rule

Reviewed the complete 2026-09-30 Chinese master report (162-page edition), its
100-source evidence index, experiment protocols and actual execution records,
effective JSON configurations, statistical summaries, original plotting/analysis
code, local completion/verification records, and repository history. Machine
results and actual run identity take precedence over stale `design_only` headers;
conflicts are recorded rather than silently repaired.

Remote main before this publication: `ed2235990fd213fcfaad30e6bde7480b08780c30`.
Local HEAD before this work: `526d245a0c8777337e34a0f1da6ab28a367e9e7c` (two
unpublished preflight/documentation commits). Existing unrelated work is retained.

## Experiment inventory and public names

| Public directory | Historical alias / experiment_id | Scientific unit and lineage |
|---|---|---|
| `geometry-aligned-conditioning` | A/B/C; geometry_alignment_20260921 | 18 fresh coordinate-control models; D/E frozen diagnostics nested here |
| `context-consistency-training` | F; adaptive_research_20260922 | Six A lineages branch into three 12k continuations; two-seed M capacity screen is development, not six-seed confirmation |
| `fixed-background-geometry-response` | fixed_geometry_joint_20260923 | New geometry-response/joint-consistency question, 36 frozen checkpoints; no new training or MC |
| `sparse-conditioning-training` | T; sparse_conditioning_20260923 | Six F0 lineages, three 4k continuations; new evaluation MC |
| `mask-query-factorial` | R; mask_query_factorial_20260924 | Six F0 lineages, four 4k continuations; **not** continuations of T |
| `independent-generation-bridge` | Bridge; generation_bridge_20260924 | Frozen R00/R10/R11; new MC and sampling RNG, not new training replications |
| `size-spacing-factorial` | G; core_geometry_factorial_20260925 | Six fresh seeds × 2×2 size/spacing arms plus span-only control |
| `fine-geometry-identification` | I; geometry_identification_20260926 | Six fresh seeds × true/span/random coordinates; larger independent reference |
| `observed-key-value-intervention` | J; observed_context_intervention_18h_20260927 | Continued I-F cohort C and independently fresh cohort S; S alone supplies the primary test |
| `exact-local-capability` | basic stages 1/2; basic_capability_stages12_20260928 | Structural O diagnostics, three fresh D models and frozen J audit; exact local truth, no MC |
| `local-context-size-diagnostic` | A4/B46; dense_multisize_control_20260928 | Frozen old-model factorial plus three fresh paired size controls; these are subsections of one capability question |
| `canonical-context-size-generalization` | N/W, 6h; dense_multisize_canonical_6h_20260930 | Six fresh pairs, fixed clock, exact G8 held-out patterns; newest completed campaign |
| `preflights/size-clock-factorial` | 12h; dense_size_clock_geometry_12h_20260930 | **No formal experiment launched**: projected 26.486h exceeded 12h; retain scratch and budget evidence separately |

## Proposed hierarchy

```text
experiments/
  README.md                    # evidence map, chronology and lineage
  CORRECTIONS.md               # original claim -> evidence -> correction
  DATA_AVAILABILITY.md         # public versus custodian-held materials
  REPRODUCING.md               # levels of reproducibility, environment, safe commands
  registry.json               # public name / alias / id / date / lineage
  <scientific-question>/
    README.md                  # question, arms, primary gates, negatives, limitations
    protocol/                  # public copies with editorial provenance
    evidence/                  # original JSON, compact NPZ/CSV, effective config
    figures/                   # original formal figures, never PDF screenshots
    provenance.json            # original and published SHA-256, transformations
    withheld-manifest.json     # large scientific files, hashes and recovery status
  preflights/size-clock-factorial/
scripts/archive20260930/       # curation, verification and numerical replay only
```

Original scientific code paths are preserved, including shared dependencies; naming
the archive does not rename frozen experiment IDs. No empty template directories.
The root README will link the new map before legacy pilots. A public reading
edition of the master report may be supplied with private communications and
infrastructure details removed, with an explicit editorial ledger. The original
local master report remains unchanged.

## Inclusion and large-file policy

Include: all primary decisions and sensitivity summaries, all secondary negative
results, point-estimate arrays and plot data, relevant actual protocols/configs,
formal figures, analysis/training/data-construction code, source/data/checkpoint
identity and completion/verification records. Publish exact synthetic/Ising labels
needed for local numerical replay. Protocol copies retain historical status fields.

Do not blindly publish checkpoint archives, MC spin fields, bulk generation arrays,
all bootstrap draws or repeated query-level files. Keep a **per-path SHA/byte manifest**
and backup archive/member identity where available; explicitly say these files are
not downloadable from ordinary Git. Compact lossless extraction of named arrays is
allowed only with source archive SHA and extraction code. No new scientific
statistic, threshold, seed selection or favorable slice is introduced by curation.

Private connection guides, host credentials, chat screenshots/transcripts, transport
helpers and unrelated poster/review packages are outside the public payload. Public
copies may replace local user/host paths with stable placeholders; each transformed
file records original SHA, published SHA and transformation. Numerical arrays and
scientific definitions are not edited. Same-drive local backups are not offsite
archival storage and are not claimed to be public data hosting.

## Scientific lineage and corrections that cannot be lost

1. A -> F0/F1/F2; F0 -> T and independently R; R -> frozen Bridge.
2. G and I start fresh; J-C continues I-F but J-S starts fresh.
3. Basic, A4/B46 and 6h are fresh exact-local capability studies, not more MC evidence.
4. D/E and reused sampling streams never count as new training seeds.
5. G H1/H2 pass; H3 fails its original joint uncertainty/practical gate and is not equivalent.
6. G W96 held-gap reversal and energy/magnetization tradeoffs remain visible.
7. T positives do not erase R/Bridge failed generation gates; new RNG is not training replication.
8. J structural invariance does not imply accuracy; local oracle failures are retained.
9. 6h has a large relative KL improvement but 0/6 W absolute passes; the full gate fails.
10. Legacy Stage2B incorrect coordinate comparison/old GO flag is annotated, not rewritten.
11. 12h formal models do not exist; design/scratch artifacts cannot be presented as completed training.

## Ambiguities and checks before release

- Public data availability is less than verified local backup completeness. Expose
  the distinction at every campaign entry; no invented download link for local archives.
- Runtime machine paths in protocols are historical references, not working public
  links. Public commands use repository-relative paths and isolate old archived outputs.
- Old LFS pointers have not been redownloaded in this task; their existence alone does
  not establish current blob availability.
- Some frozen diagnostics are exploratory despite detailed uncertainty intervals;
  do not upgrade their claims through nicer presentation.
- Checkpoint recovery and full MC regeneration require withheld data/environment;
  public numerical/table replay is a separate, testable claim.

## Validation and publication sequence

1. Build explicit allowlists and per-file provenance; inventory JSON/NPZ schema and sizes.
2. Write tailored READMEs and the global question/lineage map; preserve original figures.
3. Check numerical claims against machine outputs, source/config consistency, link
   closure, secrets, large files, figure provenance, and primary/secondary labeling.
4. Perform a newcomer reading test from root (all 14 questions in the request).
5. Inspect the exact staged payload, run `git diff --check`, `git diff --stat` and
   status; commit in scientific units without staging unrelated files.
6. Normal fast-forward push only after tool permission review; verify the remote
   HEAD and key public files. If blocked, report it; do not bypass review.

No serious map-level ambiguity prevents implementation. Remaining access and
reproducibility limitations are publication facts, not reasons to hide negative work.

## 2026-10-01 addendum: training-stage archive, not a completed efficacy study

The new campaign is `endpoint_mask_sampler_disentanglement_20261001`, publicly
named **Late-mask coverage and committed-spin correction**. Its directory is
`experiments/late-mask-coverage-and-committed-spin-correction/`. A/L/E are secondary
arm aliases, not top-level experiment names. This is one intended training × sampler
question; do not split its technical preflights or each continuation into experiments.

- **Actual lineage:** all six I-F 12k checkpoints, each continued for 8k in three
  arms; 18 final models, six paired training lineages. No new fresh training seed.
- **Executed scope:** training only, 144,000 updates, 18 resumable finals, 36 EMA
  snapshots. No new MC, formal generation, final conditional evaluation or primary
  effect estimate. The correct status is `training_complete_evaluation_pending`.
- **Hierarchy:** tailored README; historical full scientific plan and effective
  configuration; completion/audit/checkpoint metadata; technical failure records
  in a labeled subsection; per-path data-availability and provenance records.
- **Include:** frozen scientific Python modules and config, small machine-readable
  completion and full-input audit, source/data/final SHA, backup receipts and union
  coverage. Archive prose must distinguish completed training from proposed P1/P2.
- **Withhold from ordinary Git:** large final/EMA weights, raw training logs, MC
  training parent pool and tar archives. Preserve member path/bytes/SHA and archive
  identity; no new public hosting or LFS objects are implied.
- **Privacy:** do not publish private conversation/transcription, connection
  records, transport helpers, credentials or local usernames. Authorization is
  summarized scientifically rather than publishing private dialogue.
- **Historical correction:** the original BF16 inference precision gate and full
  12h feasibility gate failed. Strict continuation recovery subsequently passed,
  and an explicitly authorized training-only driver ran once. Those facts do not
  convert the failed whole-pipeline gate to a pass.
- **Open issue:** formal FP32 inference, fresh reference, generation and final
  scientific analysis still require a separately defined later execution window.
  Do not fill missing primary-result fields with training loss or synthetic fixtures.

Before publication, require full training audit, local archive-member verification,
science-manifest union, link/secret/large-file checks and a normal non-force push.
This addendum does not alter any frozen scientific source or authorize further compute.
