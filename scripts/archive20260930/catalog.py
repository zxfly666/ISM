"""Explicit scientific archive allowlist; local source roots are custodian inputs.

No credentials and no remote access. This file does not run an experiment.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research_reboot_20260921'
BACKUP = ROOT.parent / 'ISM_research_backups'

def item(slug, title, alias, eid, date, kind, parent, reference, generation,
         source, protocol, analysis, code, extra=None, backup=None):
    return dict(slug=slug, public_name=title, historical_alias=alias,
                experiment_id=eid, date=date, training_kind=kind, parents=parent,
                new_reference=reference, new_generation=generation,
                source=source, protocols=protocol, analysis=analysis, code=code,
                extra=extra or [], backup=backup)

C = [
item('geometry-aligned-conditioning','Geometry-aligned conditional prediction','A/B/C; D/E diagnostics',
     'geometry_alignment_20260921','2026-09-21','fresh: 18 models',[],True,16128,
     'artifacts/geometry_alignment_20260921_remote',
     ['REVIEW_RESPONSE_AND_FOCUSED_PROTOCOL_V2_ZH.md','EXISTING_MODEL_DIAGNOSTICS_20260922_ZH.md','MECHANISM_DIAGNOSTICS_PLAN_20260922_ZH.md'],
     '', ['scripts/research20260921/run_study.py','scripts/research20260921/evaluate_study.py'],
     [('artifacts/diagnostics_20260922_remote','diagnostics/context-joint'),('artifacts/mechanism_20260922_remote','diagnostics/mechanism')]),
item('context-consistency-training','Context-consistency continuation','F; M development screen',
     'adaptive_research_20260922','2026-09-22','continued: 18; fresh capacity screen: 2',['geometry-aligned-conditioning/A'],True,46080,
     'artifacts/adaptive_research_20260922_remote/final_evidence', ['ADAPTIVE_CAMPAIGN_20260922_ZH.md'],
     'final_analysis',['scripts/research20260921/run_adaptive_repair_20260922.py','scripts/research20260921/finalize_adaptive_campaign.py'],
     [('artifacts/adaptive_research_20260922_remote/final_evidence/repair/protocol.json','evidence/effective_protocol.json'),('artifacts/adaptive_research_20260922_remote/final_evidence/repair/main_summary.json','evidence/main_summary.json'),('artifacts/adaptive_research_20260922_remote/final_evidence/capacity','evidence/capacity'),('artifacts/adaptive_research_20260922_remote/final_evidence/reference','evidence/reference')]),
item('fixed-background-geometry-response','Fixed-background geometry response and joint consistency','fixed-geometry frozen diagnostic',
     'fixed_geometry_joint_20260923','2026-09-23','frozen: 36 checkpoints',['geometry-aligned-conditioning','context-consistency-training'],False,0,
     'artifacts/fixed_geometry_joint_20260923_remote/evidence/fixed_geometry_joint_20260923',['FIXED_GEOMETRY_JOINT_PROTOCOL_20260923_ZH.md'],
     'analysis',['scripts/research20260921/run_fixed_geometry_joint_20260923.py','scripts/research20260921/analyze_fixed_geometry_joint_20260923.py']),
item('sparse-conditioning-training','Sparse-conditioning continuation','T0/T1/T2',
     'sparse_conditioning_20260923','2026-09-23','continued: 18 models',['context-consistency-training/F0'],True,5760,
     'artifacts/sparse_conditioning_20260923_remote/final_evidence',['SPARSE_CONDITIONING_TRAINING_PROTOCOL_20260923_ZH.md'],
     'analysis',['scripts/research20260921/run_sparse_conditioning_20260923.py','scripts/research20260921/finalize_sparse_conditioning_20260923.py'],backup='20260923_sparse_conditioning'),
item('mask-query-factorial','Sparse-mask × query-placement factorial','R00/R01/R10/R11',
     'mask_query_factorial_20260924','2026-09-24','continued: 24 models',['context-consistency-training/F0'],True,5376,
     'artifacts/mask_query_factorial_20260924_remote/final_evidence',['MASK_QUERY_FACTORIAL_PROTOCOL_20260924_ZH.md'],
     'analysis',['scripts/research20260921/run_mask_query_factorial_20260924.py','scripts/research20260921/finalize_mask_query_factorial_20260924.py'],
     [('artifacts/mask_query_factorial_20260924_remote/run_protocol.json','evidence/run_protocol.json')],backup='20260924_mask_query_factorial'),
item('independent-generation-bridge','Independent-reference generation confirmation','Bridge',
     'generation_bridge_20260924','2026-09-24','frozen: 18 checkpoints',['mask-query-factorial/R00,R10,R11'],True,4032,
     'artifacts/generation_bridge_20260924_remote/final_evidence',['GENERATION_BRIDGE_PROTOCOL_20260924_ZH.md'],
     'analysis',['scripts/research20260921/run_generation_bridge_20260924.py'],backup='20260924_generation_bridge'),
item('size-spacing-factorial','Size × physical-spacing factorial','G00/G10/G01/G11/G11S',
     'core_geometry_factorial_20260925','2026-09-25','fresh: 30 models',[],True,1920,
     str(BACKUP/'20260925_core_geometry_factorial/final_review_20260925'),['CORE_GEOMETRY_FACTORIAL_PROTOCOL_20260925_ZH.md','CORE_GEOMETRY_FACTORIAL_BUDGET_V2_20260925_ZH.md'],
     'analysis',['scripts/research20260921/run_core_geometry_factorial_20260925.py','scripts/research20260921/core_geometry_factorial_analysis.py'],backup='20260925_core_geometry_factorial'),
item('fine-geometry-identification','Fine geometry versus span and randomized coordinates','I-F/I-S/I-R',
     'geometry_identification_20260926','2026-09-26','fresh: 18 models',[],True,2304,
     'artifacts/geometry_identification_20260926_remote/final_science_v1',['GEOMETRY_IDENTIFICATION_PLAN_20260926_ZH.md','GEOMETRY_IDENTIFICATION_CONFIG_20260926.json'],
     'analysis',['scripts/research20260926/run_geometry_identification_20260926.py','scripts/research20260926/identification_analysis.py'],backup='20260926_geometry_identification'),
item('observed-key-value-intervention','Observed-only key/value intervention','J: C-D/C-O; S-D/S-O; B0',
     'observed_context_intervention_18h_20260927','2026-09-27','continued: 12; fresh: 12',['fine-geometry-identification/I-F (C cohort only)'],True,0,
     'artifacts/observed_context_intervention_18h_20260927_remote/final_restore_v1/artifacts/observed_context_intervention_18h_20260927',
     ['OBSERVED_CONTEXT_INTERVENTION_PLAN_20260927_ZH.md','OBSERVED_CONTEXT_INTERVENTION_PLAN_V2_18H_20260927_ZH.md','OBSERVED_CONTEXT_INTERVENTION_CONFIG_V2_18H_20260927.json'],
     'analysis',['scripts/research20260927/run_observed_context_intervention_20260927.py','scripts/research20260927/intervention_analysis.py','scripts/research20260927/intervention_figures.py'],backup='20260927_observed_context_intervention_18h'),
item('exact-local-capability','Exact local conditional capability','Basic stages 1/2: E1/E2/E3',
     'basic_capability_stages12_20260928','2026-09-28','fresh: 3 dense D models; frozen/structural audits',['observed-key-value-intervention (audit only)'],False,0,
     'artifacts/basic_capability_stages12_20260928',['BASIC_CAPABILITY_STAGE12_PLAN_20260928_ZH.md','BASIC_CAPABILITY_STAGE12_CONFIG_20260928.json'],
     '', ['scripts/research20260928/run_basic_capability.py','scripts/research20260928/exact_ising.py'],
     [('artifacts/basic_capability_stages12_20260928/preflight_v1/structural_summary.json','evidence/structural_summary.json'),('artifacts/basic_capability_stages12_20260928/admin/independent_final_audit_v1.json','evidence/independent_final_audit_v1.json')]),
item('local-context-size-diagnostic','Local conditional context-size control','A4/B46; frozen size/clock diagnostic',
     'dense_multisize_control_20260928','2026-09-28','fresh: 6 models; frozen old-model diagnostic',['exact-local-capability (data and frozen diagnostic only)'],False,0,
     'artifacts/dense_multisize_control_20260928',['DENSE_MULTISIZE_CONTROL_PLAN_20260928_ZH.md','DENSE_MULTISIZE_CONTROL_CONFIG_20260928.json'],
     '', ['scripts/research20260928_multisize/run_multisize.py','scripts/research20260928_multisize/ms_evaluation.py'],
     [('artifacts/dense_multisize_control_20260928/diagnostic','evidence/diagnostic')]),
item('canonical-context-size-generalization','Canonical-clock context-size generalization','N/W; 6h',
     'dense_multisize_canonical_6h_20260930','2026-09-30','fresh: 12 models',['local-context-size-diagnostic (motivation only)','preflights/size-clock-factorial (exact split only)'],False,0,
     'results/dense_multisize_canonical_6h_20260930/evidence',['DENSE_MULTISIZE_CANONICAL_6H_PLAN_20260930_ZH.md','DENSE_MULTISIZE_CANONICAL_6H_CONFIG_20260930.json'],
     'analysis',['scripts/research20260930_6h/run_multisize.py','scripts/research20260930_6h/sm6_analysis.py'],backup='20260930_dense_multisize_canonical_6h'),
item('preflights/size-clock-factorial','Size × clock factorial: budget-stop preflight','12h (not launched)',
     'dense_size_clock_geometry_12h_20260930','2026-09-30','NO formal models; two scratch fits',['local-context-size-diagnostic (motivation only)'],False,0,
     'results/dense_size_clock_geometry_20260930/evidence',['DENSE_SIZE_CLOCK_GEOMETRY_12H_PLAN_20260930_ZH.md','DENSE_SIZE_CLOCK_GEOMETRY_12H_CONFIG_20260930.json'],
     '', ['scripts/research20260930/run_size_clock.py','scripts/research20260930/sg_preflight.py'],backup='20260930_dense_size_clock_geometry_12h'),
]

# Not every script in a historical directory is public scientific code.
PRIVATE_CODE = {'remote_transport.py', 'build_full_experiment_report_20260925.py',
                'build_detailed_report.py', 'check_report_pdf.py', 'check_full_report_20260925.py',
                'backup_completed_evaluation.py', 'export_adaptive_final_evidence.py',
                'verify_repair_backup.py', 'summarize_adaptive_closure.py'}
SOURCE_DIRS = ['scripts/research20260921','scripts/research20260926','scripts/research20260927',
               'scripts/research20260928','scripts/research20260928_multisize',
               'scripts/research20260930','scripts/research20260930_6h']
