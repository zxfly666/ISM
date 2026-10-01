"""Explicit training-first authorization; same scientific training, no evaluation gate."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, shutil, time, traceback
from pathlib import Path
import endpoint_common as c

AUTH=c.DOCS/'ENDPOINT_MASK_SAMPLER_DISENTANGLEMENT_AUTHORIZATION_20261001_ZH.md'

def prepare():
    c.set_clock();c.check()
    assert not (c.OUT/'run.lock').exists() and not (c.OUT/'run_protocol.json').exists()
    cpu_path=c.EXPORT/'cpu_prepare_v5/checks.json'
    cpu=c.read(cpu_path);assert cpu['status']=='passed'
    # Data, update, restoration, sampling, statistics and audit code are unchanged.
    checked={k:v for k,v in cpu['source_hashes'].items() if k!='run_endpoint.py'}
    assert all(c.sha(Path(__file__).parent/k)==v for k,v in checked.items())
    restored=c.read(c.OUT/'preflight/recovery_v3/restore.json')
    assert restored['status']=='passed' and len(restored['branches'])==18
    assert all(r['next_update_model_ema_optimizer_exact'] for r in restored['branches'])
    from endpoint_training import base_hashes
    bases={str(s):dict(path=c.base_path(s).relative_to(c.ROOT).as_posix(),sha256=c.sha(c.base_path(s))) for s in c.SEEDS}
    assert all(bases[str(s)]['sha256']==base_hashes()[s] for s in c.SEEDS)
    # No new GPU preflight, no reset of the existing absolute clock.
    b=c.read(c.OUT/'budget.json')
    assert b['deadline']==c.DEADLINE
    assert shutil.disk_usage(c.ROOT).free>=5*1024**3
    effective=copy.deepcopy(c.CFG)
    effective.update(lifecycle='authorized_training_phase_frozen',execution_authorized=True,
        execution_scope='18 continuation branches only; generation, MC and analysis deferred',
        original_full_pipeline_preflight='failed_preserved_not_relabelled_passed',
        training_completion_is_not_full_experiment_completion=True,
        stop_rules=['fixed8000updates_per_branch','original_absolute_deadline','unrecoverable_hardware_or_IO_failure',
                    'nonfinite_loss_or_gradient','frozen_source_or_data_corruption'],
        not_stop_rules=['validation_metric','scientific_significance','generation_precision_warning','old_preflight_failure'],
        runtime_environment_contract=dict(CUBLAS_WORKSPACE_CONFIG=':4096:8',deterministic_algorithms=True,
                                          warn_only=False,training_autocast='bfloat16',TF32=False))
    effective['samplers']['shared']['precision']='FP32_inference_revision_deferred_not_executed_in_training_phase'
    c.write(c.OUT/'effective_config.json',effective)
    protocol=dict(study=c.STUDY,created=time.time(),execution_scope='training_first',
        authorization=AUTH.name,authorization_sha256=c.sha(AUTH),sources=c.sources(),base_checkpoints=bases,
        deadline=b['deadline'],budget_started=b['started'],effective_config_sha256=c.sha(c.OUT/'effective_config.json'),
        cpu_receipt_sha256=c.sha(cpu_path),inherited_unchanged_scientific_modules=checked,
        recovery_receipt_sha256=c.sha(c.OUT/'preflight/recovery_v3/restore.json'),
        orchestration_change='Optional MC/diagnostic preparation dependency removed for explicitly authorized training phase',
        full_pipeline_time_gate_not_passed=True,no_scientific_sample_reduction=True,
        training_updates=144000,branches=18,updates_per_branch=8000,final_global_step=20000,
        no_new_MC_or_generation_in_this_phase=True)
    protocol['protocol_hash']=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    c.write(c.OUT/'run_protocol.json',protocol)
    from endpoint_management import archive
    paths=[c.ROOT/n for n in protocol['sources']]
    paths += [c.OUT/n for n in ['run_protocol.json','effective_config.json','budget.json','preflight/recovery_v3/restore.json',
                               'preflight/precision.json','preflight_failure.json','preflight_failure_v2.json','preflight_failure_v3.json']]
    paths += [cpu_path,c.EXPORT/'restore_diagnosis_v4.json',c.EXPORT/'fp32_revision_feasibility_v1/result.json']
    rec=archive(c.ROOT,c.EXPORT,'training_phase_initial_v1',paths)
    c.write(c.OUT/'training_phase_initial_export.json',{k:v for k,v in rec.items() if k!='files'})
    print(json.dumps(dict(status='training_phase_frozen_not_started',protocol_hash=protocol['protocol_hash'],
                         archive_sha256=rec['archive_sha256'],archive_bytes=rec['archive_bytes'],members=rec['members'])),flush=True)

def run():
    c.set_clock();c.check()
    protocol=c.read(c.OUT/'run_protocol.json');assert protocol['execution_scope']=='training_first'
    c.check_sources(protocol['sources'])
    backup=c.read(c.OUT/'training_phase_initial_backup_confirmation.json')
    assert backup['status']=='passed' and backup['archive_sha256']==c.read(c.OUT/'training_phase_initial_export.json')['archive_sha256']
    from endpoint_preflight import setup_torch
    setup_torch()
    with (c.OUT/'run.lock').open('x',encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(),pgid=os.getpgrp(),time=time.time(),protocol_hash=protocol['protocol_hash'],scope='training_first'),f)
    started=time.time()
    c.write(c.OUT/'formal_started.json',dict(started=started,pid=os.getpid(),pgid=os.getpgrp(),parent_pid=os.getppid(),
        budget_started=protocol['budget_started'],deadline=c.DEADLINE,execution_scope='training_first',
        expected_branches=18,expected_updates=144000))
    c.log('formal_training_started',scope='training_first',branches=18,updates=144000)
    try:
        from run_endpoint import train
        train(protocol,None,prepare_diagnostics=False)
        ended=time.time()
        c.write(c.OUT/'training_phase_complete.json',dict(status='training_complete_evaluation_pending',
            started=started,completed=ended,seconds=ended-started,branches=18,updates=144000,
            final_lock_sha256=c.sha(c.OUT/'final_lock.json'),scientific_conclusion_available=False,
            generation_analysis_not_executed=True,backup_pending=True))
        c.log('training_phase_complete_evaluation_pending',branches=18,updates=144000)
    except BaseException as error:
        c.write(c.OUT/'failure.json',dict(time=time.time(),exception=repr(error),traceback=traceback.format_exc(),
            execution_scope='training_first',deadline=c.DEADLINE,checkpoints='training/*/last.pt',
            requires_explicit_failure_notification=True))
        c.log('training_failed_requires_notification',error=repr(error))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','run'],required=True);a=p.parse_args()
    prepare() if a.mode=='prepare' else run()
