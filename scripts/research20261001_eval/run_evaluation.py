"""One-shot post-training evaluation, with its own explicit 22:00 deadline.

Imports frozen scientific routines but never calls training or rewrites their
sources. Precision adaptation is scoped to generation; conditional tasks stay
FP32. MC runs concurrently with fixed-order GPU work, not as a training dependency.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research20261001"))
import endpoint_common as c
import endpoint_data as d
import endpoint_evaluation as ev
import endpoint_management as mg

TRAIN = c.OUT
TRAIN_EXPORT = c.EXPORT
STAGE = "endpoint_mask_sampler_evaluation_20261001"
OUT = ROOT / "artifacts" / STAGE
EXPORT = ROOT / "artifacts" / (STAGE + "_exports")
DOC = ROOT / "docs/research_reboot_20260921/ENDPOINT_EVALUATION_WINDOW_20261001_ZH.md"
DEADLINE = 1790863200.0
BACKUP_RESERVE = 1800
COMPUTE_DEADLINE = DEADLINE - BACKUP_RESERVE
PRECISION = "FP16_autocast_FP32_parameters_output_head_probabilities"
METHODS = ("monotone-256", "reveal192-repair64")


def configure():
    c.OUT, c.EXPORT = OUT, EXPORT
    c.CFG = copy.deepcopy(c.CFG)
    c.CFG["samplers"]["shared"]["precision"] = PRECISION
    c.CFG.update(lifecycle="authorized_frozen_model_evaluation", execution_authorized=True,
                 evaluation_stage_id=STAGE, new_training=False,
                 training_inherited_complete=True, hard_evaluation_deadline=DEADLINE)
    c.DEADLINE = COMPUTE_DEADLINE


def source_files():
    return sorted((ROOT / "scripts/research20261001_eval").glob("*.py")) + [DOC]


def integrity(protocol):
    c.check_sources(protocol["sources"])
    for name, expected in protocol["locked_checkpoints"].items():
        c.check()
        assert c.sha(ROOT / name) == expected, name


def forecast(precision):
    rates = {(int(r["width"]), r["sampler"]): r["seconds"] for r in precision["timings"]}
    shards = {(96, METHODS[0]): 168, (96, METHODS[1]): 144,
              (48, METHODS[0]): 72, (48, METHODS[1]): 72}
    generation = sum(rates[k] * n for k, n in shards.items())
    # MC overlaps the GPU queue; all other allowances are explicit, not measured
    # promises. Includes conditional FP32, coverage, audits, plots, local delivery.
    gpu_queue = generation * 1.15 + 1600 + 300
    remote = max(6500, gpu_queue) + 1800
    total = remote + BACKUP_RESERVE
    now = time.time()
    return dict(time=now, generation_measured_projection_seconds=generation,
        generation_margin=1.15, conditional_allowance_seconds=1600,
        preparation_io_allowance_seconds=300, concurrent_mc_allowance_seconds=6500,
        cpu_analysis_audit_plot_allowance_seconds=1800,
        backup_visual_report_allowance_seconds=BACKUP_RESERVE,
        remaining_total_forecast_seconds=total, forecast_delivery_epoch=now+total,
        hard_deadline=DEADLINE, scientific_compute_deadline=COMPUTE_DEADLINE,
        fits=now+total < DEADLINE, estimates_not_guarantees=True,
        shard_counts={f"w{k[0]}_{k[1]}": n for k, n in shards.items()})


def freeze():
    assert not OUT.exists(), "Existing evaluation output: do not repeat launch"
    precision_path = EXPORT / "fp16_revision_v1/complete.json"
    precision = c.read(precision_path)
    assert precision["status"] == "passed" and len(precision["results"]) == 24
    assert all(x["passed"] for x in precision["results"])
    assert len(precision["timings"]) == 4
    estimate = forecast(precision)
    c.write(EXPORT / "launch_forecast_v1.json", estimate)
    assert estimate["fits"], "Full scientific scope does not fit the approved window"
    # 4 GiB for new science plus a simultaneous archive, and 2 GiB untouched slack.
    space = c.resources()
    assert space["disk_free"] >= 6 * 1024**3, space
    old = c.read(TRAIN / "run_protocol.json")
    c.check_sources(old["sources"], enforce_deadline=False)
    inherited_audit_path = TRAIN_EXPORT / "training_phase_full_audit_v1.json"
    inherited = c.read(inherited_audit_path)
    assert inherited["status"] == "passed" and inherited["updates"] == 144000
    locked = dict(inherited["finals"], **inherited["ema_snapshots"])
    assert len(locked) == 54
    for seed in c.SEEDS:
        base = old["base_checkpoints"][str(seed)]
        locked[base["path"]] = base["sha256"]
    for name, digest in locked.items():
        assert c.sha(ROOT / name) == digest
    sources = dict(old["sources"])
    sources.update({p.relative_to(ROOT).as_posix(): c.sha(p) for p in source_files()})
    for p in [precision_path, inherited_audit_path,
              TRAIN_EXPORT / "training_phase_science_manifest_v1.json"]:
        sources[p.relative_to(ROOT).as_posix()] = c.sha(p)
    protocol = dict(stage_id=STAGE, campaign_id=c.STUDY,
        status="frozen_evaluation_only", authorized_deadline=DEADLINE,
        generation_precision=PRECISION, conditional_precision="unmodified_FP32",
        training_protocol_hash=old["protocol_hash"],
        training_protocol_sha256=c.sha(TRAIN / "run_protocol.json"),
        training_audit_path=inherited_audit_path.relative_to(ROOT).as_posix(),
        inherited_training_manifest=(TRAIN_EXPORT / "training_phase_science_manifest_v1.json").relative_to(ROOT).as_posix(),
        locked_checkpoints=locked, sources=sources, config=c.CFG,
        order="new MC parallel to fixed seed/arm learning+generation; conditional after MC QA; all statistics only after complete counts",
        no_training=True, no_sample_reduction=True, no_checkpoint_selection=True,
        formal_images=6912, phase0_images=384, prediction_files=546,
        conditional_inputs=82176, reference_parents=2048, independent_lineages=6,
        training_alias="read-only use of old training directory; manifests use canonical old paths",
        freeze_time=time.time(), resources=space, launch_forecast=estimate)
    protocol["protocol_hash"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "training").symlink_to(TRAIN / "training", target_is_directory=True)
    c.write(OUT / "run_protocol.json", protocol)
    c.write(OUT / "final_lock.json", c.read(TRAIN / "final_lock.json"))
    c.write(OUT / "budget.json", dict(started=c.read(EXPORT / "precision_revision_v1/plan.json")["started"],
        deadline=DEADLINE, compute_deadline=COMPUTE_DEADLINE,
        backup_seconds_reserved=BACKUP_RESERVE, previous_training_budget_unchanged=True))
    c.write(OUT / "training_audit_inherited.json", inherited)
    c.write(OUT / "effective_config.json", c.CFG)
    technical = [p for folder in [EXPORT / "precision_revision_v1", EXPORT / "fp16_revision_v1"]
                 for p in folder.rglob("*") if p.is_file()]
    paths = source_files() + technical + [p for p in OUT.iterdir() if p.is_file()]
    package = mg.archive(ROOT, EXPORT, "evaluation_initial_v1", paths)
    print(json.dumps(dict(protocol_hash=protocol["protocol_hash"], forecast=estimate, initial=package), ensure_ascii=False))


_LAST_RESOURCE_CHECK = 0.


def check_run():
    global _LAST_RESOURCE_CHECK
    c.check()
    now = time.time()
    if now - _LAST_RESOURCE_CHECK >= 10:
        if (OUT / "reference_failure.json").exists():
            raise RuntimeError("Reference failure; preserve all products without changing the target")
        if shutil.disk_usage(ROOT).free < 2 * 1024**3:
            raise RuntimeError("Less than 2 GiB remote free space; preserve existing products")
        _LAST_RESOURCE_CHECK = now


def generation(model, seed, folder, width, method, images, source, floor=.002, phase0=False):
    import torch
    import endpoint_sampling as sam
    from fp16_adapter import fp16_mode
    folder.mkdir(parents=True, exist_ok=False)
    parts, durations = {}, []
    with fp16_mode(model):
        # Guard every formal forward, including early all-MASK states. This
        # read-only finite check is additional to validation-only agreement.
        original_forward = model.forward
        def finite_forward(*args, **kwargs):
            logits = original_forward(*args, **kwargs)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Nonfinite formal generation logits")
            return logits
        model.forward = finite_forward
        for first in range(0, images, 16):
            check_run()
            ids = np.arange(first, first+16) + (1000000 if phase0 else 0)
            torch.cuda.synchronize(); began = time.perf_counter()
            z = sam.sample(model, seed, width, ids, method, floor, amp=False, check=c.check)
            torch.cuda.synchronize(); seconds = time.perf_counter() - began
            sam.audit_repair(z)
            z.update(source_sha256=np.array(source), seed=np.array(seed),
                sampling_seconds=np.array(seconds), phase0=np.array(phase0),
                width=np.array(width), inference_precision=np.array(PRECISION))
            c.save(folder / f"shard_{first:05d}.npz", **z)
            durations.append(seconds)
            for typ, key in [("final", "spins"), ("prefix", "prefix_spins"), ("oracle", "oracle_spins")]:
                if key in z:
                    for view, stat in d.multiscale_stats(z[key]).items():
                        parts.setdefault(f"{typ}_{view}", []).append(stat)
            c.log("generation_shard", seed=seed, width=width, sampler=method,
                phase0=phase0, folder=folder.relative_to(ROOT).as_posix(),
                completed=first+16, total=images, seconds=seconds, precision=PRECISION)
    for name, rows in parts.items():
        c.save(folder / f"stats_{name}.npz", **d.concatenate(rows),
            image_ids=np.arange(images)+(1000000 if phase0 else 0), seed=np.array(seed))
    c.write(folder / "complete.json", dict(status="complete", images=images, width=width,
        sampler=method, clock_floor=floor, sampling_seconds=sum(durations),
        source_sha256=source, per_image_network_calls=256, inference_precision=PRECISION))


def predictions(model, bankpaths, destination, source, prefix=""):
    import endpoint_training as tr
    for bankpath in bankpaths:
        check_run()
        z = tr.predict(model, c.load(bankpath), check=check_run)
        c.save(destination / (prefix+bankpath.name), **z,
            source_sha256=np.array(source), bank_sha256=np.array(c.sha(bankpath)),
            inference_precision=np.array("unmodified_FP32"))
        c.log("prediction", output=(destination / (prefix+bankpath.name)).relative_to(ROOT).as_posix())


def run():
    import torch
    import platform
    import endpoint_training as tr
    from endpoint_preflight import setup_torch
    setup_torch()
    protocol = c.read(OUT / "run_protocol.json")
    integrity(protocol)
    ack = c.read(OUT / "initial_backup_ack.json")
    assert ack["status"] == "passed" and ack["independent_copy_verified"]
    assert ack["archive_sha256"] == c.read(EXPORT / "evaluation_initial_v1.manifest.json")["archive_sha256"]
    current = forecast(c.read(EXPORT / "fp16_revision_v1/complete.json"))
    c.write(OUT / "launch_forecast.json", current)
    assert current["fits"], "Elapsed preparation exhausted full-scope window"
    c.write(OUT / "run.lock", dict(pid=os.getpid(), started=time.time(), protocol_hash=protocol["protocol_hash"]))
    started = time.time()
    c.write(OUT / "formal_started.json", dict(pid=os.getpid(), pgid=os.getpgrp(), started=started,
        deadline=DEADLINE, compute_deadline=COMPUTE_DEADLINE, training=False))
    c.write(OUT / "run_environment.json", dict(python=sys.version, platform=platform.platform(),
        torch=torch.__version__, torch_cuda=torch.version.cuda,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32, torch_threads=torch.get_num_threads(),
        generation_precision=PRECISION, conditional_precision="unmodified_FP32",
        resources=c.resources()))
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = ""
    with (OUT / "reference.stdout").open("x") as f:
        mc = subprocess.Popen(["timeout", "--signal=KILL", str(max(1, math.floor(COMPUTE_DEADLINE-time.time()))),
            sys.executable, "-u", str(Path(__file__)), "--mode", "reference"],
            stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    c.write(OUT / "reference_process.json", dict(timeout_pid=mc.pid, pgid=mc.pid, started=time.time()))
    c.log("evaluation_started", mc_timeout=mc.pid, generation_precision=PRECISION)
    ev.validation_banks()
    ev.old_coverage_audit(d.load_parent(c.DATA, "train"))
    validation = sorted((OUT / "validation_banks").glob("*.npz"))
    assert len(validation) == 4
    for seed in c.SEEDS:
        for arm in c.ARMS:
            check_run(); key = f"s{seed}_{arm}"; cell = OUT / "evaluation" / key
            cell.mkdir(parents=True, exist_ok=False)
            base = TRAIN / "training" / key; path = base / "final.pt"; source = c.sha(path)
            model = tr.load_ema(path, seed)
            for step in [4000, 6000, 8000]:
                snapshot = path if step == 8000 else base / f"ema_{step}.pt"
                earlier = model if step == 8000 else tr.load_ema(snapshot, seed)
                predictions(earlier, validation, cell / "learning", c.sha(snapshot), f"step{step}_")
                if earlier is not model:
                    del earlier
            for width, images in [(96, 128), (48, 64)]:
                for method in METHODS:
                    generation(model, seed, cell / "generation" / f"w{width}_{method}", width, method, images, source)
            assert c.sha(path) == source
            c.write(cell / "generation_learning_complete.json", dict(time=time.time(), source_sha256=source))
            del model; torch.cuda.empty_cache()
    for seed in c.SEEDS:
        path = c.base_path(seed); source = c.sha(path); model = tr.load_ema(path, seed)
        for floor in [.01, .002]:
            generation(model, seed, OUT / "phase0" / f"s{seed}" / f"floor{floor}",
                       96, METHODS[0], 32, source, floor, phase0=True)
        del model; torch.cuda.empty_cache()
    # MC should have finished well before the fixed GPU queue; this is not a new
    # sampling attempt or permission to enlarge its fixed 16x128 design.
    while mc.poll() is None:
        check_run(); time.sleep(10)
    assert mc.returncode == 0, f"MC process exited {mc.returncode}"
    assert c.read(OUT / "reference/complete.json")["status"] == "passed"
    banks = sorted((OUT / "banks").glob("*.npz"))
    formal = [p for p in banks if not p.name.startswith("phase0")]
    phase0 = [p for p in banks if p.name.startswith("phase0")]
    assert (len(formal), len(phase0)) == (13, 16)
    for seed in c.SEEDS:
        for arm in c.ARMS:
            key = f"s{seed}_{arm}"; cell = OUT / "evaluation" / key
            path = TRAIN / "training" / key / "final.pt"; source = c.sha(path)
            model = tr.load_ema(path, seed)
            predictions(model, formal, cell / "conditional", source)
            c.write(cell / "complete.json", dict(status="complete", source_sha256=source,
                formal_predictions=13, learning_predictions=12, final_images=384,
                prefix_images=192, oracle_images=192))
            del model; torch.cuda.empty_cache()
        path = c.base_path(seed); source = c.sha(path); model = tr.load_ema(path, seed)
        cell = OUT / "phase0" / f"s{seed}"
        predictions(model, phase0, cell / "conditional", source)
        c.write(cell / "complete.json", dict(status="complete", predictions=16, images=64, source_sha256=source))
        del model; torch.cuda.empty_cache()
    integrity(protocol)
    for name, function in [("reference", mg.audit_reference), ("predictions", mg.audit_predictions), ("generation", mg.audit_generation)]:
        began = time.time(); result = function(); result["elapsed_seconds"] = time.time()-began
        c.write(OUT / "audit" / (name + ".json"), result); c.log("audit_complete", part=name)
    import endpoint_statistics as stats
    import endpoint_figures as figures
    c.log("analysis_started")
    stats.analyze(OUT)
    figures.render(OUT)
    assert len(list((OUT / "analysis").glob("*.png"))) == 6
    assert len(list((OUT / "analysis").glob("*.pdf"))) == 6
    integrity(protocol)
    c.write(OUT / "final_summary.json", dict(status="remote_complete_requires_backup_and_visual_review",
        started=started, completed=time.time(), elapsed_seconds=time.time()-started,
        no_new_training=True, new_reference_parents=2048, formal_images=6912,
        phase0_images=384, prediction_files=546, conditional_inputs=82176,
        precision=PRECISION, scientific_summary=c.read(OUT / "analysis/summary.json")))
    package = package_evaluation(protocol)
    c.write(OUT / "remote_export_complete.json", dict(time=time.time(), package=package,
        status="not_local_delivery_complete"))
    c.log("remote_export_complete", archive=package["archive"], archive_bytes=package["archive_bytes"])


def package_evaluation(protocol):
    omitted = {"formal.stdout", "reference.stdout", "run.jsonl", "status.json"}
    paths = [p for p in OUT.rglob("*") if p.is_file() and "training" not in p.relative_to(OUT).parts
             and p.name not in omitted and not p.name.endswith(".tmp")]
    inherited = c.read(ROOT / protocol["inherited_training_manifest"])["files"]
    combined = {r["path"]: r for r in inherited}
    initial = c.read(EXPORT / "evaluation_initial_v1.manifest.json")
    for row in initial["files"] + mg.record_paths(paths):
        if row["path"] in combined:
            assert row == combined[row["path"]], "Conflicting inherited evidence bytes"
        combined[row["path"]] = row
    c.write(OUT / "science_manifest.json", dict(files=list(combined.values()), time=time.time(),
        scientific_scope="original completed training plus new fixed evaluation",
        inherited_training_complete_not_recomputed=True,
        exclusions=["training alias duplicates already verified canonical paths", "18 redundant last.pt already excluded in inherited training manifest"],
        administrative_delta_required=sorted(omitted)))
    return mg.archive(ROOT, EXPORT, "evaluation_complete_v1", paths + [OUT / "science_manifest.json"])


def reference():
    try:
        ev.new_reference()
    except BaseException as error:
        c.write(OUT / "reference_failure.json", dict(time=time.time(), error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["freeze", "run", "reference"])
    args = parser.parse_args(); configure()
    try:
        {"freeze": freeze, "run": run, "reference": reference}[args.mode]()
    except BaseException as error:
        if args.mode == "run" and OUT.exists() and not (OUT / "failure.json").exists():
            c.write(OUT / "failure.json", dict(time=time.time(), error=repr(error), traceback=traceback.format_exc(),
                training_unchanged=True, no_automatic_restart=True))
        raise
