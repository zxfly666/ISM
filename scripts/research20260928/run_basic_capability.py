"""Single-purpose stages 1/2 runner; immutable outputs and a fixed two-hour cap."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np

import capability_common as c
import exact_ising as exact


def disk_check():
    free = {drive: shutil.disk_usage(drive + ":/").free for drive in ("C", "D")}
    if free["C"] < 2*1024**3 or free["D"] < 2*1024**3:
        raise RuntimeError("Insufficient local space; no data will be deleted")
    return free


def legacy_check():
    final = c.ROOT/c.CONFIG["J_final_local"]
    protocol = c.read(final/"run_protocol.json")
    if c.sha(final/"run_protocol.json") != "c9ee4cdc7156a93f5ae48e8fc0c6d8711d274f46491a9d7b1e5ea17d62189b22":
        raise RuntimeError("Foreign J protocol")
    checked, absent, changed = [], [], []
    for rel, expected in protocol["files"].items():
        if not (c.ROOT/rel).is_file():
            absent.append(rel)
        elif c.sha(c.ROOT/rel) != expected:
            changed.append(rel)
        else:
            checked.append(rel)
    required = {"scripts/research20260927/intervention_model.py", "ism_diffusion/model.py", "ism_diffusion/scale_model.py"}
    if changed or not required.issubset(set(checked)):
        raise RuntimeError("Frozen legacy inputs changed or required model absent: " + repr(changed))
    return dict(status="passed_present_files", checked=checked, not_locally_present=absent,
                changed=changed, all_required_model_sources_verified=True)


def prepare():
    c.OUT.mkdir(parents=True, exist_ok=False)
    start = time.time()
    c.write(c.OUT/"authorization.json", dict(user_request="请你设计实验方案，编写代码，执行1,2阶段",
        authorized_stages=[1, 2], new_architecture_authorized=False, old_J_restart_authorized=False,
        created_at=start, plan_sha256=c.sha(c.PLAN_PATH), config_sha256=c.sha(c.CONFIG_PATH)))
    import test_basic_capability
    stream = io.StringIO()
    tests = unittest.defaultTestLoader.loadTestsFromModule(test_basic_capability)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
    c.write(c.OUT/"software_tests.json", dict(tests=result.testsRun, passed=result.wasSuccessful(),
                                           output=stream.getvalue()))
    if not result.wasSuccessful():
        raise RuntimeError("Exact truth unit tests failed")
    truth = exact.enumerate_ising()
    bank, split = exact.make_bank(truth)
    c.save(c.OUT/"exact_states.npz", **truth)
    c.save(c.OUT/"bank.npz", **bank)
    c.save(c.OUT/"k4_extension_fixed.npz", **exact.k4_extension(bank))
    c.save(c.OUT/"k4_extension_natural.npz", **exact.k4_extension(bank, True))
    c.write(c.OUT/"truth_checks.json", exact.truth_checks(truth, bank))
    c.write(c.OUT/"split.json", split)
    baselines = {}
    for k in (1, 2, 4):
        selected = bank["k"] == k
        plus = (bank["tokens"] == 1).sum((1, 2))
        blind = np.zeros(len(bank["query"]), dtype=np.float64)
        for count in range(k+1):
            train = selected & bank["train"] & (plus == count)
            blind[selected & (plus == count)] = bank["target"][train].mean()
        for split_name, use in (("train", selected & bank["train"]), ("holdout", selected & ~bank["train"])):
            if use.any():
                baselines["k%d_%s" % (k, split_name)] = dict(
                    uniform=c.summarize(np.full(int(use.sum()), .5), bank["target"][use]),
                    blind_fit_on_training_only=c.summarize(blind[use], bank["target"][use]))
    c.write(c.OUT/"baselines.json", baselines)
    c.write(c.OUT/"legacy_freeze_check.json", legacy_check())
    from review_j_recipe import review
    (c.OUT/"J_review").mkdir()
    review(c.OUT/"J_review")
    c.write(c.OUT/"prepare_complete.json", dict(time=time.time(), seconds=time.time()-start,
            data_sha256=c.sha(c.OUT/"bank.npz"), disk_free=disk_check(), computational_scope="CPU truth and prior-file audit; no new training yet"))
    print("PREPARE_COMPLETE", len(bank["query"]), json.dumps(split), flush=True)


def environment(cm):
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata.get("Name")}
    return dict(python=sys.version, executable=sys.executable, torch=cm.torch.__version__,
        numpy=np.__version__, packages=dict(sorted(versions.items())), platform=platform.platform(),
        processor=platform.processor(), torch_threads=cm.torch.get_num_threads(),
        torch_interop_threads=cm.torch.get_num_interop_threads(), torch_config=cm.torch.__config__.show(),
        deterministic=cm.torch.are_deterministic_algorithms_enabled(),
        env_threads={key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(c.ROOT)).decode().strip(),
        CUDA_used=False)


def preflight():
    import capability_model as cm
    cm.configure()
    if not (c.OUT/"prepare_complete.json").is_file():
        raise RuntimeError("Prepare stage missing")
    folder = c.OUT/"preflight_v1"
    folder.mkdir(exist_ok=False)
    c.write(folder/"source_snapshot.json", c.source_files())
    c.write(folder/"environment.json", environment(cm))
    bank = c.load(c.OUT/"bank.npz")
    tb = cm.tensor_bank(bank)
    model = cm.new_model(0)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = cm.optimizer(model)
    if not (c.OUT/"budget.json").exists():
        began = time.time()
        c.write(c.OUT/"budget.json", dict(started=began, deadline=began+c.CONFIG["hard_seconds"],
            hard_seconds=c.CONFIG["hard_seconds"], starts_at="first full-model training timing", old_J_budget_reused=False))
    c.check_deadline()
    trace, elapsed = [], []
    step = 0
    for count in [c.CONFIG["timing_warmup"]] + [c.CONFIG["timing_steps_per_repeat"]]*c.CONFIG["timing_repeats"]:
        started = time.perf_counter()
        for _ in range(count):
            step += 1; c.check_deadline()
            trace.append(cm.update(model, ema, opt, tb, cm.batch_indices(bank, 0, step), step))
        elapsed.append(time.perf_counter()-started)
        print("TIMING", count, elapsed[-1], flush=True)
    cm.checkpoint(folder/"timing_recovery.pt", model, ema, opt, 0, step, "timing_only")
    restored, restored_ema, restored_opt, payload = cm.restore(folder/"timing_recovery.pt")
    idx = cm.batch_indices(bank, 0, step+1)
    first = cm.update(model, ema, opt, tb, idx, step+1)
    second = cm.update(restored, restored_ema, restored_opt, tb, idx, step+1)
    equal = cm.model_hash(model) == cm.model_hash(restored) and cm.model_hash(ema) == cm.model_hash(restored_ema) and first == second
    for a, b in zip(opt.state.values(), restored_opt.state.values()):
        equal = equal and all(cm.torch.equal(a[key], b[key]) for key in a)
    c.write(folder/"recovery_check.json", dict(passed=bool(equal), next_step=step+1,
        raw_and_ema_and_AdamW_and_loss_gradient_identical=bool(equal)))
    if not equal:
        raise RuntimeError("Exact next-step recovery check failed")
    cm.structural_suite(bank, folder)
    max_rate = max(elapsed[1:])/c.CONFIG["timing_steps_per_repeat"]
    training_prediction = max_rate*c.CONFIG["steps"]*len(c.CONFIG["seed_labels"])*c.CONFIG["timing_safety_multiplier"]
    used = time.time()-c.read(c.OUT/"budget.json")["started"]
    total = used+training_prediction+c.CONFIG["evaluation_analysis_backup_reserve_seconds"]
    timing = dict(status="passed" if total <= c.CONFIG["hard_seconds"] else "budget_gate_failed",
        elapsed_repeats=elapsed, slowest_seconds_per_update=max_rate,
        predicted_remaining_training_seconds=training_prediction, elapsed_budget_seconds=used,
        conservative_total_seconds=total, fixed_steps=c.CONFIG["steps"], fixed_seeds=len(c.CONFIG["seed_labels"]),
        time=time.time(), disk_free=disk_check(), training_trace=trace,
        trace_not_used_for_scope_or_checkpoint_selection=True)
    c.write(folder/"complete.json", timing)
    print("PREFLIGHT", json.dumps({k: v for k, v in timing.items() if k != "training_trace"}), flush=True)


def score_final(model, bank, folder, suffix):
    import capability_model as cm
    p = cm.predict(model, bank)
    c.save(folder/(suffix+"_predictions.npz"), probability=p, target=bank["target"],
           row_id=bank["row_id"], **c.metrics(p, bank["target"]))
    summary = {}
    for k in (1, 2, 4):
        for part, selected in (("train", (bank["k"] == k) & bank["train"]),
                               ("holdout", (bank["k"] == k) & ~bank["train"])):
            if selected.any():
                summary["k%d_%s" % (k, part)] = c.summarize(p[selected], bank["target"][selected])
    for view in ("fixed", "natural"):
        extension = c.load(c.OUT/("k4_extension_%s.npz" % view))
        ep = cm.predict(model, extension)
        c.save(folder/(suffix+"_k4_extension_%s.npz" % view), probability=ep,
               target=extension["target"], source_row=extension["source_row"],
               **c.metrics(ep, extension["target"]))
        summary["k4_extension_"+view] = c.summarize(ep, extension["target"])
        summary["k4_extension_"+view]["max_drift_from_small"] = float(np.max(np.abs(ep-p[extension["source_row"]])))
    summary["core_all_passed"] = all(summary[name]["diagnostic_capability_passed"] for name in ("k1_holdout", "k2_holdout", "k4_train"))
    c.write(folder/(suffix+"_summary.json"), summary)
    return summary


def run():
    import capability_model as cm
    cm.configure()
    if c.read(c.OUT/"preflight_v1/complete.json")["status"] != "passed":
        raise RuntimeError("Preflight did not pass; no automatic rescoping")
    files = c.read(c.OUT/"preflight_v1/source_snapshot.json")
    c.check_frozen(files); c.check_deadline(); disk_check()
    remaining_prediction = c.read(c.OUT/"preflight_v1/complete.json")["predicted_remaining_training_seconds"]
    if time.time()+remaining_prediction+c.CONFIG["evaluation_analysis_backup_reserve_seconds"] > c.read(c.OUT/"budget.json")["deadline"]:
        raise RuntimeError("Remaining absolute budget is insufficient")
    c.write(c.OUT/"run.lock", dict(pid=os.getpid(), time=time.time(), no_resume_or_second_run=True))
    protocol = dict(study=c.CONFIG["study"], started=time.time(), sources=files,
        config=c.CONFIG, bank_sha256=c.sha(c.OUT/"bank.npz"),
        exact_truth_sha256=c.sha(c.OUT/"exact_states.npz"), split_sha256=c.sha(c.OUT/"split.json"),
        budget=c.read(c.OUT/"budget.json"), stages=[1, 2], no_automatic_stage3_or4=True)
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol["protocol_hash"] = protocol_hash
    c.write(c.OUT/"run_protocol.json", protocol)
    c.write(c.OUT/"run_environment.json", environment(cm))
    bank = c.load(c.OUT/"bank.npz"); tb = cm.tensor_bank(bank)
    for label in c.CONFIG["seed_labels"]:
        c.check_deadline(); c.check_frozen(files)
        folder = c.OUT/("training/seed_%d" % label)
        folder.mkdir(parents=True, exist_ok=False)
        model = cm.new_model(label)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        opt = cm.optimizer(model)
        c.write(folder/"initial.json", dict(seed_label=label, initialization_seed=c.model_seed(label),
            raw_hash=cm.model_hash(model), selected_final="raw_at_2048", data_stream="SeedSequence(root,label,step,role)"))
        start = time.time(); last_step = 0
        try:
            with (folder/"trace.jsonl").open("x", encoding="utf-8", buffering=1) as log:
                for step in range(1, c.CONFIG["steps"]+1):
                    c.check_deadline()
                    indices = cm.batch_indices(bank, label, step)
                    record = cm.update(model, ema, opt, tb, indices, step)
                    record["time"] = time.time()
                    log.write(json.dumps(record, allow_nan=False)+"\n")
                    last_step = step
                    if step % 128 == 0:
                        c.write(c.OUT/"status.json", dict(stage="training", seed=label, step=step,
                            time=time.time(), elapsed_seed_seconds=time.time()-start), replace=True)
                        print("TRAIN", label, step, "elapsed", round(time.time()-start, 2), flush=True)
            cm.checkpoint(folder/"final.pt", model, ema, opt, label, last_step, protocol_hash)
            raw = score_final(model, bank, folder, "raw")
            score_final(ema, bank, folder, "ema_secondary")
            symmetries = cm.transformed_checks(model, bank)
            restored, _, _, restored_payload = cm.restore(folder/"final.pt")
            recovery_equal = cm.model_hash(restored) == cm.model_hash(model)
            if not recovery_equal or restored_payload["step"] != c.CONFIG["steps"]:
                raise RuntimeError("Final recovery verification failed")
            c.write(folder/"complete.json", dict(status="complete", seed_label=label,
                time=time.time(), seconds=time.time()-start, steps=last_step,
                raw_core_capability_passed=raw["core_all_passed"], symmetries=symmetries,
                final_sha256=c.sha(folder/"final.pt"), restored_CPU_identity_verified=recovery_equal))
            print("SEED_COMPLETE", label, "core_pass", raw["core_all_passed"], flush=True)
        except Exception:
            if last_step and not (folder/"final.pt").exists():
                cm.checkpoint(folder/("interrupted_step_%d.pt" % last_step), model, ema, opt, label, last_step, protocol_hash)
            raise
    c.check_frozen(files)
    report()


def report():
    rows = []
    for label in c.CONFIG["seed_labels"]:
        folder = c.OUT/("training/seed_%d" % label)
        if not (folder/"complete.json").is_file():
            raise RuntimeError("Cannot report full completion with an incomplete seed")
        rows.append(dict(seed=label, raw=c.read(folder/"raw_summary.json"),
                         secondary_ema=c.read(folder/"ema_secondary_summary.json"),
                         completion=c.read(folder/"complete.json")))
    j = c.read(c.OUT/"J_review/J_mixed_recipe_review.json")
    aggregate = {}
    for key in rows[0]["raw"]:
        if key == "core_all_passed":
            continue
        values = np.array([r["raw"][key]["mean_kl"] for r in rows])
        aggregate[key] = dict(per_seed_mean_kl=values.tolist(), mean=float(values.mean()),
                             seed_sd=float(values.std(ddof=1)), seed_MCSE=float(values.std(ddof=1)/np.sqrt(len(values))),
                             independent_units=3, exact_reference_MCSE=0.0,
                             confirmatory_population_claim=False)
    summary = dict(status="stage1_and_stage2_complete", time=time.time(),
        elapsed_since_budget_start_seconds=time.time()-c.read(c.OUT/"budget.json")["started"],
        new_dense_models=3, new_observed_only_training=0, new_updates=3*c.CONFIG["steps"],
        raw_all_seed_core_passed=all(r["raw"]["core_all_passed"] for r in rows),
        J_dense_original_mixed_capability_all_passed=all(v["original_gate_passed"] for v in j["groups"]["S-D"].values()),
        seeds=rows, aggregate=aggregate, local_scope_only=True,
        automatic_next_stage=False, source_mismatch=[], disk_free=disk_check(),
        important_limit="4x4 open-boundary soft-target training and reused J L1024 mixed-recipe evidence are different experiments; no causal comparison of their loss values.")
    c.write(c.OUT/"final_summary.json", summary)
    lines = ["# 第一、二阶段基础能力诊断结果", "",
        "本报告由保存结果生成。只执行阶段1/2；没有修改J架构、重启J、开展新MC或进入长程外推实验。", "",
        "## 1. 第一阶段更正", "",
        "J的观测点限制是我们另提的机制干预，不是老师dense/多尺寸/真实距离训练方案的必经步骤。保留全部J负结果；O不是已合格主线，D也必须先检验能力。", "",
        "## 2. 精确小系统与固定训练", "",
        "4×4开放边界、零场、β=log(1+sqrt(2))/2，枚举65,536状态；独立转移矩阵与解析Gibbs校验见truth_checks.json。共1,864条件行，按D4几何轨道留出，不拆符号来制造测试。", "",
        "3个从零D模型，各2,048步，原1,976,706参数架构、FP32/CPU、固定final raw。K1/K2/四邻居各占每批1/3，专门软目标训练，不是J原混合配方。EMA只作次要。", "",
        "## 3. 逐seed核心门", "",
        "门为每任务最大KL≤0.01且最大概率误差≤0.05；表中K4_train是拟合检查，不冒充独立泛化。", "",
        "| seed | 项目 | 平均KL | 最大KL | 最大概率误差 | 通过 |",
        "|---|---|---:|---:|---:|---|"]
    for row in rows:
        for key in ("k1_train", "k1_holdout", "k2_train", "k2_holdout", "k4_train", "k4_extension_fixed", "k4_extension_natural"):
            value = row["raw"][key]
            lines.append("| %d | %s | %.8g | %.8g | %.8g | %s |" %
                         (row["seed"], key, value["mean_kl"], value["max_kl"], value["max_probability_error"], value["diagnostic_capability_passed"]))
    lines += ["", "## 4. J原混合配方复核（旧开发证据）", "",
              "重新核对12个旧评价包、oracle bank及36份解析预测的SHA/输入/标签，重算CE、KL、Brier。没有新前向、训练或bootstrap；不是新的独立确认。原始区间沿用J，不用这个小任务训练替代。", "",
              "| 组 | K | 平均KL | 原98.3333%区间 | 原门通过 |", "|---|---:|---:|---|---|"]
    for group, values in j["groups"].items():
        for k, value in values.items():
            lines.append("| %s | %s | %.8g | [%.8g, %.8g] | %s |" %
                         (group, k, value["mean"], value["original_ci"][0], value["original_ci"][1], value["original_gate_passed"]))
    lines += ["", "## 5. 判定与限制", "",
              "- 三seed全部核心能力门通过：`%s`。" % summary["raw_all_seed_core_passed"],
              "- J原D混合配方三个能力门全部通过：`%s`。" % summary["J_dense_original_mixed_capability_all_passed"],
              "- 结构检查的非零输出头副本只检验距离通路和不变性，不是科学训练；O同号几何退化是预期失效对照，不作为软件故障重试。",
              "- E4局部四邻居足以给出精确条件概率，因此扩展MASK后的真值不变；这不意味着任意K1/K2改变物理域后真值也不变。固定clock与自然clock分列。",
              "- 若能力未通过，只能说这个固定有限训练配方未学会/未迁移，不能据此证明D没有表达能力。3seed不足以确认总体收益；所有seed与失败都保留。",
              "- 旧J的K4/32/512控制不替代大系统K1/K2的精确条件评估；本轮没有验证大系统稀疏条件真值或老师的长程泛化主张。",
              "- 输出包含所有逐行预测、exact_states、输入划分、训练日志与可恢复final。备份/复核以单独manifest与receipt为准。",
              "", "阶段2到此停止；不自动进入架构修复或新外推实验。", ""]
    report_path = c.OUT/"RESULTS_ZH.md"
    if report_path.exists():
        raise FileExistsError(str(report_path))
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("STAGES12_COMPLETE", json.dumps({k: v for k, v in summary.items() if k not in ("seeds", "aggregate")}), flush=True)


def verify_and_package():
    protocol = c.read(c.OUT/"run_protocol.json")
    c.check_frozen(protocol["sources"])
    c.check_deadline(); disk_check()
    bank = c.load(c.OUT/"bank.npz")
    checks = 0
    for label in c.CONFIG["seed_labels"]:
        folder = c.OUT/("training/seed_%d" % label)
        completed = c.read(folder/"complete.json")
        if c.sha(folder/"final.pt") != completed["final_sha256"]:
            raise RuntimeError("Final checkpoint SHA mismatch")
        records = [json.loads(line) for line in (folder/"trace.jsonl").read_text(encoding="utf-8").splitlines()]
        if [r["step"] for r in records] != list(range(1, c.CONFIG["steps"]+1)):
            raise RuntimeError("Incomplete training log")
        if not np.isfinite([[r[k] for k in ("loss", "grad_norm", "lr")] for r in records]).all():
            raise RuntimeError("Nonfinite training log")
        for suffix in ("raw", "ema_secondary"):
            pred = c.load(folder/(suffix+"_predictions.npz"))
            recomputed = c.metrics(pred["probability"], bank["target"])
            for key in recomputed:
                if not np.array_equal(recomputed[key], pred[key]):
                    raise RuntimeError("Prediction scores mismatch")
                checks += 1
        checks += len(records)
    c.write(c.OUT/"final_integrity_check.json", dict(status="passed", checks=checks,
        time=time.time(), legacy_check=legacy_check(), frozen_sources_unchanged=True,
        no_model_training_or_bootstrap_rerun=True))
    exports = c.ROOT/"artifacts/basic_capability_stages12_exports_20260928"
    exports.mkdir(exist_ok=False)
    paths = sorted(p for p in c.OUT.rglob("*") if p.is_file())
    paths += [c.ROOT/rel for rel in protocol["sources"]]
    paths = sorted(set(paths))
    manifest = dict(study=c.CONFIG["study"], files=[dict(path=str(p.relative_to(c.ROOT)).replace("\\", "/"),
        bytes=p.stat().st_size, sha256=c.sha(p)) for p in paths], excludes=[])
    archive = exports/"stages12_complete_v1.tar.gz"
    with tarfile.open(archive, "x:gz", compresslevel=3, dereference=True) as tar:
        for path in paths:
            tar.add(path, arcname=str(path.relative_to(c.ROOT)).replace("\\", "/"), recursive=False)
    manifest.update(archive=archive.name, archive_sha256=c.sha(archive), archive_bytes=archive.stat().st_size)
    c.write(exports/"stages12_complete_v1.manifest.json", manifest)
    expected = {row["path"]: row for row in manifest["files"]}
    seen = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise RuntimeError("Foreign or duplicate archive entry")
            seen.add(member.name)
            value = tar.extractfile(member).read()
            record = expected[member.name]
            if len(value) != record["bytes"] or hashlib.sha256(value).hexdigest() != record["sha256"]:
                raise RuntimeError("Archive member SHA mismatch")
    if seen != set(expected):
        raise RuntimeError("Incomplete archive coverage")
    receipt = dict(status="passed", members=len(seen), archive_sha256=manifest["archive_sha256"],
        archive_bytes=manifest["archive_bytes"], time=time.time(),
        elapsed_since_first_timing_seconds=time.time()-c.read(c.OUT/"budget.json")["started"],
        backup_location=str(exports), same_D_volume_not_an_independent_device=True,
        full_source_data_predictions_recoverable_weights_and_logs_covered=True)
    c.write(exports/"stages12_complete_v1.verification.json", receipt)
    print("BACKUP_VERIFIED", json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "preflight", "run", "report", "package"])
    args = parser.parse_args()
    try:
        {"prepare": prepare, "preflight": preflight, "run": run,
         "report": report, "package": verify_and_package}[args.mode]()
    except Exception:
        error = dict(time=time.time(), mode=args.mode, traceback=traceback.format_exc(),
                     no_automatic_retry=True)
        if c.OUT.exists():
            path = c.OUT/("failure_%s_%d.json" % (args.mode, time.time_ns()))
            c.write(path, error)
        raise


if __name__ == "__main__":
    main()
