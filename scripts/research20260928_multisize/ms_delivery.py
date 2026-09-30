"""Readable results plus complete same-volume archive and member verification."""
from __future__ import annotations
import hashlib
import json
import math
import tarfile
import time
from pathlib import Path
import numpy as np
import ms_common as c


def report():
    s=c.read(c.OUT/"final_summary.json")
    a=c.read(c.OUT/"audit.json")
    b=c.read(c.OUT/"budget.json")
    f=c.read(c.OUT/"diagnostic/summary.json")
    lines=["# Dense单/多尺寸配对诊断：完整结果","",
        "本报告是局部必要能力诊断，不是老师完整idea的结论。原J与上轮实验保持不变；本轮没有训练O、修改架构或追加MC/生成。","",
        "## 冻结比较与判定","",
        "三对新初始化，原dense 1,976,706参数，每臂2,048步。A4仅4×4，B46仅将奇数步四邻居微批放到6×6；K1/K2物理任务、标签、clock、初始化及所有抽样配对。K4的训练clock固定0.75。两臂均32+16微批、一次更新；不是FLOP匹配。","",
        "12×12固定clock是新主尺寸，全部六final锁定后才作模型评价；8×8复现已知问题。raw为固定主权重，EMA/自然clock均次要，没有选优。每格能力门=max KL≤0.01且max概率误差≤0.05。实质比较门=每seed平均KL(B−A)≤−0.005；不作三seed总体显著性宣称。","",
        "| 门 | 结果 |","|---|---|"]
    names={"B_new_size_ability":"B的12×12绝对能力（三seed全过）","B_retention":"B的K1/K2留出与4×4保留",
        "B_trained_size_ability":"B的6×6训练尺寸能力","paired_practical_improvement":"三对实质改善",
        "all_required_diagnostic_gates":"全部诊断门"}
    for k,v in s["gates"].items():
        lines.append("| %s | %s |"%(names[k],"通过" if v else "未通过"))
    lines += ["","主要配对平均KL差（B−A）："+", ".join("%.8f"%x for x in s["primary"]["paired_mean_kl_B_minus_A"])+"。",
        "三seed平均=%.8f，seed SD=%.8f，配对MCSE=%.8f。精确条件真值无MC误差；模式/对称位置不冒充独立seed。"%(s["primary"]["mean"],s["primary"]["seed_sd"],s["primary"]["paired_mcse"]),""]
    g=s["gates"]
    if g["all_required_diagnostic_gates"]:
        conclusion="本固定多容器训练通过所测局部门，并在三对初始化上达到事前比较幅度。支持下一步再检查原混合训练配方，但不是长程几何、联合分布或完整idea已验证。"
    elif not g["B_trained_size_ability"]:
        conclusion="B连新增训练尺寸能力都未全部过门：该固定训练配方的必要能力尚未建立，不能据此把新尺寸失效唯一归因不可表达或几何外推。"
    elif not g["B_new_size_ability"]:
        conclusion="B学会了训练尺寸，但未在三seed上全部通过12×12：本次4/6多容器训练不足以建立所测尺寸稳健性。相对改善与未达绝对门必须同时报告。"
    elif not g["B_retention"]:
        conclusion="新尺寸能力虽通过，但原能力保留门未全过，不能称无代价修复。"
    else:
        conclusion="绝对能力门通过，但配对实质改善门未全过；不能把成功明确归功于本次多尺寸变化。"
    lines += ["解释："+conclusion,"","## 全部能力表（主raw与次要EMA均保留）","",
        "| 权重 | seed | 臂 | cell | mean KL | max KL | max概率误差 | 门 |",
        "|---|---:|---|---|---:|---:|---:|---|"]
    for r in s["model_rows"]:
        lines.append("| %s | %d | %s | %s | %.8g | %.8g | %.8g | %s |"%(r["weight"],r["seed"],r["arm"],r["cell"],r["mean_kl"],r["max_kl"],r["max_probability_error"],"通过" if r["ability_passed"] else "未过"))
    lines += ["","## 只读旧D模型：因素诊断","",
        "N16/32/64 × 外围MASK坐标倍率1/2 × clock0.75/0.9375，固定四个真实单位邻居；先近/先远两个嵌套布局，N64是同一集合，不重复计为独立数据。改变N仍改变外围点集合，因此不是仅softmax分母的内部因果识别。以下汇总展示所有条件效应的范围，完整配对效应在diagnostic/summary.json。","",
        "| 因素 | 条件配对数 | 最大概率改变量范围 | mean KL差范围 |","|---|---:|---:|---:|"]
    for factor in ("valid_token_count_and_nested_set","peripheral_coordinates","clock","PAD"):
        rows=[r for r in f["effects"] if r["factor"]==factor]
        change=[r["max_probability_change"] for r in rows]
        kl=[r["mean_kl_change"] for r in rows if "mean_kl_change" in r]
        lines.append("| %s | %d | %.8g～%.8g | %s |"%(factor,len(rows),min(change),max(change),"—" if not kl else "%.8g～%.8g"%(min(kl),max(kl))))
    lines += ["","PAD阴性对照均满足2e−5概率差容差。因素变化是输入表示的受控变化，不唯一证明内部attention稀释、RoPE外推或clock失配是唯一原因。","",
        "## 训练成本与复现","",
        "实际更新计时（含前向、反向、EMA及真实输入hash，不含所有管理I/O）：A4 %.3f秒，B46 %.3f秒。"%(a["training_update_seconds_by_arm"]["A4"],a["training_update_seconds_by_arm"]["B46"]),
        "训练有效token累计：A4 %d，B46 %d；总更新各6144，监督样本各294912。"%(a["training_valid_tokens_by_arm"]["A4"],a["training_valid_tokens_by_arm"]["B46"]),
        "从prepare到审计完成实测%.3f秒；最终备份耗时在exports核验回执中另报。原始截止epoch %.6f，不重置。"%(a["time"]-b["started"],b["deadline"]),
        "9项数据单测、相同下一步raw/EMA/AdamW恢复测试、全部12288步真实输入与物理配对hash重建、96个最终预测文件逐指标重算、78个旧模型诊断文件、六final CPU恢复及冻结源检查通过。没有自动续训。","",
        "交付：final_summary.json含CE/KL/期望Brier/超额Brier及全对照；evaluation保存逐模式概率、真值与身份；六final含raw/EMA/AdamW/RNG；diagnostic保存因素输入及预测；banks保存实际输入。所有实际用于本轮的父研究权重/数据/依赖随本轮备份保留。重复last不进包但不删除。","",
        "备份目标为同D盘的artifacts/dense_multisize_control_exports_20260928，不是异盘/异地灾备。整包与逐成员SHA、科学清单覆盖以对应verification.json为准。","",
        "## 仍未验证","",
        "1. K4可由四邻居符号计数oracle解决；它不等于学会距离规律。","2. K1/K2使用4×4有限开边界精确系统，未把标签搬到大物理域；旧几何留出已被看过，属于能力保留。",
        "3. 此次不是J的混合mask/采样训练配方，不证明原J已学会。自然clock6/8/12未在本轮K4训练中使用。",
        "4. 仅三个初始化、两训练尺寸、一个新主尺寸；未测大系统联合分布、长程相关或生成。未通过不否定老师全部idea，通过也不验证完整idea。",
        "5. 本轮结束即停，不自动追加尺寸/种子/步数，不启动下一轮。",""]
    text="\n".join(lines)
    with (c.OUT/"RESULTS_ZH.md").open("x",encoding="utf-8") as handle:
        handle.write(text)
    c.write(c.OUT/"interpretation.json",dict(conclusion=conclusion,gates=g,time=time.time()))


def package():
    c.deadline()
    c.disk()
    proto=c.read(c.OUT/"run_protocol.json")
    c.check_sources(proto["sources"])
    c.EXPORT.mkdir(parents=True,exist_ok=True)
    archive=c.EXPORT/"dense_multisize_complete_v1.tar.gz"
    if archive.exists():
        raise FileExistsError(str(archive))
    paths=[p for p in c.OUT.rglob("*") if p.is_file() and p.name!="last.pt" and "__pycache__" not in p.parts]
    paths += [c.ROOT/p for p in proto["sources"]]
    paths += [c.PARENT/"bank.npz",c.PARENT/"exact_states.npz",c.PARENT/"run_protocol.json"]
    for seed in c.CFG["parent_seeds"]:
        paths += [c.PARENT/("training/seed_%d"%seed)/name for name in ("final.pt","complete.json")]
    paths=sorted(set(paths))
    files=[]
    for path in paths:
        c.deadline()
        files.append(dict(path=path.relative_to(c.ROOT).as_posix(),bytes=path.stat().st_size,sha256=c.sha(path)))
    manifest=c.EXPORT/"dense_multisize_complete_v1.manifest.json"
    c.write(manifest,dict(study=c.CFG["study"],files=files,archive=archive.name,
        excludes=[str(p.relative_to(c.ROOT).as_posix()) for p in sorted(c.OUT.rglob("last.pt"))],
        includes_all_new_scientific_files=True,same_D_volume_not_independent_device=True))
    with tarfile.open(archive,"x:gz",compresslevel=5,dereference=True) as tar:
        for path in paths:
            c.deadline()
            tar.add(path,arcname=path.relative_to(c.ROOT).as_posix(),recursive=False)
    expected={r["path"]:r for r in files}
    seen=set()
    with tarfile.open(archive,"r:gz") as tar:
        for member in tar:
            c.deadline()
            assert member.isfile() and member.name in expected and member.name not in seen
            h=hashlib.sha256()
            with tar.extractfile(member) as handle:
                for block in iter(lambda:handle.read(4*1024**2),b""):
                    h.update(block)
            assert member.size==expected[member.name]["bytes"] and h.hexdigest()==expected[member.name]["sha256"]
            seen.add(member.name)
    assert seen==set(expected)
    # Recheck originals after archive reads; no final scientific files are then modified.
    for row in files:
        assert c.sha(c.ROOT/row["path"])==row["sha256"]
    current={p.relative_to(c.ROOT).as_posix() for p in c.OUT.rglob("*") if p.is_file() and p.name!="last.pt" and "__pycache__" not in p.parts}
    assert current<=seen
    c.deadline()
    receipt=dict(status="passed",time=time.time(),archive=archive.name,archive_bytes=archive.stat().st_size,
        archive_sha256=c.sha(archive),manifest_sha256=c.sha(manifest),members=len(seen),
        all_new_scientific_paths_covered=len(current),missing_scientific_paths=[],
        elapsed_prepare_to_verified_seconds=time.time()-c.read(c.OUT/"budget.json")["started"],
        same_D_volume_not_an_independent_device=True,excludes_duplicate_last_only=True,
        final_weights=6,parent_read_only_weights=3,disk_free=c.disk(),no_automatic_next_stage=True)
    c.write(c.EXPORT/"dense_multisize_complete_v1.verification.json",receipt)
    c.write(c.EXPORT/"closed.json",dict(status="science_audit_and_backup_complete",time=time.time(),
        receipt="dense_multisize_complete_v1.verification.json",report=(c.OUT/"RESULTS_ZH.md").relative_to(c.ROOT).as_posix()))
