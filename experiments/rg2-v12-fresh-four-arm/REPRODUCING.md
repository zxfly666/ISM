# 复核与复现

先在本实验目录操作，例如仓库中的 `experiments/rg2-v12-fresh-four-arm/`。查看结果和复算保存的统计不需要 GPU；重新训练与生成需要相应 GPU 环境和时间。

## 1. 核对公开文件

```bash
python tools/verify_publication.py --root .
```

该步骤按 `publication-manifest.json` 核对逐文件大小和 SHA-256。`data-manifest.json` 单独记录从实验档案复制的科学载荷；完整公开清单还覆盖 Markdown、图注和校验工具。

## 2. 不重新训练，独立复算结果

需要 NumPy；统计审计另需要 SciPy。以下命令读取已公开的概率和自旋，并将新审计写到另一个位置：

```bash
python project/scripts/rg2_v12/audit_results.py --project project --runs project/runs --stage A --out audit_A_recheck.json
python project/scripts/rg2_v12/audit_results.py --project project --runs project/runs --stage B --out audit_B_recheck.json
python project/scripts/rg2_v12/audit_statistics.py --evaluation project/runs/B_eval --out audit_statistics_recheck.json
```

前两条用独立整数点对求和、直接 Hamiltonian 枚举及 NumPy 概率计算复核预测和物理观测；第三条逐次重算保存的 120,000 个对比 bootstrap 值，并用 SciPy 检查分布距离。B 的训练张量审计已在原 GPU 环境对24 个阶段 checkpoint 执行，记录位于 `project/runs/B_eval/training_audit.json`；上述 CPU 结果复算会核对该记录的通过状态，不把它误称为再次加载所有优化器张量。

## 3. 重画图表

需要 NumPy 和 Matplotlib。使用新输出目录，保留发布版：

```bash
python project/scripts/rg2_v12/report.py --project project --runs project/runs --out redraw_A --require-complete
python project/scripts/rg2_v12/report_b.py --project project --runs project/runs --out redraw_B
```

报告使用保存的原始数组，生成同名 PNG、PDF、CSV、图注与绘图配置。PDF 内部时间戳和不同字体版本可能改变文件字节，不能用这些差异代替数值比较。B 的效应图在发布前仅调整了过长轴标签的换行和 seed 颜色说明；原绘图源码也保留于 `project/scripts/rg2_v12/history/`。冻结训练和评价源码未因此改变。

## 4. 取回权重和完整输入

从 [Release](https://github.com/zxfly666/ISM/releases/tag/rg2-v12-20261125) 下载两个 `.tar.gz` 到同一目录，例如 `downloads/`，然后执行：

```bash
python tools/verify_publication.py --root . --assets downloads --extract
```

工具先核对压缩包及每个成员的 SHA；仅向本实验目录内解压普通文件。已有相同文件会校验后保留，已有不同内容会报错。之后 `project/inputs/ema/`、`project/runs/B_v2/` 和 `project/data/level1/` 中可找到对应权重与数据。

## 5. 重评估已训练模型

原计算环境为 Python 3.12.3、NumPy 1.26.4、PyTorch `2.7.0a0+7c8ec84dab.nv25.03`、RTX 4090 24 GB。模型及数据加载会核对身份；更换软件或硬件不自动保证逐位相同。

进入 `project/` 后，可在新输出目录重做最终 EMA 的局部预测、生成和开发统计：

```bash
python scripts/rg2_v12/phase_b_evaluate.py --training runs/B_v2 --out runs/B_eval_reproduction
```

这会产生新的计算费用；它仍只执行 B 的固定开发评价，不会启动 C。已公开的生成自旋与随机身份足以复核本文数值，无须为了阅读报告重复生成。

## 6. 从零训练同一开发配方

冻结源码的训练数据默认路径记录了原服务器位置。为保留源码 SHA，可在启动时把 `phase_b.TRAIN` 指向公开的同 SHA 文件，而不修改源码。仍在 `project/` 中执行：

```bash
python -c "import sys; from pathlib import Path; sys.path.insert(0,'scripts/rg2_v12'); import phase_b as b; b.TRAIN=Path('data/level1/parents_l1024.npz').resolve(); b.main()" preflight --out runs/B_reproduction
python -c "import sys; from pathlib import Path; sys.path.insert(0,'scripts/rg2_v12'); import phase_b as b; b.TRAIN=Path('data/level1/parents_l1024.npz').resolve(); b.main()" train --out runs/B_reproduction --gate runs/A_to_B_gate.json
python scripts/rg2_v12/phase_b_evaluate.py --training runs/B_reproduction --out runs/B_eval_reproduction_fresh
```

预检覆盖训练／验证的物理链隔离、配对输入、初始化及四臂连续／恢复的 raw、EMA、AdamW 逐位一致性。使用新目录，避免把已有完成标记当成新训练。每500 更新保存恢复状态，4k/8k/12k 另存完整阶段 checkpoint；源码或配置变化时恢复会拒绝身份不匹配。

这条路径重做相同开发 seed，用于复现实验；它不会增加独立训练证据，也不是 C 的新 seed 确认。A 的旧权重诊断和原定 A/B/C 设计见 `project/output/RG-2_v1.2.html`；HTML 下载后可直接在浏览器阅读。
