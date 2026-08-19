# 正式筛选 checkpoint

本目录保存在报告和逐样本分析中实际使用的 11 个 `best_val.pt`。它们通过 Git LFS
管理；克隆后先执行：

```bash
git lfs pull
```

目录按实验阶段分开：

- `level1/`：`T0`、`T3`、`Pphase`、`Punit`；
- `stage2a/`：参数匹配 dense controls 与 `LG-T3/LG-Punit`；
- `stage2b/`：`LG-Gap-Unit`、`LG-U-RandPE`、`LG-Gap-Matched`。

每个模型同时保存 `run_config.json`、`history.json`、`performance.json`；Stage 2B
还保存逐 update 的 `train.jsonl`。为避免重复，未保留同一训练的 `last.pt`、benchmark
和 smoke checkpoint。

权重校验见 [`SHA256SUMS`](SHA256SUMS)。这些 checkpoint 是单训练 seed 筛选资产，
不能被误写为多 seed 正式结论。
