# 数据、图表与权重

本轮实际使用的科学结果、绘图数据和源代码已公开。较大的权重与完整训练输入通过同一仓库的 [RG-2 Release](https://github.com/zxfly666/ISM/releases/tag/rg2-v12-20261125) 提供。

## 仓库中的内容

| 目录／文件 | 内容 |
|---|---|
| `stage_A/` | A 的中文报告、74 张 PNG／PDF、CSV、精确状态表、链 bootstrap、整数点对统计与图注 |
| `stage_B/` | B 的33 张 PNG／PDF、96,000 次更新表、验证／梯度记录、逐 seed 观测量、描述性区间与图注 |
| `project/runs/A_v2/` | A 精确局部预测、960 张原始生成、自旋身份与细粗统计 |
| `project/runs/A_supp/` | A 保留预测、396 组小块 P/Q、真实 MC 粗化统计及样本量诊断 |
| `project/runs/B_v2/` | 正式训练配置、逐次日志、验证、梯度、初始化及完成记录；权重见 Release |
| `project/runs/B_eval/` | 48 格局部预测、512 张生成、逐图统计、全部 bootstrap draws 与训练审计 |
| `project/inputs/` | 本轮实际使用的历史 W144 MC 裁剪、局部输入银行、模型／输入身份清单；旧 EMA 见 Release |
| `project/scripts/`、`project/ism_diffusion/` | 冻结计算源码、模型及后处理源码；原始与修复前版本保留于 `history/` |
| `verification/` | 独立复算、备份校验及历史预测复现记录 |
| `publication-manifest.json` | 本公开目录逐文件大小和 SHA-256 |
| `release-assets.json` | 两个大文件及其每个成员的大小、SHA-256 与原目录位置 |

`project/runs/` 根目录也包含 A→B 验收、数据划分审计、B 启动前编号修复，以及 A/B 独立结果复算。补充统计审计重新计算了全部 **120,000 个保存的对比 bootstrap 值**，最大差为 0，并用 SciPy 复核了 96 项 Wasserstein 距离。

公开日志仅将四条启动记录中的服务器名称替换为 `[redacted-server]`，科学数值保持原样；原始与公开文件 SHA 的对应关系见 `verification/publication_redactions.json`。复制后的 A、B 及统计审计也已重新执行，结果见 `verification/public_*_recheck.json`。

## 大文件

- [rg2-v12-checkpoints.tar.gz](https://github.com/zxfly666/ISM/releases/download/rg2-v12-20261125/rg2-v12-checkpoints.tar.gz)：B 的4k/8k/12k raw、EMA、优化器、RNG，final EMA、共同初始化、最后恢复点，以及单独标识的预检恢复状态。
- [rg2-v12-reproduction-inputs.tar.gz](https://github.com/zxfly666/ISM/releases/download/rg2-v12-20261125/rg2-v12-reproduction-inputs.tar.gz)：A 的21 份历史 EMA、35 个补充输入银行，以及完整训练／验证输入 `parents_l1024.npz`。

两个压缩包均以本实验目录为解压根；使用[复现说明](REPRODUCING.md)中的验证工具可在解压前核对压缩包和每个成员。它们没有放进普通 Git 历史，避免每次克隆都下载全部权重。

训练数据 SHA-256 为 `1f7de1ec81e82ebcfcbc4134d5670ee35711230f05d00cd037c7b9e575ef8934`；实际评估裁剪 MC 的 SHA-256 为 `4f2015591c3e0917b0d58e5473a242ade624c18cbc8e2506ecee15a03d57a4fd`。训练和参考 MC 的完整生成元数据与链种子均在 `project/runs/` 中。

原始历史评估 MC 的完整 1024×1024 父场未在本轮重复打包；本轮计算实际使用的全部 W144 裁剪、链／父场／坐标身份和输入银行已经提供。其原始全场 SHA 为 `87eadf3ac3d099ab3425d89ffc16101b9425ed7e0c0747a8175000100c36db55`。复核本文所有数值不需要重新下载该完整历史场。

原始服务器完整 A/B 快照另有逐文件核验的本地备份，回执见 `verification/`。同一台电脑上的另一个目录不等同于异地备份；本次 GitHub 公开文件与 Release 才是另一存储位置。

## 数据身份与解释边界

本轮 A 沿用三条旧训练谱系，B 是两个新的开发训练 seed；它们均不并入未来 C 的独立确认。8 条参考 MC 链、32 张配对生成与两条训练 seed 是不同层次的随机单位，不能合并成上千个“训练重复”。图像没有做事后正负翻转；所有预登记构型 ID 均保留。
