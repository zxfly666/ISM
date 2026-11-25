# 完整核心图版索引

本索引包含 36 张图：31 张来自冻结实测证据，5 张为明确标注的方法示意；每张均提供便于阅读的 PNG 与可缩放的 PDF。

共同冻结证据版本：`5516370b3ae6e98160d9468668acdcbf851dc902`。中文说明依据[实测图源记录](provenance/physical_figure_sources.json)与[方法图源记录](provenance/method_figure_sources.json)整理；完整参数、固定样本身份、脚本和输入文件哈希见相应记录。

实测图可由[绘图脚本](scripts/physical_figures.py)重建，方法图可由[示意图脚本](scripts/method_figures.py)重建；[实测图审核](provenance/physical_figure_review.json)和[方法图审核](provenance/method_figure_review.json)记录了各自的检查范围。

| 图版 | 类型 | 来源与用途 | 文件 |
|---|---|---|---|
| `F01_mc_fields` | 实测结果 | 来自固定身份的真实 MC 场，展示同一 W96 细场、中心抽点、多数投票及两种粗化的分歧位置。 | [PNG](figures/F01_mc_fields.png) · [PDF](figures/F01_mc_fields.pdf) |
| `F02_mc_statistics_W144` | 实测结果 | 来自全部真实 W144 MC 场，比较细场、中心抽点与多数投票的关联和磁化，并区分格距与物理距离。 | [PNG](figures/F02_mc_statistics_W144.png) · [PDF](figures/F02_mc_statistics_W144.pdf) |
| `F02_mc_statistics_W96` | 实测结果 | 来自全部真实 W96 MC 场，比较细场、中心抽点与多数投票的关联和磁化，并区分格距与物理距离。 | [PNG](figures/F02_mc_statistics_W96.png) · [PDF](figures/F02_mc_statistics_W96.pdf) |
| `F03a_four_arm_construction` | 方法示意 | 依据实际训练代码绘制方法示意，说明 F/P/D/M 的共同流、辅助流、输入构造与坐标区别。 | [PNG](figures/F03a_four_arm_construction.png) · [PDF](figures/F03a_four_arm_construction.pdf) |
| `F03b_hand_example` | 方法示意 | 来自明确标注的人为可手算数组，说明中央裁剪、中心抽点、多数投票及投票强度被丢弃的机制。 | [PNG](figures/F03b_hand_example.png) · [PDF](figures/F03b_hand_example.pdf) |
| `F04a_model` | 方法示意 | 依据实际模型代码绘制方法示意，说明 token、任务与时间条件、RoPE、注意力和逐位置输出。 | [PNG](figures/F04a_model.png) · [PDF](figures/F04a_model.pdf) |
| `F04b_mask_training` | 方法示意 | 依据实际训练规则及人为数组绘制示意，说明遮盖、隐藏位置监督、归一化和交替优化更新。 | [PNG](figures/F04b_mask_training.png) · [PDF](figures/F04b_mask_training.pdf) |
| `F04c_generation` | 方法示意 | 依据实际采样规则及人为过程示例绘制示意，说明从全 MASK 出发的分组揭示与随机身份配对。 | [PNG](figures/F04c_generation.png) · [PDF](figures/F04c_generation.pdf) |
| `F05_old_local_trajectories` | 实测结果 | 来自旧 I-F 暖启动谱系的固定局部诊断，比较三个 seed 在共同细任务更新进程上的条件 KL。 | [PNG](figures/F05_old_local_trajectories.png) · [PDF](figures/F05_old_local_trajectories.pdf) |
| `F06_samples_s112501_S256` | 实测结果 | 来自 Fresh seed 112501、S=256 的预登记生成 ID 0 与固定 MC 参考，展示细场及同图多数投票粗场。 | [PNG](figures/F06_samples_s112501_S256.png) · [PDF](figures/F06_samples_s112501_S256.pdf) |
| `F06_samples_s112501_S64` | 实测结果 | 来自 Fresh seed 112501、S=64 的预登记生成 ID 0 与固定 MC 参考，展示细场及同图多数投票粗场。 | [PNG](figures/F06_samples_s112501_S64.png) · [PDF](figures/F06_samples_s112501_S64.pdf) |
| `F06_samples_s112502_S256` | 实测结果 | 来自 Fresh seed 112502、S=256 的预登记生成 ID 0 与固定 MC 参考，展示细场及同图多数投票粗场。 | [PNG](figures/F06_samples_s112502_S256.png) · [PDF](figures/F06_samples_s112502_S256.pdf) |
| `F06_samples_s112502_S64` | 实测结果 | 来自 Fresh seed 112502、S=64 的预登记生成 ID 0 与固定 MC 参考，展示细场及同图多数投票粗场。 | [PNG](figures/F06_samples_s112502_S64.png) · [PDF](figures/F06_samples_s112502_S64.pdf) |
| `F07_fine_correlations` | 实测结果 | 来自 Fresh 保存样本与独立 MC 参考，比较两个 seed、两种采样步数的完整细场原始关联及长程评价区间。 | [PNG](figures/F07_fine_correlations.png) · [PDF](figures/F07_fine_correlations.pdf) |
| `F08_coarse_correlations` | 实测结果 | 来自同一 Fresh 细样本的多数投票粗场与 MC 粗场，比较完整原始粗关联并标明对应物理距离。 | [PNG](figures/F08_coarse_correlations.png) · [PDF](figures/F08_coarse_correlations.pdf) |
| `F09_coarse_abs_m` | 实测结果 | 来自同一 Fresh 细样本的多数投票粗场与 MC 粗场，使用完整 ECDF 比较绝对粗磁化分布。 | [PNG](figures/F09_coarse_abs_m.png) · [PDF](figures/F09_coarse_abs_m.pdf) |
| `F09_coarse_energy` | 实测结果 | 来自同一 Fresh 细样本的多数投票粗场与 MC 粗场，比较操作性的负近邻乘积分布，该量不等于原 Hamiltonian 能量。 | [PNG](figures/F09_coarse_energy.png) · [PDF](figures/F09_coarse_energy.pdf) |
| `F09_coarse_m` | 实测结果 | 来自同一 Fresh 细样本的多数投票粗场与 MC 粗场，使用完整 ECDF 比较有符号粗磁化分布。 | [PNG](figures/F09_coarse_m.png) · [PDF](figures/F09_coarse_m.pdf) |
| `F09_fine_abs_m` | 实测结果 | 来自 Fresh 细样本与 MC 参考，使用完整支持区间的 ECDF 比较绝对磁化分布。 | [PNG](figures/F09_fine_abs_m.png) · [PDF](figures/F09_fine_abs_m.pdf) |
| `F09_fine_energy` | 实测结果 | 来自 Fresh 细样本与 MC 参考，使用完整理论支持区间的 ECDF 比较每键能量分布。 | [PNG](figures/F09_fine_energy.png) · [PDF](figures/F09_fine_energy.pdf) |
| `F09_fine_energy_zoom` | 实测结果 | 来自与全范围能量图相同的 Fresh 和 MC 样本，以覆盖全部实测值的局部放大呈现分布位移。 | [PNG](figures/F09_fine_energy_zoom.png) · [PDF](figures/F09_fine_energy_zoom.pdf) |
| `F09_fine_m` | 实测结果 | 来自 Fresh 细样本与 MC 参考，使用完整支持区间的 ECDF 比较有符号磁化分布。 | [PNG](figures/F09_fine_m.png) · [PDF](figures/F09_fine_m.pdf) |
| `F09_fine_m_histogram` | 实测结果 | 来自 Fresh 细样本与 MC 参考，用预定等宽分箱呈现有符号磁化，需结合完整 ECDF 解读小样本峰形。 | [PNG](figures/F09_fine_m_histogram.png) · [PDF](figures/F09_fine_m_histogram.pdf) |
| `F10_energy_longrange_tradeoff` | 实测结果 | 来自逐 seed 的 Fresh 保存指标，联合展示能量分布距离与长程原始关联误差，不合成为单一分数。 | [PNG](figures/F10_energy_longrange_tradeoff.png) · [PDF](figures/F10_energy_longrange_tradeoff.pdf) |
| `F11_old_three_paths_S256` | 实测结果 | 来自旧 Fine+RG 终点 EMA、S=256 的保存结果，比较真实 MC 粗化、细生成再粗化及直接粗生成三条路径。 | [PNG](figures/F11_old_three_paths_S256.png) · [PDF](figures/F11_old_three_paths_S256.pdf) |
| `F11_old_three_paths_S64` | 实测结果 | 来自旧 Fine+RG 终点 EMA、S=64 的保存结果，比较真实 MC 粗化、细生成再粗化及直接粗生成三条路径。 | [PNG](figures/F11_old_three_paths_S64.png) · [PDF](figures/F11_old_three_paths_S64.pdf) |
| `F12_decomposition_all_boundaries` | 实测结果 | 来自旧 Fine+RG 的全部预定 MC 边界，展示小块联合 KL 的因子化与预测分解及逐边界变异。 | [PNG](figures/F12_decomposition_all_boundaries.png) · [PDF](figures/F12_decomposition_all_boundaries.pdf) |
| `F12_joint_s92601` | 实测结果 | 来自旧 Fine+RG seed 92601 的预定固定边界，比较精确小块联合分布、四种揭示计划及 KL 分解。 | [PNG](figures/F12_joint_s92601.png) · [PDF](figures/F12_joint_s92601.pdf) |
| `F12_joint_s92602` | 实测结果 | 来自旧 Fine+RG seed 92602 的预定固定边界，比较精确小块联合分布、四种揭示计划及 KL 分解。 | [PNG](figures/F12_joint_s92602.png) · [PDF](figures/F12_joint_s92602.pdf) |
| `F12_joint_s92603` | 实测结果 | 来自旧 Fine+RG seed 92603 的预定固定边界，比较精确小块联合分布、四种揭示计划及 KL 分解。 | [PNG](figures/F12_joint_s92603.png) · [PDF](figures/F12_joint_s92603.pdf) |
| `F13_mc_precision` | 实测结果 | 来自冻结的 MC–MC 重抽样校准，展示不同伪生成样本量的关联误差分位数，用于解释有限样本精度。 | [PNG](figures/F13_mc_precision.png) · [PDF](figures/F13_mc_precision.pdf) |
| `F14_fixed_probe_gradients` | 实测结果 | 来自 Fresh 固定 W48 验证 probe 的梯度日志，比较共同流与辅助流的余弦和范数比。 | [PNG](figures/F14_fixed_probe_gradients.png) · [PDF](figures/F14_fixed_probe_gradients.pdf) |
| `F14_training_s112501` | 实测结果 | 来自 Fresh seed 112501 的训练和验证日志，比较共同流损失、辅助流损失、共同细任务验证和实际裁剪比例。 | [PNG](figures/F14_training_s112501.png) · [PDF](figures/F14_training_s112501.pdf) |
| `F14_training_s112502` | 实测结果 | 来自 Fresh seed 112502 的训练和验证日志，比较共同流损失、辅助流损失、共同细任务验证和实际裁剪比例。 | [PNG](figures/F14_training_s112502.png) · [PDF](figures/F14_training_s112502.pdf) |
| `F15_connected_correlations` | 实测结果 | 来自 Fresh 保存样本与 MC 参考，展示扣除系综平均磁化平方后的连通关联，区别于原始关联。 | [PNG](figures/F15_connected_correlations.png) · [PDF](figures/F15_connected_correlations.pdf) |
| `F15_local_bonds` | 实测结果 | 来自 Fresh 保存样本与 MC 参考，比较直接测得的最近邻异号键比例及其等价的 G(1)。 | [PNG](figures/F15_local_bonds.png) · [PDF](figures/F15_local_bonds.pdf) |
