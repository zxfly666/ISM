# L=64 Ising 离散扩散实验：本地下载清单

## 1. 下载状态

- 远端来源：`root@connect.weste.seetacloud.com:/root/autodl-tmp/ISM`
- 本地工程根目录：`C:\Users\zhangxiangfei\Desktop\ISM`
- 正式训练产物与两份数据共核验 57 个文件：缺失 0，SHA256 不一致 0。
- 远端源码、配置与文档共核验 94 个文件：93 个逐字节一致，缺失 0。
- 唯一源码差异是 `ism_diffusion/__init__.py`：本地版使用延迟导入 PyTorch，属于正式训练完成后的轻量兼容性改进；训练入口直接导入具体模块，因此不改变本次训练行为。远端训练时原版已保存在 `artifacts/pilot_l64/source_snapshot/ism_diffusion/__init__.py`。

## 2. 主要内容

| 内容 | 本地位置 | 规模/说明 |
|---|---|---|
| 正式训练、里程碑与正式采样产物 | `artifacts/pilot_l64/` | 远端原始产物 55 个文件，约 1.333 GiB；另附哈希清单和源码差异快照 |
| 最终综合分析 | `artifacts/final_l64/` | 35 个文件，11,855,962 字节；含表格、图像与原始样本 |
| 正式训练数据 | `data/ising_l64_pilot.npz` | 训练 15,000；验证 2,500；测试 2,500；每张 64×64 |
| 独立 MC 参考集 | `data/ising_l64_reference_10k.npz` | 测试 10,000；8 条独立链；每张 64×64 |
| 中文最终报告 | `docs/L64_DISCRETE_DIFFUSION_FINAL_REPORT_ZH.md` | 758 行；44,697 字节 |
| 排版版 PDF 报告 | `output/pdf/L64_DISCRETE_DIFFUSION_FINAL_REPORT_ZH.pdf` | A4，34 页；2,965,188 字节；已完成逐页渲染检查 |
| 同学代码审计副本 | `external/Hackathon-3-review/` | 92 个文件；984,904 字节 |
| 完整性清单 | `artifacts/pilot_l64/REMOTE_SHA256SUMS.txt` | 57 个远端文件的 SHA256 |

## 3. 核心文件 SHA256

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| `artifacts/pilot_l64/best.pt` | 203,009,562 | `ae2f6d442d18f064494729a09cce93fa2bb45406f10e1d2fef757e724d68e183` |
| `artifacts/pilot_l64/last.pt` | 203,011,738 | `0d2ccd8070fde6a48ceec04cee3d3f43036fb468e7af8ebc2485536447ed75b9` |
| `data/ising_l64_pilot.npz` | 8,240,205 | `97d9439fa0a8aa744ace344c236ec7e7a1ddae68aa39334d89a32d070385735a` |
| `data/ising_l64_reference_10k.npz` | 4,129,609 | `d059aab12c262edca4b73bc37e5a705de0df8ea650df58b52dc1bba0ec302759` |
| `output/pdf/L64_DISCRETE_DIFFUSION_FINAL_REPORT_ZH.pdf` | 2,965,188 | `db36595a9b5a1f0937e7e0c883f4fc64a644abfc84305981d3aedea48c08d41c` |

## 4. Checkpoint 身份核验

- `best.pt`：step 55,000，最佳验证 NELBO = 0.3612483029318007。
- `last.pt`：step 60,000，保存的最佳验证 NELBO 同为 0.3612483029318007。
- 两个 checkpoint 都包含模型、EMA、优化器、历史记录、数据元数据和完整训练配置。

## 5. 数据身份核验

`ising_l64_pilot.npz` 包含：

- `train`: `(15000, 64, 64)`, `int8`
- `val`: `(2500, 64, 64)`, `int8`
- `test`: `(2500, 64, 64)`, `int8`
- 16 条独立 Wolff 链的 chain id 与生成元数据

`ising_l64_reference_10k.npz` 包含：

- `test`: `(10000, 64, 64)`, `int8`
- 8 条独立 Wolff 链的 chain id、收敛诊断和生成元数据
- 该参考集从未用于训练或模型选择

## 6. 当前尚未完成的外部步骤

本地下载和完整性校验已经完成。GitHub 发布尚未执行；需要先确定目标仓库并完成 GitHub CLI 身份验证。由于 checkpoint 超过 GitHub 普通文件 100 MB 上限，发布时应使用 Git LFS 或 Release assets。
