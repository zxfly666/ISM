# Scale-aware context 父场数据

`parents_l1024.npz` 是尺度感知坐标研究线所有窗口采样与 RandomGap 实验的共同
Monte Carlo 父场。文件通过 Git LFS 保存；克隆仓库后运行 `git lfs pull` 获取实际
内容。

## 生成设置

- 模型：二维无外场铁磁 Ising，周期边界；
- 晶格：`L=1024`；
- 逆温：`beta=0.44068679350977147`（临界点）；
- 采样器：Wolff cluster Monte Carlo，Numba 后端；
- 独立链：12 条，每条 128 个构型；
- burn-in：20 sweeps；样本间隔：目标 2 sweeps；
- 自适应：3 sweeps，pilot 128 cluster updates；
- 总 seed：`20260818`；每条链的 seed 与诊断均内嵌在 `metadata`；
- 初始化：`random` 与 `plus` 交替，用于平衡诊断；
- 划分：train/val/test-target/test-control = 6/2/2/2 条链，链级隔离。

## 文件结构

| 数组 | shape | dtype | 含义 |
|---|---:|---|---|
| `train_packed` | `(768,1024,128)` | `uint8` | 6 条训练链，bit-packed spins |
| `val_packed` | `(256,1024,128)` | `uint8` | 2 条验证链 |
| `test_target_packed` | `(256,1024,128)` | `uint8` | 2 条主测试链 |
| `test_control_packed` | `(256,1024,128)` | `uint8` | 2 条独立 MC control 链 |
| `*_chain_id` | 与样本数一致 | `int16` | 链身份，用于层级统计 |
| `metadata` | scalar JSON string | Unicode | 完整生成参数、seed、链诊断和划分清单 |

文件大小：`129,795,241` bytes。
SHA-256：`1f7de1ec81e82ebcfcbc4134d5670ee35711230f05d00cd037c7b9e575ef8934`。

## 重建命令

```bash
python generate_level1_parents.py \
  --output data/level1/parents_l1024.npz \
  --lattice-size 1024 \
  --train-chains 6 --val-chains 2 \
  --target-chains 2 --control-chains 2 \
  --samples-per-chain 128 \
  --burn-in-sweeps 20 --sweeps-between 2 \
  --adaptation-sweeps 3 --pilot-cluster-steps 128 \
  --workers 12 --seed 20260818
```

不同 Numba、NumPy 或并行运行时版本未必保证逐 bit 相同，因此正式复核应优先使用
仓库中的 LFS 文件，并用上述命令做独立再现。
