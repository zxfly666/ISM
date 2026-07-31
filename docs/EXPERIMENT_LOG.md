# ISM 实验记录

本文件记录服务器小规模实验。每次实验必须写明代码版本、配置、训练 seed、
采样 seed、耗时、显存、验证 NELBO 和物理量。

## 服务器

- 路径：`/root/autodl-tmp/ISM`
- GPU：NVIDIA GeForce RTX 5090，约 32 GB
- 初始环境：PyTorch 2.8.0+cu128，CUDA runtime 12.8

## 本地 smoke 结论（上传前）

- 模型：55,282 参数，\(L=16\)，80 training steps；
- validation NELBO：约 0.693 → 0.603；
- irreversible confidence sampler 的单 seed 能量已接近 MC，但长程量偏差明显；
- strict ancestral sampler 在该弱模型上接近随机态；
- confidence backtracking 在 3 个 sampling seed 的快速排序中，以 10% 回溯比例
  的平均综合误差最低，但改善不是每个 seed 都一致。

## 远程实验

### E1：原始 smoke

- 配置：55,282 参数，batch 8，80 step，CUDA float32；
- 总耗时：4.858 秒；
- validation NELBO：0.693153 → 0.600320；
- 结论：管线正常，但任务太小，启动与验证开销主导，不能用于吞吐比较。

### E2：固定 400-step GPU benchmark

共同设置：1,596,098 参数，\(L=16\)，batch 64，BF16，400 step。

| 实现 | 总耗时 | 训练器内部耗时 | 最终 NELBO | 备注 |
|---|---:|---:|---:|---|
| 原始 eager | 23.089 s | 未记录 | 0.354003 | 原始逐样本增强 |
| 向量化 + fused + preload | 19.959 s | 17.503 s | 0.354696 | 快 13.6% |
| 上述 + compile | 23.856 s | 19.064 s | 0.354281 | 首次编译约 14.6 s |

compile 组在首次编译后完成其余 399 step 约用 4.38 秒，稳态吞吐显著高于
eager。结论：几百步 smoke 不启用 compile；长训练启用。

修复“数据预载入后 batch index 改用不同 RNG”后再次运行优化 eager：

- step 1 的 loss、mask fraction、mean \(t\) 和 validation 与原始基线逐项一致；
- 总耗时 14.289 秒，训练器内部 11.810 秒，33.87 step/s；
- 最终 NELBO 0.355505，与原始 0.354003 接近。

由于该复验发生在 GPU 与库 kernel 已预热后，提速报告采用较保守的首次
优化结果（13.6%）；14.289 秒作为热环境实测下界保留。

### E3：2000-step 学习 pilot

- 模型：1,596,098 参数；
- batch 64，BF16，fused AdamW，数据预载入，`torch.compile`；
- 总耗时：21.736 秒；
- 训练器内部：17.329 秒，平均 115.4 step/s；
- 峰值 allocated/reserved 显存：0.538/0.832 GiB；
- 最佳 validation NELBO：0.341748（step 1500）；
- 除 \(t=1\) 外，per-t CE 约 0.24–0.33；全 MASK 端点仍约 0.69。

### E4：采样失败案例

在 E3 最佳 checkpoint 上，confidence sampler 产生接近全同向自旋：

- 三 seed 能量约 −1.974 至 −1.980，参考约 −1.49；
- \(|m|\) 约 0.986–0.993，参考约 0.75；
- \(\xi/L\) 约 4.45–11.90；
- confidence corrector 会进一步强化过度有序。

该结果证明 NELBO 改善不保证生成分布正确，也证明置信度排名提交带来明显
低温偏置。

### E5：祖先采样校准与稳健复验

先在 48 张样本上搜索方向：

- ancestral，24 step，\(T=1.0\)：综合快速误差 0.407；
- ancestral，24 step，\(T=0.9\)：0.093；
- ancestral，48 step，\(T=1.0\)：0.133；
- 12 step 不足，\(T\leq0.8\) 开始过度有序。

最终使用每组 256 张、3 个 sampling seed、每次独立 4-chain Wolff 参考：

| sampler | seed | E | \|m\| | U4 | xi/L | 快速总误差 |
|---|---:|---:|---:|---:|---:|---:|
| ancestral 24/T0.9 | 1234 | −1.5127 | 0.7462 | 0.6119 | 0.9941 | 0.1956 |
| ancestral 24/T0.9 | 2345 | −1.5377 | 0.7691 | 0.6172 | 1.0980 | 0.1608 |
| ancestral 24/T0.9 | 3456 | −1.5062 | 0.7368 | 0.6054 | 0.9421 | 0.2073 |
| ancestral 48/T1.0 | 1234 | −1.4025 | 0.7060 | 0.6138 | 0.9648 | 0.3308 |
| ancestral 48/T1.0 | 2345 | −1.4095 | 0.7107 | 0.6133 | 0.9875 | 0.1591 |
| ancestral 48/T1.0 | 3456 | −1.4104 | 0.6973 | 0.5982 | 0.8946 | 0.3855 |

结论：当前 checkpoint 的推荐是 `ancestral, steps=24, temperature=0.9`。
confidence 系列保留为诊断，不再作为默认。

### E6：完整 L32 模型 capacity benchmark

- 数据：4 条小型 chain，共 train/val/test = 128/64/64，仅用于容量测试；
- 模型：12,674,434 参数，\(L=32\)，batch 32，BF16；
- 优化：fused AdamW、预载入、`torch.compile(reduce-overhead)`；
- 首次编译至 step 1：20.87 秒；
- step 1→100：约 3.34 秒，即稳态约 29.6 step/s；
- 全流程内部耗时 24.75 秒，外部总耗时 29.63 秒；
- 峰值 allocated/reserved 显存：3.51/4.03 GiB；
- validation NELBO：0.693142 → 0.431792。

按稳态线性估计，60,000 step 核心训练约 34 分钟。加入固定网格验证、
checkpoint、首次编译和正式数据生成，单张 RTX 5090 的 1.5 小时预算仍有
余量。该估计需在正式数据生成后用前 1000 step 再校准一次。

后续结果继续按时间追加，不覆盖以上记录。
