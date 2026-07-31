# 配置文件说明

| 配置 | 用途 |
|---|---|
| `pilot_l64.json` | 正式 `L=64`、60k-step 训练；最终结论对应此配置 |
| `benchmark_l64_compact.json` | `L=64` 紧凑模型性能测试 |
| `benchmark_l64_full.json` | `L=64` 完整模型性能测试 |
| `gpu_benchmark_*.json` | 编译、预加载和优化器设置的 GPU 性能对照 |
| `phase1.json` | 早期 `L=32` 实验方案 |
| `gpu_l32_capacity.json` | `L=32` 容量/吞吐测试 |
| `pilot_l16_compiled.json` | 编译训练链路 smoke test |
| `smoke.json` | 最小 CPU/功能 smoke test |

正式复现入口：

```bash
python train.py --config configs/pilot_l64.json
```

