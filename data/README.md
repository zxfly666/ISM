# Monte Carlo 数据说明

所有 `.npz` 文件均保存取值为 `-1/+1` 的二维 Ising 自旋构型，并附带生成
元数据或 split/chain 标识。正式数据使用周期边界和 Wolff cluster 更新。

| 文件 | 用途 |
|---|---|
| `ising_l64_pilot.npz` | 正式训练集 15,000、验证集 2,500、测试集 2,500；16 条独立链 |
| `ising_l64_reference_10k.npz` | 从未参与训练/选模的独立 10,000 样本参考集；8 条独立链 |
| `ising_l32_pilot_fixed.npz` | 修正后的早期 `L=32` pilot 数据 |
| `ising_l32_pilot.npz` | 原始 `L=32` 诊断数据，保留用于审计 |
| `ising_l16_smoke.npz` | 最小端到端 smoke 数据 |

正式训练读取：

```python
import numpy as np

dataset = np.load("data/ising_l64_pilot.npz")
train = dataset["train"]  # (15000, 64, 64), int8
val = dataset["val"]      # (2500, 64, 64), int8
test = dataset["test"]    # (2500, 64, 64), int8
```

