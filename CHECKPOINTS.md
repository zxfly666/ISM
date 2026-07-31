# L=64 模型 checkpoint

仓库保留训练代码、配置、数据、日志、图像和评估结果，并提供正式实验的最佳 PyTorch checkpoint。`best.pt` 约为 203 MB，使用 Git LFS 管理，而不是写入普通 Git 对象。

下载后请恢复到以下相对位置：

| 文件 | 项目内位置 | SHA-256 |
|---|---|---|
| `best.pt` | `artifacts/pilot_l64/best.pt` | `ae2f6d442d18f064494729a09cce93fa2bb45406f10e1d2fef757e724d68e183` |

完整克隆（包含最佳 checkpoint）：

```bash
git lfs install
git clone https://github.com/zxfly666/ISM.git
```

如果已经在没有 LFS 对象的情况下克隆了仓库，可执行：

```bash
git lfs pull
```

正式评估默认使用 `artifacts/pilot_l64/best.pt`。

`last.pt` 和 5 个里程碑 checkpoint 仅用于断点续训及训练轨迹研究，继续保留在实验机器/本地备份中，不上传 GitHub。对应的 loss、验证指标、采样结果和 SHA-256 清单仍保存在仓库内。
