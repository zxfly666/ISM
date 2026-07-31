# L=64 模型 checkpoint

仓库保留训练代码、配置、数据、日志、图像、评估结果和所有小型实验产物。由于单个 PyTorch checkpoint 约为 203 MB，7 个 checkpoint 不作为普通 Git 对象提交，而是随 GitHub Release `l64-pilot-checkpoints` 发布。

下载后请恢复到以下相对位置：

| Release 文件 | 项目内位置 | SHA-256 |
|---|---|---|
| `best.pt` | `artifacts/pilot_l64/best.pt` | `ae2f6d442d18f064494729a09cce93fa2bb45406f10e1d2fef757e724d68e183` |
| `last.pt` | `artifacts/pilot_l64/last.pt` | `0d2ccd8070fde6a48ceec04cee3d3f43036fb468e7af8ebc2485536447ed75b9` |
| `step_005000.pt` | `artifacts/pilot_l64/checkpoints/step_005000.pt` | `e796c27482cec55f3c594a9da270a09876668bf7bf73937f4109cf0f5a5253f5` |
| `step_015000.pt` | `artifacts/pilot_l64/checkpoints/step_015000.pt` | `acf01a7e6e857cd0a0c9daa2628b58001f3c72f6eff779222c6bfa5108cd2d7e` |
| `step_030000.pt` | `artifacts/pilot_l64/checkpoints/step_030000.pt` | `5809b2ca5bad4c0325bfb5b4ec54d77f6ae53c81e56f1d4328c09e181e0493a8` |
| `step_045000.pt` | `artifacts/pilot_l64/checkpoints/step_045000.pt` | `67512b8c19a7abc9dcbec3c6113fb0a1989fa42ff097f114b7b631f01697d4ba` |
| `step_060000.pt` | `artifacts/pilot_l64/checkpoints/step_060000.pt` | `8889c8bf917a34a4114bbc909987146894cdb6558af70eb1d138d50a4d34af0b` |

使用 GitHub CLI 下载全部 checkpoint：

```bash
gh release download l64-pilot-checkpoints --pattern "*.pt" --dir checkpoint_downloads
```

随后按上表将文件放回项目目录。正式评估默认使用 `best.pt`。
