# 复现路线与边界

## 三级复现，不把它们混称“完全可复现”

| 层级 | 当前公共仓库能做什么 | 额外要求 |
|---|---|---|
| 1：核对叙述/表/图 | README→统计JSON/CSV→NPZ/plot_data→原protocol/source；检查公开SHA | Python、NumPy；配对t重放需要SciPy |
| 2：重新做全部原分析 | 若所需逐父场/逐图数组已公开，可重算；未公开的大容器需先取回 | 见每轮withheld manifest与analysis输入；不可默认全部齐全 |
| 3：从头训练或恢复 | 代码/config/数据构造/版本记录公开，恢复身份有SHA | 大MC训练池、权重、原冻结文档身份、匹配设备/软件、独立新输出目录与预算 |

## 无训练的公共验证

在仓库根目录，使用Python 3.10+与NumPy/SciPy运行：

```bash
pip install -r experiments/requirements-replay.txt
python scripts/archive20260930/validate.py
python scripts/archive20260930/replay.py
```

两者不连接服务器、不运行GPU前向、不写原科学目录。第一条核对公开文件/链接/
NPZ可读性与来源；第二条从G主bootstrap数组和最新N/W配对效应等机器证据重放
主结论，输出声明对应的数值而不是凭README文本“验证自己”。

## 原代码入口与环境

每轮`EVIDENCE.md`链接原driver/analysis；原source文件保持历史路径，
公开语义目录名不改写实验ID。共享dense定义位于
[scale_model.py](../ism_diffusion/scale_model.py)、
[geometry_study.py](../ism_diffusion/geometry_study.py)、
[intervention_model.py](../scripts/research20260927/intervention_model.py)。

原服务器通常单RTX4090、Linux、PyTorch2.7/CUDA；精确CPU轮记录另有本地Torch2.3。
这些是各轮runtime记录，不是任意新版本的兼容保证。精确复现需依每轮effective
protocol固定NumPy、Torch、BLAS/CUDA、线程、精度、RNG和数据。BF16研究不能
悄悄改成FP32并声称位级重现；同理精确局部FP32不使用AMP/TF32。

科学脚本包括Linux `fcntl` 等运行管理依赖，Windows读结果与完整Linux训练环境
是不同层级。原checkpoint的raw、EMA、AdamW、Python/NumPy/Torch CPU/CUDA RNG
及step/data counter必须一起恢复。只载EMA做推理不叫精确续训恢复。

## 从干净clone做新复现之前

1. 先跑上述只读验证；阅读原门和失败结果，别先挑最好checkpoint。
2. 从`EVIDENCE.md`获取有效config/协议；`design_only`是历史设计快照，实际运行以
   run_protocol与final/verification身份为准。脱敏后的协议**不与原冻结SHA等同**。
3. 按withheld manifest获取缺失数据和历史权重，核SHA。保留原数据拆分，不能
   用公开test或evaluation MC构建训练标签。
4. 原driver常默认`artifacts/<historical_id>`且有run.lock/源SHA检查。**不要直接
   在原研究目录执行`--mode run`**。在隔离的新checkout/output配置新的复现身份，
   恢复其记录的相对输入路径，记录环境差异及新的source manifest；这不再是原运行本身。
5. 先独立检查数学、PAD/query、配对数据和下一步恢复；再单次运行固定计划，不按
   中途显著性加步/换seed。跨软件版本不承诺位级相同。

本次发布不声称一个公开命令即可取回所有私人/大型材料并重训全部campaign。
这项限制在各轮可用性里显式披露，而不是用漂亮README遮住缺失数据。

## 图、统计和独立单位

正式图复用原PNG/PDF，配套plot_data和原绘图函数；没有重绘改变门或选择seed。
G的三个主CI按原多重性规则，J唯一主属于fresh S，6h主为配对t而非MC bootstrap。
旧模型重采样、同父场多个query、四种随机坐标、对称/平移副本都不增加training
seed数。公开bootstrap敏感性不能替代原预定义主决定。
