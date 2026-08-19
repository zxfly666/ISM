# 第一级 Scale-Aware Ising 快速判断

## 结论

自动化、预注册规则给出的标签为：**NO-GO**。

该标签仅用于决定是否进入多 seed 与更大尺度实验，不构成论文结论。

## 有效性

- MC 有效：True
- 四组初始化一致：True
- PAD 最大绝对 logit 漂移：8.225e-06
- 64/128-step 排名稳定：True
- 四组均达到 8,000 steps：True

## 三类信号

- ID guard：False（T3-T0 anchor NLL = +0.001527）
- Coordinate signal：True
- Pollution signal：False（T0 中可检测污染：True；T3 改善：False）
- Generation signal：True

## 解释边界

这是单 seed pilot。即使标签为 GO，也只能说明“训练时随机化真实坐标尺度”的研究假设值得继续验证；必须经过多 seed、W=128/更大 context 与更强基线后，才能写成稳定有效的方法结论。
