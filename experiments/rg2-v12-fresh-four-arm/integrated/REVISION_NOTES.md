# 整合报告修订与证据记录

修订日期：2026-11-25。科学源固定为RG-2提交`5516370b3ae6e98160d9468668acdcbf851dc902`，历史源为`7e608a594c06ead3c484cdd5b38e1d8f2c4d56e3`。本次是科学写作、CPU复算与图文重建，没有新训练、GPU生成或阶段C确认。

## 本次实质修订

将原结果摘要重写为12章及附录A–D的完整中文报告。正文包含历史pilot与能力保留、四臂数据和手算例、输入张量、网络结构、随机遮盖、逆时间损失、交替优化、生成规则、完整细粗物理分布、联合分解、十种竞争机制及后续对照优先级。README仅保留问题、主要发现和正式入口。

新增36组PNG/PDF（5张方法示意、31张真实数据图），完整展示两个Fresh训练seed和两档步数。构型使用预定ID0，另保留原全部图版；未根据观感选择有利样本。能量同时保留理论全范围和覆盖全部实际值的标注放大图。

Markdown为唯一正文权威。HTML在构建时使用固定KaTeX 0.16.22渲染数学，保留MathML及全部本地字体，不依赖CDN或浏览器脚本；主文、目录与表图均从同一Markdown生成。[构建记录](provenance/html_build.json)保存输入、输出及资源指纹。

图版已按审核记录进行视觉检查；HTML已完成公式、链接、资源和样式规则的静态核验。可用浏览器的安全策略拒绝打开本地文件，因此没有进行桌面、移动端或打印预览的实机视觉检查，也未绕过该限制。响应式显示和打印分页仍存在这一验证范围限制，详见[最终编辑审阅](provenance/editorial_review.json)。

## 影响科学解释的澄清

1. 旧geometry任务的20项中只有13项非均匀、7项连续。本次描述性重新分组显示二者变化方向不同，不将其伪称新增预注册检验。
2. 旧Fine+RG的三个W96局部KL终点均高于自身初始，三个W48终点均弱于replay；相对缓解replay退化不等于超越初始能力。
3. 旧replay辅助使用level0、旧RG使用level1，旧训练没有Fresh的2%全MASK端点。旧／Fresh比较同时改变训练历史和配方。
4. P/D/M具有相同扩大编码布局；P仍监督连续细自旋。M仅接收二值多数场，既不接收票差，也没有配对粗细条件或层次化采样。
5. 8.25%块分歧的相关界解释按点对端点概率成立；不能以全格平均分歧直接替代每距离端点权重。
6. 三路径与512态联合实验使用旧Fine+RG等旧权重，不归属Fresh M；A2三点方案只有首组三点并行。
7. forward KL分解的条件预测项按真实P前缀加权。checker无直接边不代表边缘化其他隐藏位后独立。
8. 固定梯度probe中F/P更负，D/M更接近零或为正；不支持简单的“多数任务更冲突”解释，也不能代表整个训练分布。
9. MC–MC校准阴影为5%–95%经验分位，不是95%置信带或数学下界；B已保存12项联合区间只覆盖M−D/F/P。
10. 未执行阶段C，物理保留界限尚未冻结；M−P候选门槛同样为−δ_G，粗尺度门槛才为0；数值复核通过不能写成科学假设得到独立确认。

## 数值及来源核验

| 范围 | 结果与记录 |
|---|---|
| 原发布3,457文件 | 逐文件重新核验；基线清单原样保存于[publication-manifest.baseline.json](provenance/publication-manifest.baseline.json) |
| 方法实现 | 64批人工输入、6,240位置、16组日程配对及参数数目；[method_mechanics_audit.json](provenance/method_mechanics_audit.json) |
| 历史概率／轨迹／保留 | 297 CE、72精确KL、126轨迹项、735保留预测与30,720更新；[history_audit.json](provenance/history_audit.json) |
| 精确联合分布 | 396组Hamiltonian、Oracle与P前缀分解，最大残差约2.01e−14；同上 |
| Fresh汇总 | 32设置、96项Wasserstein及能量恒等式，最大差1.11e−16；[report_values_audit.json](provenance/report_values_audit.json) |
| 实测图 | 1,111输入源与5,000数值检查，最大差约2.00e−14，另22项显示范围；[physical_figure_review.json](provenance/physical_figure_review.json) |
| 最终报告结构 | 链接、公式、36对图及来源；[report_validation.json](provenance/report_validation.json) |

原科学载荷逐字保留。只替换实验根`REPORT_ZH.md`和`README.md`作为新的阅读入口；原文精确副本为根目录`REPORT_ZH.initial-20261125.md`和`README.initial-20261125.md`，保持原相对图链可读。新版发布清单包含报告、派生图表、源码及阅读资源，清单自身不自哈希；编辑来源另存于`provenance/editorial_provenance.json`。原`data-manifest.json`保持不变。

## 本地复建

在完整仓库副本中安装Python的numpy、scipy、matplotlib、Pillow、markdown、markdown-it-py、beautifulsoup4及pypdf，并准备Node.js。以下命令以实验根目录为当前目录；脚本也支持通过完整路径从其他目录运行：

```text
python integrated/scripts/report_tables.py
python integrated/scripts/history_audit.py
python integrated/scripts/audit_method_mechanics.py
python integrated/scripts/method_figures.py
python integrated/scripts/physical_figures.py
python integrated/scripts/build_html.py
python integrated/scripts/validate_report.py
```

其中图形需要可用的中文字体，构建HTML需要Node.js；科学复算不要求GPU。修改正文后须重建HTML，并重新记录发布清单。构建／核验JSON给出实际检查范围，视觉审读不声称逐个原始文件被人工阅读。

当前科学限制仍是两个开发seed、每设置32图、有限MC参考、缺少D1与比例消融、固定梯度probe以及未测试的显式粗细生成机制。报告保留这些限制，不以增加图文质量替代新的科学证据。
