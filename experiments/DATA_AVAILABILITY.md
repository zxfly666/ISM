# 数据可用性：公开证据 ≠ 完整本地备份

本目录是**可核对的公共科研证据子集**，不是所有原始数据/模型的下载镜像。
每轮的`provenance.json`列出原文件SHA、公开文件SHA、字节和编辑操作；
`withheld-manifest.json`列出科学路径、大小、SHA、归档身份及公开副本映射。
`backup-verification.json`是既有本地逐成员核验回执，不是公开托管凭证。

| 材料 | 普通Git公开范围 | 没有整体上传的部分 / 原因 |
|---|---|---|
| 主/次要/反例结果 | JSON、CSV、关键NPZ、全部正式图及plot_data | 个别大型逐query数组/全部bootstrap draws；避免普通Git膨胀 |
| 协议与config | 每轮完整科学协议公开副本、effective run记录、原config和源码 | 连接身份/本地用户名脱敏；私人通信与传输脚本不公开 |
| 代码 | 原科学训练、数据、统计、绘图、测试，共享依赖 | 远程登录、私有报告/通信处理不属于复现科研代码 |
| 小型精确真值 | K1/K2/K4/G8构造代码、拆分、必要标签/预测 | 模型权重依然单列 |
| checkpoint | 身份/步数/恢复字段的审计、SHA与成员路径 | 大型raw/EMA/AdamW/RNG权重不进普通Git |
| MC与生成原数组 | 参考QA、协议、聚合量、SHA、样本数与谱系 | L1024父场、bulk generation、部分中间bank存于保管者备份 |
| 历史LFS | 保留仓库原有pointer与说明 | 本次未重新下载验证其服务端blob，不作可用保证 |
| 私人上下文 | 科学问题的独立转述 | 不上传老师聊天截图、逐字通信、凭据、连接指南 |

## 怎样请求或恢复未公开材料

通过本仓库issue向维护者提交：`experiment_id`、`withheld-manifest.json`中的
科学路径、所需SHA-256和用途。提供归档basename/member（若清单有）可避免
把错误版本当成所需模型。维护者核实权限后另选正式数据仓库或release分发；
**本次没有创建release/LFS新对象，也没有公开永久下载链接或保留期限保证。**

保管者当前保存本地独立备份目录，原运行服务器另有科学产物。公开文档里的
`<CUSTODIAN_BACKUPS>`是隐去机器路径的标识，不是URL。同D盘目录不是异盘或
异地灾备。恢复时先核归档整包SHA，再核成员路径/字节/SHA；恢复权重使用审计过的
兼容Torch环境，不能把未知pickle当可信代码执行。

## 避免误解

1. “本地科学manifest全覆盖”不意味着所有文件都已上传GitHub。
2. 原始/公开SHA不同的文字副本有逐文件编辑记录；脱敏不改变数值数组。
3. G大型conditional bootstrap只无损提取`primary`、`retention`及标识数组，
   `arm_means`原容器留在清单。主统计可重放；全二阶重新绘图可能仍需原容器。
4. 一个图可能依赖大数组；即使图已公开，完整从权重重跑仍可能需要保管者材料。
5. 计划/预算/软件fixture和正式神经网络产物在目录及README中分开标记。

每轮具体已公开文件见自动生成的`EVIDENCE.md`，未公开文件见各轮清单。

## 2026-10-01训练阶段补充

[末期MASK覆盖](late-mask-coverage-and-committed-spin-correction/README.md)公开训练完成、
全144k输入审计、18final/36EMA身份、原定协议、effective config、冻结科学源码和技术失败记录。
新包128成员、联合173路径本地核验通过；权重、原始逐步日志和大训练池仍为manifest-only。
以上是训练交付时点范围。随后在独立授权时窗已完成新MC/正式生成/P1/P2统计，不用新状态抹去旧失败。

## 2026-10-01正式评价追加

公开1086条来源记录（训练63+评价1023），含546个条件/学习预测NPZ、完整分析和敏感性数组、
通过及失败precision候选、六图PNG/PDF、384图phase0诊断（含24个spin shard）、协议和CPU审计/备份回执。
七个新冻结评价源码逐SHA发布；数值与图片不修改。详见[证据索引](late-mask-coverage-and-committed-spin-correction/EVIDENCE.md)。

大权重、144k原训练日志、大训练池、新MC原场、输入bank与6912正式图spin/RNG轨迹**已计算并已核验备份，但未公开托管**。
完成包382721565字节/1811成员、退出后管理包385648字节/7成员两处逐成员核验；
与六基座/训练/初始/失败预检共12包联合2256/2256路径，正式final无排除。
[联合回执](late-mask-coverage-and-committed-spin-correction/evaluation/closure/evaluation_independent_union_v1.json)与
[逐文件归档映射](late-mask-coverage-and-committed-spin-correction/withheld-manifest.json)区分公开副本与保管者材料。
仍是同D盘独立目录，不是异盘/异地灾备，没有新release/LFS或永久公开下载保证。
