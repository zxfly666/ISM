> Public archival copy. Historical planning/status text is preserved; consult the campaign README for actual execution and final decisions. Infrastructure identifiers are redacted. Relative links below are historical source identifiers. Original SHA and every transformation are recorded in provenance.json.

# 完整方案保留：12小时执行窗口（预算修订v2）

2026-09-25。用户在取消次要生成或保留全部并增加到12小时的选择后明确回复“12h”，并协助重新认证服务器。本修订只改变这一轮的运行预算，不改变科学假设、五组、六个新seed、每模型12k步、900预测、1920生成图、采样器或统计次数。仅授权这一轮，不追加下一轮。

## 原记录与新的执行窗口

原budget.json与两次失败预检完整保留：started1790280756.5880508，原10小时截止1790316756.5880508（14:12:36.588 UTC+8）；原门9.5小时，结果10.41485和9.68903小时，均失败，不能改写为通过。

用户决定预算及重新登录期间没有正式训练或新MC，没有后台待启动队列。12:29:30实测原主机GPU0%、1MiB、磁盘33GB可用，无实验计算进程。

本次明确登记第二执行窗口，不把原04:12连续时钟静默延长或隐去等待。budget_v2.json在本次GPU重新预检实际开始时独占创建，记录当前started、deadline=started+43200秒、旧budget SHA与本修订SHA。新的12小时包含重新预检、完整scratch、冻结准备、训练/MC/评估/统计；此后审批、断线、重试和等待同样计时，不再重置。原预检已用时间单独保留报告，不计作从未发生。

保守启动门11.5小时（41400秒），保留20%安全乘数、统计至少1800秒、末尾半小时空间。完整scratch和启动前等待加入实际预测。预算不通过则停，不缩减科学内容、不自动加时。

## 科学与启动门

原CORE_GEOMETRY_FACTORIAL_PROTOCOL_20260925_ZH.md保持原字节；其10小时/9.5小时预算条款由本修订替代，其他科学内容仍有效。新预检目录preflight_authorized12h_v3；旧preflight及preflight_buffered_v2不动。新的入口冻结原科学协议、本修订、所有依赖与数据、新预算证据。

须通过科学/恢复/全部几何原生输入/MC软件流程/统计六图scratch并实际看图，且启动预算通过，才单次创建run.lock。900文件scratch采用旧参考每bank前两个父场和同一临时模型，明确不是正式训练或新MC；正式样本量仍128父场/bank。

正常启动已有用户授权，不重复询问许可；工具权限仍按审批。开跑后核实PID、有效更新、独立MC、锁、外部timeout、绝对截止并更新原l64。逐阶段备份SHA，最终核验看图和解释后暂停监控，不归档、不关机、不上传GitHub、不追加新轮。
