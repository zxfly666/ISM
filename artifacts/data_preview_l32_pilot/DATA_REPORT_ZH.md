# L=32 训练数据采样审计

- 数据：`C:\Users\zhangxiangfei\Desktop\ISM\data\ising_l32_pilot.npz`
- 临界逆温：0.44068679
- 训练/验证/测试链：8/2/2
- 每个 split 样本数：4096/1024/1024
- 训练集 `<e>`：-1.454248
- 训练集 `<|m|>`：0.696905
- 训练集 Binder U4：0.632586
- 训练集 xi/L：1.117136
- energy / |m| split-R-hat：1.0002 / 0.9999
- 最小单链 ESS：421.4

## 自动检查

- PASS: `chain_splits_disjoint`
- PASS: `binary_spins_only`
- PASS: `g_zero_identity`
- PASS: `energy_correlation_identity`
- PASS: `structure_sum_rule`
- PASS: `critical_energy_sanity`
- PASS: `spin_flip_symmetry`
- PASS: `binder_sanity`
- FAIL: `xi_over_l_sanity`
- PASS: `split_rhat`
- PASS: `minimum_chain_ess`
- PASS: `local_gibbs_calibration`

这些检查借鉴了同学项目的链级审计和局域物理校准，但这里针对完整的周期 L×L 格点计算，没有沿用其 open-window crop 假设。
