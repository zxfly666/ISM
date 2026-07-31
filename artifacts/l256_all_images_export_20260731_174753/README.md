# L=256 图片导出清单

本目录汇总本轮完整晶格生成与采样机制诊断产生的全部图片，不包含更早的
L=32 训练和 sampler smoke 实验。

1. `01_our_original_l256_preview.png`：我们的原始单张 L=256 预览。
2. `02_burnin_sensitivity.png/.pdf`：1、10、50、500 burn-in 对物理量的影响。
3. `03_correlation_burnin.png/.pdf`：不同 burn-in 的两点关联函数。
4. `04_magnetization_and_samples.png/.pdf`：磁化分布及平衡、典型、高磁化样本。
5. `05_final_chain_traces.png/.pdf`：四条独立链的轨迹和自相关函数。
6. `06_finite_size_magnetization.png/.pdf`：L=16 至 256 的有限尺寸标度。
7. `07_peer_wolff_l256.png`：同学 Hackathon-3 Wolff 实现生成的单张 L=256 图。
8. `00_contact_sheet.png`：上述七张 PNG 的总览缩略图。

所有晶格图均为完整周期边界 L=256，没有裁剪。
