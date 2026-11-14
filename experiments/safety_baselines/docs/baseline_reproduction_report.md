# 复现进度与限制

已实现：纯公式更新（Projected/PD/ALM）、共同CBF QP、PDM正文T/M循环、VP score换算、独立地图SDF与控制点梯度。公式测试包含硬/软QP及无解、粒子期望、slack与rho、种子复现、投影调用次数、梯度有限差分。

尚未完成七方法的完整Runner与S0验收，不能对外称为完整算法复现。CBF robust gamma合法选择及ReS/TVS时间表尚未封装验证；PDM VP/VE适配未锁定。局部非凸最近点投影、独立碰撞评价、统一数据/模型接入、失败/超时记录、端到端计时及S1–S3未完成。当前无CARLA对比结果。

所有新代码在 experiments/safety_baselines，不调用Ours ALM冒充外部方法。用户工作树存在未提交修改，未自动commit。正式阶段入口失败关闭；dry-run可查看缺失条件。

## 当前实际阻塞

D-C48 checkpoint 与当前网络不匹配：20个旧地图解码层权重unexpected，严格加载已拒绝。Ours v4_map16_l1严格加载成功。完整记录见 locked_assets.json。修复基座兼容性之前不进入真实采样实验。

验证：新测试8项通过，连同原有回归共38项通过。只代表公式与局部接口，不代表七方法S0完整验收。


## 2026-11-13 新基座接入

map_only_control_diffuser_v1已完成100轮训练，改用其epoch88 best_task作为外部共同基座，旧D-C48兼容问题通过独立从零训练解决。完整Ours改为1+8/100轮epoch84 best_task。已新增独立DDIM采样、严格加载及20场景冒烟；3项相关测试通过。固定chunk32下与Ours采用相同seed+chunk起始下标噪声协议；该历史协议依赖chunk，不能声称batch不变性。完整两模型对比输出results/map_only_vs_ours_20261113；不代表七论文方法已复现。外部七方法原文采样/投影/QP时间表集成与完整S0仍未通过，正式安全算法S1–S3保持阻塞。
