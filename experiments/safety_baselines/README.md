# 独立安全扩散基线实验

当前交付为审计、原文映射和公式内核，不是完整七算法 CARLA 复现。旧审计以 v4_map16_l1 / D-C48 为对象。当前新增完整 Ours 1+8 深度试验，以及不含骨架/安全分支的新地图条件1+8共同基座；新基座从零训练100轮，详见 docs/map_only_baseline_training.md。既有 Ours 代码与结果不变。

Python 3.12 入口：`python experiments/safety_baselines/run.py --stage S0`。
使用安装了 numpy/scipy/pytest 的环境；本机补充依赖位于 `.deps312`，入口自动加载。
`--stage S1 --dry-run` 显示尚缺关卡；实际 S1/S2/S3 当前返回非零，不生成伪实验结果。

实现：`algorithms/kernels.py` 包含 Projected、Primal-Dual、ALM 原文单步更新、SafeDiffuser 共同QP、PDM正文双层循环；`adapters/score_adapter.py` 提供VP x0到score换算；测试为人工数值对照。尚需地图/模型Runner、CBF时间条件、投影器、统一独立评价、S0几何测试及S1/S2验收，才允许正式实验。

论文来源及适配限制见 docs/baseline_source_map.md；环境与状态见 docs/baseline_audit.md。

## 当前的入口（2026-11-14）

七方法原生点实验的可运行入口，优先看第一个：

|用途|命令|
|---|---|
|**七种对比方法怎么实现的（约束作用在哪一层 / 怎么更新 / 约束是什么）**|`docs/comparison_methods_implementation.md`|
|修复报告与全部证据|`docs/DSH_FIX_REPORT_20261114.md`|
|**二十案例小批量（论文原样口径：PDM 用 App.C.3 整段线段约束）**|`scripts/validate_point_methods.py --config configs/seven_native_points_paper_t256.json`|
|二十案例小批量（全顶点诊断口径，八方法可直接比较）|`scripts/validate_point_methods.py --config configs/seven_native_points_vertexall_t256.json`|
|最终协议应加 `--split test`（val 参与过 checkpoint 选择）|`... --split test`；`--ddim-steps` 可对齐 DDIM 预算|
|配置生成（所有派生常数由日程与论文值反算，禁止手写）|`scripts/make_aligned_configs.py`|
|采样器/基座无约束兼容性门|`scripts/audit_sampler_match.py`|
|线段约束与投影的真实地图审计|`scripts/audit_segment_projection.py`|
|骨架 vs 约束对象对照（旧修正器表示逻辑，仅诊断）|`scripts/audit_backbone_effect.py`|
|SafeDiffuser 的 Δτ/子步审计|`scripts/audit_safediffuser_dt.py`|
|256 级基座重训（~46 min）|`scripts/train_map_only_points.py --run-id map_only_points_t256_e100 --config configs/map_only_points_l1_final8_e100_t256.json`|

**为什么要 256 级**：waypoint 间距 0.95 m，而 16 级 cosine 日程的最低训练噪声是 8.77 m/坐标，任何以 p_0 收尾的采样器必然交付 13× 锯齿折线；256 级把它降到 1.10 m（下限 1.56 m）。论证与前后对照见修复报告 §3.0/§3.1/§4b。

`run.py --stage S1/S2/S3` 与 `scripts/campaign.py` 仍是旧控制点路线，不是新点模型入口。

