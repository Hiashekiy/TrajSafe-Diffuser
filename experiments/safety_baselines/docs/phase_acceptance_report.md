# Phase A–F 验收报告（G1–G3）

判定基准：任务书 G 节。`✅` 完成 / `⚠️` 部分完成 / `❌` 未完成。**结论：不允许启动正式 test 实验。**

## G1 共同生成模型

|项|状态|证据|
|---|---|---|
|DDIM-16/64/128/256 全部完成|✅|`results/phase_a_base_model/sampler_steps_*.jsonl`，20 案例 × 4 步数|
|256 步采样索引与公式验证|✅|`tests/test_ddim_steps.py`（7 项，含与手写参考 DDIM 逐元素一致）|
|checkpoint 选择重新审查|⚠️|已量化（prefix-192 的排序不稳定、`best_task` 比 `best.pt` 差一倍），已把冻结基座换成 `best.pt`；**但没改训练脚本的选择公式**（需重训验证）|
|raw/decoded 差异量化|✅|`docs/boundary_decoder_consistency_audit.md`（decoded 在每级略好 0.3–1.8%）|
|score 质量分析|✅|`docs/score_calibration_audit.md`（末级 cos 0.359、范数 3.2e3）|
|无约束生成可视化|✅|`results/phase_f_visualization/case_0_trajectories.png`、`case_0_denoise.gif`、`case_0_denoise.csv`|

关键数字：无约束 DDIM 在 20 个 val 案例上 **14/20 路径穿障**（16 步）到 **14/20**（256 步），
换成 `best.pt` 后 **7/20**、成功率 13/20、平均最小净距 −0.6 m。

## G2 算法实现

|项|状态|证据|
|---|---|---|
|PD/ALM 四粒子初始噪声配对|✅|`tests/test_phase_b_protocol.py`（数组逐元素相等、粒子 0 = 共享噪声）|
|PDM 步长与投影口径明确|✅|`docs/pdm_fidelity_audit.md`；`level_gamma` 两约定 + trace 逐级记录|
|PDM 部分投影不冒充完整投影|✅|`pdm_stats` + 末端精确门槛 + `test_pdm_records_projection_status_...`|
|PDM 圆覆盖几何检查|✅|判据 `‖c_i−c_j‖+r_cell ≤ r_i` + 4 角审计；`test_circle_cover_..._SQUARE_not_just_centre`|
|CD beta 与乘子缩放数学测试|✅|`tests/test_phase_d_cd.py`（9 项，含 V∞=v/(1−β/4v) 与 λ/η/ρ 缩放）|
|SafeDiffuser 离散安全性解析测试|✅|`tests/test_phase_e_safediffuser.py`（4 项，单圆解析）| 
|SafeDiffuser 在真实地图上变安全|❌|**实测做不到**：761/768 级证书不通过、侵入 36.44 m（详见 `safediffuser_discrete_barrier_audit.md`）|

## G3 实验协议

|项|状态|证据|
|---|---|---|
|val/test 索引严格分离|✅|`resolve_indices` + 越界/重复测试 + `sample_id` 交集为 0|
|所有随机种子可复现|✅|`docs/paired_protocol_audit.md`；四类种子明确作用域|
|所有方法共享规定的初始噪声|✅|`initial_noise` / `initial_override` / `sample_points(init=)`|
|统一连续路径评价器|✅|`adapters/point_evaluator.py`（10 类指标）|
|所有方法保存逐场景结果|✅|`results/*/results.jsonl` 逐条，含 trace 与 `pdm_projection_stats`|
|代码/配置/结果可追溯到 commit|⚠️|配置哈希、源码哈希、checkpoint SHA 都记录；本轮 commit 见文末，**推送到远端尚未完成**|

## 未完成项（明确标记）

1. **训练脚本的验证子集（192/1550）与选择指标**未改；换 `best.pt` 只是绕过，不是修复。
   → 需要重训 + 完整验证集才能验证新选择公式。
2. **CD 的粒子数/超参敏感性**（D5）未做。
3. **计算预算 B 组**（相近预算对照）未做。
4. **Ours 未进入统一评价**（需要 B 样条密集采样后过同一评价器）。
5. **正式 test 表未跑**：`--split test` 已可用，但 G2 有 ❌ 项，按任务书门槛不得启动。

## 允许的下一步（建议）

按任务书的"如果基础模型 256 步后依然很差，优先独立模型诊断"：
基座已确认是主要瓶颈（无约束即 7–14/20 穿障），因此**不应**继续堆叠安全约束参数。
建议顺序：
1. 用完整验证集重训一次，并把选择指标改成"端到端 DDIM-256 的连续路径成功率"（成本：1 次训练 + 少量采样评估）；
2. 若重训后基座仍 35% 穿障，则问题在数据/表示，需要独立诊断（见 `deepseek_diagnostic_handoff.md`）；
3. 安全方法侧先只保留"顶点约束"这一档，另行安排"整段约束"作为任务适配实验。
