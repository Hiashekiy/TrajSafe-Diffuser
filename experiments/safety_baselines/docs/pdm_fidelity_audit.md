# PDM 保真度审计（Phase C）

## C1 `mixing_ratio` 公式（已修）

- 原来记的是 `gamma/(1-alphabar_t)`（VP 归一化量），**错**。
- VE 下的逐级混合率应为 `r_t = gamma_t / sigma_t^2`，`sigma_t^2=(1-alphabar_t)/alphabar_t`。
- `algorithms/pdm/runner.py` 已改为 `gamma/variance[t]`，并在 trace 里同时记 `drift_norm=gamma*||s||`、
  `noise_std=sqrt(2 gamma)`、`score_norm`。
- `docs/sampler_compatibility_gate_t256.json` 已用修正后的公式重新生成（`best.pt` 基座，3 案例）：

|约定|逐级 mixing ratio|无约束 RMSE 均值|
|---|---|---|
|`paper`（Alg.1 第 3 行）|1.85e-8（256 级日程，恒定）|**546,391 m**|
|`official_ald`|0.167（恒定，按 ε/(2σ_min²)=1/6 反算 ε）|**11.90 m**|

`paper` 约定的 mixing ratio 在 256 级日程上是 1.85e-8 —— 即 `γ_t/σ_t²` 恒定但极小，M 步内样本不动；
这不是实现错误，而是"原文公式要求 VE 噪声上界是 O(1)"与本基座 σ_max≫1 的冲突。

## C2 两种步长分别记录（已实现）

`level_gamma(variance,t,levels,convention,ald_eps)`：

```
paper        : gamma_t = sigma_t^2 / (2 sigma_T^2)                       # σ_T = 梯级最大噪声
official_ald : gamma_t = delta_t / 2 = eps * sigma_t^2 / (2 sigma_min^2)  # 官方 scoremodel.py
```

- 两者**不是**同一配置，trace 每步都写 `step_convention`，配置里是显式字段，不会混用。
- 单步量级（`best.pt`、案例 0、t=0 级）：`||s||≈194`，official 的 `gamma·||s||≈0.5 m`、噪声 `sqrt(2γ)≈0.4 m`；
  paper 约定的同位置 `gamma·||s||≈1e-9 m`（等于不动）。
- 端到端：official 11.9 m vs paper 546 km（见上表）。

**不要把 `official_ald` 标成"原论文原样复现"**：它是官方参考实现的步长约定 + 本日程下的单位换算，
报告中必须写成"task-adapted, official-step"。

## C3 部分投影不得冒充完整投影（a17a667 已修，本轮补充记录）

- 内层 `allow_partial=True` 允许返回 `segment_projection_partial`（残差未达标），但只在**非末端**级别使用；
- 末端级别 `allow_partial=False`：达不到 `projection_tol`（1e-6）就抛 `SolverFailure`；
- 每条记录带 `pdm_projection_stats = {projection_calls, partial_projections, final_status, final_residual, tolerance, exact_final}`；
- 交付前的残差复检再独立算一次，超容差即失败。

区分"算法失败"与"预算不足"：残差 > tol 且 `final_status` 含 `maxiter`/`partial` → 预算/收敛问题；
`fixed path intersects obstacle` → 任务本身不可行（端点被占）；`segment_sqp_qp_failed` → QP 数值失败。

## C4 圆覆盖几何（a17a667 已修，本轮给出判据与实测）

判据（`obstacle_circles`）：某占据格被判"已覆盖"当且仅当

```
||c_i - c_j|| + r_cell <= r_i        # r_cell = 0.7072 * cell  （半格对角）
```

且审计额外检查每格的 4 个角点与中心（`covered_corner_fraction`）。实测（val 0/81/163/244）：

|指标|数值|
|---|---|
|圆数量|215–370 个/场景（判据收紧后从 249 增到 370）|
|覆盖率（中心）|1.000|
|覆盖率（4 角 + 中心）|1.000|
|自由格被误禁比例|3.5–5%|
|半径中位数|1.7–2.7 格|

注意：即使 `covered_corner_fraction=1.0`，这只证明"每个占据格方块被某个圆覆盖"，
不等于"圆并 = 占据区"（圆并会侵入自由空间，误禁比例即其度量）。要更紧的几何需用矩形/多边形约束，属于改方法。

## C5 复现边界（必须写进论文）

1. **训练目标不同**：原论文用 score matching，本基座是 x0 MSE + coarse/smooth/boundary 辅助项（`models/point_losses.py`）。
   由 x0 换算 score 在数学上成立，但前提是 x̂0 为条件均值最优；`docs/score_calibration_audit.md` 实测末级（t=0）
   条件噪声 cos 只有 0.359、score 范数 3.2e3。
2. **状态坐标**：只允许 VE（论文坐标）；VP 变体因投影集需随级缩放、且要跨级换算 y 而被删除。
3. **预算**：T=10 与原文一致；M=100→20，依据是原文 App.F 明确允许减少 M 或调整投影频率。
4. **约束**：主配置 `seven_native_points_paper_t256.json` 用原文 App.C.3 整段线段约束；`..._vertexall_*` 是全顶点诊断口径，
   **不能**当作 PDM 原样结果。
5. 在这些差异下**能说**："PDM 的 Alg.1 + App.C.3 约束在任务适配后的实现"；
   **不能说**："PDM 原论文结果"或"score 质量已达标"。
