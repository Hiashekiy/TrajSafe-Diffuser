# 统一评价与可视化审计（Phase F）

## 1. 评价器（`adapters/point_evaluator.py`）

所有方法都用这**一个**评价器打分，输入是"交付的 128 点折线"，不是控制点、不是中间状态。

对每一段折线：按 256×256 网格的所有格边界求出交点，把每段切成"落在单一格内"的小区间，再取区间中点与端点
一起判碰（`evaluation_points` 记录采样点数）。因此**线段穿越**不会被顶点采样漏掉。

输出（每个方法每条轨迹）：

|字段|含义|
|---|---|
|`vertex_collision`|任一顶点在障碍内或出图|
|`segment_only_collision`|顶点都安全但连线穿越|
|`collision`|任一（= 上面两者的或）|
|`planning_success`|无碰撞且端点误差 ≤ 0.01 m|
|`min_clearance_m`|沿整条折线的最小有符号净距（负=侵入）|
|`violation_max_m`|最大侵入深度|
|`endpoint_error_m`|首末点与条件的最大偏差（钉死端点时为 0）|
|`path_length_m` / `path_length_ratio`|绝对长度与相对 GT 的比值（识别锯齿/绕行）|
|`mean_turning_deg` / `max_turning_deg`|相邻段夹角 |
|`mean_second_difference_m` / `max_second_difference_m` / `mean_jerk_m`|平滑性|
|`rmse_m`|与 GT 的逐点 RMS|
|`representation` / `continuous_curve_certificate`|口径声明（折线，非拟合曲线，未给连续证书）|

Ours 不在本轮范围；若纳入，必须把 B 样条曲线**密集采样**后送入同一评价器，不能只评控制点。

## 2. 计算成本（`results/phase_a_base_model/sampler_steps_*.jsonl`）

DDIM 在不同步数下的 NFE 与耗时（20 案例均值，`best.pt`）：

|steps|NFE|耗时 s|耗时/NFE ms|
|---:|---:|---:|---:|
|16|16|0.22|13.8|
|64|64|0.82|12.8|
|128|128|1.46|11.4|
|256|256|3.10|12.1|

各安全方法的单次成本（同一 20 案例、256 级基座，`results/review_fix_vertexall_t256/`）：
RoS 14.4 s、ReS 12.9 s、TVS 13.2 s、Projected 7.4 s、PD/ALM 9.3 s、PDM 4.5 s、Diffuser 0.34 s。
即**约束循环的 NFE 是 256 的整数倍**（SafeDiffuser 每级 1 次 + 微步内多次 QP；CD 320 次；PDM 200 次 SGLD），
单次 QP/SQP 与投影的时间已经在逐级 trace 里（`iterations`、`integration_steps`、`seconds`）。

预算公平性：本轮**只**做了"A：各算法原生配置"这一组；"B：相近计算预算"需要额外定义预算口径（例如
以 DDIM-256 的 NFE=256 为上限裁剪各循环），尚未做。

## 3. 可视化（`scripts/visualize_case.py`）

每个场景输出到 `results/phase_f_visualization/`：

|产物|内容|
|---|---|
|`case_<idx>_trajectories.png`|占据图 + 起终点 + GT + 各方法最终折线，**碰撞顶点用红叉标出**|
|`case_<idx>_denoise.gif`|DDIM 逆向链逐帧（每帧一个 NFE）|
|`case_<idx>_denoise.csv`|逐帧 `t / next / min_clearance_m / 顶点碰撞 / 路径碰撞 / 长度比 / 最大转角`|
|`case_<idx>_<method>_steps.json`|该方法逐级 trace（QP 状态、残差、迭代数、积分步数）|
|`case_<idx>_record.json` / `summary.json`|完整记录：配置、checkpoint、种子、逐方法指标|

### 视觉证据（case 0，DDIM-32 对比）

- 蓝色 DDIM 与绿色 ReS：沿走廊的光滑曲线，净距 +3.9 m / +4.6 m；
- 橙色 Projected、红色 PDM：在走廊里**横向反复乱窜**，转角 179–180°、长度比 5.5–9.6×，最大侵入 −16 m；
- 红叉标出的碰撞点集中在横向尖刺上。

这就是"顶点安全、路径不安全"的直观形态：投影只约束 128 个顶点，Langevin 的随机步把它们横向甩进走廊两侧的障碍，
而连线自然穿障。**failure 从第一级高噪声就开始了**（GIF 里能看到轨迹在 t≈200 之后就散开）。

## 4. 仍未做

- "B 组：相近预算"对照。
- Ours 的加密采样评价。
- 多随机种子的可视化（当前 case 0/81、种子 0）。
