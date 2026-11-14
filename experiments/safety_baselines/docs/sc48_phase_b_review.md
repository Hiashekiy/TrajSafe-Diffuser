# Phase B-Review：S-C48 控制点空间适配的二次审查

日期 2026-11-14。回答外部审查（对 `55def29` 的 6 条意见）。**本轮结论修正了上一份报告的两处判断**，
依据是 `results/sc48_phase_b_review/diagnostics*.json` 的实测，脚本 `scripts/diag_sc48_phase_b.py`。

审查对象：`55def298`。三个代表场景：61001（安全）、74014（轻碰 −0.6 m）、86008（重碰 −5.7 m）。
索引映射：`ds[0]=61001`、`ds[163]=74014`、`ds[407]=86008`（`data/processed/160m-k4p-c48/val/sample_id.npy`）。

**可核对的原始数据已随文档提交**（`results/` 被 gitignore，所以复制到 `docs/sc48_seven_results/`）：

|文件|内容|
|---|---|
|`docs/sc48_seven_results/seven_sc48_summary.json`|主表 8 行的官方汇总（20 场景）|
|`docs/sc48_seven_results/seven_sc48.csv`|20×8 = 160 条逐场景记录|
|`docs/sc48_seven_results/diag_scene_61001.json`|本文 §1–§5 的 61001 诊断原始值|
|`docs/sc48_seven_results/diag_scenes_74014_86008.json`|74014 / 86008 的诊断原始值|

---

## 0. 结论速览

|审查意见|核实结果|处理|
|---|---|---|
|① SafeDiffuser 报告根因与实际执行的求解器不一致|**成立**（且比审查指出的更严重）|已修：求解器显式门控 + OSQP 收敛状态检查 + 逐级诊断；报告改写|
|② "92 < 128 ⇒ 控制点空间不可行"不成立|**成立，且实测反证我方结论**|已修：`control_gt` 实测可行、`J` 是带状稀疏；报告改写|
|② 附：PDM 的 warm start 没传进投影器|**成立**|已修：`start=warm` 已接线；`CurveControlTask.project` 支持 `start`|
|③ "128 倍放大"推理不成立|**成立**|已修：改为实测 `‖J‖₂ / ‖J‖_F / ‖J‖ᵀλ‖`；报告改写|
|④ NFE 统计错误|**成立**|已修：真实网络前向计数器（`model_forward_calls`）|
|⑤ "12.4 m 是 RMSE 下限"表述不当|**成立**|已改称"终端噪声尺度"|
|⑥ 命名 `Diffuser` 掩盖了候选拓扑条件|**成立**|已改名 `S-C48 (DDIM, no inference-time safety correction)`|

## 1. 控制点空间的可行域**非空**（推翻我方上一份结论）

用数据集自带的 `control_gt`（B 样条标签，只用于诊断、从不作为解）检验 5 个场景：

|场景|`max_k g_k`（128 点约束，≤0 可行）|512 点碰撞|最小净距|
|---|---:|---|---:|
|61001（安全）|**−0.0833**|无|+6.66 m|
|61003|**−0.1094**|无|+8.75 m|
|74014（主表轻碰）|**−0.0184**|无|+1.49 m|
|86008（主表重碰）|**−0.0531**|无|+4.25 m|

即**每个场景都存在一条真实的、满足全部 128 条约束的 B 样条曲线**。因此"92 个自由变量 < 128 条约束
⇒ 可行域为空"的推理是错的（审查意见②正确）。上一份报告 §3.2 的这一结论作废。

## 2. `J_Q` 是**带状稀疏**的，不是稠密

5 个场景、多个噪声级实测：

|量|实测值|
|---|---|
|每行非零元|min 0 / **median 4**（个别场景 8）/ max 8|
|`‖J‖₂`|**1.567 / 1.573 / 1.629 / 1.643**|
|`‖J‖_F`|7.35 – 7.63|
|活跃行（`g > −1e-6`）|82 – 104 / 128|

median 4、max 8 正是三次 B 样条的局部支撑（一个曲线点只依赖 3–4 个控制点 × 2 坐标）。
`‖J‖₂ ≈ 1.6` 说明**不存在"128 倍放大"**：算子范数在 1.6 附近，与点空间同量级。
上一份报告 §3.3 的倍数推断作废（审查意见③正确）。

## 3. SafeDiffuser：机制说清了，但不是我上次写的那个

三个求解器路径的逐级实测（RoS，16 级；`slack` 已换算成**真实松弛**以便比较——
`legacy` 的 `slack_norm` 是 `r = neg/w`，乘以 w 才是实际松弛）：

|场景|路径|状态|NFE|真实松弛|硬残差 `min(Jv+b)`|
|---|---|---|---:|---:|---:|
|61001|`legacy`（w=100）|ok|16|**0.02 ×100 = 2.0**|−0.82|
|61001|`osqp` ρ=1/w²=1e−4|ok|16|1.997|−0.822|
|61001|`osqp` ρ=1e4（w=0.01）|**failed**（2 NFE）|2|0.377|−0.171|
|74014|`legacy`|ok|16|0.031 ×100 = 3.1|−1.33|
|86008|`legacy`|ok|16|0.029 ×100 = 2.9|−0.92|

两条路径在**同一 ρ 下结果一致**（slack 2.0 vs 1.997，残差 −0.82 相同），说明我的 `cbf_solve_osqp`
与仓库的 `cbf_velocity` 是同一个问题的两种解法，不是两个不同的模型。

**首轮主表跑的是 `legacy` 路径**（`cbf_solve` 当时还不存在），它给出的速度**只满足软约束**
（硬残差 −0.8…−1.3 scene）、松弛 2–3 scene。所以"松弛几乎免费 ⇒ CBF 等于关闭"这个解释是**错的**。
真实机制是：

1. 逐级 trace 里 `min_h_scene` 的平均值是 **−0.53 … −0.64 scene ≈ −42…−51 m**——高噪声级别下状态本来就深不可行；
2. SafeDiffuser 是**预防型**条件（`ġ ≥ −ε(g−γ)`），它限制违反的**变化率**，不负责把已违反的状态拉回来；
3. 每微步位移被 `cbf_max_point_step`（0.156 m）限制，需要的外向速度是数十 m/步，远超可执行范围 ⇒ 松弛吸收；
4. 末级交付时，早先累积的违反已经无法消除 ⇒ 74014/86008 仍碰撞。

也就是说：**SafeDiffuser 在本基座上"能算、算得对、但不负责修复既有违反"**。这与"实现是 no-op"是两件不同的事，
上一份报告 §3.1 的说法作废。**要真正修复那 2 个场景，必须改算法**（可行初值 / 每级末投影），本轮不动公式。

同时修掉两个真实的代码问题（审查①）：
- `cbf_solve` 只检查 `res.x is None`，不检查 `status_val` ⇒ 现在只接受 OSQP `1/2`，否则抛错；
- `kkt_dual` 不是真对偶残差 ⇒ 改为记录 `primal_residual`/`dual_residual`（OSQP 原生），并加
  `velocity_change_norm`、`velocity_ratio`、`slack_norm/max`、`cbf_residual`、`cbf_residual_soft`；
- 求解器改为显式门控：`SC48ControlTask(cbf_solver='legacy'|'osqp')`，默认 `legacy`，
  即**已提交的 20 场景主表可复现**，`osqp` 路径需显式开启。

## 4. 投影：warm start 是真的 bug，可行域是真的非空

|场景|冷启动|以 `control_gt` 为 warm start|
|---|---|---|
|61001|`sc48_projection_failed:1:residual=0.6151`|**`identity`，残差 −0.0833，0 次迭代**|
|61003|`failed:1:residual=0.8564`|**`identity`，−0.1094，0 迭代**|
|74014|`failed:18:residual=0.7973`|**`identity`，−0.0184，0 迭代**|
|86008|`failed:2:residual=0.7785`|**`identity`，−0.0531，0 迭代**|

两点：
1. 从可行点出发，投影立即成功且残差为负 ⇒ **可行域非空**（再次反证"结构化不可行"）；
2. 冷启动在 1–18 次迭代内残差 0.6–0.86 就停滞 ⇒ 局部 SQP 对初值高度敏感，这正是 warm start 该起作用的地方，
   而审查②指出的缺陷确实存在：`pdm/runner.py` 里 `warm = x.copy()` 虽然算了，
   但 `task.project(target, budget)` 没有把它传进去。现在：
- `CurveControlTask.project(target, budget, start=...)` → 透传给 `nearest_projection(warm=start)`；
- `pdm/runner.py` 与 `constrained_diffusers/runner.py` 都改为 `start=warm`；
- `SC48ControlTask.project` 本来就支持 `start`。

修完后仍有一个待查项：PDM 的投影在冷启动下残差依旧很大（`diagnostics*.json` 里的 `pdm` 段），
需要下一轮确认是"目标漂移太远"还是"局部 SQP 收敛不足"。

### 4b. 修复 warm start 后重跑 20 场景：**结果逐位不变**（0/20）

`results/sc48_seven_warmstart/`（20 场景 × Projected + PDM）：Projected **0/20**、PDM **0/20**，
失败残差与修复前**逐位相同**（61001 Projected `residual=0.5983, 32 iters`；69032 `1.143, 1 iter`）。

原因很直接：**失败发生在第 0 级（t=15），而那一级 `warm=None`**。warm start 只在"上一级已经成功"时
才有用，可链条的第一个环节就断了。

### 4c. 那么到底是"够不着"还是"求解器不行"？——按噪声级分开看

`scripts/diag_sc48_reachability.py`（`results/sc48_phase_b_review/reachability.json`），
在每个噪声级上取"该级模型自己的干净预测 + 该级噪声尺度"作为目标，做两件事：
(a) 调用 Projected/PDM 真正用的那个局部 SQP；(b) 一个**仅用于诊断**的两段动作
（曲线空间逐点夹出障碍 → 最小二乘拟合回控制点）来判断"可行解离目标有多远"。

|场景|t=15（σ=325 scene）|t=8（σ≈1.4 scene）|t=0（σ=0.11 scene）|
|---|---|---|---|
|61001|**失败，残差 50.5**|失败，残差 **0.00105**|**成功，残差 −0.0033**|
|74014|**失败，50.2**|失败，**0.00113**|**成功，1.4e−12**|
|86008|**失败，49.8**|**成功，4.4e−16**|**成功，2.8e−9**|

三点结论：

1. **在 t=0（以及 t=8 的一部分）投影是成功的**，残差 ≤ 0。控制点空间的"最近可行曲线"是找得到的。
2. **t=8 的两次失败只差 0.001 scene = 8 cm**（容差 1e-6），是**收敛/容差**问题，不是不可行。
3. **t=15 的残差 ≈ 50 scene = 4000 m**：那一级的状态是 σ=325 的纯噪声，"投影到可行集"这件事本身没有意义。
   点空间之所以从不报错，是因为它的投影是**逐点可分离**的——每个点单独挪到最近的自由格，
   永远能"成功"，但那 128 个点已经不再是同一条曲线了。

**因此 Projected/PDM 的 0/20 不是"控制点空间不可行"的证据，而是"在状态还是噪声的级别上做投影、
并且要求 1e-6 的精确可行"这件事本身不成立。** 这与点空间的差别不在可行性，而在于点空间的投影
退化成了 128 个互不相关的小问题。

据此，让 Projected/PDM 在控制点空间跑通的路径是**明确且可验证的**（属改算法/改日程，需授权）：

|选项|内容|代价|性质|
|---|---|---|---|
|P1|只在 `σ_t ≤ σ_阈值`（例如 ≤1 scene）的级别开始投影；高噪声级用未投影的 x₀ 预测|1 天|改日程，必须声明（PDM App.F 允许调整投影频率）|
|P2|容差改为相对尺度（例如 `1e-3 scene` = 8 cm）|半天|改数值口径，需说明与论文 `C` 的差别|
|P3|t=15 用"曲线空间逐点夹出 + 拟合回控制点"作为**初始化**，再跑 SQP|2 天|改初始化，影响算法语义|

三者都不改论文的约束集合与更新式，只改"在哪些级别、以什么初值与容差做投影"。**本轮未实施**。

## 5. PD / ALM：额外采样步**不是**振荡的原因

三场景、`cd_extra_steps ∈ {0,16,64}`（NFE 实测 16/32/80）：

|场景|方法|extra=0|extra=16|extra=64|
|---|---|---|---|---|
|61001|Primal-Dual|len 7.28 / curv 181 / −15.35 m|len 3.82 / curv 40 / −7.71 m|len 4.13 / curv 139 / −11.28 m|
|61001|ALM|len 6.98 / curv 80 / −15.22 m|len 4.24 / curv 49 / −4.10 m|len 4.25 / curv 287 / −11.08 m|
|74014|Primal-Dual|len 3.75 / curv **643** / −24.82 m|len 2.36 / curv 37 / −7.27 m|len 2.42 / curv 15 / −9.30 m|
|74014|ALM|len 3.64 / curv 61 / −22.60 m|len 2.13 / curv 31 / −11.51 m|len 2.05 / curv 58 / −15.87 m|
|86008|Primal-Dual|len 3.88 / curv 79 / −22.88 m|len 2.58 / curv **1261** / −10.13 m|len 2.98 / curv 36 / −17.81 m|
|86008|ALM|len 4.22 / curv 13 / −17.56 m|len 2.46 / curv 31 / −12.33 m|len 2.95 / curv 30 / −14.28 m|

**`extra=0` 也一样爆（长度比 3.75–7.28、曲率 79–643、净距 −15…−25 m）**，所以"64 次末端更新导致振荡"
这一假设被实测排除（审查③提出的检查项已做）。振荡来自每级的乘子/惩罚更新本身，与额外步数无关。

结合 `‖J‖₂ ≈ 1.6`，"128 倍放大"也不成立。正确表述是：
**乘子与惩罚系数沿用了点空间（每点一行）的推导，尚未针对 B 样条控制变量的 Jacobian 结构重新标定**——
这是"未完成适配"，不是"算法不适合"。

## 6. NFE：已改为真实计数

原实现 `nfe = cd_levels if method != 'Diffuser' else len(schedule)` 会把所有方法记成 16。
现在 `run_method` 用计数器包住 `task.clean`，返回值新增 `model_forward_calls`。实测：

|方法|真实 NFE（`cd_extra_steps=64`）|
|---|---|
|Diffuser / RoS / ReS / TVS|16|
|Projected / Primal-Dual / ALM|16 + 64 = 80|
|PDM|`pdm_levels × pdm_inner` = 10 × 20 = 200|

（`Primal-Dual extra=16` 实测 32、`extra=64` 实测 80，与公式一致。）

## 7. 表述修正

- `√2·σ_min = 12.4 m` 改称**终端噪声尺度（terminal noise scale）**，不再称为"RMSE 下限"
  （score 修正与投影都会改变最终误差）。
- 主表的 `Diffuser` 行更名为 **`S-C48 (DDIM, no inference-time safety correction)`**：
  该模型的 `forward_all` 接收 `candidate_xy / candidate_mask / candidate_geometry / geometry_lengths`，
  含候选拓扑条件，写成无条件 Diffuser 会误导。
- 2/20 与 18/20 是 **20 个 val 场景**上的初步结果，不是独立 test 集结论。

## 8. 修正后的总结

> S-C48 显著改善了基础生成质量（2/20 碰撞、+3.5 m 净距、16 NFE、0.68 s），
> 但把现有安全扩散算法直接迁移到 B 样条控制点空间**尚未实现有效的安全约束增强**。
> 失败同时涉及求解器实现（warm start 未传递、OSQP 状态未检查、NFE 统计错误）、
> 参数尺度（乘子/惩罚沿用点空间推导）与状态表示适配（预防型 CBF 无法修复高噪声段的既有违反），
> **不能归结为算法本身的局限，也不能归结为"控制点空间不可行"**（已用 `control_gt` 反证）。

## 9. 本轮改动清单（代码）

|文件|改动|
|---|---|
|`adapters/sc48_control.py`|`cbf_solve` → `cbf_solve_osqp`（ρ=1/w²，OSQP 状态检查，完整诊断字段）；新增 `cbf_solver`/`slack_weight` 构造参数，默认 `legacy`；修正 `project` 的"仿射"注释为"单元内二次 + 局部线性化 SQP"|
|`adapters/curve_control.py`|`project(..., start=None)` 透传 warm start|
|`algorithms/pdm/runner.py`|`task.project(..., start=warm)`|
|`algorithms/constrained_diffusers/runner.py`|`task.project(..., start=warm)`|
|`algorithms/runner.py`|真实网络前向计数器 → `model_forward_calls`|
|`scripts/diag_sc48_phase_b.py`|新增：可行域/J 谱/求解器对比/投影 warm start/extra-step 敏感性/NFE 一次跑完|

## 10. 下一步（按审查建议的 Phase B-Review 收尾）

1. PDM/Projected 在**修复 warm start 后**重跑三场景，确认残差是否收敛（当前冷启动仍失败）。
2. SafeDiffuser 若要真正修复"既有违反"，需要改算法（投影/可行初值），需单独授权——本轮不动公式。
3. PD/ALM 的乘子与惩罚系数需要按 `‖J‖` 重新标定，需单独授权（属改公式）。
4. 以上完成前**不实施 B3**，也不合并两张表做横向排名。
