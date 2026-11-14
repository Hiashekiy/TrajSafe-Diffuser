# 七种对比方法的实现说明（约束作用在哪一层、怎么更新、约束是什么）

对应代码：`experiments/safety_baselines/algorithms/{safediffuser,constrained_diffusers,pdm}/runner.py`、
`algorithms/{runner,kernels,solvers}.py`、`adapters/{native_points,segment_points,paper_segment_constraints,point_sampler,point_evaluator}.py`。
配置：`configs/seven_native_points_aligned_{t16,t256}.json`。证据与数字：`docs/DSH_FIX_REPORT_20261114.md`。

---

## 0. 先回答两个问题

**问题一：现在修正的是"预测的干净轨迹"吗？**

不是。七个论文方法里，被约束/投影的对象是**带噪声状态** `x_t`：

- `Projected` / `PDM`：对**加噪状态走了一步 Langevin/SGLD 之后的结果**做投影；
- `RoS / ReS / TVS`：对**由 clean 预测导出的 DDIM 名义跳转**求一个速度 `ν`，再对 `ν` 施加 CBF-QP；
- `Primal-Dual / ALM`：把约束梯度作为**漂移项修正**加到加噪状态上（乘子是状态量）。

只有 `scripts/audit_backbone_effect.py` 里的**对照实验**（旧 SDF 修正器的表示逻辑，不是七方法之一，也不是 Ours）才在 clean 预测上做修正。

**问题二：修正后的轨迹会再经过一次扩散，生成下一阶段的带噪状态吗？**

不会。论文循环里"修正后的结果"**本身就是**下一阶段状态：

| 方法 | 被修正 / 被约束的对象 | 修正后是否再经"加噪"得到下一状态 | 下一状态 x_{t-1} 的来源 |
|---|---|---|---|
| 旧 SDF 修正器（对照，非论文方法） | **clean 预测 x̂₀** | **是** | 修正后的 x̂₀ 回代 DDIM 递推式重加噪 |
| Diffuser 基线（无约束） | — | — | DDIM 递推（最后一级直接输出 x̂₀） |
| RoS / ReS / TVS | 速度 ν（由 x̂₀ 导出的 DDIM 名义跳转） | 否 | `x_{t-1} = x_t + Δτ·ν*`（CBF-QP 修正后的积分） |
| Projected | 加噪状态（Langevin 一步的 target） | 否 | `x_{t-1} = Π_C(target)` |
| Primal-Dual | 加噪状态（乘子修正漂移） | 否 | Langevin 一步的结果本身 |
| ALM | 加噪状态（乘子+罚项修正漂移） | 否 | Langevin 一步的结果本身 |
| PDM | 加噪状态（SGLD 一步的 target） | 否 | `x_{t-1} = Π_C(target)` |

```
旧修正器（对照）                       七个论文方法
────────────────────                   ────────────────────────────────────
x_t ──denoise──► x̂₀                     x_t ──denoise──► x̂₀
                 │                                     │  (只用来算 score / 名义跳转)
                 ▼ 修正（数据空间）                     ▼
                x̂₀'  （安全）          ┌─ Projected/PDM：x_t + s·score + noise → Π_C ─┐
                 │                        ├─ PD/ALM：x_t + β/2(score − λ∇g) + noise
                 ▼  DDIM 重加噪           └─ RoS/ReS/TVS：ν = (名义跳转 − x_t)/Δτ → CBF-QP
              x_{t-1}（带噪）                          │
                                    x_{t-1} = 修正后的结果（不再加噪）
```

**这个差别是量级性的**：旧修正器在数据空间做一次小位移，随后 DDIM 重加噪会把修正"平滑"进下一步；论文循环让修正与噪声在同一层正面相遇，投影要在锯齿上求解（报告 §3.0 的实测：无约束时 PDM/CD 折线长度是 GT 的 14–21 倍）。

---

## 1. 七个方法共用的骨架

### 1.1 状态

```
H = 128 个轨迹点，每点 2 维
x  = 内部 126 个点展平 → 252 维（自由变量）
首末点由 pin_endpoints(x, cond) 钉死，永远不进优化变量，也不参与任何更新
输出 = unpack(x)：把 252 维还原成 128×2，首末点填 cond → 交付折线 = 优化变量本身
```
`adapters/native_points.py: unpack/points/initial/clean/score/constraints/project`。
校验：`scripts/validate_point_methods.py` 断言 `internal_state_trajectory == trajectory`（逐元素）、首末点未被改动、输出形状 (128,2)。

### 1.2 模型与 score

模型 `map_only_point_diffuser_v1`（`models/map_only_points.py`）：输入带噪点 + 占据图 + 起终点 + 时间 t，输出 **x₀ 预测**（不是 ε 预测）。没有骨架、走廊、安全特征。4,366,146 参数。

score 由 x₀ 预测精确换算（`native_points.py: score`）：

```
VP:  score(x,t) = −(x − √ᾱ_t·x̂₀) / (1 − ᾱ_t)
VE:  y = x/√ᾱ_t,  σ_t² = (1−ᾱ_t)/ᾱ_t,  score(y,t) = −(y − x̂₀)/σ_t²
```
两者是同一条 SDE 的两种坐标；`s = γ·ᾱ_t` 是精确变量替换（`pdm/runner.py` 的注释）。

噪声日程：基座 `NoiseSchedule(timesteps, squaredcos_cap_v2)`。**σ_min 决定一切**：

|timesteps|σ_min（m）|Langevin 类方法交付轨迹的必然残差（两维 RMSE）|
|---:|---:|---:|
|16（交接时的基座）|8.767|12.40 m|
|**256（当前）**|**1.099**|**1.56 m**|

waypoint 间距 0.95 m。16 级时残差噪声是间距的 9.2 倍 ⇒ 交付折线必然是锯齿（实测 CD 21×、PDM 14.4×，无约束 DDIM 0.97×）。

### 1.3 约束对象（两套，注意它们不是一回事）

**A. 逐顶点自由空间约束**（RoS/ReS/TVS/Projected/PD/ALM 用）

```
h(p) = 到"自由格并集"边界的解析有符号距离（scene 单位，1 = 80 m，1 格 = 0.625 m）
       h > 0 在自由格内；h < 0 在障碍内或地图外
g = −h ≤ 0 即该顶点安全
∂g/∂x 稀疏：每个顶点一行，只有它自己 2 列（`native_points.py: constraints`）
可行 ⇔ 128 个顶点全部在自由格内
```
实现 `adapters/exact_map.py`：把自由格边界拆成线段集合，用 STRtree 求最近边、返回精确距离与梯度。**只在真正最近特征处不可微**，不是双线性插值（双线性插值在像素中轴处梯度会消失）。

**B. 整段线段 vs 圆并约束**（PDM 原文 App. C.3；本轮新实现）

```
障碍 = 占据格并集 → 用"内切极大圆覆盖 + 半格对角垫"转成一组圆 {(c_j, r_j)}
         （圆并 ⊇ 占据区；实测 215–249 圆/场景，自由格被误禁 3.5–3.9%）
g_ij = r_j − dist(线段_i, 圆心_j) ≤ 0  ⇔ 该线段与该圆不相交
约束行数 = 127 段 × M 圆（稀疏：每行最多 4 个非零）
可行 ⇔ 所有线段都不穿过任何障碍圆
```
实现 `adapters/paper_segment_constraints.py`（`obstacle_circles` + `CircleSegmentConstraints`），
task 封装 `adapters/segment_points.py`（`SegmentPointTask`，同一网络/同一起终点/同一输出，只换约束对象与投影）。

**评价器（与约束对象独立）** `adapters/point_evaluator.py`：对交付折线做**逐格跨越**检测，同时报告
`vertex_collision`（顶点在障碍内）、`segment_only_collision`（顶点都安全但连线穿障）、`collision`（任一）。

### 1.4 投影与 QP 求解器

```
Π_C(z) = argmin ‖x − z‖²  s.t. g(x) ≤ 0
```
- 逐顶点约束集：约束按点可分 ⇒ 精确解 = 每个顶点沿 SDF 梯度做一次牛顿步（`native_points.py: project`），
  推完**自检**残差 ≤ 1e-5，否则抛 `SolverFailure('native point projection residual')`。不做迭代最近点。
- 整段约束集：最小范数 SQP（OSQP 解线性化 QP + 精确 L1 罚 merit 回溯 + 罚权 continuation + 行修复兜底），
  最后对**全部**圆做可行性校验（`CircleSegmentConstraints.project`）。可行时返回恒等（原文 App. C.4 的定义）。
- CBF-QP：`J ν + offset ≥ 0`；RoS 用可分精确解（`NativePointTask.cbf_solve`），ReS 用带松弛变量 w 的
  消元精确解，通用情形回落 OSQP（`solvers.cbf_velocity`，含 KKT/不可行性独立校验）。

---

## 2. 逐方法

约定：`t` 从大到小遍历（N−1 → 0）；`β_t = schedule.betas[t]`；`ᾱ_t = schedule.alphas_cumprod[t]`；
`clean = task.clean(x,t)` 是 x̂₀ 的内部点；`pin(·)` 只钉首末点。

### 2.1 Diffuser 基线（无约束参照，不是七方法之一）

`adapters/point_sampler.py: sample_points`

```python
points = pin(randn(1,128,2))
for t, nxt in [(N-1,N-2), ..., (t0, -1)]:
    clean = model(points, occ, cond, t)              # x̂0
    if nxt < 0:  points = clean                      # 最后一级直接输出 clean
    else:        points = a[nxt]*clean + s[nxt]*(points − a[t]*clean)/s[t]   # DDIM 递推
    points = pin(points)
```
**约束：无。** 用于给出"不做任何安全处理"的参照。

### 2.2 RoS-diffuser / ReS-diffuser / TVS-diffuser（SafeDiffuser, ICLR 2025）

原文：eq (5)-(6) 可控形式、eq (7) SafeDiffuser 证书、eq (8) RoS / (9)(10) ReS / (11)(12) TVS、eq (13)(14) QP、Thm 3.4–3.6 的 γ/σ 条件。代码 `algorithms/safediffuser/runner.py`。

每一步（每个扩散级 t）：

```
1) clean  = task.clean(x, t)                                  # x̂0（只用来构造名义跳转）
2) nominal = √ᾱ_{t-1}·clean + √(1−ᾱ_{t-1})·(x − √ᾱ_t·clean)/√(1−ᾱ_t)   # DDIM 名义下一状态
3) desired ν = (nominal − x) / Δτ          (Δτ = cbf_dt = 1)
4) 微步积分：while elapsed < Δτ:
       h = −g(x),  J = ∇h(x)                                  # 逐顶点 SDF
       RoS :  offset = ε(h − γ),                 w = 0        # 硬约束，γ 按 horizon=N·Δτ
       ReS :  offset = ε(h − γ_used),            w = 100 (t > N0) / 0 (t ≤ N0)
              γ_used = γ (t > N0) ； γ_used = legal_gamma(·, (N0+1)Δτ) (t ≤ N0)
       TVS :  offset = −σ̇ + ε(h − σ),            w = 0        # σ,σ̇ 见下
       解 ν* = argmin‖ν − desired‖²  s.t.  J ν + offset ≥ 0   (+ w·r 松弛项)
       x += step·ν*    (step ≤ Δτ/substeps，再被 cbf_max_point_step/speed 掐)
5) x 就是下一级状态 x_{t-1}（不再加噪）
```

参数来源：

- `γ = legal_gamma(h, horizon, ε, floor)` = `max(floor, −min(h)/(e^{ε·horizon}−1), max(h)/(e^{ε·horizon}+1))`。
  这是 Thm 3.4 条件 `γ ≥ max_k|γ−b_k|e^{−εN}` 的**精确解**（本轮已重新推导核对）。
  RoS 用 `horizon = N·Δτ`；ReS 在进入硬区间时改用 `horizon = (N0+1)·Δτ`（论文用 N0，差 1）。
- `hard_steps = 10 → N0 = 9`：对应原文 ReS-DIFFUSER-**l10**（最后 11 步硬约束）。
- TVS：`σ(j) = min(h₀,0)·j/N`，`σ̇ = −min(h₀,0)/(N·Δτ)`，其中 `j = t+1 − elapsed/Δτ`。
  σ 的端点条件 `σ(N) ≤ b(τ^N)`、`σ(0)=0` 满足；**σ̇ 的符号与它自己 σ 的导数差一个负号和 Δτ，尚未定论**（报告 §4c）。
- `ε = cbf_epsilon = 1`、`Δτ = cbf_dt = 1`、`floor = cbf_gamma_floor = 1.25e-4`、`qp_tol = 1e-5`。

**约束是什么**：每个轨迹点相对自由空间边界的距离 `h`；RoS 要求 `h` 按 `e^{−ε}` 速率衰减（ε=1、N=16/256 时几乎等于"不许退让"），ReS 在前段用松弛变量软化、最后 11 步变硬，TVS 用一个随时间变化的 σ 把要求"抬回"到 0。

**忠实 / 偏差**：QP 目标、barrier 形式、γ 闭式、ReS 的 w(j) 结构都对得上原文；不在原文的是 `cbf_max_point_step`（每微步 0.156 m 的位移上限）、固定 Δτ=1、以及 `cbf_substeps` 空转（实测 16→1 结果完全不变，见报告 §4c）。

### 2.3 Projected（Constrained Diffusers, NeurIPS 2025, §5.1 eq 10-11）

代码 `algorithms/constrained_diffusers/runner.py`（`method == 'Projected'` 分支）。

```
每个 level t（cd_levels=64 个 level 访问）：
    target = x + β_t/2·score(x,t) + √β_t·z                 # Langevin 一步（加噪状态）
    x = Π_C(target)                                        # 投影后的结果直接是 x_{t-1}
最后 cd_extra_steps=64 次在 t=0 上重复同样的更新
返回 x（无挑粒子、无按 GT/可行性选种子）
```
**约束**：逐顶点自由空间（1.3 A）。**约束作用层**：加噪状态。**不再重加噪。**

忠实：eq (10)(11) 完全一致。偏差：`cd_levels/extra` 用 `linspace(N−1,0,64).round()` 实现（16 级基座时每级被重复 4 次，256 级时是 64 个不同级）；`Π_C` 是单次牛顿步而非迭代最近点；做约束的是顶点不是线段。

### 2.4 Primal-Dual（同文 §5.2 eq 13-14）

```
x = [4 个粒子 × 252] 初始化 N(0,I)
每个 level t：
    score = ∇log p_t(x)                            # 每个粒子各自
    g = max(0, −h)·scale,  J = ∂g/∂x·(g>0)·scale   # 只保留违反项（Remark 1 的 [g]+）
    x ← x + β/2·(score − λᵀJ) + √β·z               # 约束梯度进漂移
    λ ← [λ + η_λ·mean_over_particles(g)]₊          # eq (14)，η_λ = dual_eta
返回粒子 0
```
**约束**：逐顶点；`scale = dual_constraint_scale = 128`（把 scene 单位换成格单位，使 η_λ、ρ 与论文 Appendix C "large" 设置的量级对齐）。**约束作用层**：加噪状态的漂移项。**不再重加噪。**

忠实：eq (13)(14)、Remark 1 的 hinge 期望都对得上。偏差：`scale=128` 是原文没有的单位换算；4 粒子估 `E[g]` 噪声很大；"固定返回粒子 0"是仓库选择。

### 2.5 ALM（同文 §5.3 eq 15-18）

```
每个 level t：
    mean = mean_particles(g);  slack s = max(0, −mean − λ/ρ)
    coeff = λ + ρ(mean + s)
    x ← x + β/2·(score − coeffᵀJ) + √β·z
    λ ← λ + ρ(mean + s);   ρ ← 1.01·ρ
返回粒子 0
```
**约束**：逐顶点。**约束作用层**：加噪状态的漂移项。**不再重加噪。**

忠实：eq (16)(17)(18) 一致。偏差：`ρ` 增长率 c=1.01（原文只要求 c>1）；4 粒子；粒子 0。

### 2.6 PDM（Constrained Synthesis with Projected Diffusion, NeurIPS 2024, Alg.1）

代码 `algorithms/pdm/runner.py`。

```
x = N(0,I)（VP 坐标；VE 坐标下是 N(0,σ_T)）        # Alg.1 第 1 行，不投影
levels = linspace(0, N−1, 10).round() 的降序（T = 10）
for t in levels:                                     # σ_t 由大到小
    for i in range(M = 20):                          # Alg.1 第 4-6 行
        s = γ_t·ᾱ_t            (VP)                  # = γ_t (VE)；γ_t 见下
        target = x + s·score(x,t) + √(2s)·z          # SGLD，没有漂移项（原文如此）
        x = Π_C(target)                              # 逐点/整段约束集
x 就是 x_0（不再加噪）；末端级要求精确可行，否则 SolverFailure
```
γ_t 两种约定，均已实现并记录来源（`level_gamma`）：

|约定|公式|本日程下的每步相对混合率 γ_t/σ_t²|
|---|---|---|
|`paper`（Alg.1 第 3 行）|`γ_t = σ_t²/(2σ_T²)`|`1/(2σ_T²)` = 4.7e-6（16 级）/ ~1e-9（256 级）|
|`official_ald`（官方 `projected_diffusion/scoremodel.py`）|`γ_t = ε·(σ_t/σ_min)²`|`ε/σ_min²` = 1/6（两者都按此反算 ε）|

**这是必须做选择的差异**：原文公式要求 VE 噪声上界是 O(1)，而本基座 σ_T 在 16 级时是 325、256 级时更大，所以原文公式的混合率低到 M=100 步内样本不动（实测 RMSE 3.4e4 m）；官方实现约定才可用（修正后 17.8 m）。当前配置用 `official_ald`。

**约束**：两种都实现了——`configs/seven_native_points_aligned_t16.json` 用逐顶点（与其它方法可比），
`configs/seven_native_points_aligned_segment_pdm_t16.json` 用原文整段线段约束（结果见 `results/native_points_pdm_segments_20261114/notes.md`：预算内解不出可行投影）。**约束作用层**：加噪状态。**不再重加噪。**

忠实：Alg.1 的层级/内层结构、无漂移 SGLD、投影即返回。偏差：M=100 → 20（原文 App.F 明确允许）；T=10 ✓ 原文。

---

## 3. 七个方法在同一基座上的结果（20 案例，256 级基座，种子 0）

|方法|约束对象|折线长度/GT|转角|顶点碰撞|仅线段碰撞|路径碰撞|planning success|RMSE m|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|RoS-diffuser|逐顶点|50.97|91°|20|0|20|0/20|51.2|
|ReS-diffuser|逐顶点|2.43|19°|10|2|12|8/20|20.0|
|TVS-diffuser|逐顶点|31.15|74°|20|0|20|0/20|38.0|
|Projected|逐顶点|3.83|106°|0|13|13|7/20|24.7|
|Primal-Dual|逐顶点|3.37|103°|14|0|14|6/20|29.3|
|ALM|逐顶点|3.40|102°|14|0|14|6/20|31.6|
|PDM|逐顶点|5.35|108°|0|18|18|2/20|21.4|
|Diffuser（无约束）|无|0.99|1.2°|10|0|10|10/20|17.4|

读法：`长度/GT ≫ 1` 且转角大 ⇒ 交付折线是锯齿（采样器末级残差噪声，见 1.2）；`顶点 0 / 仅线段 18` ⇒ 顶点都被推进了自由格，但连线横穿障碍（约束对象与评价对象不一致）。

---

## 4. 原文没有、由实现自己添加的部分（可能的实现问题）

|#|东西|位置|影响|状态|
|---|---|---|---|---|
|1|`cbf_max_point_step = 0.001953125`（0.156 m/微步）|safediffuser 微步积分|使 `cbf_substeps` 完全失效（16→1 结果相同），交付侵入 36.9 m|已测出|
|2|`Δτ = cbf_dt = 1` 固定|同上|单级位移可达数十米，一阶 barrier 线性化失效；原文靠"Δτ 足够小"回避|已测出|
|3|`dual_constraint_scale = 128`|PD/ALM|λ、ρ 的单位被人为缩放|未验证|
|4|`particles = 4` + 固定返回粒子 0|PD/ALM|`E[g]` 只有 4 样本，乘子更新噪声大|未验证|
|5|`cd_levels/extra` 用 `linspace().round()`|Projected/PD/ALM|16 级基座时每级重复 4 次；换 256 级基座后变成 64 个不同级（步数含义变了）|已知|
|6|六方法用"逐顶点"约束、评价用"整段"|1.3 A vs 评价器|顶点 0 / 路径 100% 碰撞的直接原因|已知|
|7|TVS 的 `σ̇ = −min(h₀,0)/(N·Δτ)`|safediffuser|与其自身 `σ(j)` 的导数差一个负号和 Δτ|待核|
|8|`ρ` 每级 ×1.01|ALM|原文只要求 c>1|已知|

---

## 5. 如果要把"修正"改到有效的位置（候选，未实施）

1. **SafeDiffuser**：对 `ν*` 做 backtracking，直到离散 barrier `b(x+s·ν*) ≥ γ + (b−γ)e^{−εs}` 成立
   （= 把原文的"Δτ 取足够小"自适应化，不改 QP、不改 barrier）。可验证指标：RoS 最大侵入 36.9 m → ?
2. **对齐 `cbf_max_point_step` 与 `cbf_substeps`**，让"细化"真的产生细化。
3. **统一约束对象**：给六个方法也提供整段线段约束集（`SegmentPointTask` 已就绪，只是 CBF 的可分 QP 捷径要换成通用 OSQP）。
4. **`dual_constraint_scale` / 4 粒子**：做一次消融，确认乘子法不是在拼单位。
