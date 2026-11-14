# 官方实现 vs 我们：逐条差异表（§6.0 交付物）

> 状态：**只读对照**。本文件生成过程中**没有修改 `algorithms/` 或 `adapters/` 里的任何一行约束代码**
> （唯一写入就是这个 .md）。表中"按官方改"的条目是要**等你确认后才动手**的候选清单。
>
> 对照对象（本机 `.refs/`，只读原文）：
>
> | 方法族 | 官方路径 | 读的关键文件 |
> |---|---|---|
> | PDM (NeurIPS 2024) | `.refs/Projected-Diffusion/` | `examples/trajectories/mpd/models/diffusion_models/{scoremodel,projection,solver,sample_functions}.py`、`projected_diffusion/scoremodel.py`、`examples/trajectories/mpd/models/diffusion_models/train_score_model.py` |
> | SafeDiffuser (ICLR 2025) | `.refs/SafeDiffuser/` | `diffuser/models/diffusion.py`（`#CBF` 共 34 处）、`scripts/plan_maze2d.py` |
> | Constrained Diffusers (NeurIPS 2025) | `.refs/NeurIPS-2025-…-Supplemental-Conference/code/` | `{umaze-large,umaze-small,Hopper,BallRunning,swimmer}/diffuser/models/diffusion.py` |
>
> 我们这边：`experiments/safety_baselines/`（行号为当前 HEAD `e29475a` 工作树）。

---

## 0. 三条一句话结论（先看这个）

1. **官方三份实现里，没有任何一份把"段×圆"全展开成一个大 QP/NLP 的稠密行集。**
   - SafeDiffuser：QP 是**逐点、2 变量、障碍数个约束**（`Q=I₂`，`(B,H,2)` batch），段根本不进约束；
   - PDM：行是"每段×每圆"（63×16=1008 行 + 8×63 额外），但**只在一个 128 变量的 IPOPT NLP 里**，且半径只有 16 个圆；
   - Constrained Diffusers：**一个障碍 = 一个约束函数**，梯度用 autograd 得到与 x 同形的张量，**从不构造稠密 Jacobian**。
   → 我们的 `MemoryError` / `qp_infeasible_certified` 不是"实现 bug"，是**把官方三种结构里的任意一种都换了形状**（46k 行稠密 J）。
2. **PDM 官方确实会在高噪声段跳过障碍投影**（`scoremodel.py:571-594`：`idx > 5` 才打开 `traj_p`），
   所以"高噪声门控"有官方先例，但**官方用的是噪声级下标判据，不是残差幅值判据**（我们是后者）。
3. **`Projected` 的归属应当维持 "Constrained Diffusers"，不能改判给 PDM**：
   官方 CD 的 `projected_gradient` 是**逐点解析径向投影到椭圆边界**（`_project_to_feasible_region`，一行闭式解），
   而 PDM 的投影是**整条轨迹的 NLP**。两者不是同一个算子，详见 §2 表 A6。

---

## 1. 官方实现的事实清单（写表之前先固定事实）

### 1.1 PDM — `.refs/Projected-Diffusion/`

| 事实 | 证据（文件:行） |
|---|---|
| 采样循环：外层遍历 10 个噪声级 `process`，**每个 Langevin 迭代后立刻投影一次** | `projected_diffusion/scoremodel.py:540-566`（`x = self.projection_step(x)` 在 `for step_num in range(self.annealed_step)` 内） |
| 步长 `step_size = eps * (σ/σ_min)²`，`σ_min = process[-1]`；`eps` 默认 `1e-1`，trajectory 训练脚本里是 `1.5e-5` | `.../mpd/.../scoremodel.py:509-510`；`train_score_model.py:20`；`main.py:24` |
| 日程：`n_steps=10` 个级（log 均匀，σ_max=10 → σ_min=0.005），`annealed_step=25`（`main.py:28`）/`100`（`train_score_model.py:28`） | `scoremodel.py:509`；`train_score_model.py:22-28` |
| **轨迹版**：`if idx > 5: self.annealed_step = 10`，且 `idx>5` 才把 `projection.traj_p=True`（即只有**后半段低噪声级**做障碍投影；前 6 级 `apply()` 只钳端点） | `.../mpd/.../scoremodel.py:571`、`581-594`；`projection.py:49-76`（`traj_p` 才进 solver）、`86-90`（`project_hard_conds` 只钳首末点） |
| 投影失败的处理：**不当作可行解**，`violation=True` → 外层 `while violation:` 再跑 **15 步 Langevin** 换初值重投 | `solver.py:70-76`；`scoremodel.py:586-591` |
| 投影算子 = `argmin ||x - gt||² s.t. g(x) <= 0`，CasADi+IPOPT（`ipopt.max_iter=150`, `tol=1e-3`），变量 128、box `[0,2]`，**每次从漂移目标 `x0=gt` 起步（没有跨级 warm start）** | `solver.py:29-62` |
| 约束行构造：**每段×每圆一行**，`(radius+eps) - dist(center, segment)`，`dist` 用 `t=clip(·,0,1)` 的**线段**最近点 → **段约束，非点约束** | `solver.py:143-172`（`for i in range(64-1)` 内层 `for circle_center`）、`243-250`（`line_circle_intersection`） |
| 点约束分支存在但**默认关闭**（`point_v = False`） | `solver.py:120-136` |
| 行数量级：63 段 × (16 配置圆 + 8 硬编码额外圆) ≈ **1512 行**；`eps=0.01` 是余量 | `solver.py:140-155` |
| 官方仓库自带示例把障碍**硬编码**成 16 圆 + 14 矩形（maze2d 演示），不是通用的占用栅格→圆转换 | `projection.py:21-46`；`solver.py:95-114` |

### 1.2 SafeDiffuser — `.refs/SafeDiffuser/diffuser/models/diffusion.py`

| 事实 | 证据（行） |
|---|---|
| CBF 是**离散时域**改写：`h = Lfb + k*b`（`k=1`，`Lfb=0`），QP 约束 `G u <= h`，`G = -∇b` ⇒ 等价于 `b(x+u) >= 0` | `451-506`（RoS maze2d-large）、`466`（`b = ellipse - 1 - 0.01` 稳健项） |
| **QP 变量 = 2（位置增量 u∈R²），行数 = 障碍数（2）**；`Q = I₂`，`q = -(xp1 - x)`；在**轨迹 horizon 上 batch 化**（`nBatch = x.shape[0]`，`x=x.squeeze(0)` 后第一维就是时间步） | `456`、`501-506`、`508-509` |
| 因此约束对象是**每个时间点（顶点）**，不是段：`rt[:,2:4] = x[:,2:4] + out` 对每个时间点各自加各自 2 维修正 | `508-509` |
| 每级只解 **1 次** QP（`p_sample` 里 `xp1` 生成后调用一次），没有子步积分、没有步长上限 | `1030-1097` |
| ReS：`t >= 10` 时 `sign=100`（放松：G 加 2 列 slack，`Q=I₄`，取 `out[:,0:2]`），`t < 10` 时 `sign=0`（硬，`Q=I₂`） | `588-658`（`610-616`、`649-656`） |
| TVS：`b = ellipse - sigmoid(t_bias-t) - 0.01`，`t_bias=5`，`Lfb = σ(1-σ)`（把时变项当 Lie 导数进 h），**不放松（无 slack）** | `895-956`（`896`、`911-921`） |
| 形式：`QPFunction(QPSolvers.PDIPM_BATCHED)`，无 slack 时 `e = Variable(torch.Tensor())`（空等式） | `506`、`370`、`443` 等 |
| **发布版默认把全部 SafeDiffuser 调用注释掉了**（`x = xp1  # training only`），要手工选一个 uncomment；另有"只对最后 10 步施加"的注释分支 | `1062`、`1063-1069`、`1071-1090` |
| 采样：1000 级 DDPM，每级一次 `p_sample` | `1100-1131` |
| 障碍：**2 个闭式椭圆/超椭圆**，参数写在 `norm_maxs/norm_mins` 归一化坐标里 | `459-490` |

### 1.3 Constrained Diffusers — `.../code/<env>/diffuser/models/diffusion.py`

| 事实 | 证据（umaze-large 行号） |
|---|---|
| 三方法同一实现，靠 `self.algorithm ∈ {primal_dual, projected_gradient, augmented_lagrangian}` 切换 | `65-69` |
| **该文件是坏的**：`algorithm = [3 个元素]` 却写 `self.algorithm = algorithm[4]` → 发布版直接 IndexError；其它环境目录里是 `algorithm[0..2]` 的连续赋值（后者生效） | `65-67`；`BallRunning/Hopper/swimmer:363-372` |
| 各环境的实际选择：umaze-small = `projected_gradient`；Hopper/BallRunning/swimmer = `primal_dual`（最后一行 `algorithm[0]` 生效）；umaze-large = 崩 | `umaze-small:68`；`Hopper:372` |
| 约束函数 `g_x1(state,t)`：归一化坐标下 `sqrt(((x2-off)/yr)² + ((x3-off)/xr)²) - 1`，**正数 = 安全**（与我们的 `g<=0` 相反）；`g_x_funcs` 长度 = 障碍数（umaze-large 只有 1 个） | `72-87` |
| 梯度：`autograd.grad(g_x.sum(), state)` → **与 x 同形的逐元张量**，形状 `(C,B,H,D)`；**全程没有稠密 Jacobian** | `306-322` |
| 乘子：`dual_vars` 形状 `(C, B, horizon)` —— **每个时间点一个乘子**（不是每行一个） | `95-99` |
| Primal-Dual：`posterior_mean += Σ_i λ_i ∇g_i`；`λ ← clamp(λ − lr·g, min=0)`，`lr=1e-3`（umaze-large）/`2.5e-2`（umaze-small）/`1e-1`（其余） | `188-202`；`325-348` |
| ALM：`s ← clamp(λ/ρ + g, min=0)`，`λ ← λ + ρ(g − s)`，`ρ ← ρ·1.004`（**只在 `cbf=True` 分支**；默认分支不涨 ρ）；`penalty` 在 `__init__` 是 `2.5e-4`，**`p_sample` 里每步重置为 `2.5e-2`** | `350-380`；`62`、`472` |
| Projected：`_project_to_feasible_region(xp1)` —— 对每个 `g<0` 的点做**闭式径向投影到椭圆边界**（`scaling=sqrt(1/r)`），无求解器、1 次到位 | `382-466`、`481-482` |
| 采样阶梯：`reversed(range(-200, n_timesteps))`，`i<=0` 时用 `i_=1` ⇒ **t=1 这一级重复 200 步**（不是 t=0） | `503-510`；`479`（`nonzero_mask` 用 `t==1`） |
| 强制项：`g_x` 在每个 `p_sample` 的 `xp1` 之后统计，每步都施加约束，**没有噪声级门控** | `481-485` |

---

## 2. 差异表

列说明：**结论** ∈ {按官方改 / 保留并说明 / 待定}。"我们"一列的行号来自当前工作树。

### A. 状态空间与约束对象（跨三份共性）

| # | 项目 | 官方实现（文件:行） | 我们（文件:行） | 差异 | 结论 |
|---|---|---|---|---|---|
| A1 | 状态空间 | PDM `scoremodel.py:557-627`：VE 数据空间 y；SD `diffusion.py:1030-1097`：x_t 直出；CD `diffusion.py:469-487`：VP 后验均值 | `adapters/native_points.py:10-19`（128 顶点，端点钉死，D=252）；PDM 额外走 VE（`pdm/runner.py:38-48`） | 我们把三种状态空间统一到"原生 128 点 + 端点钉死"，官方各在自己的空间里 | **保留并说明**（这是 `safety_points_t256` 重训的目的，报告必须写 "state lifted to native points"） |
| A2 | 端点约束 | PDM `projection.py:86-90`：每次投影前后**钳制**首末点（等式，不是罚项） | `native_points.py:21-25`（`unpack` 里直接覆写端点）、`native_points.py:58-60`（J 的 0/末行全 0） | 等价（都是硬钳制） | 保留（无需改） |
| A3 | **约束对象：点 vs 段** | PDM `solver.py:143-172` = **段**（`line_circle_intersection`，`t∈[0,1]`）；SD `diffusion.py:466/501-509` = **点**（逐时间步）；CD `diffusion.py:72-83` = **点**（逐时间步） | 顶点版 `native_points.py:57-60`；段版 `segment_points.py:29` + `paper_segment_constraints.py:112-128` | 我们两种都做了，但**七方法全部切段**（`configs/seven_safety_points_t256_segments.json:40-48`），而官方只有 PDM 是段 | **保留并说明**：段版本必须单独一列并标 "task-adapted segment variant"（口径红线 §8.2） |
| A4 | 段行数 | PDM：63 段 × 24 圆 ≈ **1512 行 / 128 变量**（`solver.py:140-172`） | **127 段 × 362 圆 = 45 974 行 / 252 变量**（配置注释按 128×362 记 46 355 行，`configs/…_segments.json:113`；HANDOFF §5 里崩溃的数组是 220 752 元素） | **行数比官方多 30 倍**。根源是**障碍表示**（官方 16 圆 vs 我们的内接最大盘圆并 215–370 圆，`paper_segment_constraints.py:23-71`），不是行构造方式 | **按官方改**：官方没有"每段取最紧一个圆"的约简，但有"障碍=少量圆"的前提。可选两条路——(a) 粗化圆并到 ~16–32 个/场景；(b) 保留 362 圆但只在**工作集**（当前违反/接近的圆）上建行。两者都必须在报告里声明 |
| A5 | 稠密 Jacobian | PDM：IPOPT 自己算导数；SD：不建 J（只用 `h=-g` 与解析 `∇b`）；CD：`autograd` 逐元梯度 `(C,B,H,D)`，**永不物化** | `constrained_diffusers/runner.py:67-69`：`np.stack([v[1] for v in vals])` → 稠密 `(P, rows, D)`；段版 rows=45974 ⇒ `4×45974×252×8 ≈ 370 MB`，加掩码即 `MemoryError` | **结构性差异**，不是调参能解决的 | **按官方改**（必做）：PD/ALM 的更新式只用到 `Σ_k λ_k J_k` 和 `Σ_k λ_k g_k`——这是 `Jᵀλ` 的一个矩阵-向量积，用**稀疏 J**（段版已有 `csr_matrix`）或"逐行回代"即可，不需要稠密 (rows,D) |
| A6 | `Projected` 的归属 | CD `diffusion.py:382-466`：**逐点闭式径向投影**（`sqrt(1/r)` 缩放），无求解器；PDM `solver.py:29-78`：**整轨迹 NLP** | `constrained_diffusers/runner.py:58-64` → `solvers.py:31-69`（SLSQP 局部最近点） | 我们的 Projected 用的是**通用局部 SLSQP**，既不是 CD 的闭式径向投影（那只能表达"每点一个椭圆"），也不是 PDM 的 IPOPT NLP | **待定 + 建议**：家族归属**保持 Constrained Diffusers**（paper 名 + 我们复现的是它的"每步投回可行集"算法骨架），但**必须写明"projection operator 被换成了通用局部 SLSQP"**；PDM 已有独立行，不重复计。若要严格对齐 CD，只能退回**顶点行**（`native_points.py:61-70` 的 `nearest_free_cell_union` 也不是闭式径向，需另写），段版本永远无法闭式 |
| A7 | 每级解几次 | PDM：每个 Langevin 内步 1 次投影（`scoremodel.py:573-579`，M=25/100 次/级）；SD：**每级 1 次 QP**（`diffusion.py:1030-1097`）；CD：每级 1 次（`469-487`） | SD：`safediffuser/runner.py:49-96` 每级**积分到 `dt_full=1`**，子步数由 `cbf_max_point_step` 决定（`86-89`）；PDM：`pdm/runner.py:53-69` 每内步 1 次投影（`pdm_inner=20`）；CD：`constrained_diffusers/runner.py:55-77` 每级 `cd_substeps=1` 次 | SD 是唯一结构不同的：官方"一步到位"，我们"连续时间积分" | **保留并说明**（这是论文连续时间 CBF 的读法，但要写清"官方发布码是离散一步"） |

### B. PDM

| # | 项目 | 官方 | 我们 | 差异 | 结论 |
|---|---|---|---|---|---|
| B1 | 步长 | `step_size = eps·(σ/σ_min)²`，`x += (δ/2)s + √δ z`（`scoremodel.py:509-510, 522-531`） | `pdm/runner.py:19-36` `official_ald`：`γ = eps·σ²/(2σ_min²)`，`x += γ s + √(2γ) z` | **完全等价**（γ = δ/2）。`pdm_ald_eps=6.29541e-05` 使 `γ/σ²=1/6`（官方 σ_max=10/σ_min=0.005 时该比值是 `eps/2/σ_min²=3e-2`，量级不同但公式一致） | 保留（已对齐；`docs/pdm_fidelity_audit.md` C1-C2 已记录） |
| B2 | 每级步数 | `annealed_step=25`（main.py）/`100`（train_score_model.py）；轨迹版 `idx>5` 降为 10 | `pdm_inner=20`（`configs/…json:30`） | 数量级一致；官方论文允许减 M（我们引 App. F） | 保留并说明 |
| B3 | 噪声级数 | `n_steps=10`（log 均匀，σ 10→0.005） | `pdm_levels=10`（在 256 级 VP 日程上均匀取 10 个） | 级数一致；**σ 取值来自我们的基座日程**（σ_max≈5195 scene），官方是 O(10) | 保留并说明（`pdm_fidelity_audit.md` §C1 已解释） |
| B4 | **高噪声是否跳过投影** | **是**：`idx>5` 才 `traj_p=True`（`scoremodel.py:571,581-594`）⇒ 10 级里只有 4 级做障碍投影；判据是**噪声级下标** | **是**，但判据是**残差幅值** `projection_gate_residual=0.05 scene`（`safety_point_task.py:107,117-119`），且 `info.status='skipped_over_gate'` | 机制不同：官方按级号，我们按可达性 | **按官方改（建议）**：改成"级号/σ 阈值"判据更可复现（σ > τ 跳过），把残差门降级为诊断字段；两种都写进差异表。**注意这是本项目唯一有官方直接先例的"门控"** |
| B5 | 投影失败策略 | 失败 → 再跑 15 步 Langevin 换初值重投，**永不把不可行点当交付**（`scoremodel.py:586-591`；`solver.py:70-76`） | 末级 `allow_partial=False` 抛 `SolverFailure`（`pdm/runner.py:81-89`；`paper_segment_constraints.py:282-286`） | 官方是"重试到成功"，我们是"失败即失败" | **保留并说明**（我们的更保守，报告里写成"交付前强制可行"） |
| B6 | 投影初值 | **无** warm start：每次 `x0=gt`（当前漂移目标）（`solver.py:40,60`） | `pdm/runner.py:62-64`：`start=warm`（上一级可行点） | 我们多了一个可选加速/鲁棒化 | **保留并说明**（已在 docstring 记为 2026-11-14 review 项） |
| B7 | 投影求解器 | CasADi + IPOPT 内点法，`max_iter=150`，`tol=1e-3`，`constr_viol_tol=1e-3`（`solver.py:50-51`） | OSQP 信赖域 SQP + `finish_feasible` 行修复（`paper_segment_constraints.py:138-198, 200-229`） | 求解器不同，可行集定义相同（都验终态残差） | 保留并说明（容差我们更严：1e-6 vs 1e-3） |
| B8 | 圆并几何 | 硬编码 16 圆（演示用） | `obstacle_circles`：内接最大盘 + 半格对角覆盖（`paper_segment_constraints.py:23-71`），实测 215–370 圆、误禁自由格 3.5–5% | 我们更通用也更保守，但行数爆炸 | **待定**：见 A4，先决定圆数策略 |

### C. SafeDiffuser（RoS / ReS / TVS）

| # | 项目 | 官方 | 我们 | 差异 | 结论 |
|---|---|---|---|---|---|
| C1 | 证书形式 | 离散一步：`G u <= h`，`h = b`（`k=1, Lfb=0`）⇒ `b(x+u) >= 0`（`diffusion.py:471-474, 501-506`） | `safediffuser/runner.py:81`：`offset = eps·(h − γ)`，`γ = legal_gamma(...)`；`cbf_epsilon=1.0`、`cbf_gamma_floor=1.25e-4`（`configs/…json:17-19`） | 在 `eps=1` 下 `∇h·v >= −(h−γ)` ⇔ `h_next >= γ ≈ 1.25e-4`，**与官方同形**，只多一个 1.25e-4 余量 | 保留（已等价；把"γ≈floor"写进表） |
| C2 | QP 规模 | **2 变量 × 障碍数(2) 行**，horizon 上 batch（`diffusion.py:501-506`） | 顶点版：`native_points.py:71-79` 有精确可分离解（每行独立，`v = nominal + Jᵀα`）；段版：`cbf_solve=None`（`segment_points.py:28`） ⇒ 落到 `solvers.py:71-118` 的**耦合 OSQP**，`J` 是 127×362 稀疏行 | 官方 QP **天然可分离**；段版**结构上不可分离**，行数 46k，OSQP 判不可行（`qp_infeasible_certified`） | **按官方改**：RoS/ReS/TVS 官方只有"点"约束（`diffusion.py:466`）。**不应把段约束硬塞进 CBF QP**——要么三者在顶点行上跑（官方口径），要么把段版本单列为 "task-adapted segment variant" 并接受它需要工作集 QP（不是官方内容） |
| C3 | slack 权重 | ReS：`t>=10 → sign=100`（G 加 2 列 slack，`Q=I₄`），`t<10 → sign=0`（`diffusion.py:610-616, 649-656`） | `relax_weight=100.0`，`w = relax_weight if (t>N0 or cbf_hard_slack) else 0`，`N0=hard_steps-1=9`（`configs/…json:27-28`；`runner.py:76,104`） | **完全一致**（官方 t≥10 放松、t<10 硬；我们 t>9 放松、t≤9 硬；权重都是 100） | 保留（已对齐，可作为"按官方核对通过"的正面条目） |
| C4 | slack 求解实现 | qpth 的 `-w·I` 块（`G = [G_b, rx1, rx0]`，`diffusion.py:618`） | w>0 时消元：`r = min(Jz+b,0)/w`，避免 `-w·I` 病态块（`solvers.py:73-84`） | 数学等价，数值路径不同 | 保留并说明 |
| C5 | TVS 时变水平 | `b = ellipse − σ(t_bias−t) − 0.01`，`t_bias=5`，`Lfb = σ(1−σ)`（`diffusion.py:911-921`） | `tvs_schedule`：`σ(j)=min(h0,0)·j/N`，`σ̇=−min(h0,0)/(N·dt)`，`offset=−σ̇+ε(h−σ)`（`runner.py:32-35, 56-65`） | **同向不同形**：都是"高噪声松、末端紧"；官方的 σ 是归一化椭圆半径上的 sigmoid（末端多出 ≈1 个半径的余量），我们是场景单位初始违反量的线性退火（末端余量 0）；官方把 σ̇ 记为 `Lfb` 进 h，我们单独进 offset（等价项） | **待定**：若要严格对齐，改成 sigmoid 形式并换算到场景单位；若保留，必须在表里写真"σ(j) 曲线为自定，官方为 sigmoid(t_bias−t)" |
| C6 | 每级 QP 次数 | 1 次/级（`p_sample` 内一次），1000 级 | `_integrate_level` 在一个 `dt_full=1` 窗口内**多次**：`step=min(剩余, dt_level, cbf_max_point_step/speed)`（`runner.py:85-90`） | 官方无子步、无步速上限 | **保留并说明**，但必须删掉**虚假配置**：`cbf_substeps=16` / `cbf_min_step` / `cbf_max_halvings` / `cbf_max_rejects`（`configs/…json:20,23-25`）**在 runner 里从未被读取**（grep 仅 audit 脚本用过）→ 会误导复现者 |
| C7 | 离散证书 & 重试 | 无（官方本来就只保证一步线性化） | `cbf_discrete_check=true`：每级末查 `h(x₁)−γ >= e^{−εdt}(h(x₀)−γ)`，不过则**把该级 dt 减半重跑**（`runner.py:105-123`；`discrete_residual` 43-47） | 纯属我们新增的"保证交付不穿透"机制 | **保留并说明**（口径红线 §8.3 已要求；但要在表里写"官方无此机制"） |
| C8 | 障碍表示 | 2 个闭式椭圆（`diffusion.py:459-490`） | 顶点版：ExactMap 距离场 `exact_map`；段版：362 圆并 | 官方是玩具规模 | 保留并说明 |
| C9 | 发布码的坑 | 官方 `p_sample` 里 SafeDiffuser 调用**全部注释**，默认 `x = xp1`（`diffusion.py:1062-1069`） | 我们默认开启 | 复现时必须说明"官方需手工 uncomment 指定变体" | 保留并说明（写进复现章节） |

### D. Constrained Diffusers（Projected / Primal-Dual / ALM）

| # | 项目 | 官方 | 我们 | 差异 | 结论 |
|---|---|---|---|---|---|
| D1 | 约束符号 | `g = dist−1`，**g≥0 安全**（`diffusion.py:72-83`） | `g ≤ 0` 安全（`native_points.py:60`、`paper_segment_constraints.py:113`） | 互为取反；乘子符号同步取反后**更新式等价** | 保留并说明（换算已在 §2 里验证） |
| D2 | hinge | **无 hinge**：直接用原始 `g` 和 `∇g`（`diffusion.py:316-322, 337-340`） | `J=J*(g>0)`、`g=max(g,0)`（`runner.py:69`），依据 paper Remark 1 | 可行集相同，**梯度/乘子轨迹不同** | **待定**：建议做一次单变量对照（hinge on/off），否则不能说"按 Remark 1"；当前状态写"我们按 Remark 1 加了 hinge，官方码没有" |
| D3 | 乘子结构 | `(C,B,H)`，**每个时间点一个乘子**（`diffusion.py:95-99`） | 每**行**一个乘子（`lam=np.zeros(len(task.constraints(x[0])[0]))`，`runner.py:46`） | 官方行数 = 障碍数 × 时间点（隐式稠密但在每点 2−4 维），我们行数 = 段×圆 | 保留并说明（这是 A3/A4 的直接后果） |
| D4 | PD 更新 | `posterior_mean += Σλ∇g`；`λ←clamp(λ−lr·g, min=0)`，`lr=1e-3`（umaze-large）/`2.5e-2`（umaze-small）/`1e-1`（其余）（`188-202`、`325-348`） | `x += β/2(score−Σλ∇g)+√β z`；`λ←max(0, λ+η·mean_particles(g))`（`kernels.py:13-18`） | 符号等价；**两处真实差异**：(i) 我们对 particles 求均值（官方无 particle 概念）；(ii) 我们的修正乘 `β/2`，官方直接加到后验均值上 | **按官方改（机制层）**：`λ` 的单位换算已核对——`dual_eta=6.1035e-08`、`dual_constraint_scale=128` ⇒ 等效 `η_official = 6.1035e-8 × 128² = 1.0e-3`，**正好是 umaze-large 的 lr**。这一条应写进表作为"已对齐"。particles 均值保留并说明 |
| D5 | ALM | `s←clamp(λ/ρ+g,0)`；`λ←λ+ρ(g−s)`；`ρ←ρ·1.004` **仅在 cbf=True 分支**（`350-380`） | `slack=max(0,−mean−λ/ρ)`；`λ+=ρ·residual`；`ρ*=rho_growth=1.01`（`kernels.py:21-28`） | 代数等价（已逐项验证）；差异：**ρ 增长 1.01 vs 1.004**，且官方默认分支根本不涨 ρ 而是每步把 `penalty` 重置为 `2.5e-2`（`472`）；我们 `rho=1.525879e-08 ×128² = 2.5e-4`（= 官方 `__init__` 的 `2.5e-4`，`62`），**不是** `p_sample` 里的 `2.5e-2` | **按官方改（建议）**：把 `rho_growth` 对齐到 1.004 或（更好）用官方"每级重置 ρ"的做法；当前 1.01 × 320 级 ⇒ ρ 放大 24 倍，官方是常数。**这一条要单变量实验** |
| D6 | Projected 算子 | 闭式径向投影（`382-466`），只对 `g<0` 的点动 | SLSQP 局部最近点（`solvers.py:31-69`），带 KKT/Clarke 校验 | 算子不同（见 A6） | 待定（见 A6） |
| D7 | 级阶梯 | `range(-200, T)`，`i<=0 → i_=1` ⇒ **t=1 重复 200 次**（`503-510`） | `level_ladder`：`linspace(T-1,0,visits)` 后追加 `[0]*cd_extra_steps`，`cd_extra_steps=64`（`runner.py:22-24`；`configs/…json:36`） | 重复的**级号不同**（官方 t=1，我们 t=0），次数不同（200 vs 64）。`β₀=1.889e-4`、`β₁=2.630e-4`（实测，clip 下限 1e-4），单步量级相近，差异主要在 200 vs 64 | **按官方改**：把额外步放在 `t=1` 且默认 200（或至少在表里显式声明 64@t=0）。另：官方 `nonzero_mask` 用 `t==1`，我们 `effective_beta` 也会因此在 t=1 正常出噪声 ✅ |
| D8 | β 的处理 | 每级用该级 `β_t`（`q_posterior` 直接取 `self.betas`）；t=1 重复 200 次时也用 `β_1` | `cd_beta="level"` → `task.betas[t]`；`cd_levels=256`=全阶梯，未子采样（`runner.py:26-32,47-53`） | 一致（我们还有 `interval` 兜底模式并记录 `ladder_subsampled`） | 保留（已对齐，比官方更显式） |
| D9 | 噪声级门控 | 无（每步都施加） | 无（`cd_run` 每级都调用投影/更新） | 一致 | 保留 |
| D10 | 官方发布码的坑 | umaze-large `algorithm[4]` → IndexError；三方法必须手工选（`65-67`） | 我们的 `METHODS` 显式三行（`algorithms/runner.py:9`） | 无实质差异 | 保留并说明（复现章节写明我们选了哪个 `algorithm[i]` 语义） |

### E. 训练/日程/预算/报告

| # | 项目 | 官方 | 我们 | 差异 | 结论 |
|---|---|---|---|---|---|
| E1 | 基座训练目标 | PDM/SD/CD 都是 score matching（ε-prediction） | `models/safety_point_losses.py`：x0 MSE + 9 项辅助（`docs/pdm_fidelity_audit.md` §C5.1） | 目标不同；score 由 `x̂0` 换算 | 保留并说明（已在 §C5 记录：不能声称"原论文结果"） |
| E2 | 日程 | PDM：10 级 log 均匀、σ∈[0.005,10]；SD：1000 级 cosine；CD：1000 级 cosine + 200 次 t=1 | VP T=256，σ_max≈5194.6 scene、σ_min≈0.0137 scene（`configs/…json:119-127`） | 日程完全不同；官方是 O(1) 数据尺度 | 保留并说明（这就是 `official_ald` 换算存在的原因） |
| E3 | NFE | PDM 5×100+4×10 ≈ 540；SD 1000；CD 1000+200 | `pdm_levels×pdm_inner=200`；SD ~256×(子步数)；CD 256+64（`algorithms/runner.py:20-31` 已计真实 forward） | 数量级可比 | 保留（我们已计真实 NFE，官方未报） |
| E4 | 数值口径 | 官方 `tol=1e-3`（PDM IPOPT） | `projection_tol_scene=1e-7`、`segment_projection_tol=1e-6`（`configs/…json:10,52`） | 我们严 3–4 个数量级 | 保留并说明（更严只会更容易失败，不会"刷分"） |
| E5 | 报告的"方法名" | — | 七行表 | 段版本必须单列（红线 §8.2） | 保留 |

---

## 3. 直接回答 §6.1 的五个假设

| 假设（HANDOFF §6.1 原文） | 官方代码的回答 | 判定 |
|---|---|---|
| 段约束应约简为"每段一行、取最紧的圆"（127 行） | **官方没有这种约简。** PDM 是"每段×每圆一行"全保留（`solver.py:143-172`），因为它只有 16–24 个圆。官方不约简，是因为**不需要**约简 | **假设不成立**（至少不是官方做法）。真正的差异是圆数：16–24 vs 362。可选：粗化圆并（A4-a）或用工作集（A4-b），二者都必须标为 task-adapted |
| SafeDiffuser 的 CBF 每步该解多大的 QP | **2 变量、障碍数行（2 行）、`Q=I₂`**，在 horizon 上 batch；ReS 放松时 4 变量、4 行（`diffusion.py:501-506, 649-656`） | **官方 QP 极小**。我们的 46k 行耦合 QP 没有任何官方对应物 → 三者回到点行，或段版本另立条目 |
| PD/ALM 需要稠密 J，所以只能承载少量行 | **官方根本不建 J**：`autograd.grad(g.sum(), state)` 得到与 x 同形的逐元梯度，更新只用到 `Σλ∇g`（`diffusion.py:306-322, 188-224`） | **前提错误**。我们只需 `Jᵀλ` 与 `g` 的加权和 ⇒ 稀疏 J 或逐行回代即可，`MemoryError` 是**可以按官方结构消掉的**（A5） |
| 高噪声级不该做投影（`projection_gate_residual`） | **有官方先例**，但机制是"按噪声级下标"：`idx>5` 才 `traj_p=True`（`scoremodel.py:571,581-594`）；PDM App.F 的"调整投影频率"在本仓库是靠这个实现 | **方向成立，判据不同**（级号 vs 残差）。建议改成 σ/级号阈值，残差门降为诊断 |
| `lambda_smooth` 0.08 需加大 | 与三份官方实现无关（属基座标定） | **保持"待你决定"**，不动 |

---

## 4. 结论汇总（不改代码，等确认）

### 4.1 已经"按官方核对通过"的条目（不要动）

- PDM 步长 `γ = eps·σ²/(2σ_min²)` ⇔ 官方 `δ = eps·(σ/σ_min)²`，`x += (δ/2)s + √δ z`（B1）。
- ReS 的 slack 权重 100 与"t≥10 放松 / t<10 硬"的分段（C3）。
- 端点硬钳制（A2）。
- `λ` 单位换算：`dual_constraint_scale=128` + `dual_eta=6.1e-8` ⇒ 官方 `lr=1e-3`；`rho=1.5e-8` ⇒ 官方 `2.5e-4`（D4）。
- CD 的级阶梯"全 256 级、不做子采样"（D8）。

### 4.2 建议按官方改（按优先级；每项都要单独跑可复现的单变量实验）

1. **A5 — 去掉稠密 J（必做，修 `MemoryError`）**：PD/ALM 只用 `Σλ_k J_k` 与 `Σλ_k g_k`；段版 `CircleSegmentConstraints.constraints` 已经返回 `csr_matrix`，改成对稀疏行做加权和，内存 O(rows) → O(nnz)。
2. **C2 — RoS/ReS/TVS 回到点行**：官方三者都是逐点约束（`diffusion.py:466`），且 QP 可分离；段版本如果保留，必须单列并说明"需要工作集 QP，官方无对应"。
3. **D5 — ρ 对齐**：`rho_growth 1.01 → 1.004`（或改成官方"每级重置 penalty=2.5e-2"）。
4. **D7 — 额外步放回 t=1**：`cd_extra_steps` 的重复级号 `0 → 1`，并考虑 200（默认可先保持 64，但必须显式声明）。
5. **B4 — 门控判据改成级号/σ 阈值**（保留残差门作诊断）。
6. **C6 — 删掉/接上无效配置键**：`cbf_substeps`、`cbf_min_step`、`cbf_max_halvings`、`cbf_max_rejects`。

### 4.3 保留并说明（写进差异表，不改行为）

- 原生点状态替换 VE/B 样条（A1）、段约束作为 task-adapted 变体（A3）、连续时间 CBF 积分（A7、C6）、
  离散证书 + dt 减半重试（C7）、PDM 的 warm start 与"失败即失败"（B5、B6）、x0-MSE 基座（E1）、更严容差（E4）。

### 4.4 待定（需要你拍板）

- **A4/B8**：圆并粗化到 ~16–32 个/场景（牺牲保守性换行数），还是保留 362 圆改工作集？
- **A6/D6**：`Projected` 是否要在顶点行上改成 CD 的闭式径向投影（只有"每点一个椭圆"能闭式，圆并做不到）。
- **C5**：TVS 的 σ(j) 是否改成官方 sigmoid 形式。
- **D2**：hinge 是否保留（需一次 on/off 对照实验才能下结论）。
- **§2 的 `lambda_smooth` 0.08 → 0.5 重训**：与本文件无关，仍是你的决定项。

---

## 5. 换装记录：修正代码已全部换成官方（2026-11-14）

> 决定：**修正这一块全部用官方原文，场景和模型用我们的。** 本节记录实际做法、两处必须声明的
> 替换、以及换完后的第一次实跑。

### 5.1 新增（官方原文的落点）

| 文件 | 内容 | 来源 |
|---|---|---|
| `algorithms/official/safediffuser.py` | RoS / ReS / TVS 的 QP 与 slack / level 规则 | `diffusion.py:451-512, 588-658, 895-956` |
| `algorithms/official/constrained_diffusers.py` | Primal-Dual / ALM 的乘子步与投影 | `umaze-large/diffusion.py:188-224, 306-380, 382-466` |
| `algorithms/official/posterior.py` | 后验系数与官方级阶梯（含 t=1 重访） | `diffusion.py:117-131, 503-510` |
| `algorithms/official/pdm.py` | 步长、Langevin、`idx>5` 门控、失败补 15 步重投 | `mpd/.../scoremodel.py:509-531, 557-627` |

### 5.2 删除（我们自己的实现，已全部不再使用）

- SafeDiffuser：连续时间速度场积分、`cbf_dt` 子步、离散证书、dt 减半重试、点速度上限、`legal_gamma`、高噪声残差门控；
- Constrained Diffusers：`cd_beta` 的 level / interval 两种 β、对粒子求违反量均值、hinge `J*(g>0)`、附加步放在 t=0；
- PDM：warm start（带上一个可行点）、`allow_partial`、`projection_gate_residual`；
- `algorithms/kernels.py` 的 `primal_dual_step` / `alm_step` / `cbf_qp` / `pdm` / `projected_step`（只剩官方更新要用的 `row_contribution`）；
- 配置里 14 个死键（`cbf_substeps` / `cbf_dt` / `cbf_gamma_floor` / `cbf_discrete_check` / `hard_steps` / `cd_beta` / …）；
- 两个审计脚本 `audit_discrete_barrier.py`、`audit_safediffuser_dt.py`（量的是已删除的代码）。

### 5.3 两处必须声明的替换（论文里没有）

| 项 | 官方 | 我们 | 为什么必须换 |
|---|---|---|---|
| 障碍几何 | 写死 maze2d 的 2 个椭圆 / 16 圆 | `task.point_barrier` / `task.constraints` 读我们的占用图 | 场景是我们的 |
| QP 后端 | `qpth.QPFunction(PDIPM_BATCHED)` | 同一问题陈述的闭式解（单位阵 Q、每点 1 行、至多 1 个 slack 列），已用 SLSQP 逐点比对 | qpth 不是本仓依赖 |

### 5.4 两处官方有原文、但默认值需要你确认

1. **`sd_apply_below`（默认 3，T=256 的最后 3 级）**
   官方 `p_sample` 同时存在"每级都施加"与"只在 `t <= 10` 施加"两套（`diffusion.py:1071-1078`，后者是注释状态）。
   用前者时 σ_max≈5195 scene，RoS/TVS 在 t=255 被硬投影到几何半空间 → 样本爆掉（实测 len_ratio 19–27、
   rmse 26–34 m）。换成后者即可用。**这条等价于 PDM 的 `idx>5`，也是假设 4（"高噪声不该投影"）的官方出处。**
2. **`sd_tvs_level_scene`（默认 0.125 = 一个栅格）**
   官方 TVS 的 σ∈[0,1] 是归一化量（1 = 障碍半径），我们的 b 是米。不换算时 TVS 在 4 m 地图上要求
   0.9 m 额外余量（实测 −15.7 / −40.7 m）；换算成"一个栅格"后 5.08 / 1.51 m，两个场景都成功。

### 5.5 换完后的第一次实跑（61001 / 74014，seed 0，index 0 / 163）

| 方法 | 61001 碰撞 / 最小净距 / RMSE | 74014 碰撞 / 最小净距 / RMSE |
|---|---|---|
| RoS | 否 / **+1.65 m** / 8.64 | **是** / −2.06 m / 8.22 |
| ReS | 否 / **+1.40 m** / 9.00 | **是** / −1.19 m / 8.15 |
| TVS | 否 / **+5.08 m** / 8.04 | 否 / **+1.51 m** / 7.38 |
| Projected | 否 / +1.97 m / 8.97 | 是 / −0.65 m / 8.46 |
| Primal-Dual | 否 / +4.00 m / 7.69 | 是 / −1.96 m / 7.57 |
| ALM | 否 / +2.12 m / 8.86 | 是 / −2.14 m / 8.21 |
| **PDM（段约束）** | 否 / +0.72 m / 8.71 | 否 / **+0.36 m** / 8.11 |

读法：**PDM 是唯一两个场景都干净的，因为它是唯一用 App.C.3 段约束的**；其余六种是论文原样的逐点
约束，74014 上的 −0.65…−2.14 m 穿透正是 §A3 预言的"顶点安全、线段切角"。这是原方法的已知局限，
应作为结论报告（红线 §8.2 要求单列），不是实现错误。

### 5.6 尚未做

- 20 个冻结 val 场景的全表（上面只有 2 个场景，不能当结论）；
- `sd_apply_below` / `sd_tvs_level_scene` 的敏感性各需一次单变量实验；
- `lambda_smooth 0.08→0.5` 重训（与本文件无关，仍是你的决定项）。

---

## 6. 参考：官方文件行号速查

```
.refs/Projected-Diffusion/examples/trajectories/mpd/models/diffusion_models/
  scoremodel.py:509-510   step_size = eps*(process/process[-1])**2
  scoremodel.py:557-627   pgd_diffusion_step（571: idx>5→annealed_step=10；573-579: 每内步投影；581-594: traj_p+solver+15步重试）
  projection.py:49-76     apply()（traj_p 才进 solver）；86-90 project_hard_conds
  solver.py:14-78         CasADi/IPOPT 最近点 NLP；29-62 变量/objective；70-76 失败→violation
  solver.py:120-136       point_v=False（点行是死代码）
  solver.py:140-172       段行：每段×每圆一行；243-250 line_circle_intersection(t∈[0,1])
projected_diffusion/scoremodel.py:540-566   规范化 PGD：Langevin 后立即投影
examples/trajectories/mpd/.../train_score_model.py:20-34  eps=1.5e-5, σ∈[0.005,10], n_steps=10, annealed=100

.refs/SafeDiffuser/diffuser/models/diffusion.py
  451-512  invariance（RoS maze2d-large）：466 b=ellipse-1-0.01；471-474 G=-∇b,k=1,h=b；501-506 Q=I₂
  588-658  invariance_relax（ReS）：610-616 sign=100/0；649-656 Q=I₄ + out[:,0:2]
  895-956  invariance_time（TVS）：896 t_bias=5；911-921 σ 进 b、Lfb=σ(1-σ)
  1030-1097 p_sample：1062 默认 x=xp1；1063-1069 官方把变体全注释
  1100-1131 p_sample_loop（1000 级）

.refs/NeurIPS-2025-…/code/umaze-large/diffuser/models/diffusion.py
  65-69    algorithm 列表 + algorithm[4]（越界 bug）
  72-87    g_x1 = 归一化距离 - 1（逐时间点）
  95-99    dual_vars (C,B,H)
  188-230  q_posterior 里 PD/ALM 的后验均值修正
  306-322  calc_grad（autograd 逐元梯度，无 J）
  325-348  dual_update（lr=1e-3）
  350-380  dual_update_aug（slack + ρ*1.004）
  382-466  _project_to_feasible_region（逐点闭式径向投影）
  468-487  p_sample（481-482 Projected；472 penalty=2.5e-2）
  503-510  range(-200,T) + i<=0→i_=1（t=1 重复 200 步）
```
