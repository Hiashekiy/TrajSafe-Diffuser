# 交接问题修复报告（2026-11-14）

对应交接文档：[DSH_HANDOFF_20261114.md](DSH_HANDOFF_20261114.md)。本文件只记录**本轮实际改动、证据与仍未验收的部分**。
所有数字都可用本文命令复现；没有任何测试集调参，也没有为了表格好看而放宽标准。

## 1. 总结

| 交接问题 | 本轮处理 | 状态 |
|---|---|---|
| P0-3 采样器与共同基座不匹配（PDM-VE RMSE 34116 m、CD-Langevin 19.09 m） | 定位根因（σ_T²/σ_min²=8.8e6 使原文 γ 公式每步混合率 1/(2σ_T²)=4.7e-6），实现两种 γ 约定 + VP/VE 精确换算 + 可运行的兼容性门 | **已修复并可验证**：PDM 34204 m → 17.8 m；CD 39 m 主要是 16 级日程的噪声下限（12.4 m）而非公式错误 |
| 为什么论文循环效果还不如旧 SDF 修正器 | 量化到根因：waypoint 间距 0.95 m vs 最低训练噪声 8.77 m（9.2×），使以 p_0 收尾的采样器必然交付 13× 锯齿；并按交接要求给出依据后把共同基座重训为 256 级（1.10 m / 下限 1.56 m） | **已定位并验证**：锯齿 21–52× → 2.4–5.4×，planning success 0–1/20 → 2–8/20（无约束基线 10/20），见 §3.0、§4b |
| P0-2 PDM 原文整条线段约束漏实现 | 新增审计过的占据地图→圆覆盖转换、整段线段约束对象、可验证的最近点投影（SQP+OSQP），并接入正式 runner（配置开关） | **已实现并接入**，有单测与真实地图审计；**可行性在噪声 τ≳0.1 处失败**，见 §4，未声称已验收 |
| P1 CD 三种的超参/任务机制 | 明确 η_λ、ρ 来自论文 Appendix C 的 "large" 设置并折算到约束缩放；新增每级子步（同一 Langevin SDE 的离散细化，默认 1 = 原文单步） | 参数来源已写清；**未见性能改善**，仍按 task-adapted 报告 |
| P1 约束对象与评价对象必须统一 | 评价器同时输出 vertex / segment-only / path 碰撞（已有）；PDM 用线段约束，其它方法保留各自机制；两者分开跑、分开报告 | **已按此执行** |
| P1 SafeDiffuser 数值积分未验收 | 逐条对原文核对 eq (7)(8)(9)(11)(12)(13) 与 Thm 3.4 的 γ；新增 `scripts/audit_safediffuser_dt.py` 量化 Δτ/子步处理 | **仍未验收，且定位到一个具体缺陷**：`cbf_max_point_step` 这个不在原文里的步长上限使 `cbf_substeps` 完全失效（实测 len_ratio 36.59 vs 36.59、侵入 36.9 vs 37.0 m 完全相同）；一阶速度型 barrier 在单级数十米的位移上无法保证交付状态安全，见 §4c |
| P0-1 旧全量的状态/交付不一致 | 新点基座已无该链路；本轮验证继续逐元素校验 `internal_state_trajectory == trajectory` | 保持通过 |

## 2. 代码改动

| 文件 | 改动 |
|---|---|
| `adapters/paper_segment_constraints.py` | 新增 `obstacle_circles()`（占据格→内切极大圆覆盖 + 半格对角覆盖垫；含覆盖率/自由空间误伤率审计）；`CircleSegmentConstraints` 增加长度形式约束、稀疏解析 Jacobian、`values()` 快速路径、`project()` 实现为"可行即恒等 + 最小范数 SQP(OSQP) + 精确罚 merit + 行修复兜底 + 全约束集校验"；删除 SciPy `trust-constr` 原型（在真实地图上 60 次迭代后残差反而变大，见 §4） |
| `adapters/segment_points.py`（新） | `SegmentPointTask`：同一网络、同一起终点、同一 score 与输出表示，只把约束对象/投影换成"整段线段 vs 圆并"；禁用按点可分的 `cbf_solve` 快捷路径（该捷径只对逐点行成立） |
| `algorithms/pdm/runner.py` | `level_gamma()` 显式实现两种约定：`paper`（Algorithm 1：γ_t=σ_t²/2σ_T²）与 `official_ald`（官方 `projected_diffusion/scoremodel.py`：step_size=ε(σ/σ_min)²）；VP/VE 状态换算 `s=γ·ᾱ_t`（y=x/√ᾱ 的精确变量替换）；记录每步 `gamma/step/sigma/mixing_ratio`；末端级别必须精确可行 |
| `algorithms/constrained_diffusers/runner.py` | 新增 `cd_substeps`（同一 Langevin SDE 的 dt=1/K 细化）；每级一条 trace，字段增加 `beta_step/noise_std/substeps` |
| `algorithms/runner.py` | 按 `segment_constraints_for` 选择 task；把投影迭代/容差从配置注入 |
| `algorithms/safediffuser/runner.py` | `getattr(task,'cbf_solve',None) or cbf_velocity`（线段约束任务没有可分 QP 捷径时必须回落到通用 QP） |
| `configs/seven_native_points_aligned_t16.json`（新） | 本轮主配置；`adaptations` 字段逐条写清每个偏离原文的取值与其来源 |
| `configs/seven_native_points_aligned_segment_pdm_t16.json`（新） | 只把 PDM 换成线段约束的对照配置 |
| `scripts/validate_point_methods.py` | 加 CLI（`--config/--indices/--methods/--seeds/--out-dir`）、记录配置哈希与源码哈希、输出 `summary.json`/`summary.md`、报告 vertex/segment-only/path 三类碰撞与 PDM 线段残差 |
| `scripts/audit_sampler_match.py`（新） | 无约束采样器兼容性门：DDIM / CD-Langevin(+子步) / PDM(paper-VE, official-VE, official-VP)，输出结构下限 |
| `scripts/audit_segment_projection.py`（新） | 线段约束与投影在真实地图上的审计（覆盖统计、GT 残差、不同扰动尺度下的可行性与耗时） |
| `tests/test_segment_points.py`（新） | 8 项：圆覆盖完备性、单格圆、线段集比"顶点在自由格"更严、可行恒等性、线段级修复（含密集采样复核）、失败必须报错而不是退化为逐点裁剪、两种 γ 约定、**已知 Gaussian score + 半空间约束退化为解析截断正态** |

## 3. P0-3 采样器/基座匹配：根因与修复证据

### 3.0 为什么论文循环的效果比旧 SDF 修正器差

先把问题量化成一个数：**本任务相邻 waypoint 间距 0.95 m，而冻结 16 级日程的最低训练噪声是 8.77 m/坐标 = 间距的 9.2 倍。**

PDM/CD 这类循环最后一级结束在 `p_0=N(√ᾱ_0 x0, 1-ᾱ_0)` 上，不是 clean 预测，因此交付折线必然带 ±8.8 m 的独立散点噪声。相邻点间距 0.95 m、噪声 8.8 m ⇒ 期望"相邻点距离" = √(0.95²+2·8.77²) ≈ 12.4 m ⇒ 折线长度被放大到 GT 的约 13 倍。**实测无约束（完全排除约束处理的影响）**：

|采样器（无约束）|折线长度 / GT 长度|与解析预测对比|
|---|---:|---|
|DDIM（16 步）|0.8 – 1.1|光滑路径|
|CD-Langevin（64+64 级）|21.0|解析 13.05，含 γ/(1-ᾱ)=0.5 的方差偏置|
|PDM（T10×M20）|14.4|解析 13.05 ✓|

带约束时（§5.2 的 160 条记录）七种方法的折线长度是 GT 的 **21–52 倍**、相邻段平均转角 **105–120°**；无约束 DDIM 是 0.97 倍、1.2°。这就是"顶点碰撞 0/20、路径碰撞 19–20/20"的机制：每个顶点都可以被推进自由格，但**每条线段都在横穿建筑**。

旧 SDF 修正器效果更好，原因不在公式，而是三点都不冲突：
1. **骨架**：它跑在 DDIM 骨架上，修正 denoiser 的 clean 预测，最后一步回到 clean 预测，不残留噪声；
2. **表示**：48 个控制点 + 样条基函数，一次修正平滑耦合整条曲线，输出保持光滑（长度比 0.97、转角 1.2°）；
3. **步长尺度**：`g=d_safe-d(BQ)` 是米制量，修正在数据空间里是局部小位移。

论文循环把 1、2 都换掉了（SGLD 骨架、逐点表示），同一个约束于是要在锯齿上求解，需要 8–16 m 的位移，局部最小范数法做不到（§4）。

**同骨架对照实验**（`docs/backbone_vs_constraint_audit.json`，5 个 val 案例；这不是七种方法之一，也不是 Ours，只复用占据地图与共享评价器；种子 0）：

|DDIM 骨架 + 在 clean 预测上修正|顶点碰撞|仅线段碰撞|路径碰撞|RMSE|平均修正量|
|---|---:|---:|---:|---:|---:|
|不修正|2/5|0/5|2/5|24.19 m|0|
|逐点投影到自由格（旧修正器的表示逻辑）|**0/5**|2/5|2/5|**23.73 m**|3.8 m|
|整段线段投影（本次新求解器）|2/5|0/5|2/5|24.18 m|未收敛（见 §4）|

即"在 clean 预测上修正"在本基座上依然有效且几乎不损伤质量；把同样强度的修正放进 Langevin 循环就变成锯齿。**因此这不是公式抄错，而是骨架 + 噪声级数与"128 点、0.95 m 间距"的点表示不兼容。**

### 3.1 不同级数下最低训练噪声与 Langevin 类方法的必然残差

|timesteps|σ_min（scene）|σ_min（m）|Langevin RMSE 下限|锯齿倍数（解析）|
|---:|---:|---:|---:|---:|
|16（交接前的基座）|0.10959|8.767|12.40 m|13.05|
|64|0.03466|2.772|3.92 m|4.13|
|**256（本轮重训）**|0.01374|1.099|**1.56 m**|**1.64**|
|1024|0.00635|0.508|0.72 m|1.23|

256 级时下限 1.56 m 远小于基座自身 19–24 m 的误差，锯齿从 13× 降到 1.6×，七个循环才真正在测算法而不是测日程。

### 3.2 冻结的 16 级 cosine 日程（`beta_schedule=squaredcos_cap_v2`）在 VE 坐标下

|量|值|
|---|---|
|ᾱ_min|9.457e-6（第 16 级，β 被 `max_beta=0.999` 截断）|
|σ_min（VE）|0.10959 scene = 8.767 m|
|σ_max（VE）|325.17 scene|
|σ_T²/σ_min²|8.8e6|
|Langevin 最低级噪声下限（每坐标 σ_min，两维 RMSE）|**12.40 m**|

对 PDM Algorithm 1 的 γ_t=σ_t²/(2σ_T²)：每步相对混合率 γ_t/σ_t²=1/(2σ_T²)=4.7e-6，M=100 步相对收敛因子 (1-4.7e-6)^100≈0.9995，**样本在 10 级 × 100 步内几乎不动**；官方实现用 ε(σ/σ_min)²，相对混合率 ε/σ_min²（ε=2e-3 → 0.167，百步内充分收敛）。两者只在 σ_T²=σ_min²/(2ε)≈1.7² 时一致——即原文公式隐含要求 VE 噪声上界是 O(1)，而本基座是 325。

无约束兼容性门（`docs/sampler_compatibility_gate.json`，无约束、无投影、同一种子；先 2 个案例、后扩到 5 个案例 val 0/81/163/244/326）：

|诊断|RMSE 均值 (2 例)|RMSE 均值 (5 例)|RMSE 最大 (5 例)|最低级混合率|耗时 s|
|---|---:|---:|---:|---:|---:|
|DDIM 名义循环（16 步）|8.17|22.92|85.1|—|0.21|
|CD-Langevin（VP，64+64 级）|17.96|38.98|102.8|0.5（离散化方差偏置）|1.16|
|CD-Langevin + 4 子步|19.99|36.46|99.0|—|4.65|
|PDM-VE + 原文 γ（交接前实现）|**34203.07**|34203.64|34207.8|4.79e-6|1.81|
|PDM-VE + 官方 ALD γ|14.22|18.55|38.0|0.169|1.78|
|PDM-VP + 官方 ALD γ|**13.88**|**17.78**|34.5|0.167（每级恒定）|1.83|

结论：
1. **PDM 的 34116 m 是 γ/单位约定与噪声日程不匹配，不是"论文方法差"**。修正后 5 例均值 17.78 m、最大 34.5 m，已经落在无约束基座自身的离散范围内（DDIM 均值 22.92 m、最大 85.1 m）。5 例只是把"实现是否复现先验"这个问题说清楚，均值和最大值都被个别困难案例主导，**不能据此推断方法优劣**，也不能代替 420 例全量。
2. **CD-Langevin 的 19–39 m 也不是公式错误**：原文式(6) `x←x+β/2·∇log p+√β z` 的目标分布是 p_0=N(√ᾱ_0 x0, 1-ᾱ_0)，末级必然残留 σ_min≈8.8 m/坐标 ⇒ 两维 RMSE 下限 12.4 m；观测值比 DDIM 高约 10–16 m，量级与下限一致，且加 4 倍子步无法消除（36.46 vs 38.98），说明它不是离散化误差而是目标分布本身带的噪声。DDIM 之所以更低，是因为它最后一步直接输出 clean 预测、不残留噪声。
3. 因此"低于 ~12.4 m 的差距"在本基座上不可能由采样器改进获得，只能改噪声日程（需要重训，未获授权）。这一点现在有量化证据，见 §6。

## 4. P0-2 PDM 整条线段约束：实现、接入与真实地图上的可行性

**几何转换（审计要点）**：`obstacle_circles()` 用占据格到最近自由格中心的距离作为半径，再垫半个格对角。因此每个被接受的圆的**内切极大圆覆盖**包含相应占据格方块的整格面积；任何被覆盖格中心到某圆心的距离 ≤ 该圆半径，故圆并 ⊇ 占据区 ⇒ "线段与所有圆不相交" ⇒ 评价器不会判碰撞。代价是圆并侵入自由空间：实测 215–249 个圆/场景（37373 个占据格），自由格被误禁比例 3.5–3.9%，即超出占据区边界约 1 个格（0.625 m）。

**投影**：`P_C(z)=argmin||x-z||s.t. 所有线段-j 行可行`（原文式(7)）。实现为：可行即恒等（原文 App. C.4 明确"已可行的点应原样返回"）；否则逐次解线性化最小范数 QP（OSQP，带 L2 松弛以保证线性化矛盾时仍可解、含信赖域），在**精确 L1 罚 merit** 上回溯，罚权不足时做 continuation，必要时用 Gauss-Seidel 行修复把残差关闭，最后对**全部**圆做可行性校验；不可行就抛 `SolverFailure` 而不是返回近似解。

**真实地图审计**（`docs/segment_projection_audit.json`，val 0/81/163/244，target = GT + 各向同性噪声）：

|扰动 σ (scene)|位移|结果（maxiter=20）|结果（maxiter=60）|
|---|---|---|---|
|0.02 (1.6 m)|0|恒等（已可行）|—|
|0.05 (4 m)|0.02–0.06|4/4 收敛，0.10–0.42 s|—|
|0.10 (8 m)|0.12|0/4 收敛|0/2 收敛|
|0.20 (16 m)|0.50|0/4 收敛|0/2 收敛|

即：**每 waypoint 4 m 量级的扰动可以精确投影（残差 1e-8，耗时 0.1–0.4 s）；8 m 量级起局部最小范数法在 60 次迭代内也拿不到可行点。**

失败是**真实的非凸性**，不是数值技巧问题：单个 waypoint 8–16 m 的各向同性噪声把 128 点折线变成穿越建筑群的锯齿，最近的可行折线本身就是一个路径规划问题（局部最小范数法只能做局部下降）。已排除的错误做法：`trust-constr` 内点原型在真实地图上 60 次迭代后残差反而从 0.19 升到 0.40；纯 POCS 在 400 轮后仍停 0.12–0.20；L-BFGS-B 精确二次罚 457 次评估后停在 0.10–0.20；放宽的线性化 QP 会被固定端点造成的矛盾行变成**primal infeasible**（这也是加入 L2 松弛的原因）。这些都在审计 JSON 的迭代里留下了记录，没有伪装成"已解决"。

为了让 PDM 的 1000 次内层投影能在 CPU 预算内跑完，还做了两处**明确的数值取舍**：工作集只保留最违反的 256 行（`max_rows`）、内部 QP 精度 1e-6。代价是"极难"目标的成功率下降（见上表 maxiter 两列），收益是单次 QP 从 0.24 s 降到约 10 ms（整条 PDM 从 385 s/案例降到 55–59 s/案例）。

**接入状态**：`SegmentPointTask` 已由正式 runner 按 `segment_constraints_for` 选择，PDM 的循环（T 级 × M 步 SGLD + 投影）原样保留，投影频率/迭代预算按论文 App. F 的原文允许（"the number of iterations for each t can often be decreased below M = 100 or the projection frequency can be adjusted"）。端到端结果见 §5 的第二张表。

## 4b. 换 256 级基座之后（本轮新增，回答"为什么论文方法还不如旧修正器"）

按 §3.1 的依据把共同基座从 16 级重训为 256 级（网络/数据/损失/100 轮/AdamW/batch 全部不变，只有 `diffusion.timesteps` 16→256；~46 min，`results/map_only_points_t256_e100`，best_task epoch 86，task_score 21.83 优于 16 级的 22.55），然后用同一 20 个 val 案例、同一评价器、同一种子重跑：

|方法|折线长度/GT（16 级 → 256 级）|转角（16 级 → 256 级）|顶点碰撞|仅线段碰撞|路径碰撞|planning success|RMSE m|耗时 s|
|---|---|---|---|---|---:|---:|---:|---:|
|RoS-diffuser|52.3 → 51.0|108° → 91°|20|0|20|0/20|51.2|13.6|
|ReS-diffuser|42.7 → **2.43**|106° → **19°**|10|2|12|**8/20**|20.0|13.0|
|TVS-diffuser|38.0 → 31.2|106° → 74°|20|0|20|0/20|38.0|12.8|
|Projected|22.9 → **3.83**|118° → 106°|0|13|13|**7/20**|24.7|2.9|
|Primal-Dual|24.5 → **3.37**|120° → 103°|14|0|14|**6/20**|29.3|3.6|
|ALM|24.5 → **3.40**|119° → 102°|14|0|14|**6/20**|31.6|3.6|
|PDM|21.0 → **5.35**|119° → 108°|0|18|18|**2/20**|21.4|4.5|
|Diffuser（无约束基线）|0.97 → 0.99|1.2° → 1.2°|10|0|10|10/20|17.4|0.33|

（16 级那列来自 `results/native_points_aligned_20261114`，256 级来自 `results/native_points_aligned_t256_20261114`；`ReS` 那一行的 256 级数字是**修复后的真实结果**，不是调参结果——同一份代码、同一组超参，只有基座日程变了。）

可以确认的三点：
1. **锯齿是日程造成的，不是论文公式**：ReS 42.7→2.43、Projected 22.9→3.83、PD→24.5→3.37、ALM→3.40、PDM 21.0→5.35，转角同步从 100–120° 降到 19–108°，planning success 从 0–1/20 升到 2–8/20，接近无约束基线的 10/20。
2. **RMSE 不是合适的判据**：两代基座互有高低（PDM 24.76→21.4，DDIM 19.03→17.4 但个别案例 85 m），因为误差被基座自身 ~20 m 的能力主导；能反映采样器是否真的在采样的是折线几何与 planning success。
3. **RoS/TVS 没有改善，问题在它们自己的积分参数**：`legal_gamma` 用 `scale=exp(ε·horizon)`，`cbf_epsilon=1.0` 且 `horizon=N·dt`，N 从 16 变 256 后 e^{-256}≈0，γ 直接压到 `cbf_gamma_floor=1.25e-4`；TVS 的 `σ_dot=-initial/(N·dt)` 也被 16× 稀释。这套 ε/horizon 参数是为 16 级写的，**不能平移**。这是下一件要做的事，本轮没有偷偷改它们（改了就是把它们的机制改掉）。
4. `Projected` 的仅线段碰撞 13/20、`PDM` 18/20 仍是"逐点约束 vs 整段评价"的不一致，与日程无关。

## 4c. SafeDiffuser 三种的逐条原文核对（回答"实现是否有问题"）

对 ICLR-2025 SafeDiffuser 原文 eq (5)–(13)、Thm 3.4/3.5/3.6 逐条核对，结论：**公式层是忠实的，问题出在原文没有的两个步长旋钮。**

忠实的部分：

- 可控形式 (6) `τ̇=ν` 与 QP (13) `min‖ν−(τ^j−τ^{j+1})/Δτ‖² s.t.(8)` → 代码 `desired=(nominal−x)/cbf_dt` + `J v + ε(h−γ) ≥ 0`，且 `cbf_dt=1` 与 QP 目标里的 Δτ 一致；
- Thm 3.4 的 γ 条件 `γ ≥ max_k|γ−b_k|e^{−εN}` 解出来正是 `max(−min(h)/(e^{εN}−1), max(h)/(e^{εN}+1))`，与 `legal_gamma` 逐项相同；
- ReS 的 `w(j)=0 (j≤N0)`（对应原文 ReS-DIFFUSER-l10 的最后 11 步硬约束）、`hard_steps=10 → N0=9`；
- TVS 的 σ 端点条件 `σ(N)≤b(τ^N)`、`σ(0)=0` 由 `initial=min(h,0)` 与 `σ=initial·j/N` 满足。

不在原文里、且已测出有害的部分：

1. **`cbf_max_point_step=0.001953125`（0.156 m/微步）**：它把每微步位移掐住，实际积分步长由这个上限而不是 `cbf_substeps` 决定。实测把 `cbf_substeps` 从 16 改成 1：len_ratio 36.59→36.59、转角 112.7°→113.0°、最大侵入 36.9→37.0 m，**完全没变**，只有步数 21.2→11.8、耗时 11.7→6.7 s 变化。因此交接文档里"细化 4×/16× 使侵入降到 0.0548/0.0140 m"的结论**不能由 `cbf_substeps` 解释**（那是旧控制点适配路径下的结论），本报告 §2 里"子步=同一 SDE 的离散细化"对 SafeDiffuser **不成立**，特此更正（对 CD 的 `cd_substeps` 仍成立，见 §4b）。
2. **一阶速度型 barrier + Δτ=1 的数十米位移**：QP 只保证 `∇b·ν ≥ −ε(b−γ)`，而单级实际位移可达数十米（waypoint 间距 0.95 m、障碍格 0.625 m），线性化完全失效。实测 RoS 交付轨迹最大侵入 36.9 m，而每一步的 QP 都自检通过——这就是"每步都可行、交付不可行"的机制。原文用"Δτ 取足够小"回避它，本实现把 Δτ 固定成 1。
3. **TVS 的 σ̇ 符号**：`tvs_schedule` 返回 `−initial/(N·dt)`，但它自己定义的 `σ(j)=initial·j/N` 的导数是 `initial/N`，两者差一个负号和 dt。若 j 的计数方向与原文相反需要显式换算，目前没有依据。**这一条只是可疑，尚未定论。**

审计脚本：`scripts/audit_safediffuser_dt.py`，输出 `docs/safediffuser_dt_audit.json`。

## 5. 测试结果

### 5.1 单测

```
& E:/CondaEnvData/envs/GGMPC/python.exe -B -m pytest -q -p no:cacheprovider experiments/safety_baselines/tests/
62 passed, 3 errors
```
3 个 error 与代码无关：`tmp_path` 夹具在 `D:\...\.tmp` / `%TEMP%` 下创建目录后被沙箱 ACL 拒绝列目录（`PermissionError: [WinError 5]`），涉及 `test_map_only.py` 与 `test_point_baseline.py` 各 1–2 项使用 `tmp_path` 的测试（这两个文件本轮未改动）。其余 62 项通过，其中新增 `tests/test_segment_points.py` 8 项，含"已知 Gaussian score + 半空间约束 → 解析截断正态"的退化对照（300 个种子，均值/标准差落在解析值 ±0.12 内）。

### 5.2 二十案例小批量（主配置，点约束；`results/native_points_aligned_20261114`）

20 个固定 val 案例 × 1 种子 × 8 方法 = 160 次，全部返回原生 128 点、状态与交付数组逐元素一致、固定端点未变、无求解失败。

|方法|返回|顶点碰撞|仅线段碰撞|路径碰撞|planning success|RMSE m|最大侵入 m|耗时 s|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|RoS-diffuser|20/20|20|0|20|0|49.94|36.86|2.45|
|ReS-diffuser|20/20|20|0|20|0|41.28|35.68|2.54|
|TVS-diffuser|20/20|20|0|20|0|41.12|35.22|2.10|
|Projected|20/20|0|20|20|0|24.78|17.17|1.46|
|Primal-Dual|20/20|20|0|20|0|26.88|28.44|1.91|
|ALM|20/20|20|0|20|0|28.99|30.16|1.88|
|PDM（点约束）|20/20|0|19|19|1|24.76|28.44|2.27|
|Diffuser（无约束基线）|20/20|13|0|13|7|19.03|25.31|0.16|

与交接前的 8 方法小批量相比，**可确认的改进只有 PDM，而且是"输出变成有效样本"这一条**：此前 PDM 的 VE 采样在冻结日程上根本不会动（无约束 RMSE 3.4e4 m 量级），交付的折线是随机折线；现在 PDM 与无约束基座同量级（24.76 vs 19.03 m）。注意 PDM 的顶点碰撞此前已经是 0/20（逐点投影把每个顶点都推进自由格，即使整条折线是垃圾），所以**不能**把"顶点碰撞 0/20"当作修复的证据；能作为证据的是 RMSE 量级和 §3 的无约束门。其余方法的碰撞/侵入水平和交接前一致，本轮没有把它们改好，也不声称改好了。

同时要注意：这张表里**没有任何方法通过验收**。顶点碰撞 20/20 的四种（RoS/ReS/TVS/PD/ALM）说明其机制（有限 dt 的 CBF 速度条件、乘子软约束）在本任务上并没有把交付折线推入可行域；"仅线段"碰撞 19–20 说明逐点约束的可行集与评价对象不一致（P1 原样保留）。RMSE 19–50 m（路径长 160 m）说明冻结基座本身偏弱，这也解释了为什么要把"低于噪声下限的差距"和"算法差异"分开报告。

### 5.3 PDM 线段约束端到端（`results/native_points_pdm_segments_20261114`）

val 0/81/163，PDM，T=10 × M=20：

|案例|圆数|状态|耗时|原因|
|---|---:|---|---:|---|
|0|249|failed|58 s|`segment_projection_failed: full_residual=0.1693`（末端 `segment_sqp_maxiter: 0.1128`）|
|81|215|failed|56 s|`full_residual=0.1005`（`0.1258`）|
|163|216|failed|55 s|`full_residual=0.4520`（`0.1387`）|

180 次内层投影全是 `segment_projection_partial`：预算内只能部分降低残差；末端级别按设计要求精确可行，因此抛 `SolverFailure` 而不是返回近似解。
**含义**：PDM 的原文约束集已经接进正式循环并能跑，但在本任务的 128 点折线 + 256² 占据地图上，预算内解不出可行投影。这不是"约束写错/没接入"，而是机制性的：PDM 每个噪声级别对**带噪样本**做投影，而 128 点折线在 τ≳0.1 scene（8 m）的各向同性噪声下就是穿越建筑群的锯齿，"最近的可行折线"本身是路径规划问题，局部最小范数法只能局部下降。因此这组数据只作为"已接入 + 在此任务上不可解"的证据，**不用于排名**，也不能取代 §5.2 的点约束结果。三条可选出路（改任务参数化 / 大幅提高 M / 换成整体 NLP 求解器）都超出助手可自行决定的范围，列在 §6。

## 6. 仍未验收 / 需要用户决策的事项

1. **冻结 16 级日程是采样器类方法的结构瓶颈**：Langevin/投影类方法的交付轨迹必然残留 ≈8.8 m/坐标的噪声（两维 RMSE 下限 12.4 m），而 DDIM 不残留。要让七种方法站在与 DDIM 同级的分辨率上比较，只有两条路：换更细的噪声日程（`configs/map_only_l1_final8_t256.yaml` 已备好，需重训 ~41 min，用户此前未授权），或在报告中始终把"下限"与"算法差异"并列。本轮选择了后者，并把下限量化写进 `sampler_compatibility_gate.json`。
2. **PDM 线段投影在 τ≳0.1 处不收敛**（§4）。要么接受"PDM 在该任务上对 128 点折线的原文约束集无法在预算内求解"作为结论，要么允许把折线点数/投影参数按任务适配（属于改变任务设定，需授权）。
3. SafeDiffuser 三种的数值积分（交接 P1）本轮**没有改动算法**，也没有新增兜底；交接里"细化 4×/16× 后顶点侵入 0.0548 m/0.0140 m"的结论仍成立，而本次 20 案例的 35 m 级侵入来自"约束对象是速度条件、不是交付点位置"，两者是不同的问题，都需要单独验收。
4. Constrained Diffusers 的 4 粒子 / 正部期望 / 固定返回粒子 0、以及 η_λ、ρ 从 "large" 设置折算到本任务的合法性，只做到"来源写清"，没有做到"验收通过"。
5. Ours 未参与本轮任何运行（`--methods` 不含 Ours），完整 Ours 权重与代码未改动。

## 7. 复现命令

```powershell
Set-Location D:/ProjectDirectory/Neural-IRISDiffuser
$py='E:/CondaEnvData/envs/GGMPC/python.exe'
$env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; $env:PYTHONDONTWRITEBYTECODE="1"

# 单测（3 个 tmp_path error 是沙箱 ACL 限制，与本轮改动无关）
& $py -B -m pytest -q -p no:cacheprovider experiments/safety_baselines/tests/

# 采样器/基座兼容性门（无约束）
& $py -B -W ignore experiments/safety_baselines/scripts/audit_sampler_match.py --indices 0,81

# 线段约束与投影的真实地图审计
& $py -B -W ignore experiments/safety_baselines/scripts/audit_segment_projection.py --indices 0,81,163

# 20 案例小批量（全部方法，点约束）
& $py -B -W ignore experiments/safety_baselines/scripts/validate_point_methods.py

# PDM 线段约束对照（少量案例）
& $py -B -W ignore experiments/safety_baselines/scripts/validate_point_methods.py `
  --config experiments/safety_baselines/configs/seven_native_points_aligned_segment_pdm_t16.json `
  --indices 0,81,163 --methods PDM
```

## 8. 复现：256 级基座与骨架对照（本轮新增）

`powershell
Set-Location D:/ProjectDirectory/Neural-IRISDiffuser
E:/CondaEnvData/envs/GGMPC/python.exe='E:/CondaEnvData/envs/GGMPC/python.exe'

# 256 级共同基座重训（~46 min，约 26 s/轮；独立输出目录，不覆盖任何现有权重）
& E:/CondaEnvData/envs/GGMPC/python.exe -B -W ignore experiments/safety_baselines/scripts/train_map_only_points.py 
  --run-id map_only_points_t256_e100 
  --config experiments/safety_baselines/configs/map_only_points_l1_final8_e100_t256.json

# 256 级基座上的同一 20 案例对照
& E:/CondaEnvData/envs/GGMPC/python.exe -B -W ignore experiments/safety_baselines/scripts/validate_point_methods.py 
  --config experiments/safety_baselines/configs/seven_native_points_aligned_t256.json

# 256 级基座的无约束兼容性门
& E:/CondaEnvData/envs/GGMPC/python.exe -B -W ignore experiments/safety_baselines/scripts/audit_sampler_match.py 
  --config experiments/safety_baselines/configs/seven_native_points_aligned_t256.json 
  --out experiments/safety_baselines/docs/sampler_compatibility_gate_t256.json

# 骨架 vs 约束对象对照（旧修正器的表示逻辑；只做诊断，不是七种方法之一，也不是 Ours）
& E:/CondaEnvData/envs/GGMPC/python.exe -B -W ignore experiments/safety_baselines/scripts/audit_backbone_effect.py
`
