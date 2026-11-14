# 七方法复现审查：全部尚未通过

原始实验记录保留作为调试证据，不进入论文方法排名。当前不启动全量。

**2026-11-14 更新：** 本轮修复与证据见 [DSH_FIX_REPORT_20261114.md](DSH_FIX_REPORT_20261114.md)。要点：PDM 的 γ 约定与 VP/VE 换算已按两种来源分别实现并通过无约束兼容性门（34203 m → 13.9 m，下限 12.40 m）；PDM 原文整段线段约束已实现并接入正式 runner，但在本任务上预算内解不出可行投影；CD 的 19 m 被证明主要是冻结 16 级日程的噪声下限而非公式错误；其余方法的碰撞水平未改变，仍未验收。2026-11-14 追加：共同基座已按交接要求重训为 256 级（依据见修复报告 §3.1：waypoint 间距 0.95 m vs 16 级最低训练噪声 8.77 m），锯齿 21–52× → 2.4–5.4×、planning success 0–1/20 → 2–8/20；RoS/TVS 的 ε/horizon 参数不能跨日程平移，仍是待办。

|方法|原文依据|当前验收|
|---|---|---|
|RoS-diffuser|SafeDiffuser Sections3-4,AppendixC.1|未通过|
|ReS-diffuser|SafeDiffuser Sections3-4,AppendixC.1|未通过|
|TVS-diffuser|SafeDiffuser Sections3-4,AppendixC.1|未通过|
|Projected|Constrained Diffusers Sections5-7,AppendixC(page22)|未通过|
|Primal-Dual|Constrained Diffusers Sections5-7,AppendixC(page22)|未通过|
|ALM|Constrained Diffusers Sections5-7,AppendixC(page22)|未通过|
|PDM|PDM Algorithm1,Section6.3,AppendixC.3(page15),AppendixF/G(page19)|未通过|

## RoS-diffuser

- Current finite-step integration returns vertex constraint violations; refinement changes residual substantially.
- Exact occupancy-union SDF is nonsmooth; original continuous-time CBF guarantees have not been established for this map implementation.
- Coarse16-level common prior and DDIM nominal are task adaptations, not an unchanged original experiment.

后续核对：
- Match the chosen nominal reverse transition and physical/diffusion time indexing to source.
- Verify method-specific gamma, hard interval or sigma against source independently.
- Check original local-trap evaluation and distinguish discrete planning from lower-level continuous execution.

## ReS-diffuser

- Current finite-step integration returns vertex constraint violations; refinement changes residual substantially.
- Exact occupancy-union SDF is nonsmooth; original continuous-time CBF guarantees have not been established for this map implementation.
- Coarse16-level common prior and DDIM nominal are task adaptations, not an unchanged original experiment.

后续核对：
- Match the chosen nominal reverse transition and physical/diffusion time indexing to source.
- Verify method-specific gamma, hard interval or sigma against source independently.
- Check original local-trap evaluation and distinguish discrete planning from lower-level continuous execution.

## TVS-diffuser

- Current finite-step integration returns vertex constraint violations; refinement changes residual substantially.
- Exact occupancy-union SDF is nonsmooth; original continuous-time CBF guarantees have not been established for this map implementation.
- Coarse16-level common prior and DDIM nominal are task adaptations, not an unchanged original experiment.

后续核对：
- Match the chosen nominal reverse transition and physical/diffusion time indexing to source.
- Verify method-specific gamma, hard interval or sigma against source independently.
- Check original local-trap evaluation and distinguish discrete planning from lower-level continuous execution.

## Projected

- Repeating16 trained noise levels into64 updates is not the original64-level trained noise process.
- Unconstrained LMC diagnostic already differs strongly from the native DDIM prior.
- Map-SDF constraints and previous-task dual/rho settings were not validated for this new point prior.
- Current vertex constraints do not implement whole-polyline collision avoidance.

DCBF and inverse dynamics/receding-horizon execution must be checked by task. They are explicitly online mechanisms; do not blindly add online control modules to the offline Maze planning task.

后续核对：
- Check score target, normalization and per-level beta against source.
- Check projected objective or PD/ALM expectation, multiplier/slack/rho updates separately.
- Reproduce an original analytical constraint task before treating CARLA task adaptation as paper reproduction.

## Primal-Dual

- Repeating16 trained noise levels into64 updates is not the original64-level trained noise process.
- Unconstrained LMC diagnostic already differs strongly from the native DDIM prior.
- Map-SDF constraints and previous-task dual/rho settings were not validated for this new point prior.
- Current vertex constraints do not implement whole-polyline collision avoidance.

DCBF and inverse dynamics/receding-horizon execution must be checked by task. They are explicitly online mechanisms; do not blindly add online control modules to the offline Maze planning task.

后续核对：
- Check score target, normalization and per-level beta against source.
- Check projected objective or PD/ALM expectation, multiplier/slack/rho updates separately.
- Reproduce an original analytical constraint task before treating CARLA task adaptation as paper reproduction.

## ALM

- Repeating16 trained noise levels into64 updates is not the original64-level trained noise process.
- Unconstrained LMC diagnostic already differs strongly from the native DDIM prior.
- Map-SDF constraints and previous-task dual/rho settings were not validated for this new point prior.
- Current vertex constraints do not implement whole-polyline collision avoidance.

DCBF and inverse dynamics/receding-horizon execution must be checked by task. They are explicitly online mechanisms; do not blindly add online control modules to the offline Maze planning task.

后续核对：
- Check score target, normalization and per-level beta against source.
- Check projected objective or PD/ALM expectation, multiplier/slack/rho updates separately.
- Reproduce an original analytical constraint task before treating CARLA task adaptation as paper reproduction.

## PDM

- The pointwise free-cell projection omitted the paper explicit whole-segment obstacle constraints.
- Unconstrained finite-budget VE sampling has not recovered the learned prior.
- Paper Algorithm1 and generic official ALD code use different step conventions; provenance must be resolved, not mixed.
- The common x0/coarse/smooth/boundary objective is not the original score-matching training recipe.

2026-11-14：整段约束已实现（`adapters/paper_segment_constraints.py` + `adapters/segment_points.py`，占据地图→圆覆盖有覆盖率与自由空间误伤率审计，投影为最小范数 SQP(OSQP)+精确 L1 罚+全约束集校验，不可行即报错），并已由 `segment_constraints_for` 接入正式 runner；`tests/test_segment_points.py` 8 项通过，含解析截断正态退化对照。γ 两种约定均已实现并记录来源：原文 `γ_t=σ_t²/2σ_T²` 在本日程上每步混合率 4.7e-6（M=100 无法移动样本），官方 ALD `ε(σ/σ_min)²` 为 0.167；VP/VE 用 `s=γ·ᾱ_t` 精确换算。无约束门：34203 m → 13.88 m（下限 12.40 m）。**端到端线段版仍失败**（val 0/81/163 全部 `segment_projection_failed`），原因是 128 点折线在 τ≳0.1 scene 时"最近可行折线"本身是路径规划问题，见 `results/native_points_pdm_segments_20261114/notes.md`。原文 score-matching 训练目标仍未复现（共同基座是 x0 预测 + 混合辅助损失）。

后续核对：
- Audit original trajectory-specific official implementation, not generic image ALD.
- Audit training sigma range, score convention, initialization, gamma and fixed iteration budgets.
- Integrate segment geometry and solver with explicit audited obstacle representation.
