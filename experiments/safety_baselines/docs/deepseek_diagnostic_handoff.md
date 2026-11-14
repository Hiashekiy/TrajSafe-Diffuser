# 独立诊断交接（Phase H）：给 DSH 的材料与待答问题

本文件只交接**证据与待答问题**，不做结论替代。所有数字可在报告列出的脚本/结果目录中复现。

## 0. 资产清单

|项|值|
|---|---|
|仓库 / 分支|`Hiashekiy/TrajSafe-Diffuser` / `v4-map16-fbdelta-safeloss`|
|本轮起点 commit|`a17a667`（本轮修复见文末 commit）|
|冻结基座（t256，当前）|`results/map_only_points_t256_e100/ckpt/best.pt`（epoch 79）|
|对照基座（a17a667 时代）|同目录 `ckpt/best_task.pt`（epoch 86）|
|基座架构|`map_only_point_diffuser_v1`，4,366,146 参数，主干 1 + 末端 8，地图特征 16×16，128 点直接输出，x0 预测|
|训练日程|256 级 `squaredcos_cap_v2`，σ_min=1.099 m，σ_max≈5195 scene|
|数据|`data/processed/160m-k4p-c48`：train 3850 / val 1550 / test 420（15 episodes，无跨集重叠）|
|配置|`configs/seven_native_points_paper_t256.json`（论文口径）、`..._vertexall_t256.json`（全顶点诊断）、`..._vertexall_t256_besttask.json`（旧口径复现）|
|种子协议|初始噪声共享（`initial_seed`），逐级噪声按方法派生（`derived_seed`），见 `docs/paired_protocol_audit.md`|

## 1. 关键结果（同一 20 个 val 案例、种子 0、配对初始噪声）

### 1.1 DDIM 步数（`best_task` 与 `best` 两个 checkpoint）

|checkpoint|steps|NFE|顶点碰撞|路径碰撞|success|RMSE m|最小净距 m|
|---|---:|---:|---:|---:|---:|---:|---:|
|best_task|16|16|12|12|8|18.4|−3.3|
|best_task|256|256|14|14|6|19.6|−4.1|
|**best**|16|16|8|8|12|14.2|−0.7|
|**best**|256|256|7|7|13|14.3|−0.6|

### 1.2 七个安全方法（`best_task` 与全顶点口径；本轮修复后全顶点口径见 `results/review_fix_vertexall_t256/`）

`a17a667` 时代（`results/review_fix_vertexall_t256/`，256 级、配对噪声、`best_task.pt`）：
RoS 20/0/20（顶点/仅线段/路径）、ReS 13/1/14、TVS 20/0/20、Projected 0/13/13、PD 14/0/14、ALM 13/0/13、PDM 0/18/18、Diffuser 14/0/14。

在 `best.pt` 上重跑同一套（`results/` 下同名目录，尚无完整表）已明显不同：DDIM 7/20，其余方法相应下降。

### 1.3 PDM 论文口径（VE + App.C.3 整段线段约束）

`results/review_fix_paper_pdm_t256/`：val0 **成功**（三类碰撞全 False，160 s），val81 `segment_projection_failed: full_residual=0.105`。

## 2. 逐场景材料（可视化/逐步）

`results/phase_f_visualization/`：
- `case_0_trajectories.png`、`case_81_trajectories.png`：地图 + GT + 各方法折线 + 碰撞点；
- `case_*_denoise.gif` / `.csv`：DDIM 反向链逐帧与逐帧指标（`t`、净距、碰撞、长度比、转角）；
- `case_*_<method>_steps.json`：逐级 trace（QP 状态、残差、迭代数、积分步数、`discrete_ok`）；
- `case_*_record.json`、`summary.json`：完整记录。

**视觉结论（case 0）**：DDIM/ReS 沿走廊光滑（长度比 1.14、转角 14°），Projected/PDM 在走廊里横向乱窜
（转角 179–180°、长度比 5.5–9.6×、最大侵入 −16 m），红叉标出的碰撞点全在横向尖刺上。

## 3. 求解日志

- `results/phase_e_safediffuser/discrete_barrier.jsonl`：离散证书开/关对照（含 `level_attempts`、`discrete_ok`、
  `integration_steps`、`qp_status`、`cbf_residual`、`kkt_dual`）。
- `results/review_fix_vertexall_t256/results.jsonl`：每条记录含 `pdm_projection_stats` 与逐级 trace。

## 4. 待 DSH 独立回答的问题

### H1 基础模型
1. 这个基座是否学到了"沿走廊的轨迹分布"？（证据：长度比 0.99–1.07、最大转角 35–48°、但 7–14/20 穿障）
2. x0 预测能否可靠恢复干净轨迹？（t=0 的 x0 RMSE 4.1 m，而 waypoint 间距 0.95 m）
3. score 是否足以支撑 Langevin？（末级 cos(ε̂,ε)=0.36、‖s‖≈3.2e3，中高噪声段 cos≥0.996）
4. BoundaryDecoder 是否破坏生成分布？（实测每级略好 0.3–1.8%，见 A4）
5. 256 步完整采样是否仍无法生成有效路径？（是，14/20 穿障，与 16 步几乎相同）

### H2 安全约束
6. 当前约束集合是否与表示匹配？（顶点 vs 整段：`segment_only_collision` 13–18/20 是直接证据）
7. 安全梯度是否正确？（`adapters/exact_map.py` 解析距离梯度；`adapters/paper_segment_constraints.py` 稀疏解析 Jacobian；
   `tests/test_paper_segment_constraints.py` 有限差分校验；不可微只在最近特征切换处）
8. 约束修正是否改变生成分布？（Projected/PDM 的 θ=180°、长度比 5–9× 说明修正把样本推离了数据流形）
9. 是否存在投影不可行/局部最优？（PDM 线段投影在 8 m 级扰动起失败；`docs/segment_projection_audit*.json`）
10. 是否存在约束过强导致轨迹严重偏移？（`cbf_epsilon=1` + γ≈1.25e-4 在 256 级下约等于"不许退让"）

### H3 算法复现
11. 哪些实现与原文一致？见 `docs/comparison_methods_implementation.md` 的逐条对照（QP 目标、barrier 形式、
    γ 闭式、eq(13)(14)(16)(17)(18)、PDM Alg.1 结构）。
12. 哪些属于任务适配？同文件 §4 的 11 项表（含 `dual_constraint_scale`、`pdm_ald_eps`、M=20→各方法预算、
    `cbf_max_point_step`、`Δτ=1`）。
13. 哪些修改改变了理论机制？（SafeDiffuser 的"按级离散证书"是工程加固，不是原文机制；
    若按逐微步强制则会冻结轨迹——实测 176 m）
14. 哪些失败是方法局限、哪些是实现问题？（PDM 的 546 km 是步长/日程不匹配；Projected 的 180° 乱窜是
    "Langevin 在 256 级上混合不足 + 顶点投影"的组合；SafeDiffuser 的 36 m 是高噪声段一阶条件无意义）

### H4 实验设计
15. 128 点轨迹表示是否合理？（间距 0.95 m vs 障碍格 0.625 m；投影只动 126 个自由变量）
16. 训练数据是否适合安全扩散规划？（GT 自身在圆并覆盖下残差 −0.7…−6.2 m，即 GT 也不满足"线段-圆"约束）
17. 是否存在尺度/归一化问题？（坐标 [-1,1]²，1 unit=80 m；σ_min 1.1 m 与间距 0.95 m 同量级）
18. 安全集合与训练分布是否严重不匹配？（无约束 DDIM 7–14/20 穿障 ⇒ 训练分布本身大量穿障）
19. 当前实验能否公平比较七种方法？（顶点口径可比较；整段口径只有 PDM 有，且 1/2 可用）

### H5 证据要求（已在文档中逐项给出代码位置/公式/失败案例/数据）

## 5. 已排除的假设（避免重复劳动）

|假设|排除依据|
|---|---|
|"16 级日程太粗导致锯齿"|换 256 级后锯齿 21–52× → 2.4–5.4×，确实修好了；但穿障率没变（7–14/20），所以锯齿不是主因|
|"采样步数不足"|16/64/128/256 步的碰撞数 12/13/14/14，步数增加还略差|
|"PDM 的 VP/VE 混用"|已改 VE-only，端到端从 546 km 变 11.9 m（无约束），但带约束仍 1/2 成功|
|"乘子单位/噪声重复抽取"|已修，量级变化 <15%|
|"BoundaryDecoder 破坏分布"|实测每级略好 0.3–1.8%|
|"SafeDiffuser 只差离散证书"|实测证书在 761/768 级不通过且减步长无法修复|
