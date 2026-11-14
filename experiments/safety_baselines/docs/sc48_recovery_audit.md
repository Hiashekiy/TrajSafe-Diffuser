# S-C48 找回与验证审计

日期 2026-11-14。范围：只做取证、加载验证、统一评价与兼容性分析。**未重训、未改算法、未跑正式 test 集。**
数据目录 `results/sc48_recovery/`；脚本 `scripts/recover_sc48_sample.py`（历史代码内运行）、
`scripts/audit_sc48_vs_maponly.py`（统一评价）。

---

## 1. 是否找到了原始 S-C48 checkpoint？—— 找到了，但当前代码不能直接加载

|项|值|
|---|---|
|文件|`outputs/models/oneshot_k4p_c48_s/ckpt/best_task.pt`（90,808,984 B，2026-09-25 15:28:57）|
|目录内容|只有 `ckpt/best_task.pt` + `train.log`（902 KB）+ `training_summary.json`（163 KB）|
|**有没有 epoch_65.pt**|**没有**。仓库内所有 `epoch_*.pt` 都属于 `experiments/safety_baselines/results/*`，与 S-C48 无关|
|epoch 来源|`training_summary.json` 的 `best_task_epoch = 65`；checkpoint 内 `epoch=65`（已加载核对）→ **"epoch 65" 就是这个 `best_task.pt`**，没有独立的 epoch_65 文件|

同目录还有两个容易混淆的权重，**不能互相冒充**：

|目录|epoch|参数量（历史代码）|区别|
|---|---:|---:|---|
|`oneshot_k4p_c48`（A-C48）|55|7.5205 M|`train.feedback.rollout=True`（训练时跑两步真实 rollout）|
|`oneshot_k4p_c48_s`（S-C48）|**65**|**7.5205 M**|`train.feedback.rollout=False`（无历史反馈训练）← 本次对象|
|`oneshot_k4p_c48_d`（D-C48）|96|6.34 M|`model.feedback.enabled=False` + `model.safety_features.enabled=False`，所有安全损失权重为 0|

A/S 的 `config` 逐项 diff 只有 4 处不同：`train.feedback.rollout`（True→False）、`train.epochs`（200→100）、
`alm.max_curve_step_scene`（0.03→0.2）、`alm.final_project`（None→True）。
**S-C48 的模型结构与 A-C48 完全相同**（`num_controls=48`、`num_safety_queries=128`、`skeleton_blocks=2`、
`traj_blocks=8`、`final_blocks=3`、`feedback{enabled:true,zero_init:true}`）——"S" 的差别在**训练侧不做 rollout**，
不在结构。这与任务书描述一致。

### 兼容性：必须用历史代码

用**当前 HEAD** 加载：

```
missing: 1   ['feedback_encoder.net.0.weight']
unexpected: 20  ['scene_cnn.up_fuses.0.net.0.0.weight', … 'scene_cnn.merge.1.bias']
params = 6.52 M（应为 7.52 M）
```

原因：commit `c1d97bf`（2026-10-26，v4 改造）重写了 `SceneCNN`（去掉 `up_fuses`/`merge` 上采样融合）、
`FeedbackEncoder` 与 `planner`。S-C48 训练于 2026-09-25，**早于**该提交。
用 `c1d97bf^` 的代码加载：`missing=0, unexpected=0, adapted=0`，参数 **7.5205 M** ← 唯一忠实的方式。

复现（只读抽出历史代码树，不触碰 `src/`）：

```powershell
# 1) 抽历史代码（103 个文件，1.1 MB，仅 src/ + configs/）
git archive c1d97bf^ src configs | tar -x -C experiments/safety_baselines/legacy_sc48_code
# 2) 在历史代码里采样（guidance=none ⇒ 无走廊/无反馈/无 ALM/无投影，16 级原生日程）
& $py -B -W ignore experiments/safety_baselines/scripts/recover_sc48_sample.py
# 3) 用当前统一评价器比较
& $py -B -W ignore experiments/safety_baselines/scripts/audit_sc48_vs_maponly.py
```

## 2. 结构确认（加载后逐项核对）

|项|要求|实测|
|---|---|---|
|控制点 C|48|`num_controls=48`，`bspline.basis=(128,48)` ✓|
|曲线解码密度|128|`bspline.curve_points=128` ✓|
|最终碰撞评价密度|512|历史 `alm.dense_verify_points=512`；本次统一评价同样用 512 基网格 + 近障碍细化 ✓|
|安全几何分支|开启|`num_safety_queries=128`、`geo_mem_res=32`、`geo_decode_res=64`、`geo_sigma=0.25`；`safety_features_enabled=True` ✓|
|候选拓扑分支|正常|`skeleton_blocks=2`，采样时需喂 `candidate_xy/candidate_mask/candidate_geometry` ✓|
|反馈模块零初始化|是|`zero_init=True`；`train.feedback.rollout=False` ⇒ `feedback_valid≡0`，门控乘 valid ⇒ 梯度恒 0，停在恒等位 ✓|
|推理期历史反馈|不执行|`guidance="none"` 不走 corridor/feedback 分支 ✓|
|推理期 ALM|不执行|同上（`sample(..., guidance="none")` = plain DDIM，无走廊、无 ALM、无末端硬投影）✓|
|扩散日程|**T=16，原生**|`diffusion.timesteps=16`，`steps=16` ⇒ `pick_times(16,16)=None` ⇒ 逐级访问全部 16 级 ✓（未改成 256）|

## 3. 统一评价：S-C48 vs 当前 map-only best.pt

协议：同样的 20 个 val 场景（按 `sample_id` 对齐，61001…106100）、同一占据栅格、同一起终点、同一种子 0
（两边都是 `torch.manual_seed(0)` 后各自抽自己的状态维度），**同一个评价器**（512 基网格 + 近障碍细化 +
逐格跨越判碰 + 解析净距）。S-C48 用 16 步原生 DDIM，map-only 用 256 步 DDIM（其训练日程的全部级）。

|指标|**S-C48**|map-only `best.pt`|判读|
|---|---:|---:|---|
|**连续曲线碰撞率**|**2/20**|12/20|S-C48 好 6×|
|顶点碰撞率（曲线 128 采样点）|2/20|12/20|同上|
|控制多边形（48 点）碰撞|2/20|—|控制点≠曲线，仅作参考|
|**规划成功率**|**18/20**|8/20|S-C48 好 2.25×|
|平均最小净距|**+3.40 m**|−1.77 m|S-C48 全程不侵入|
|最大侵入深度|5.94 m|24.33 m|—|
|RMSE|**7.63 m**|16.04 m|—|
|路径长度比|0.969|0.995|都不是锯齿|
|最大曲率（1/m）|**1.64**|12.54|map-only 有大尖角|
|二阶差分均值（m）|0.00226|0.00145|两者都平滑|
|**NFE**|**16**|256|—|
|单场景耗时|**0.58 s**|4.98 s|**快 8.6×**，NFE 少 16×|

逐场景（前 6 个，`unified_eval.csv` 有全部 20 个）：

|scene|S-C48 净距|S-C48 碰撞|map-only 净距|map-only 碰撞|
|---|---:|---|---:|---|
|61001|7.1|否|8.8|否|
|69032|3.4|否|−4.2|是|
|74014|−0.6|**是**|−16.1|是|
|75045|2.6|否|2.1|否|
|80027|3.7|否|0.0|是|
|86008|−5.9|**是**|−24.3|是|

**交叉验证**：S-C48 自带的 512 点 `dense_validation.final_collision` 与本次统一评价器的判碰结论 **20/20 一致**
（不碰撞 18 个、碰撞 2 个 = 74014 / 86008）。说明统一评价器与历史评价器在"是否碰撞"上无分歧。

## 4. 是否优于当前 map-only best.pt？—— 是，且是量级差异

碰撞 12/20 → 2/20，成功率 8/20 → 18/20，净距 −1.8 m → +3.4 m，RMSE 16.0 → 7.6 m，
NFE 256 → 16，耗时 4.98 → 0.58 s。所有指标同一方向，**不是抽样噪声**（同一批场景、同一种子、同一评价器）。

两点必须同时写进报告：
1. S-C48 的优势**主要来自表示 + 训练目标**（48 控制点的 B 样条是强平滑先验，训练带 `L_safe`/`L_curve`/`L_shape`），
   而不是"16 级日程更好"。16 级日程的 σ_min=8.77 m 对 Langevin 类方法是负担（见 §5.5）。
2. S-C48 仍有 2/20 碰撞，且最深侵入 −5.9 m：**不是"安全模型"**，只是"好得多的基座"。

## 5. 七种算法与 S-C48 的兼容性（只检查，未修改）

### 5.1 CBF 需要 B 样条基矩阵

曲线 `p(u)=B(u)Q`（`B(u)` 为 1×48 权重）。安全约束 `h(p(u))` 对控制点的梯度是

```
∂h/∂Q = (∂h/∂p) · B(u)        # B(u) 与 p 无关，是精确的伴随映射
```

本仓库历史里已有该实现：`legacy_sc48_code/src/diffusion/sdf_guidance.py` 的
`_constraint_rows()`（`p = einsum('hk,bkd->bhd', basis, q)`）与 `_grad_q()`（把 `grad_p` 映回控制点）。
**结论：可实现，且已有参考实现。** 但当前的 `NativePointTask.cbf_solve` 假设 Jacobian 是按点对角
（每行只含该点自己的 2 列），**不能直接复用**；需要改用通用 OSQP 路径并喂 `J_Q = grad_p @ basis`。

### 5.2 Projected / Primal-Dual / ALM 能否在控制点空间表达连续曲线约束

**能，而且本仓库历史代码已经这么做过了**：`sdf_guidance.py` 提供
`penalty` / `projection`（顺序凸化 + OSQP）/ `primal_dual` / `alm_sdf` 四种处理器，
签名 `apply_sdf_guidance(mode, q_ref, sdf, codec, cond, …)` 直接在 48 控制点上更新，
约束在 128 点解码曲线上求值。这正是任务书第 4 节第 2 条要求的形态。
未验证的部分：这四种处理器是"外部 SDF 基线"语义，与七方法里 CD 的 Projected/PD/ALM 在
**乘子继承、预热、末端硬投影**上的细节不同，需要逐项对齐后才能称为"七方法之一"。

### 5.3 PDM 的 score 转换是否适用

适用。S-C48 同为 VP 前向 + x0 预测（`diffusion.prediction_type="x0"`），
`score = -(x-√ᾱ x̂0)/(1-ᾱ)` 是同一个恒等式，无需改动。注意 T=16 ⇒ σ_min=8.77 m、σ_max=325，
PDM 的 `paper` 步长约定在此日程上同样不可用（mixing ratio `1/(2σ_T²)≈4.7e-6`），
必须用 official-ALD 约定并按 `ε/(2σ_min²)=1/6` 反算 ε（等价于 `pdm_ald_eps≈4.0e-3`）。

### 5.4 PDM 的线段约束**不能**直接作用在相邻控制点上 —— 这是最容易踩的坑

48 个控制点是**控制多边形**，不是曲线；相邻控制点的连线**不是**轨迹线段。
把 `CircleSegmentConstraints`（我的实现假设状态就是折线）直接套到 `Q` 上会约束错对象：
控制多边形可以在障碍里而曲线是安全的，反之亦然。
正确做法：约束定义在解码曲线的相邻采样点 `[B(u_i)Q, B(u_{i+1})Q]` 上，
把每行的梯度经 `B(u_i), B(u_{i+1})` 映回 48 维。**当前 `SegmentPointTask` 不能复用。**

### 5.5 256 级算法要不要重推到 16 级

要。七种循环里凡是 Langevin/SGLD 的（PDM、CD 三种的噪声注入、SafeDiffuser 的微步积分）都直接受
σ_min 影响：

|基座|σ_min|Langevin 末级必然残差（两维 RMSE 下限）|waypoint/曲线采样间距|
|---|---:|---:|---:|
|S-C48（16 级）|8.77 m|**12.40 m**|曲线 512 采样 ≈0.31 m|
|map-only（256 级）|1.10 m|1.55 m|0.95 m|

也就是说：**把 PDM/CD 直接搬到 S-C48 的 16 级日程上，会重新引入 12.4 m 的噪声下限**，
而 S-C48 之所以好看，恰恰是因为它的基线是 DDIM（末步直接输出 clean，不残留噪声）。
SafeDiffuser 三种跟随 DDIM 名义跳转，不受此限；PDM/CD 受影响。

## 6. 需要注意的两处口径差异

1. **数据根**：checkpoint 里记的是 `data/carla_processed_160k4p_c48`（已不存在），
   但 `data/processed/160m-k4p-c48` 含全部所需数组（`control_gt(1550,48,2)`、`candidate_xy`、
   `alm_*`、`sample_id` 61001… 完全一致）。本次用后者，`sample_id` 逐一对齐，无场景错配。
2. **种子**：两边都是 `torch.manual_seed(0)` 后各抽自己的维度（48×2 vs 128×2），
   即"同一 RNG 方案、同一种子、不同状态维度"。这符合任务书"相同的随机种子"，
   但不能解释为"同一噪声向量"；若要多种子稳健性，需另跑 seed 1/2。

## 7. 结论与建议

### 直接回答任务书的 7 个问题

1. **找到了吗**：找到了 —— `outputs/models/oneshot_k4p_c48_s/ckpt/best_task.pt`，但**当前代码不兼容**（21 个张量），必须用 `c1d97bf^` 的历史代码。
2. **是 epoch 65 吗**：是。`best_task_epoch=65` 且 checkpoint 内 `epoch=65`；没有独立的 `epoch_65.pt` 文件。
3. **统一评价下的真实碰撞率**：**2/20**（74014、86008），最大侵入 5.94 m，平均最小净距 +3.40 m。
4. **优于当前 best.pt 吗**：**显著优于**（碰撞 2 vs 12、成功率 18 vs 8、RMSE 7.6 vs 16.0、NFE 16 vs 256）。
5. **适合作为七种方法的共同基座吗**：**适合作为"生成先验"**，但需要做两件事才公平：
   (a) 约束必须定义在解码曲线上、梯度经基矩阵映回控制点（§5.1/5.4）；
   (b) PDM/CD 的 Langevin 类循环在 16 级上会受 12.4 m 噪声下限限制（§5.5）。
6. **16 级日程能否支撑七种方法的合理适配**：DDIM-backbone 类（SafeDiffuser 三种）可以；
   Langevin/SGLD 类（PDM、CD 三种）**不合适**，除非接受 12.4 m 的残差下限，
   或者把 S-C48 重训到更细的日程。
7. **是否需要重训 256 级 S-C48**：**如果要保留 PDM 与 CD 三种方法，需要**。理由即 §5.5 的噪声下限；
   若只做 SafeDiffuser 三种 + PDM 的 DDIM 变体，则可以先用 16 级跑通。

### 明确的建议（三选一）

**建议：采用 S-C48 作为共同基座，并按需重训一个 256 级版本。**

|方案|代价|收益|风险|
|---|---|---|---|
|A. 保留 map-only|0|已有 256 级日程，Langevin 类下限 1.55 m|**基座本身 12/20 穿障**，结论没有解释价值|
|**B. 采用 S-C48（16 级）**|0（权重已存在）|碰撞 2/20、成功率 18/20、快 8.6×|Langevin 类方法受 12.4 m 下限；约束必须改到曲线上|
|C. 重训 256 级 S-C48|1 次训练（估 2–3 h，7.5 M 参数 + 16× 级数）|B 的收益 + Langevin 类下限降到 ~1.5 m|需要重训、要重新选 checkpoint（选择准则也要按 Phase A 的证据改）|

推荐路径：**B 先行（本周可出七方法对比的"任务适配版"表），同时排期 C**；
在 C 完成前，PDM/CD 三者的结果必须标注"受 16 级日程噪声下限限制"，不得与 S-C48 自身基线直接比较。
A 只保留为"点表示消融"的一行，不再作为主基座。

### 明确不做的事（本阶段）

- 未重训任何模型；未修改七种算法的任何公式；未跑正式 test 集。
- 未把 S-C48 的权重复制/改名到别处，未用 A-C48 的权重冒充 S-C48。
- 未修改 `src/`；历史代码只读抽到 `experiments/safety_baselines/legacy_sc48_code/`（已 gitignore）。
