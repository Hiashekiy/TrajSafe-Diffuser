# Native Point 重训方案（含已有权重审计）

日期 2026-11-14。决策前提：**放弃"把七种算法塞进 B 样条控制点空间"**，改用原生 128 轨迹点作为共同扩散状态；
S-C48 保留为独立的 B 样条生成器实验，不再为七种基线做适配。

本文件只做审计与方案，**尚未开始训练**。

---

## 1. 已有 Native Point 权重审计

### 1.1 两个权重的结构与训练信息

|项|`map_only_points_l1_final8_e100_20261114_000521`|`map_only_points_t256_e100`|
|---|---|---|
|`d_model` / `final_blocks`|128 / **8**|128 / **8**|
|`map_res` / `global_res` / `coord_hidden`|256 / 16 / 256|同左|
|`dropout`|0.0|0.0|
|预测目标|`x0`|`x0`|
|扩散日程|**T=16**|**T=256**|
|数据根|`data/processed/160m-k4p-c48`|同左|
|epochs / best_task epoch|100 / 96|100 / 86|
|`best_task_score`|22.55|21.83|
|参数量|4.37 M|4.37 M|

结论：**两者都是"SceneCNN 编码 + 8 层 FinalDenoiser"的点模型**（即"1 层编码 8 层去噪"），
只是日程不同；它们都还没有安全几何分支。

### 1.2 统一评价下的实测（同一 20 个 val 场景、同一 512 点曲线评价器）

|权重|DDIM 步数|曲线碰撞|成功率|平均最小净距|RMSE|长度比|最大曲率|NFE|耗时|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`t256/best.pt`|256|**7/20**|**13/20**|−0.6 m|14.3 m|0.99|12.5|256|3.10 s|
|`t256/best_task.pt`|256|14/20|6/20|−4.1 m|19.6 m|1.07|46.8|256|2.49 s|
|`t256/latest.pt`|256|10/20|10/20|−1.9 m|16.4 m|1.00|67.3|256|2.51 s|
|`l1_final8`（T=16）|16|12/20|8/20|−3.3 m|18.4 m|1.05|48.3|16|0.17 s|

（`best_task` 与 `latest` 的数字来自 Phase A 审计；`best.pt` 是当前冻结基座。）

**判断**：现有 Native Point 权重的最好成绩是 **7/20 碰撞、13/20 成功**，明显弱于
S-C48（2/20、18/20）。但注意两者的差别不只在表示：S-C48 多了**安全几何分支 + 候选拓扑条件**，
而这两个点模型是纯地图条件。因此"点表示本身更差"这个结论**不成立**——
缺的是安全特征，不是点表示。

## 2. 目标模型：Native Point + 安全几何特征（无反馈、无推理期 ALM）

### 2.1 结构（在现有 `MapOnlyPointDiffuser` 上加一个分支，不动主干）

|模块|现状（点模型）|新模型|说明|
|---|---|---|---|
|`scene_cnn`（Map + Global 编码）|✅|不变|1 层编码|
|`final_denoiser`（8 层，128 点 token）|✅|不变|8 层去噪|
|`head_p`（128×2 输出）|✅|不变|直接输出轨迹点|
|**安全几何分支**|❌ 无|**新增**|移植 S-C48 的 `EllipseGeometry`：`num_safety_queries=128`、`geo_mem_res=32`、`geo_sigma=0.25`、`geo_bias_clip=8`|
|候选拓扑条件|❌ 无|**可选**（建议先关）|`candidate_xy/mask/geometry` 已在数据里，但会引入 `topology_head` 与路由逻辑，先不加，保持单变量|
|`feedback` 模块|❌ 无|**不加**|对应 S-C48 的 `rollout=false` 消融口径|
|B 样条解码|—|**不要**|直接输出 128 点|

参数量估计：4.37 M（点模型）+ 约 0.6 M（EllipseGeometry + AdaLaN）≈ **5.0 M**。

### 2.2 数据：**不需要重新准备**

`data/processed/160m-k4p-c48/{train,val,test}` 里已经有安全分支需要的全部标签
（这正是 S-C48 用的同一份缓存）：

|标签|形状|用途|
|---|---|---|
|`curve_gt.npy`|(N,128,2)|点表示的监督目标（不是 `control_gt`）|
|`alm_cell_a/b/valid.npy`|(N,128,12,2)/(N,128,12)/(N,128)|走廊单元（安全几何记忆）|
|`alm_anchor_s.npy`|(N,128)|走廊进度参数|
|`ellipse_shape4_gt.npy` / `shape_valid.npy`|(N,128,4)/(N,128)|椭圆标签（若加形状损失）|
|`candidate_xy/mask/geometry/lengths`|—|候选拓扑（本轮先不用）|

### 2.3 日程：**建议 T=256**

|理由|依据|
|---|---|
|七种方法的安全约束要在"状态还算一条轨迹"的级别上起作用|Phase B-Review 实测：16 级 σ_min=8.77 m，高噪声级 `min_h ≈ −0.4 scene`，逐级投影无意义|
|Langevin 类（PDM/CD）的终端噪声尺度直接由 σ_min 决定|σ_min(16 级)=8.77 m vs σ_min(256 级)=1.10 m|
|现有 256 级点模型已经跑通整套训练/采样流程|`map_only_points_t256_e100` 100 epoch 约 48 min|

### 2.4 损失

```
L = L_x0(curve)                      # 主监督，作用在 128 点上
  + 0.1 * L_smooth,   L_smooth = mean_{i=1..126} ||p_{i+1} - 2 p_i + p_{i-1}||^2
  + 0.2 * L_boundary                   # 端点硬约束的软版本（现有 boundary 逻辑）
  [可选] + w_safe * L_safe(椭圆/走廊)   # 若启用安全标签监督
```

平滑项用**二阶差分**（用户指定形式），系数 0.1 与点模型的 `lambda_smooth` 生态一致，训练前会做一次 1 epoch 的
量级核对（保证 `L_smooth` 与 `L_x0` 同量级，防止被淹没）。

### 2.5 训练配方（沿用已验证的设置）

|项|lr|batch|epochs|优化器|AMP|时间估计|
|---|---|---|---|---|---|---|
|值|2e-4（cosine 到 1e-5）|16|100|AdamW(wd=0.01)|bf16 可选|**约 50–60 min**（参照 t256 的 48 min + 安全分支开销）|

## 3. 七种基线要改什么？—— **什么都不用改**

### 3.0 已确认：候选拓扑**不是可选项**

`src/models/trajsafe/planner.py` 的 forward 链是硬的：

```
C_G, C_E = SceneCNN(occ)                    # 记忆来自地图
H_S      = SkeletonEncoder(candidate_xy)    # 候选拓扑
R        = MatchBlock(H_traj, H_S, h_t)
pi       = TopologyHead(R, candidate_mask);  m = m* / argmax(pi)
H_Gamma  = safety_query_head(H_S[m])        # 安全查询 = 沿"被选中候选"的骨架
c_i      = Γ_m(s_i),  s_i = i/(Q-1)         # 查询锚点 = 该候选的弧长参数
H_ell    = EllipseGeometry(H_Gamma, c_i, C_G, h_t, ab)
```

即"沿一条候选路径取 Q 个锚点，逐点问椭圆形状"。没有候选就没有 `H_S[m]` 与 `c_i`，
安全分支无从定义 ⇒ **候选必须保留**（早期把它当"可选"是我的判断错误）。

**已实现并通过冒烟测试**：`models/safety_points.py::SafetyPointDiffuser`，5.396 M 参数，
输出 `points (B,128,2)` / `coarse` / `pi (B,4)` / `shape4{raw,shape4,a,b,theta} (B,128,·)` /
`ellipse_attention (B,128,128)`，端点钉死、`select_index` 专家选择生效。

### 3.1 改点表示后必须对应修改的六处（照抄 S-C48 损失但逐条适配）

|S-C48（控制点）|新模型（原生点）|
|---|---|
|状态 `Q∈R^{48×2}`，`curve = B Q`|状态 `p∈R^{128×2}`，**`curve = p`**，无 B 样条解码|
|`lambda_control` 作用在 predicted `Q`|改名 `lambda_points`，作用在 predicted `p`（128 点）|
|`lambda_coarse` 作用在粗阶段 `Q`|作用在粗阶段 `p`|
|`lambda_smooth` = 控制多边形二/三阶差分（**控制点的光滑 ≠ 曲线的光滑**）|**改为轨迹点按索引参数的二阶差分**：`mean‖p_{i+1}-2p_i+p_{i-1}‖²`（+jerk 项），系数 0.08 / acc 0.25 / jerk 1.0 照旧|
|`lambda_alm = alm_corridor_loss(curve, cell_a, cell_b, anchor_s, ...)`，`curve = B Q`|`alm_corridor_loss(p, ...)` **直接传点**，无需解码（更简单，且语义完全一致：走廊约束作用在交付曲线）|
|端点：`hard_control_endpoints` 钉 48 控制点里的首末 2 个（**解码后端点仍精确**，因为 B 样条端点插值）|`pin_endpoints` 钉 128 点里的首末 2 个（本身即端点，不需要插值性质）|
|`lambda_shape / lambda_iou / lambda_safe` 作用在 `shape4`（Q=128 锚点，沿候选弧长）|**完全不变**：它们只依赖候选几何 + 地图，与扩散状态表示无关|
|`SafetyControlFusion` 的 C = 48|C = **128**（查询集从控制点换成轨迹点）|
|`lambda_feedback_safe / lambda_curve_smooth`（仅 rollout 时有效）|**不启用**（`rollout=false`，与 S-C48 同口径）|

参数、日程、损失权重：`lambda_points 0.2 / coarse 0.5 / smooth 0.08 / boundary 0.2 / topology 0.25 /
shape 0.08 / iou 0.25 / safe 0.15 / alm 0.3`，`ellipse_safe_res 64 / mask_tau 10 / cvar_fraction 0.2`，
T=**256**（你已确认），100 epoch。

这一点是本次转向的主要收益：

|方法|要求的约束形式|Native Point 下|
|---|---|---|
|RoS / ReS / TVS|`J` 按点对角（每点一行）|`NativePointTask` 已是该形式，且带**精确可分离 QP**（`cbf_solve`）|
|Projected|投影到 `C`|`NativePointTask.project` 逐点投影，天然可行|
|Primal-Dual / ALM|`Jᵀλ` 逐点|同上，乘子系数沿用点空间推导本来就是对的|
|PDM|App.C.3 **整段线段**约束|`segment_constraints_for: ["PDM"]` → `SegmentPointTask` **已实现并测过**（`results/review_fix_paper_pdm_t256/`）|

即：**控制器/投影器/整段约束都不需要新的求解器**，这正是"原生点表示"的核心价值。

## 4. 需要你确认的三件事

|#|问题|我的建议|
|---|---|---|
|1|日程 T|**256**（理由见 §2.3）。若你坚持 16 以对齐 S-C48，则 Langevin 类方法会重新受 12.4 m 终端噪声尺度限制|
|2|是否加候选拓扑条件|**先不加**（单变量：只加安全几何分支），第二轮再加|
|3|是否加安全损失 `L_safe`|**先加 0.2 权重**，与 S-C48 对齐；同时保留一个 `w_safe=0` 的对照，明确"安全特征是结构还是监督带来的"|

## 5. 验收标准（训练前先定好，避免事后挑指标）

|指标|门槛|对照|
|---|---|---|
|DDIM-256 曲线碰撞|**≤ 4/20**|现 best.pt 7/20；S-C48 2/20|
|成功率|**≥ 16/20**|现 13/20；S-C48 18/20|
|平均最小净距|**≥ +2.0 m**|现 −0.6 m；S-C48 +3.5 m|
|长度比 / 最大曲率|≤ 1.05 / ≤ 5 1/m|现 0.99 / 12.5|
|七方法可运行性|**7/7 返回，无新求解器**|S-C48 上 4/7 失败|

## 6. 交付物

1. `experiments/safety_baselines/models/safety_points.py`（新模型，`EllipseGeometry` 移植到点 token）；
2. `configs/safety_points_t256.json` + `scripts/train_safety_points.py`（复用 `train_map_only_points.py` 的循环）；
3. 训练日志 + `training_summary.json` + 三份 checkpoint；
4. `docs/native_point_retrain_report.md`：训练曲线、验收表、与 S-C48 / 旧 best.pt 的统一对比；
5. 七方法在新基座上的 20 场景表（`configs/seven_native_points_safety_t256.json`）。

## 7. 风险

|风险|缓解|
|---|---|
|安全分支在点表示上"用不上"（走廊标签是按 128 点存的，正好对齐）|数据形状已核对：`alm_cell_*` 的第一维就是 128 点，无需重采样|
|T=256 下 100 epoch 不够|先跑 30 epoch 看验证曲线，再决定是否延长（训练可断点续训）|
|平滑损失压死细节（拐角变圆）|同时记录 `max_curvature` 与 `rmse`，若 RMSE 恶化 > 10% 则把系数降到 0.03|
|训练后七方法仍不改善|那才是真正的"安全约束对该表示无效"，届时结论有据可依|
