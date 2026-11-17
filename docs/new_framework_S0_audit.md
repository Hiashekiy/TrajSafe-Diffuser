# TrajSafe-Diffuser 新框架 S0 现状审查

> 依据：《TrajSafe-Diffuser 新框架整体改造实施方案.md》§11 S0（L651-667）与 §14（L812-831）
> 性质：**只读审查**。本文件是本次唯一产物，未修改任何 `.py`。
> 分支：`v4-map16-fbdelta-safeloss`；HEAD `b30c2d5`。

## 0. 审计快照与"工作树在动"的警告

审计时工作树**正在被并发编辑**（`planner_v5.py` mtime 01:19:35、`carla_spline_dataset.py` mtime 01:20:04，而审计时间 01:20:22）。因此本文件的每个行号都绑定到下表的 md5 快照；若行号漂移，以 md5 + 行数为准重新定位。

| 文件 | md5(前8) | 行数 |
|---|---|---|
| `src/models/trajsafe/planner.py` | 5F30034C | 787 |
| `src/models/trajsafe/planner_v5.py` | EC89A68B | 379 |
| `src/models/factory.py` | 8F644087 | 32 |
| `src/models/trajsafe/fusion.py` | 50AF3D5C | 113 |
| `src/models/trajsafe/heads.py` | 50F9B59D | 121 |
| `src/models/trajsafe/encoders.py` | F7C3E9F0 | 102 |
| `src/models/trajsafe/boundary.py` | 8D3C8C43 | 121 |
| `src/datasets/carla_spline_dataset.py` | B38614F5 | 310 |
| `train.py` | 487D7C27 | 1070 |
| `src/diffusion/sampler.py` | 30C69491 | 971 |
| `src/geometry/safety_corridor.py` | F813342B | 525 |
| `src/geometry/bspline_constraints.py` | 3114EF18 | 468 |
| `src/diffusion/bspline_alm.py` | 86F44C3F | 312 |
| `scripts/data/carla_full/06_refit_controls.py` | 441D051E | 333 |
| `scripts/data/carla/01_build_candidates.py` | 7D87225D | 297 |
| `scripts/data/carla/02_build_ellipse_labels.py` | 9708DE69 | 436 |
| `scripts/data/new_framework/10_make_c32_dataset.py` | 70B81948 | 219 |

**本次审查最意外的事实（改变了 S0 的结论形态）**：工作树里已经存在一套**未提交的新框架实现**，不是从零开始：

- `src/models/trajsafe/planner_v5.py`（`TrajSafePlannerV5`，单路径 + C32/Q64 + heading + 可学习空流）
- `src/models/factory.py`（`model.framework: v5` → V5，默认 legacy）
- `configs/config_v5_c32q64.yaml`（C=32 / Q=64 / traj_blocks=1 / final_blocks=4 / heading / rollout 4 步）
- `scripts/data/new_framework/10_make_c32_dataset.py`（C32 标签）
- `src/datasets/carla_spline_dataset.py` 已被改动，支持 `heading.npy` / `pdg_path_points.npy`（`carla_spline_dataset.py:86-89`）
- `src/diffusion/sampler_v5.py`（348 行，**审计过程中新出现**，见 §10.4）：四步推理采样器 + A* 接入 + 碰撞区间诊断

审计结束时（01:2x）`git status` 里 `src/diffusion/sampler_v5.py` 仍是 `??`（未跟踪）。本文件对它以"只读观察"记录，不视为已验收实现。

同时**缺口也很明确**：`train.py` 仍然直接 `from src.models.trajsafe import TrajSafePlanner`（`train.py:71`）并在 `train.py:792` 构造 legacy 模型，**没有任何地方调用 `build_model`**（全仓库 grep 只命中 `factory.py` 自身）；`13_build_ellipse64_labels.py` 被 `10_make_c32_dataset.py:18-20,208-210` 引用但**并不存在**；`data/processed/160m-k4p-c32q64` **不存在**。

---

## 1. 当前真实网络结构（`planner.py`）

### 1.1 `__init__` 逐段与配置键

| 段 | 位置 | `model_cfg` / `bspline_cfg` 键 | 默认 |
|---|---|---|---|
| C（控制点数）| `planner.py:85-93` | `bspline.num_controls`、`model.num_controls`（两者都有则必须相等，否则 `ValueError`）| 32 |
| 曲线密度 `curve_points`/`horizon` | `planner.py:96-99` | `bspline.curve_points` → `model.horizon` | 128 |
| 安全查询数 `num_safety_queries` | `planner.py:101-102` | `model.num_safety_queries` | = `curve_points` |
| 相对位置 bias 表长 | `planner.py:106-108` | `model.rel_bias_len` | `max(C, Q, H)` |
| 架构开关 | `planner.py:110` / `111` / `112` | `model.control_space` / `model.path_features.enabled` / `model.safety_query.center_encoding` | True |
| 宽深 | `planner.py:116-121` | `d_model` 128、`num_heads` 4、`traj_blocks` **8**、`final_blocks` **3**、`skeleton_blocks` 2、`ffn_dim` 512 | |
| 地图 | `planner.py:122-124` | `map_res` 256、`global_res` ← `global_mem_res`/`mem_res` 16 | |
| 安全分支总开关 | `planner.py:145-146` | `model.safety_features.enabled` | True |

模块清单：

- `bspline = BSplineCodec(...)` `planner.py:154-161`；`traj_to_control`（仅 legacy LS 投影）`planner.py:163`
- `scene_cnn` `planner.py:165-166`（唯一的图记忆，输出 16×16 token，见 `scene_tokens` `planner.py:263-268`）
- 三个 PE `planner.py:169-171`
- **控制分支**：`traj_encoder`（类名 `TrajectoryEncoder`，实际作用在 C 个控制点上，见 `encoders.py:40-49` 的说明）`planner.py:176-178`；`traj_backbone = traj_blocks × TrajBlock` `planner.py:179-184`；**唯一** `head_p = Linear(D,2)` `planner.py:186`（coarse 与 final **共用**）
- **反馈分支**：`feedback_encoder` / `feedback_fusion` `planner.py:192-203`（`model.feedback.*`）
- **骨架/拓扑分支**：`skeleton_encoder` `planner.py:206-209`、`match_block` `planner.py:210-211`、`topology_head` `planner.py:212`、`path_feature_head`（`path_features.enabled` 时）`planner.py:213-214`
- **安全分支**：`safety_query_head`（复用 `PathFeatureHead` 类）+ `safety_cross_attention`（`SafetyControlFusion`）`planner.py:218-221`；`safety_features.enabled=false` 时二者为 `None` `planner.py:222-224`
- **边界**：`boundary_decoder` `planner.py:227-230`（无参数、固定 profile）
- **椭圆分支**：`ellipse_geometry` + `ellipse_shape_head` `planner.py:233-241`（或 `None` `planner.py:242-245`）
- **融合/去噪**：`fusion_mlp(num_streams = 3/2/1)` `planner.py:248-251`；`final_denoiser(final_blocks)` `planner.py:252-255`
- 固定进度 buffer `fixed_progress = linspace(0,1,Q)` `planner.py:259-260`

### 1.2 各文件职责（含"名不副实"的地方）

- `fusion.py`
  - `FusionMLP` `fusion.py:28-61`：仅做**通道拼接** `[h_traj; h_path; h_ell]`（3D）→ `LN → 2D → GELU → D`；`num_streams ∈ {1,2,3}` 在**构造期**定死，改它会改 `LayerNorm/Linear` 宽度（`fusion.py:44-48`）。
  - `SafetyControlFusion` `fusion.py:64-93`：C 个控制 token 对 Q 个安全 token 做交叉注意力，输出层零初始化（`fusion.py:85-86`）→ 起点等价恒等。
  - `FinalDenoiser` `fusion.py:96-113`：N_F 个**独立参数**的 `TrajBlock`，无轨迹/椭圆交错 token 流。
- `heads.py`
  - `TopologyHead` `heads.py:37-74`：`R[B,M,H,D]` → 对 token 轴 mean → 打分 → masked softmax `pi[B,M]`（全 invalid 行为 0）。
  - `PathFeatureHead` `heads.py:77-100`：纯特征变换，**不是**进度分布（`heads.py:80-86` 明确弃用了旧的 cumulative softplus 进度 MLP）。同一个类被复用成 `safety_query_head`（`planner.py:219`）。
  - `EllipseShapeHead` `heads.py:103-121`：`[raw(4)] → [log a, log b, cos2θ, sin2θ]`。
- `encoders.py`
  - `CoordMLP` `encoders.py:26-37`；`TrajectoryEncoder`（实为 Control Encoder，索引 PE 现算，所以与 token 数无关）`encoders.py:40-62`；`SkeletonEncoder`（只看坐标，永不看 `h_t`/`C_G`）`encoders.py:65-102`。
- `boundary.py`
  - `boundary_targets` `boundary.py:39-49`；`BoundaryDecoder` `boundary.py:52-121`：固定 profile `[1,0.75,0.5,0.25]`（`boundary.py:36`），`span = min(4, C//2)`（`boundary.py:87`），把 `Q0/Q[-1]` **构造性地**钉到 start/goal，邻点同向平移（`boundary.py:111-121`）。因为端点钳制（clamped knots），这等价于曲线端点条件。

**注意（S2 的关键依赖）**：`head_p` 只有一个（`planner.py:186`），coarse 与 final 是两个**不同的前向**共用同一线性头；删 `q_coarse` 不需要删参数，只需删那条 forward 支路与全部 loss/where 使用点。

---

## 2. `q_coarse` 的全部引用

### 2.1 产生（3 处 / 2 条链）

| 位置 | 代码 |
|---|---|
| `planner.py:477` | `q_coarse_raw = self.head_p(h_ctrl)` |
| `planner.py:478` | `q_coarse = self.boundary_decoder(q_coarse_raw, cond)` |
| `planner.py:479` | `coarse = self.bspline.decode_controls(q_coarse)` |
| `planner.py:630-632` | legacy 链：`coarse_raw = head_p(h_traj)` → `traj_to_control` LS 投影 → `coarse` |
| `planner.py:22, 630` | docstring 中的链路描述（非代码） |

### 2.2 参与 loss

| 位置 | 用途 |
|---|---|
| `train.py:457` | `l_coarse = control_x0_loss(out["q_coarse_raw"], q0)` |
| `train.py:464-465` | `boundary_coarse_weight × boundary_control_loss(out["q_coarse_raw"], ...)` |
| `train.py:550` | 第二步：`w2 × control_x0_loss(out2["q_coarse_raw"], q0)` |
| `train.py:552-554` | 第二步 boundary 的 coarse 项 |
| `scripts/debug/profile_step.py:101` | profile 脚本里的 coarse 项 |
| `tests/test_control_space.py:312`、`tests/test_model.py:395`、`tests/test_bspline_carla.py:179` | 单测/回归里的 coarse loss |

### 2.3 参与 `torch.where` 回退（**这是删不干净就会残留语义的地方，共 2 处 runtime**）

| 位置 | 回退语义 |
|---|---|
| `planner.py:505-506` | `fallback = self._resample_curve(coarse, Q)`；`center = torch.where(has_cand[...,None,None], center, fallback)` —— **椭圆中心**在"无候选"时退回 coarse |
| `planner.py:535` | `final = torch.where(has_cand[...,None,None], final, coarse)` —— 最终轨迹退回 coarse |
| `planner.py:647`、`planner.py:664` | legacy 链的同一对 where |

### 2.4 仅日志 / 可视化 / 断言

| 位置 | 用途 |
|---|---|
| `planner.py:555-558`、`669-671` | 写进 `out` dict（`q_coarse_raw`/`coarse_raw`/`q_coarse`/`coarse`）|
| `planner.py:754` | `assert out["q_coarse"].shape == (B,C,2)` |
| `sample.py:105` | 画 `step["coarse"]` |
| `sample.py:282`、`296` | 存 trace / 图注 "orange=coarse" |
| `tests/test_model.py:80,82,100,101,140`、`test_control_space.py:153`、`test_dc48_model.py:141`、`test_alm_state_machine.py:139`（手工构造 `"q_coarse": straight`）| 断言 / 造桩 |
| `configs/config.yaml:8`、`config_160.yaml:15`、`config_160k8.yaml:15`、`config_160k8p.yaml:29` | 配置注释里的链路图 |
| `src/models/trajsafe/planner_v5.py:19` | 新链 docstring 明说"没有 q_coarse" |
| `src/models/factory.py:4` | "legacy, q_coarse chain" |
| `README.md`、`docs/项目架构说明.md`、`docs/项目总报告.md`、`docs/控制点空间改造记录.md`、`docs/历史安全反馈说明.md` | 文档（各 1-2 处） |

**结论**：删 `q_coarse` 是 **planner.py 12 处 + train.py 4 处 + 2 个 where 回退 + 8 个测试文件**的工作量。技术上删除无阻碍（`head_p` 保留、`boundary_decoder` 保留），但 §2.3 的两处 where 回退必须换成"无路径就不建椭圆/直接输出 final"，否则删掉 coarse 会留下 `has_cand=false` 分支无处可退。

---

## 3. Topology / 候选骨架的全部引用

按用途分类（`M` = `topology.num_candidates` = 4 in 现有数据，见 `data/processed/160m-k4p-c48/train/candidate_xy.npy` 形状 `(3850,4,128,2)`）：

### 3.1 模型内部（训练/推理共用）
- `planner.py:212` `self.topology_head = TopologyHead(...)`
- `planner.py:213-214` `path_feature_head`
- `planner.py:359-387` `_route_topology(r, topo, candidate_mask, select_index)`：`argmax(pi)` / 显式 `select_index` / 负数 = 该样本走 argmax（**per-sample 冻结路由**，`planner.py:368-374`），无候选 → slot 0（`planner.py:384-385`）
- `planner.py:482-489` 前向里的骨架链：`h_s → r → topo → idx → r_use → h_path`
- `planner.py:497-498` `gamma = geometry[ar, idx]`（安全查询挂在被选骨架的 dense chain 上）
- `planner.py:637-644` legacy 链的同一套
- `heads.py:37-74` `TopologyHead` 本体

### 3.2 训练
- `train.py:446-447` `forward_all(..., select_index=best)`（专家路由 m*）
- `train.py:434` `best = batch["topology_best"]`
- `train.py:466` `l_topo = topology_ce(out["topo"]["pi"], best, has_cand)`
- `train.py:320, 337-344` 第二步路由 `select2`（`expert`=best / `pi`=argmax）
- `train.py:345-346, 349-362` `topo_match` / `corridor_fit`（诊断：选定骨架是否落在离线走廊里）
- `train.py:522-525` `Ltopo` 进入 loss 字典

### 3.3 推理（采样器）
- `sampler.py:430-436` `sample(...)` 的 `candidate_xy / candidate_mask / geometry / geometry_lengths` 形参
- `sampler.py:593` `has_cand = candidate_mask.any(dim=1)`
- `sampler.py:597` `tried`（每样本已试过的候选）
- `sampler.py:668` `sel = where(guided, frozen_idx, -1)` —— **冻结行必须保住自己的骨架**
- `sampler.py:687` `top1 = out["topo"]["pi"].argmax(-1)`
- `sampler.py:688-725` `try_activate_corridor(...)`（`sampler.py:166-184` 签名），成功则写 `frozen_idx`、重建 pack、重置 `lam`
- `sampler.py:702-709` `topology_fallback` 诊断（fallback = 最终选中的骨架 ≠ top1）
- `sample.py:228-231`、`evaluate.py:194-196, 211-213` 传参

### 3.4 评估
- `evaluate.py:185-196` `--max-candidates` 消融（把 M 裁到 1 再采样）
- `evaluate.py:217` `best = batch["topology_best"]`，`221-222` dense geometry
- `evaluate.py:273-278` 选中的候选 vs GT 的 nDTW
- `evaluate.py:375-376` 汇总里报 `M`
- `scripts/eval_random_endpoints.py`(4)、`scripts/compare_modified_geometry.py`(3)、`scripts/grid_search/extract_diffusion_x0.py`(2)、`scripts/ch4_plot.py`(3)、`scripts/dashboard_app.py`(4)

### 3.5 数据
- `scripts/data/carla/01_build_candidates.py` 产出：`candidate_xy[N,M,128,2]`、`candidate_mask[N,M]`、`candidate_lengths`、`candidate_geometry[P,2] i16`、`candidate_geometry_offsets`、`candidate_geometry_lengths`、`topology_best[N]`、`candidate_branch_counts`（docstring `01_build_candidates.py:8-15`）
- `scripts/data/carla/02_build_ellipse_labels.py:224-229` 读 `candidate_mask/offsets/glen/geom/topology_best`；`256-263` 用 `m*=topology_best` 的 dense chain 定椭圆中心 → **椭圆标签是"候选骨架的产物"**，不是"路径的产物"
- `scripts/data/carla_full/04_build_alm_constraints.py`（走廊缓存，1 处）
- `src/datasets/carla_spline_dataset.py:74-80` 加载全部候选数组；`:216-224`（快照 `B38614F5`）返回 `candidate_*`/`topology_best`/`has_candidate`
- `scripts/data/carla/03_validate_processed.py`（5 处，契约校验）

### 3.6 测试
`tests/test_model.py`(15)、`tests/test_alm_state_machine.py`(20)、`tests/test_feedback.py`(14)、`tests/test_dc48_model.py`(7)、`tests/test_dataset.py`(3)、`tests/helpers.py`(3)、`tests/test_control_space.py`(3)、`tests/test_hpath_ablation.py`(1)

**结论**：删除候选骨架 = 拆掉 `planner.py` 的 1 个头 + 1 个 match 分支 + 1 个路由函数、`sampler.py` 的**激活机制整体**（168 行）、`train.py` 的 4 处、数据集 8 个数组、2 个数据脚本、7 个测试文件。**`sampler.py` 的"激活"语义与候选骨架是强耦合的**：一旦没有 M 个候选，"试哪个骨架"这个问题就消失，走廊必须改成"从唯一路径直接建"。

---

## 4. 新单路径输入要改的接口（签名 + 张量形状）

现状（`planner.py`，全部是 `M` 个候选的批）：

| 接口 | 位置 | 签名 / 形状 |
|---|---|---|
| `forward_all` | `planner.py:390-396` | `(q_t[B,C,2], occ[B,1,R,R], cond[B,2,2], t[B], ab[B], candidate_xy[B,M,L,2], candidate_mask[B,M], geometry[B,M,G,2], geometry_lengths[B,M], select_index[B]|None, feedback_delta[B,C,2]|None, feedback_valid[B]|None) -> dict` |
| `forward_controls` | `planner.py:426-434` | 同上（`control_space=true` 的实现体）|
| `_route_topology` | `planner.py:359-361` | `(r[B,M,C,D], topo{pi[B,M]}, candidate_mask[B,M], select_index[B]|None) -> (idx[B], has_cand[B], ar[B])` |
| `match_block` | `blocks.py:145-147` | `(h_traj[B,H,D], h_skel[B,M,L,D], h_t) -> R[B,M,H,D]`（`H`=token 数，control 链里是 C）|
| `path_feature_head` | `heads.py:96`（实现 `heads.py:90-100`）| `(r_use[B,H,D]) -> [B,H,D]` |
| `safety_query_head` | 实例 `planner.py:219`，调用 `planner.py:500` | `(h_s[ar,idx][B,L,D]) -> H_gamma[B,Q,D]` |
| `ellipse_geometry` | `ellipse.py:67-70` | `(h_path[B,H,D], center[B,H,2], map_tokens[B,N_G,D]|None, h_t, ab[B]) -> (H_ell[B,H,D], A_E|None)`；`H` 在这里是 Q |
| `iter` 侧证据 | `planner.py:500-509` | `h_gamma[B,Q,D]`、`center[B,Q,2]`、`h_ell[B,Q,D]`、`shape4[B,Q,4]` |
| 融合 | `planner.py:512-515` | `SafetyControlFusion(query[B,C,D], h_ell[B,Q,D], h_t) -> A[B,C,D]`，再 `FusionMLP(h_ctrl, h_path, A)` |
| 形状断言 | `planner.py:740-773` | 全部形状契约所在，改造时是主要回归网 |

**新框架的目标签名已经在 `planner_v5.py` 里存在**（可作为 S2 的合同）：

- `forward(q_t[B,C,2], occ[B,1,R,R], cond[B,2,2], t[B], ab[B], active_path[B,P,2]|None, path_valid[B]|None, feedback_delta[B,C,2]|None, feedback_valid[B]|None, heading[B,2,2]|None, heading_valid[B,2]|None, use_safety=True)` `planner_v5.py:287-295`
- 单路径编码：`path_encoder(ap[:,None]) -> [B,1,L,D]`，`match_block` 后取 `[:,0] -> [B,C,D]` `planner_v5.py:330-337`
- 安全查询：`resample_tokens(h_path_tokens[B,L,D], Q)` → `safety_query_head` `planner_v5.py:344-346`
- 端点 heading：`apply_heading` / `hard_control_endpoints` `planner_v5.py:237-284`

**改造要点（最小改动面）**：`forward_all` 保留旧签名做 legacy 复用（`planner.py:415-423` 已按 `control_space` 分派），新增路径走新签名；`_route_topology` 整体删除而不是改造（单路径没有 M 维）；`match_block` / `path_feature_head` / `ellipse_geometry` **不需要改**（只吃 `[B,H,D]`，与 M 无关）；`safety_query_head` 需要新增一次 `resample_tokens`（legacy 里是 `h_s[ar,idx]` 本来就是 L=Q 长度，见 `planner.py:500` + `geometry` 的 `G=Q` 约定），否则 Q≠L 时形状直接崩。

---

## 5. 第一阶段"无路径、无安全特征"前向的落点

三条现有机制，能力各不相同：

1. **构造期开关（legacy，能力最弱）**
   - `path_features.enabled` → `path_feature_head = None` `planner.py:213-214`，`FusionMLP(num_streams=2 或 1)` `planner.py:248-251`
   - `safety_features.enabled` → 椭圆/交叉注意力**根本不构造** `planner.py:218-224, 242-245`，`FusionMLP(2 流)` `planner.py:248-251`
   - 运行时前向走 `planner.py:516-526`，并且**输出零占位** `planner.py:537-549`（注释明确写 "ZERO placeholders … NOT predictions"）；`_assert_common_shapes` `planner.py:732` 也照旧断言
   - 也就是说：**legacy 的"缺省流"就是全零张量冒充**，只是把模块整体删掉以后零占位只用于给共享管线（sampler/trace/dashboard）保键位。
2. **运行时开关（legacy 只有 feedback 支持）**
   - `feedback_features` `planner.py:307-348`：`valid=0` 时 delta 先被清零再编码（`planner.py:345-347`），`FeedbackFusion` 是门控残差 → **精确恒等**。这是唯一"运行时缺省流不靠零张量"的例子（严格说是"零特征 + 门控"，但仍然会计算 encoder）。
3. **可学习空 token + absent 嵌入（`planner_v5`，能力最强，已是正确落点）**
   - `path_null` / `safety_null` / `absent_emb` 三个**可学习参数** `planner_v5.py:180-185`
   - `has_path = active_path is not None` `planner_v5.py:328`；无路径 → `h_path = path_null.expand(B,C,D)` `planner_v5.py:339`
   - `has_safety = has_path and use_safety` `planner_v5.py:341`；无安全 → `h_safety = safety_null.expand(B,C,D)` `planner_v5.py:354`
   - 融合后加 absent 偏置 `planner_v5.py:357-360`
   - `FusionMLP` 固定 3 流 `planner_v5.py:187`：**宽度不变**，所以三阶段（无/半/全）共享同一套权重，这正是"第一阶段不是零张量冒充"的实现方式
   - `FusionMLP` 本身没有"缺省流"分支（`fusion.py:50-61` 只能按 `num_streams` 拼接，且 `h_path is None` 直接 `ValueError` `fusion.py:54-55`）→ **不要指望在 fusion 里开缺省流**

**结论**：第一阶段的前向落点 = `TrajSafePlannerV5.forward(active_path=None, use_safety=False)`。`train.py` 现在没有任何地方传这两个参数（`train.py:446-447` 走 legacy 签名），所以 S2 必须先接 `factory.build_model`。legacy `planner.py` 不能承担这个任务——它的 `path_features/safety_features` 是构造期开关，训练时"随机缺流"会把两种架构混在一个 ckpt 里。

---

## 6. 原始 un-eroded occupancy 可用性

### 6.1 路径存在性（实测）

| 路径 | 状态 |
|---|---|
| `data/carla_processed_160` | **不存在** |
| `data/carla_processed_160k4p` | **不存在** |
| `data/processed/160m-k4p-c48` | 存在（train 11550 / val 4650 / test 1260 个 `_cache` npz）|
| `data/raw/160m-256px` | 存在（115 scenes、5820 tasks、`occupancy_train`/`occupancy_full`）|
| `data/raw/80m/samples/*/*.npz` | 存在（train/val/test）|

**你的初判正确**：唯一可用的 processed 快照只有 k4p（已腐蚀 4 格）。

### 6.2 原始图为**未腐蚀**，可 100% 重建

- `data/raw/160m-256px/scenes/scene_0001.npz` 实测键：`occupancy_full (640,640) uint8`、`occupancy_train (256,256) uint8`（未腐蚀，含边缘带）
- `scripts/data/carla_full/00_build_processed.py:209-211`：
  `with np.load(_scene_npz(scene_id)) as z: occ_cache[scene_id] = np.flipud(z["occupancy_train"]).copy()` —— 即 processed 的 `occupancy.npy` **就是**未腐蚀原图（canonical flipud）
- 腐蚀发生在**另一个脚本**：`data/processed/160m-k4p-c48/erode_report.json` 记录 `source=…carla_processed_160`、`out=…carla_processed_160k4p`、`k=4`、`border_mode=protect`、`free_before_mean=0.1864 → free_after_mean=0.2830`（即 free 面积被"放大"约 52%）

**结论**：§13 第 17 条（"不得声称恢复了被腐蚀的障碍边界"）在数据上是可满足的——原始边界从没丢。可选两条路：
1. 重跑 `00_build_processed.py`（`--source data/raw/160m-256px --out …`，需要重新拟 control/C 与全部下游标签，最贵）；
2. **只换 occupancy 来源**：用现有 `clean_manifest.jsonl`（1.6 MB，`data/processed/160m-k4p-c48/clean_manifest.jsonl`）恢复"样本 → scene_id"映射，把 `flipud(occupancy_train)` 重新写进新 root。这条路不碰任何标签，S1 的"正确膨胀"必须走它。

⚠ 注意 `erode_report` 的 `border_mode=protect`：k4p 把外圈 0..3 环恢复成原图，所以 k4p 与原图**不是**简单的"嵌套"关系（`docs/历史实验记录_160k8p.md` 第 9 节亦有记）。用它做膨胀基准会引入一条 4 格宽的"免费走廊"。

---

## 7. 160m heading 数据可用性

### 7.1 processed 快照里到底有什么（`data/processed/160m-k4p-c48/train/`，实测形状）

```
occupancy                 (3850, 256, 256) uint8
control_gt                (3850, 48, 2)    float32
curve_gt                  (3850, 128, 2)   float32
candidate_xy              (3850, 4, 128, 2)
candidate_mask            (3850, 4)        bool
candidate_lengths         (3850, 4)        float32
candidate_branch_counts   (3850, 4)        int32
candidate_geometry        (2383292, 2)     int16
candidate_geometry_offsets/lengths, topology_best, ellipse_shape4_gt (3850,128,4),
shape_valid (3850,128) bool, alm_anchor_s (3850,128),
alm_cell_a (3850,128,14,2), alm_cell_b (3850,128,14), alm_cell_valid, alm_valid,
conditions (3850,2,2), sample_id, episode_id
_cache/alm_*.npz  (idx, a, b, cell_offsets, cell_valid, anchor_s, valid, reason)
```

**没有**：原始轨迹点列、时间戳、速度、yaw、heading。

### 7.2 上游 raw 里有什么

- `data/raw/160m-256px/tasks/scene_0001/task_0001.npz` 实测键：
  `start(2)`, `goal(2)`, `trajectory_128(128,2)`, `bspline_controls(32,2)`, `bspline_dense_512(512,2)`, `planner_path(200,2)`, `pair_id`, `variant_id`, `scene_id`
  → **同样没有原始轨迹/时间戳**；但有两样值得注意：`bspline_controls(32,2)`（现成的 C=32！）与 `planner_path(200,2)`（参考规划路径，可当路径条件/切线代理的额外来源，注意它没有被 `00_build_processed.py` 写进 processed）。
- `data/raw/80m/samples/train/sample_00000001.npz` 实测键：
  `occupancy(256,256)`, `trajectory_128(128,2)`, `bspline_controls(32,2)`, `bspline_dense_512(512,2)`, `reference_route_local(55,2)`, **`trajectory_raw_local(332,2)`**, `start`, `goal`, `episode_id`, `raw_start_idx`, `raw_end_idx`
  → 80m **有**原始轨迹（332 点 + 起止索引），可以真正"从原始轨迹拟合 heading"。

### 7.3 结论

- **160m（当前训练集）：heading 只能从 GT 曲线切线代理。** 具体可取 `curve_gt[:,1]-curve_gt[:,0]`、`curve_gt[:,-1]-curve_gt[:,-2]`，或用 `bspline.dense_derivative`/`derivative_op`（`src/geometry/bspline.py:287`）在 u=0/1 求精确切线（更稳，因为 GT 曲线是从 B 样条采样出来的）。
- **80m 可以拟合真实 heading，但 80m 不是当前训练集**（没有 processed 快照、没有 candidate/ALM 标签、crop 也小一半），把它当作 heading 监督源会引入第二个数据集偏差，不建议作为 S3 的方案。
- `planner_v5.apply_heading` `planner_v5.py:237-284` 已经把 heading 定义成 `[B,2,2] = (t0, t1)` 单位向量 + `ell` 几何长度（`planner_v5.py:229-235`），**数据契约已定**，只差 `heading.npy`/`heading_valid.npy` 生成脚本（dataset 侧已就绪：`carla_spline_dataset.py:86-87, 237-241, 284-287`）。

---

## 8. C=32 与 Q=64 标签生成方案

### 8.1 `scripts/data/carla_full/06_refit_controls.py`（C 无关的完整重刷）

- 输入：一个 processed root（`--source`），只要求 `curve_gt`、`conditions`、`control_gt` + 全部 C 无关数组（`06:191-200`）
- 输出：新 root（`--out`），**只重写 `control_gt.npy`**（`REFIT_FILES = ("control_gt.npy",)` `06:69`），其余按 `COPY_FILES` `06:71-80` 逐字节复制（含 `candidate_*`、`topology_best`、`ellipse_shape4_gt`、`shape_valid`、`alm_*`）
- 核心拟合：`numpy_fit_curve_to_controls(knots, C, degree, curve_i, start_i, goal_i, H)` `06:206-209`（端点约束 LS 投影，与 `00_build_processed.py --fit` 同一 codec/同一参数化）
- CLI：`--source`（必填）、`--out`（必填）、`--config`（必填，提供目标 C/H/degree/knots）、`--overwrite` `06:248-256`
- 是否依赖候选标签：**计算上不依赖**（只用 curve/cond），但**产物里会把候选/拓扑/椭圆/ALM 标签原样带过**（`06:74-79`）—— 所以 C32 产物里仍是 `M=4`、`Q=128`
- 自证：`06:29-31` 记录"用自身 `curve_gt` 重拟 C=32 可复现原 32 标签到 3e-8"

### 8.2 `scripts/data/carla/01_build_candidates.py`（候选骨架）

- 输入：processed split（`occupancy`/`conditions`/`curve_gt`）+ `--config` 的 `topology.*`/`skeleton.*`
- 输出（docstring `01:8-15`）：`candidate_xy[N,M,128,2]`、`candidate_mask[N,M]`、`candidate_lengths`、`candidate_geometry[P,2] i16`、`candidate_geometry_offsets[N,M+1]`、`candidate_geometry_lengths[N,M]`、`topology_best[N]`（`argmin_m nDTW(curve_gt, S_m)`）、`candidate_branch_counts[N,M]`
- CLI：`--processed`、`--config`、`--splits`、`--workers`、`--limit`、`--resume/--no-resume`、`--force` `01:208-215`
- 是否依赖椭圆标签：**不依赖**

### 8.3 `scripts/data/carla/02_build_ellipse_labels.py`（Q 个椭圆）

- 输入：`occupancy`、`conditions`、`candidate_mask`、`candidate_geometry_offsets/lengths`、`candidate_geometry`、`topology_best` `02:222-229`
- 输出：`ellipse_shape4_gt.npy [N,Q,4]`（`[log a, log b, cos2t, sin2t]`）、`shape_valid.npy [N,Q]` `02:18-19, 310-311`
- 中心来源：**强依赖候选骨架** —— `m* = topology_best`（`02:256-258`）→ dense cell chain 解成 scene 坐标（`02:235-240`）→ 端点强制成 start/goal（`02:262-263`）→ `s = linspace(0,1,Q)` 上做 **arc-length** 插值（`02:273-274`）
- 形状计算：复用 `occupancy_safe / shape_at_center`（`02:64-207`，36 方向 + 二分 + 射线间隙 + 内部环）
- `Q` 从 config 的 `model.num_safety_queries` / `topology.candidate_points` 读（`02:21-22, 52-55`）
- CLI：`--processed`、`--config`、`--splits`、`--workers`、`--limit`、`--resume/--force`、`--num-orientations`、`--local-radius`、`--dilation`、`--boundary`、`--interior-rings`、`--interior-angles`、`--binary-iters`、`--min-semi-axis`（`02:321-336`）

### 8.4 已经有的一半：`10_make_c32_dataset.py`

- 默认 `--source data/processed/160m-k4p-c48`、`--out data/processed/160m-k4p-c32q64`、`--num-controls 32`、`--limit`、`--overwrite` `10:45-47, 139-147`
- 通过 `importlib` 复用 `06_refit_controls` 的 `_copy_split/_solve_knots/_decode_rmse/_copy_root` `10:15-17, 61-66, 167-171`
- 额外做 **md5 字节一致性校验**（`CHECK_FILES` `10:51-53`，`10:116-120`），输出 `c32_controls_report.json` `10:211-213`
- 自我声明缺口：`ellipse_shape4_gt/shape_valid` 仍是 **128** 查询，"由 `13_build_ellipse64_labels.py` 替换"，而该脚本不存在 `10:18-20, 208-210`

### 8.5 "椭圆 64 个沿 A* 路径"需要新写什么

必须新写 **一个脚本**（建议 `scripts/data/new_framework/13_build_ellipse64_labels.py`），它 = `02` 的椭圆数学 + `src/grid_search` 的路径 + 63 行新的中心/缓存/报告逻辑：

1. **路径来源（新）**：每样本跑一次 `build_safe_grid → build_neighbor_table → prepare_reference_path(GT 或 x0) → build_progress_guidance → pdg_focal_astar`（API 全在 `src/grid_search/__init__.py:16-33, 37-49`；现成探针 `scripts/data/new_framework/_probe.py:41-53` 就是这套顺序和计时骨架）
2. **中心采样（改）**：把 `02:256-274` 的"候选 dense chain + arc-length"换成"**A* cell path + arc-length**"（`skeleton_paths.interpolate_path` 可直接复用；cell→scene 换算 `02:51, 240`）
3. **Q=64 常量（改）**：`HORIZON` 从 config 读（`02:52-55` 已有机制），但 `_cache/ell_*.npz` 的数组形状会变 → 必须新 root，不能复用 128 的 `_cache`
4. **额外的路径产物（新）**：`pdg_path_points.npy [N,128,2]` + `pdg_path_valid.npy [N,128]` —— dataset 已经在等它们（`carla_spline_dataset.py:88-89, 242-245, 288-291`），现在没人生成
5. **口径决议（新，且是阻塞项）**：A* 用哪张图做膨胀？k4p 已腐蚀 4 格 + `erode_report` 的 `border_mode=protect` 会造 4 格免费走廊，而 `configs/config_v5_c32q64.yaml` 里 `grid_search.already_inflated_cells: 0.0`（即"当作没膨胀过"）—— 这两者互相矛盾，见 §11.3
6. **成本**：`preprocess_labels_report.json` 记录 128 查询标签在 12 进程下 train 用了 **3227 s**（3850 样本，≈0.84 s/样本），三 split 合计 ≈50-55 min；A* 每样本再叠加（`_probe.py` 是计时脚本，尚无落盘结果）

**结论**：**C=32 已经是现成的（跑 `10_make_c32_dataset.py` 即可，全量 LS 拟合是分钟级）**；**Q=64 是唯一真正"从零写"的数据工作**，且它必须和"PDG 路径缓存"一起做，否则 `config_v5_c32q64.yaml` 里 `rollout.use_pdg_path: true` 无数据可用。

---

## 9. 动态局部 corridor 与 B 样条约束包

### 9.1 `build_safety_corridor` 的输入与 region 构造

签名 `src/geometry/safety_corridor.py:346-355`：

```
build_safety_corridor(builder: EllipseRegionBuilder, centers [H,2], shape4 [H,4],
                      anchors [H], gamma [G,2]|None, gamma_lengths, config, gamma_fn)
                      -> SafetyCorridor
```

构造顺序（全部在 `safety_corridor.py`）：

1. `region_stride` 决定"每几个 anchor 共享一个 cell" `:366-368`；`keep_idx` 生成 `:385-396`（**最后一个 anchor 必独自成 cell** `:386-390`）
2. `builder.build_from_ellipse(keep_centers[1], keep_shape4[1])` → `A,b,mask,valid` `:400-402`
3. 每 cell 求顶点与面积：`halfspaces_to_vertices` + `polygon_area` `:418-419`
4. 相邻 cell 交叠判定：**直接把两 cell 的 face 拼起来求交**（凸集交的性质，不需要裁剪）`:427-434`
5. 任一 base cell 无效 → **整个 corridor 作废** `:435-439`
6. 桥接 seed（断点补 cell）`:369-372, 441-456`（`gamma_fn` 在 Skeleton 上按进度取点）

### 9.2 `build_constraint_pack` 的 anchor → 责任区间分配

- 入口 `build_constraint_pack(codec, corridors, knot_boundaries, device, dtype, margin)` `bspline_constraints.py:369-374`；`corridor is None / not valid / num_cells < 2` → 全 mask（ALM 降级为 no-op）`:386-390`
- 离线路径（训练用）：`build_constraint_pack_from_regions(codec, cell_A, cell_b, cell_mask, anchors, …, sample_valid)` `:399-409`，锚点默认 `linspace(0,1,R)` `:439-443`
- **anchor → 责任区间**：`responsibility_intervals(anchors)` `:114-135`
  `tau_0 = 0`、`tau_M = 1`、`tau_j = (s_{j-1}+s_j)/2`（中点分界）；`order = argsort(s)`（anchor 不单调会先排序）
- **区间细分**：`exact_subdivision(tau, knot_boundaries)` 把 tau 与"原始 knot 值"取并集 `:138-147`
- **piece → region**：`piece_region_assignment` 用每个 piece 的**中点** `searchsorted(tau)` `:150-157`，再 `clip(0, M-1)`
- 每 piece 一条**精确 Bezier 抽取** `E[4,C]`（`bezier_extraction` `:165-180`），约束变成 `A_j E_{l,r} Q <= b_j`（4 个 Bezier 控制点的凸包 → 连续约束）
- 组装 `_assemble_pack` `:304-366` 缓存 piece 结构（`_piece_structure` `:72-106`，`_sample_pieces` 注释说明训练里 pack 重建是主要开销 `:286-291`）

### 9.3 双变量 `lam` 的形状与来源

- `constraint_state` `bspline_alm.py:98-110`：`beta [B,P,4,2]`、`g [B,P,4,F]`、`mask [B,P,1,F]`
- `lam` 形状 = **`[B,P,4,F]`**（`bspline_alm.py:153, 205-212`）：`P` = piece 数（由 anchor 划分的区间数），`4` = 三次 Bezier 控制点，`F` = 最大面数
- 顺序来源：`pack` 里的 piece 顺序 = `_piece_structure(codec, anchors, knot_boundaries)` 的 `region` 顺序（`bspline_constraints.py:292-300`），**完全由 anchors 表决定**；`lam` 在激活时被重置（`sampler.py:718` `lam = None`），跨步继承受 `inherit_dual_across_reverse_steps` 控制（`sampler.py:748`，`config_v5: alm.inherit_dual_across_reverse_steps: true`）

### 9.4 "只对碰撞区间建局部区域"要在哪一层切

三个候选切点，只有第三个不破坏现有不变量：

| 切法 | 结果 |
|---|---|
| 在 `region_stride` 上切（`safety_corridor.py:385-396`）| ❌ 只能均匀抽稀，不能表达"只覆盖 [i0,i1]" |
| 在 `build_constraint_pack` 里按 region 过滤（`bspline_constraints.py:386-396`）| ❌ `tau_0=0, tau_M=1` 的端点假设（`:129-131`）会让未覆盖的控制点**完全没有约束**，ALM 会把轨迹尾巴甩出走廊 |
| **在 `build_safety_corridor` 之前把 `centers/shape4/anchors` 裁成局部区间，并显式携带"局部区间 → 全局 u 范围"** | ✅ 但要新增：①局部 `tau` 必须映射回全局 [0,1]（否则 responsibility 语义错）②"链式相邻交叠"检查在局部区间是断的（`safety_corridor.py:427-434` 只保证局部连续）→ §5.3 的"局部走廊连续性"需要新写 ③两端要加"收口"区域（否则局部包不约束区间外控制点） |

**可复用 / 必须新写**

- 可复用（不改一行）：`EllipseRegionBuilder.build_from_ellipse`、`halfspaces_to_vertices`、`overlap_ratio`、`region_table`（`bspline_constraints.py:253-275`）、`bezier_extraction`、`_assemble_pack`、`bspline_alm_correct`（`bspline_alm.py:135-…`）、`pack.index_select`（`:232-241`）
- 必须新写：局部区间的锚点/参数区间映射；局部连续性检查；"局部包变了就不能盲目继承旧 dual"的判定（§13 第 14 条）——现在是 `inherit_dual` 一个**全局布尔**（`sampler.py:539, 748`），没有"包变了"的检测。

### 9.5 并发出现的"局部区间"代码是否真的局部化？——**没有**

新出现的 `src/diffusion/sampler_v5.py` 里有 `_collision_intervals(hit, pad=4, min_q=8, n_query=64)`（`sampler_v5.py:157-179`），把 dense 曲线上的连续碰撞段映射成**椭圆查询下标区间** `[(qa,qb)]`，语义与 `configs/config_v5_c32q64.yaml` 的 `corridor.local.pad_queries: 4 / min_queries: 8` 一致。

但该区间**只用于诊断**，没有进入走廊构建：

- `_alt_corridor_and_alm` 建走廊时把**全部 Q 个**中心和 anchor 一次性传给 `build_safety_corridor(builder, centers, shape4, anchors, None, None, config=...)`（`sampler_v5.py:196-207`），没有任何切片；
- 局部区间只在 `step["collision"][b]["intervals"]` 里落 trace（`sampler_v5.py:337-338`），`out["queries_collision"]` 永远是 `[]`（`sampler_v5.py:186`）。

**结论**：§5.2「只在碰撞区间建局部区域」在现有代码里是**未接线**状态。要真做，仍须按 §9.4 的"入参层切片 + 局部 tau 映射回全局 [0,1]"，并把 `pad_queries/min_queries` 从 YAML 送进 `build_safety_corridor` 的 `anchors` 裁剪，而不是只做 trace。

---

## 10. 四步训练 rollout 的落点

### 10.1 现在只有**两步**（1 次 ALM + 1 次反馈前向）

- `feedback_rollout` `train.py:279-417`：
  `Q0_raw = out1["control"]` → `ALM(Q0_raw, pack)` `:365-367` → 判定 `reliable/already_ok` `:368-377` → `fb_delta = where(already_ok, 0, Q0_safe - Q0_raw)` `:375-376` → **DDIM 到 s=t-1** `:387-397` → 第二次 `forward_all(..., select_index=select2, feedback_*)` `:400-402`
- **`no_grad` 段**：`train.py:364-397`（ALM + 统计 + DDIM 全部在 `no_grad` 内），梯度只从第二次预测回流
- 时间步抽样：`rollout_timesteps` `train.py:201-212`（两步时 `t>=1`），`rollout_pair` `train.py:215-225`（**硬编码 `s = t-1`**，docstring `:221-225` 已预留"改成 `pick_times`"的位置）
- **loss 组合** `train.py:542-570`：
  - 第一步：`raw1 = {Lctrl, Lcoarse, Lsmooth, Lboundary, Ltopo, Lshape, Liou, Lsafe, Lcurve}` `:522-525`，`total1 = Σ w_k · raw1_k` `:526`
  - 第二步：`raw2 = {Lctrl, Lcoarse, Lboundary, Lcurve, (Lsafe)}` 全部乘 `step2_weight` `:548-566`；`raw[k] = raw1[k] + raw2[k]` `:567-568`；`total2 = Σ w_k raw2_k` `:570`
  - `total = total1 + total2` `:590`，但**分两段反传**（`stats["loss_parts"] = [total1, total2]` `:594`，注释 `:591-593` 说明是为了省一半激活内存）
- pack 来源：离线走廊 `alm_cell_a/b` `train.py:267-276`（`build_constraint_pack_from_regions`），与推理 ALM 的 pack 语义逐位一致

### 10.2 sampler 的状态机（每步 5 个动作）

`src/diffusion/sampler.py:658-809`，每个 reverse pair 依次：

1. **网络前向** `:671-673`，路由 `sel = where(guided, frozen_idx, -1)` `:668`
2. **激活** `:683-735`（`reverse_forward_count >= warmup` 且未 exhausted）：`try_activate_corridor` `:688-691` → 成功则 `frozen_idx/corridors/alm_status/activation_step` `:698-713`、**重建 pack + `lam=None`** `:714-718`、`topology_fallback` 诊断 `:708-709`；`max_activation_delay_steps` 超时 → `exhausted` `:726-734`
3. **ALM** `:736-750`，`budget` 按"刚激活 = activation_inner / 其余 = inner_steps / 未 guided = 0" `:739-745`
4. **反馈刷新** `:786-796`：只有 `max_violation_after <= tol` 的步才写历史（`feedback_step` `:300-330`），`fb_force_zero` 消融开关 `:631-645`
5. **DDIM** `:802-809`：`q0_ddim = q0_used`（默认）或 raw（ablation）

### 10.3 四步 rollout 的改造点清单

| # | 改造点 | 现状证据 |
|---|---|---|
| 1 | **第 1 步必须"无路径/无安全/无反馈"** → 需要模型支持运行时缺流 | legacy `planner.py` 只有构造期开关（§5）；`planner_v5.py:287-295` 已具备，但 `train.py` 没用它 |
| 2 | **第 1 步后 A\* 只跑一次** | `train.py` 全文件**没有任何** `grid_search` 相关代码（grep 无命中）；要新增路径生成 + 缓存（`pdg_path_points.npy`）|
| 3 | **第 2 步 path+椭圆，无反馈** | 可复用 `fb_force_zero` 语义（`sampler.py:635-645`），但训练侧需显式 `feedback_valid=0` |
| 4 | **第 2 步 ALM 后才产反馈** | `train.py:364-397` 的 no_grad 段可整体复用，只需把"第二步前向"前移 |
| 5 | **第 3/4 步路径更新 + 反馈** | **不存在**：`feedback_step`（`sampler.py:300-330`）只更新 delta/valid，没有"ALM 修正后重算路径/椭圆中心"的逻辑；这是全新代码（且需要 §13 第 12 条"验证通过才更新 active_path"）|
| 6 | **训练时间步抽样扩到 4 段** | `rollout_pair` 硬编码 `t-1`（`train.py:215-225`）；`rollout_timesteps` 只保证 `t>=1`（`:211-212`），四步最多需要 `t>=3` |
| 7 | **loss 组合扩到 4 步** | `train.py:542-570` 是"1 步 + 1 步"的加法结构；四步要么 4 段分别反传（内存 4×），要么分段累加后统一反传 |
| 8 | **ALM pack 语义** | 训练用**离线固定 pack**（`train.py:267-276`），推理用**动态局部 pack**（`sampler.py:714-717`）；四步训练如果引入 A* 路径就必须同时缓存"按该路径建的局部走廊"，否则训练/推理约束集不一致（违反 §13 第 18 条）|

---

### 10.4 并发出现的 `sampler_v5.py`：四步**推理**已写出，训练侧仍是两步

（`src/diffusion/sampler_v5.py`，348 行，审计期间新出现，未跟踪）

已实现（推理侧，`sample_v5` `sampler_v5.py:246-348`）：

| 步骤 | 落点 |
|---|---|
| 4 步时间表 | `pick_times(T, steps)` `:259-263`（默认 `diffusion.inference_steps: 4`）|
| 第 1 步：无路径/无安全/无反馈 | `active_path=None` 初值 `:270`；`feedback_delta=None if i<2` `:281-282` |
| 第 1 步后**只搜一次** | `i == 0` 分支逐样本 `_search_once` `:295-305`（`_search_once` `:83-140`：`pdg_focal_astar` → `focal_astar` → `astar` 三级回退）|
| 第 2 步：路径 + 64 椭圆，无反馈 | 由 `i>=2` 的反馈门控自然满足 `:281-282` |
| 第 ≥2 步：无碰撞 → 不建走廊/不 ALM/无反馈 | `:307-319` |
| ALM + **全局验证**后才更新 `active_path` | `:321-328`（`_alt_corridor_and_alm` `:182-242` 里的 `_collision_mask` 全局校验 `:216-236`）|
| 碰撞区间诊断 | `:332-338`（但未接线，见 §9.5）|
| 计时 | `net_ms / grid_ms / alm_ms / total_ms` `:284, 265, 237, 342` |

**与 legacy 的语义分歧（S5 验收时必须先裁定）**：

1. **ALM 修正没有进入 DDIM 转移**：`q = ddim_step_v5(q, x0, tb, s, sched, cond)`（`sampler_v5.py:340`）用的是 **raw `x0`**；legacy 是 `q0_ddim = q0_used`（`sampler.py:802-803`，受 `alm.safe_prediction_to_ddim` 控制）。后果：ALM 只通过"下一步的 `active_path`/feedback"间接生效，当前反向前向里"修好的轨迹"被丢掉——ALM 的消融结论会和 legacy 不可比。
2. **A\* 在 batch 维顺序执行**：`for b in range(B): _search_once(...)`（`:298-300`）+ `_build_safe_grids` 逐样本建图（`:265`），batch=16 就是 16 次串行搜索 + 16 张膨胀图，§12 要求的"端到端搜索耗时"会被 batch 维度放大。
3. **训练侧没有任何对应物**：`train.py:279-417` 仍是"1 次 ALM + 1 次前向"，`rollout_pair` 仍是 `s=t-1`（`train.py:215-225`）。即 §10.3 的 #1/#5/#6/#7/#8 全部未动，训练出来的模型仍然是"两步反馈模型"，拿它去跑四步推理会有**训练/推理步数不一致**（§13 第 18 条的同一类风险）。

**结论**：S5 的"推理骨架"已经有人在写，S6 的"四步训练 rollout"（§10.3）完全空白。今晚不要碰 §10.3。

---

## 11. 风险与硬阻塞

### 11.1 会破坏旧 ckpt 兼容（按严重度排序）

| 风险 | 证据 | 后果 |
|---|---|---|
| **`BSplineCodec.knots` / `basis [H,C]` 是 persistent buffer** | `src/geometry/bspline.py:282-283`（`register_buffer` 无 `persistent=False`；对比 `:287-289` 的 `derivative_op` 明确写了 `persistent=False`）| **改 C（48→32）或改 H 都直接破坏 `load_state_dict(strict=True)`**。旧 ckpt 只能 `strict=False` + `detect_architecture` 分流（`src/utils/checkpoint.py:69-76, 172-204`）|
| 删除 `TopologyHead` | `planner.py:212`（参数在 ckpt 内）| 旧 ckpt → `unexpected keys: topology_head.*` |
| `FusionMLP.num_streams` 变化 | `fusion.py:44-48`（宽度随流数变）| 旧 ckpt → shape mismatch |
| legacy `TrajSafePlanner` 与 `TrajSafePlannerV5` 是**两套 state_dict** | `planner_v5.py:154-208`（`control_encoder/control_backbone/path_encoder/path_head/path_null/absent_emb/fusion` vs legacy 的 `traj_encoder/traj_backbone/topology_head/path_feature_head/fusion_mlp`）| 只能靠 `model.framework` 显式分流；`factory.py:20-31` 已实现，但**没有任何调用者**（`train.py:71, 792` 仍直接 import legacy） |
| `head_p` 共用 | `planner.py:186` | 保留 `head_p` 即可让 legacy ckpt 的 final 头继续可用；但删了 coarse loss 后**指标不可比**（旧 run 的 `Lcoarse` 监督消失）|
| 训练 target 语义变化（C 48→32 / Q 128→64）| §8 | 旧 ckpt 的**所有椭圆与 ALM 相关指标**都不可比；`ellipse_shape4_gt` 行数从 128→64 |

### 11.2 需要重跑数据（不可回避）

| 数据 | 成本 | 是否阻塞训练 |
|---|---|---|
| `data/processed/160m-k4p-c32q64`（C=32）| `10_make_c32_dataset.py` 全量，LS 拟合分钟级 + 拷贝 ~1.5 GB | 是（`config_v5` 指向它）|
| `13_build_ellipse64_labels.py`（Q=64）| 约 50-60 min（对齐 `preprocess_labels_report.json` 的 3227 s/3850 样本）+ A* 开销 | **是，且是唯一"从零写"的脚本** |
| `pdg_path_points.npy` / `pdg_path_valid.npy` | 与上条同一次运行最好 | 是（`rollout.use_pdg_path: true`）|
| `heading.npy` / `heading_valid.npy` | 分钟级（GT 曲线切线）| S3 才需要 |
| 未腐蚀 occupancy 重建（若 S1 要"正确膨胀"）| 分钟级（按 manifest 回读 `scenes/*.npz`）| 是（S1 的前置）|
| 若改腐蚀口径 k 或 border_mode | 腐蚀 + 椭圆 + ALM 约束全部重算（`erode_report` 显示 free 面积 0.186→0.283，改了会整体平移）| 会成为 1-2 天级工作 |

### 11.3 硬阻塞（必须先决，不然写代码是白写）

1. **膨胀口径互相矛盾（最高优先级）**：k4p 数据**已经**被腐蚀 4 格（`erode_report.json`：`k=4, border_mode=protect, free_before_mean=0.186 → free_after_mean=0.283`），而 `configs/config_v5_c32q64.yaml` 写着 `grid_search.already_inflated_cells: 0.0`（"当作没有膨胀过"），同时又要求 `robot_radius_m: 0.8 / safety_margin_m: 0.3`。三者叠加会得到一张**过度膨胀**的图（0.8+0.3 m ≈ 3.5 格 + 已腐蚀 4 格），A* 在窄路必失败。这不是代码问题，是**必须先定的数字**。
2. **Q=64 标签与 Q=128 快照的兼容检查太弱**：`carla_spline_dataset.py:153-159` 只检查 `shape4_gt.shape[1] ∈ {num_safety_queries, candidate_points}`。Q=64 且数据仍是 128 时 → `128 == candidate_points` **通过**，错误被推迟到 loss 处（`train.py:506-518` 的 `ellipse_shape_loss/ellipse_iou_loss` 形状不匹配），甚至可能广播成静默错位。训练前必须补一条"Q 必须等于标签行数"的硬断言。
3. **`train.py` 没有接 `factory`**：`train.py:71, 792`。S2 的"接新模型"不是改一行 import —— `train.py` 还硬编码了 legacy 契约：`out["q_coarse_raw"]`（`:457, 465, 550, 554`）、`out["topo"]["pi"]`（`:466`）、`select_index=best`（`:447`）、`forward_all` 12 参数（`:446-447`）。`planner_v5.forward` 的返回值只有 7 个键（`planner_v5.py:368-379`），**loss 函数会全线 KeyError**。
4. **四步 + 局部走廊在训练侧不可离线复现**（§10.3 #8）：训练 pack 来自固定离线表（`train.py:267-276`），而"只对碰撞区间建走廊"依赖**当步的预测**（碰撞区间由 `Q0_raw` 与 occupancy 决定）。两条路都不便宜：①缓存"GT 路径的局部走廊"（与推理分布不一致）②训练里在线建走廊（`04_build_alm_constraints.py` 注释说 ~240 ms/样本，`bspline_constraints.py:412-413`）→ 4070 单卡上 batch=16 就是 3.8 s/step，直接吃掉吞吐。
5. **`sample.py` / `evaluate.py` 全部按 legacy 契约写死**（`sample.py:228-231, 282`；`evaluate.py:185-226`），且 `evaluate.py` 的 `--max-candidates` 消融（`:185-196`）在新框架下语义为空。§7 的三方对比（旧候选骨架 vs 普通 A* vs PDG-A*）**必须保留 legacy 路径可跑**，所以 `planner.py` 不能删，只能"不再被新配置引用"——这与 §10 表里"删除 q_coarse"的字面要求冲突，建议按"新链不带、旧链保留"执行。

### 11.4 建议的执行顺序（依赖倒置后的最小闭环）

1. 定膨胀口径（数字）→ 修 `grid_search` 接法（S1）→ 出 `build_safe_grid` 基准
2. 重建未腐蚀 occupancy 新 root（保标签）
3. 跑 `10_make_c32_dataset.py` → C=32 root（分钟级，可当场验证）
4. 写 `13_build_ellipse64_labels.py`（含 PDG 路径缓存）
5. 通 `factory` 把 `train.py` 接到 V5，跑 1 个 overfit batch（形状/键位全绿）
6. 才谈四步 rollout 与局部走廊

---

## 12. 今晚可做 / 不可做（假设 8 小时，单卡 4070）

工时按"1 人 + 1 agent 并行"粗略估计，扣掉评审与返工 ~25%。

### ✅ 可做（预计 5.5-7 h）

| 任务 | 预计 | 依据 |
|---|---|---|
| 膨胀口径决议 + `build_safe_grid` 一次真实基准（含端到端耗时）| 1.0 h | `scripts/data/new_framework/_probe.py:26-61` 骨架已就绪，只需定数字与跑 |
| 未腐蚀 occupancy 新 root（按 manifest 回贴）| 0.5 h | `00_build_processed.py:209-211` + `clean_manifest.jsonl` |
| `10_make_c32_dataset.py` 全量（C=32 root）| 0.5-1.0 h | 纯 LS 拟合 + 字节拷贝；`--limit 8` 先自证 |
| `13_build_ellipse64_labels.py` **写 + 小样本跑通**（`--limit 64`，不开全量）| 2.5-3.5 h | 复用 `02:64-207` 与 `grid_search`；全量留给第二天 |
| V5 形状/键位单测（3 种模式：无路径/有路径/全开；+ heading）| 1.0 h | 目前 `tests/` 里**没有任何 v5 测试**（`tests/` 29 个文件全是 legacy）|
| `train.py` → `factory` 的最小接线 + 1 个 overfit batch 跑通 | 1.0-1.5 h | 需同时补 loss 键位缺失（§11.3 #3）|
| **只读验收 `sampler_v5.py`**（用未训练/小 ckpt 跑 4 步，核对 `has_path/has_safety/feedback_valid` 的步序与 §3.1-3.5 一致）| 0.5 h | 骨架已存在（§10.4），只差"按设计核对"，不需要训练 |

### ❌ 不可做（今晚别碰）

| 任务 | 原因 |
|---|---|
| **四步 rollout 训练（S6）** | 推理侧虽有 `sampler_v5.py`（§10.4），但训练侧仍是两步 + 离线固定 pack；要 #1-4 全部就绪 + 新训练循环；`config_v5` 自己写 `epochs: 120 / max_hours: 6.0`，一晚连"跑完并验证"都不可能 |
| **局部 corridor + 局部约束包（S4）** | 需要新的"局部 anchor → 全局 u 映射" + 局部连续性 + dual 失效判定（§9.4）；`sampler_v5` 的碰撞区间目前只进 trace、未接线（§9.5）；训练侧 pack 不可离线复现（§11.3 #4）|
| **全量 Q=64 标签** | 50-60 min CPU（12 进程）会与 GPU 训练抢核；且口径未定前跑了就是白跑 |
| **正式对比实验（S7）** | 依赖 S1-S6 全绿；旧 ckpt 兼容性也未验证 |
| **改 `evaluate.py` 全量指标** | `evaluate.py` 与 `sample.py` 都按 legacy 契约写死，改它要等 V5 契约冻结 |
| **删任何 legacy 代码** | §7 的三方对比要求 legacy 可跑（§11.3 #5）|

---

## 13. 十项结论（一句话）

1. **网络结构**：legacy 是 `8×TrajBlock 控制骨干 + 1×head_p 共用 + 3 层 FinalDenoiser + 4 候选路由`，C/Q/H 分别读 `bspline.num_controls`(`planner.py:85-93`)、`model.num_safety_queries`(`:101-102`)、`bspline.curve_points`(`:96-99`)。
2. **q_coarse**：12 处产生/输出、train.py 4 处进 loss、**2 处 torch.where 回退**（`planner.py:505-506, 535`）是删除的真难点。
3. **Topology**：跨模型/训练/推理/评估/数据/测试 6 层共 ~20 个文件，且与 sampler 的"走廊激活"语义强耦合。
4. **接口**：`match_block/path_feature_head/ellipse_geometry` 可原样复用，`_route_topology` 应整体删除，新签名已在 `planner_v5.py:287-295`。
5. **第一阶段前向**：legacy 只能构造期开关 + 零占位冒充；正确落点是 `planner_v5` 的可学习空 token + absent 嵌入（`:180-185, 357-360`）。
6. **未腐蚀 occupancy**：processed 只剩 k4p，但原始图 100% 可从 `data/raw/160m-256px/scenes/*.npz::occupancy_train` 重建（`00_build_processed.py:209-211`）。
7. **160m heading**：processed 无原始轨迹/时间戳，只能取 GT 曲线切线；80m 有 `trajectory_raw_local` 但不是当前数据集。
8. **C32/Q64**：C32 现成（`10_make_c32_dataset.py`），Q64 必须新写（`13_build_ellipse64_labels.py` 不存在），且旧椭圆标签强依赖候选骨架。
9. **局部 corridor**：`build_safety_corridor` 全链可复用，切分点必须在 `centers/anchors` 入参层；`lam[B,P,4,F]` 的顺序完全由 anchors 的 piece 划分决定；`sampler_v5` 虽算了碰撞区间却未接进走廊构建（§9.5）。
10. **四步 rollout**：现有两步结构（`train.py:279-417`）可复用 ALM/DDIM/no_grad 段；推理侧 `sampler_v5.py` 已有 4 步骨架（§10.4），**训练侧完全空白**，且训练 pack 不可离线复现。

---

## 14. 我认为最危险的 3 个改动

1. **改 C=48 → 32**。`BSplineCodec.knots/basis` 是 persistent buffer（`src/geometry/bspline.py:282-283`），C 一变旧 ckpt 直接 strict-load 失败，且 `control_gt` 是**离线数据**（`carla_spline_dataset.py:140-145` 会直接 `ValueError`）。同时 `head_p` 的输出语义、`boundary_decoder` 的 `span=min(4, C//2)`（`boundary.py:87`）都会变——这是"一行配置 + 一次数据重刷"就悄悄改掉 3 个不变量的改动。
2. **在 `sampler` 里把"激活式走廊"换成"单路径走廊"**。现在走廊构建、`tried/frozen_idx/topology_fallback`、`lam=None` 重置、`max_activation_delay_steps` 全部围绕"M 个候选"设计（`sampler.py:683-735`）。删掉候选后这些状态机没有对应物，一旦改成"第 1 步后直接建走廊"，`warmup`/`activation`/`exhausted` 的语义会静默失效——而 `train.py` 的两步 rollout 又假定了 ALM 一定作用在 pack 上（`train.py:365-377`），训练/推理的约束集从此分叉。
3. **训练侧"四步 rollout + 动态局部走廊"同时上**。它把两个独立的不确定性乘在一起：①路径由当步预测决定 → 约束集随预测漂移；②`lam` 顺序/形状由 anchors 决定（`bspline_alm.py:153, 205-212`）→ 包一变旧 dual 就必须作废（§13 第 14 条），而现在只有一个全局 `inherit_dual` 布尔（`sampler.py:539, 748`）。再加训练用的**离线固定 pack**（`train.py:267-276`），训练与推理的约束集在语义上必然不一致，这正是 §13 第 18 条明令禁止的状态。
