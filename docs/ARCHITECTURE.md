# TrajSafe-Diffuser（实现说明）

> 当前实现规格（**优先**）：`docs/CONTROL_SPACE_REFACTOR.md`（控制点空间重构）。
> `docs/TrajSafe-Diffuser_网络结构设计报告_实现细化版.md` 描述的是重构前的
> "128 曲线点 token" 版本；两者冲突时以重构说明为准。
> 旧版本的代码、配置、入口、测试与文档已删除，只保留本实现依赖的共享组件
> （Skeleton 几何、B 样条编解码、ALM、数据准备）。

---

## 1. 唯一 diffusion state

只有一个 diffusion state：B 样条**控制多边形** `Q_t ∈ R^{B×C×2}`，`C` 来自配置
（`model.num_controls`，与 `bspline.num_controls` 必须一致，默认 32），不硬编码。
网络内部**没有任何 128 点轨迹张量**；`Q = model.num_safety_queries`（默认 128）
只用于 Skeleton / 椭圆安全 query。椭圆不是扩散状态，椭圆中心由选中骨架的
**固定**进度给出（`c_i = Γ_m(i/(Q-1))`），**没有 center head、没有 progress head**：

```
Q_t [B,C,2]                 扩散状态；端点硬条件（clamped knot ⇒ 曲线端点精确）
  -> H_ctrl                  (ControlEncoder = planner.traj_encoder, [B,C,D])
  -> H_ctrl                  (ControlBackbone = planner.traj_backbone, N_T 个 TrajBlock)
  -> Q~_coarse               (head_p，直接输出控制点 [B,C,2]，无 LS 拟合)
  -> Q_coarse                (BoundaryDecoder，固定零参数 -> 端点精确)
  -> H_S / R                 (SkeletonEncoder -> [B,M,L,D]; MatchBlock -> [B,M,C,D])
  -> pi = TopologyHead       -> m = m*（训练）/ argmax pi（推理）
  -> H_path                  (PathFeatureHead(R[m]), [B,C,D])
  -> H_safety                (SafetyQueryHead(H_S[m]), [B,Q,D])
  -> c = Γ_m(i/(Q-1))        (固定进度 buffer，无 head)
  -> H_ell -> shape4         (EllipseGeometry + EllipseShapeHead, a>=b>0)
  -> A_safety                (SafetyControlFusion：cross attention 控制 <- 安全)
  -> F = FusionMLP([H_ctrl, H_path, A_safety])
  -> H_clean                 (FinalDenoiser, N_F 个 TrajBlock)
  -> Q~_final -> Q_final     (head_p -> BoundaryDecoder，[B,C,2]，只有 Q_final 进 DDIM)
  -> B_128 @ Q_final         解码曲线（仅绘图 / 指标 / 控制器）
```

`head_p` 同时给出 coarse 与 final 的 raw 控制点；`out["control"] = Q_final`（端点精确），
`out["q_raw_final"] = Q~_final`（网络自由预测）。详细链路见
`docs/CONTROL_SPACE_REFACTOR.md`。

共享条件编码：`SpatialPE`、`PE_1D`、sinusoidal timestep + 2 层 MLP（AdaLN）。
场景编码：`C_G`（16×16 全局）与 `C_E`（32×32 细粒度）。

---

## 2. 文件

```
configs/config.yaml     训练 / 数据 / 模型 / 损失 + 推理期 `alm` / `corridor` 段

src/models/trajsafe/
    planner.py        TrajSafePlanner：forward_controls（默认）/ forward_curve_tokens（legacy）+ forward_all
    blocks.py         TrajBlock / MatchBlock / TrajSelfAttention / CrossAttention
    encoders.py       TrajectoryEncoder（控制点 / 曲线 token 共用）/ SkeletonEncoder / CoordMLP
    boundary.py       BoundaryDecoder（固定、零参数端点修正 + boundary_targets）
    ellipse.py        EllipseGeometry（Γ_m 可微插值 + 栅格化辅助）
    fusion.py         FusionMLP / FinalDenoiser / SafetyControlFusion
    geometry.py       CurveDecoder / dense_arclength / gather_dense_path_points
    heads.py          TopologyHead / PathFeatureHead / EllipseShapeHead
                      （PathFeatureHead 复用为 safety_query_head；无 ProgressHead）
src/models/common/     blocks.py（AdaLN/MHA）、scene_cnn.py（SceneCNN）
src/models/position_encoding.py

src/datasets/carla_spline_dataset.py  当前训练数据集（控制点快照，校验 C / Q）
src/datasets/skeleton_dataset.py      scenes 数据集（lazy GT soft mask）
src/diffusion/schedule.py            NoiseSchedule（squaredcos_cap_v2, T=16）
src/diffusion/sampler.py          sample（DDIM + 安全走廊 / ALM 状态机）
src/diffusion/alm_guidance.py        alm_correct（旧 waypoint 语义，sample() 已拒绝该入口）
src/diffusion/bspline_alm.py         bspline_alm_correct（控制空间 ALM）
src/losses/losses.py              8 项损失

src/geometry/
    skeleton_graph.py / skeleton_paths.py   骨架图 + 候选路径生成
    thinning.py / topology.py               Guo-Hall 细化 + 拓扑工具
    bspline.py / bspline_constraints.py     B 样条编解码（knots: "auto"）+ 约束包
    ellipse_shape.py                        稳定椭圆参数化（共享）
    ellipse_raster.py                       可微/软椭圆栅格化
    convex_region.py / safety_corridor.py   EllipseRegionBuilder + 安全走廊

train.py / sample.py / evaluate.py
scripts/data/{01,02,03}_*.py                scenes 数据准备流水线
scripts/data/carla/{00,01,02,03}_*.py       CARLA v1 数据准备流水线
scripts/debug/{candidate_recall,profile_step,visualize_labels}.py
tests/test_{model,dataset,geometry}.py, test_thinning.py,
tests/test_alm_guidance.py, tests/test_convex_region_validity.py
```

---

## 3. 损失（`train.py` 默认权重）

控制点空间下损失固定为 8 项（权重在 `configs/config.yaml` 的 `loss:` 段）：

```
L = 0.20 L_ctrl + 0.50 L_coarse + 0.08 L_smooth + 0.20 L_boundary
  + 0.25 L_topo + 0.08 L_shape + 0.25 L_iou + 0.15 L_safe
```

* `L_ctrl`：`MSE(Q~_final[1:-1], Q_GT[1:-1])`，用 **Boundary Decoder 之前**的 raw 控制点；
* `L_coarse`：`MSE(Q~_coarse[1:-1], Q_GT[1:-1])`，coarse head 直接输出控制点（无 LS 拟合）；
* `L_smooth`：控制多边形的二阶 / 三阶差分（`α=0.25`、`β=1.0`），按 detached 的 GT 控制点
  平均步长缩放，**不解码任何轨迹点**；
* `L_boundary = L_b(Q~_final) + 0.5 L_b(Q~_coarse)`，`L_b` 是端点邻域的加权 MSE：
  `T^s_i = S + (Q^GT_i - Q^GT_0)`、`T^g_j = G + (Q^GT_j - Q^GT_{C-1})`，权重 `w^s` / `w^g`
  从 `model.boundary_decoder` 读取（单一来源，不在 YAML 重复定义）；
* `L_topo`：`CE(pi, m*)`，`m*` 由 nDTW 与 GT 轨迹确定；
* `L_shape` 只作用于 `EllipseShapeHead`，**不包含圆心**（圆心是固定骨架中心）；
* `L_iou` / `L_safe`：lazy GT soft mask 的 IoU 与逐轨迹 CVaR 覆盖率。

`L_traj`（`trajectory_x0_loss` / `trajectory_smoothness_loss`）与 `L_align` 已不存在：
没有 128 点解码曲线进入损失、没有 GT 骨架投影、没有 `s*` / `progress_gt` /
`ellipse_center_gt`。

---

## 4. 数据

训练使用 CARLA v1 处理快照：`data.root: data/carla_v1` ->
`data.processed_root: data/carla_processed`（`train / val / test`），由
`CarlaSplineDataset` 读取。离线控制点标签是**数据**，改 `bspline.num_controls`
后必须重跑流水线，否则数据集会直接报错。

```bash
python scripts/data/carla/00_clean_dataset.py        --config configs/config.yaml
python scripts/data/carla/01_build_candidates.py     --config configs/config.yaml
python scripts/data/carla/02_build_ellipse_labels.py --config configs/config.yaml
python scripts/data/carla/03_validate_processed.py   --config configs/config.yaml
```

* 原始轨迹 / 占据图：`data/carla_v1`（仓库已提供）；
* 候选缓存 / 椭圆标签 / 控制点 GT：`data/carla_processed/{train,val,test}`；
* 旧的 Maze2D（`data/scenes` / `data/skeleton`）工具链仍保留，但默认配置不再使用。

---

## 5. 命令

```bash
python train.py    --config configs/config.yaml
python sample.py   --config configs/config.yaml --ckpt outputs/ckpt/best.pt
python evaluate.py --config configs/config.yaml --ckpt outputs/ckpt/best.pt

python -m pytest tests/test_model.py tests/test_dataset.py tests/test_geometry.py \
                 tests/test_thinning.py tests/test_alm_guidance.py \
                 tests/test_convex_region.py -q
```

`sample.py` / `evaluate.py` 另有 `--steps`（DDIM 子采样）、`--device` 与
`--arch {auto,control,legacy}`（默认 `auto`：按 checkpoint 自动选择控制点链路或
重构前的 128 曲线 token 链路，详见 `docs/CONTROL_SPACE_REFACTOR.md`）。

Dashboard：见 `diffusion-dashboard/README.md`。

---

## 6. 推理期安全走廊 + ALM 修正

`sample.py` / dashboard 可选开启（`configs/config.yaml` 的 `alm` / `corridor` 段）：
`sample()` 内部是三阶段状态机 `WARMUP` -> `TRY_ACTIVATE`（用候选骨架构造安全走廊，
处理区域重叠 / 间隙桥接 / 冻结）-> `GUIDED`，在反向扩散的每个 step 用
`bspline_alm_correct` 对**控制点**做一阶修正（约束包只构建一次后冻结）。旧 waypoint
语义的 `alm_guidance` 入口已被 `sample()` 拒绝。这是**推理期扩展**，不改变网络、
checkpoint 与训练损失。
