# TrajSafe-Diffuser（实现说明）

> 架构唯一规格：`docs/TrajSafe-Diffuser_网络结构设计报告_实现细化版.md`。
> 旧版本的代码、配置、入口、测试与文档已删除，只保留本实现依赖的共享组件
> （Skeleton 几何、ALM、数据准备）。

---

## 1. 唯一 diffusion state

只有 `P_t ∈ R^{B×H×2}` 是扩散状态。椭圆不是扩散状态，椭圆中心由选中骨架
曲线 `c_i = Γ_m(s_i)` 给出，**没有 center head**：

```
P_t
  -> H_traj                    (Trajectory Backbone, N_T 个 TrajBlock)
  -> {R_m}                     (Skeleton Transformer, N_S 个 MatchBlock, 共享)
  -> pi = TopologyHead         -> m = argmax pi（每个 reverse step 重选）
  -> H_prog -> s               (ProgressHead, softplus 间隔 -> 严格单调)
  -> c = Γ_m(s)
  -> H_ell -> shape4           (EllipseShapeHead, 稳定参数化 a>=b>0)
  -> H_clean -> P0_hat         (Final Denoiser, N_F 个 TrajBlock)
  -> DDIM 一步
```

粗解码 `P~_0` 与最终解码 `P^_0` 共用同一个 `Head_P`；只有 `P^_0` 进入 DDIM。

共享条件编码：`SpatialPE`、`PE_1D`、sinusoidal timestep + 2 层 MLP（AdaLN）。
场景编码：`C_G`（16×16 全局）与 `C_E`（32×32 细粒度）。

---

## 2. 文件

```
configs/config.yaml     训练 / 数据 / 模型 / 损失 + 推理期 `alm` 段

src/models/trajsafe/
    planner.py        TrajSafePlanner：三入口 + forward_all
    blocks.py         TrajBlock / MatchBlock / TrajSelfAttention / CrossAttention
    encoders.py       TrajectoryEncoder / SkeletonEncoder / CoordMLP
    ellipse.py        EllipseGeometry（Γ_m(s) 可微插值 + 栅格化辅助）
    fusion.py         FusionMLP / FinalDenoiser
    geometry.py       CurveDecoder / dense_arclength / gather_dense_path_points
    heads.py          TopologyHead / ProgressHead / EllipseShapeHead
src/models/common/     blocks.py（AdaLN/MHA）、scene_cnn.py（SceneCNN）
src/models/position_encoding.py

src/datasets/skeleton_dataset.py  数据集（lazy GT soft mask）
src/diffusion/schedule.py            NoiseSchedule（squaredcos_cap_v2, T=16）
src/diffusion/sampler.py          sample（DDIM + 可选 ALM 引导）
src/diffusion/alm_guidance.py        alm_correct（推理期一阶修正）
src/losses/losses.py              全部损失

src/geometry/
    skeleton_graph.py / skeleton_paths.py   骨架图 + 候选路径生成
    thinning.py / topology.py               Guo-Hall 细化 + 拓扑工具
    ellipse_shape.py                        稳定椭圆参数化（共享）
    ellipse_raster.py                       可微/软椭圆栅格化
    convex_region.py / convex_region.py   EllipseRegionBuilder + 凸区域

train.py / sample.py / evaluate.py
scripts/data/{10,13,14}_*.py                数据准备流水线
scripts/debug/{candidate_recall,profile_step,visualize_labels}.py
tests/test_{model,dataset,geometry}.py, test_thinning.py,
tests/test_alm_guidance.py, tests/test_convex_region_validity.py
```

---

## 3. 损失（`train.py` 默认权重）

```
L = 1.00 L_traj + 0.50 L_coarse + 0.08 L_smooth + 0.25 L_topo
  + 5.00 L_align + 0.08 L_shape + 0.25 L_iou + 0.15 L_safe
```

* `L_traj` / `L_coarse`：内部 waypoint 的 MSE（端点硬条件）；
* `L_smooth`：几何加速度 + jerk，按 GT 平均步长缩放；
* `L_topo`：`CE(pi, m*)`，`m*` 由 nDTW 与 GT 轨迹确定；
* `L_align = mean SmoothL1(Γ(s_i), p_i^GT)`，**直接对 GT 轨迹**，
  没有 GT 骨架投影、没有 `s*`、没有 `progress_gt`、没有 `ellipse_center_gt`；
* `L_shape` 只作用于 `EllipseShapeHead`，**不包含圆心**；
* `L_iou` / `L_safe`：lazy GT soft mask 的 IoU 与逐轨迹 CVaR 覆盖率。

---

## 4. 数据

* 轨迹 / 条件 / 地图：`data/scenes`（`positions / conditions / maze_id / maps`）。
* 候选缓存与椭圆标签：`data/skeleton`。
* 训练只用 large：`data.mazes: ["large"]`（5400 train / 300 val / 300 test）。

离线流水线：

```bash
python scripts/data/01_build_skeletons.py          --config configs/config.yaml
python scripts/data/02_build_candidates.py --config configs/config.yaml
python scripts/data/03_build_ellipse_labels.py  --config configs/config.yaml
```

`data/scenes` 已随仓库提供；从 d4rl hdf5 重建它的 V1 数据准备脚本
已在本分支删除（需要时从 git 历史取回）。

---

## 5. 命令

```bash
python train.py    --config configs/config.yaml
python sample.py   --config configs/config.yaml --ckpt outputs/ckpt/best.pt
python evaluate.py --config configs/config.yaml --ckpt outputs/ckpt/best.pt

python -m pytest tests/test_model.py tests/test_dataset.py tests/test_geometry.py \
                 tests/test_thinning.py tests/test_alm_guidance.py \
                 tests/test_convex_region_validity.py -q
```

Dashboard：见 `diffusion-dashboard/README.md`。

---

## 6. 推理期凸区域 + ALM 修正

`sample.py` / dashboard 可选开启：对 `t <= start_t` 的帧，用预测椭圆在线构造
verified convex region，并用 `alm_correct` 对 `x0` 做一阶修正后再走 DDIM。
这是**推理期扩展**，不改变网络、checkpoint 与训练损失。参数在
`configs/config.yaml` 的 `alm` 段。
