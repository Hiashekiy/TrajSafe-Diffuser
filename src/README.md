# src 代码说明

`src/` 按职责分 6 个包：`datasets / geometry / diffusion / models / losses / utils`。
架构规格见 `docs/TrajSafe-Diffuser_网络结构设计报告_实现细化版.md`，概览见 `docs/ARCHITECTURE.md`。

---

## src/models —— 网络

| 文件 | 作用 |
|---|---|
| `trajsafe/planner.py` | `TrajSafePlanner`：唯一入口，`forward` / `forward_all` |
| `trajsafe/blocks.py` | `TrajBlock` / `MatchBlock` / `TrajSelfAttention` / `CrossAttention`（AdaLN） |
| `trajsafe/encoders.py` | `TrajectoryEncoder` / `SkeletonEncoder` / `CoordMLP` |
| `trajsafe/ellipse.py` | `EllipseGeometry`：`c = Γ_m(s)` 的可微插值与栅格化辅助 |
| `trajsafe/fusion.py` | `FusionMLP` / `FinalDenoiser` |
| `trajsafe/geometry.py` | `CurveDecoder` / `dense_arclength` / `gather_dense_path_points` |
| `trajsafe/heads.py` | `TopologyHead` / `ProgressHead` / `EllipseShapeHead` |
| `common/blocks.py` | 共享 `AdaLN` 与 MHA 基类 |
| `common/scene_cnn.py` | `SceneCNN`：全局 `C_G` + 细粒度 `C_E` |
| `position_encoding.py` | `SpatialPE` / `PE_1D` / sinusoidal timestep embedding |

## src/datasets

| 文件 | 作用 |
|---|---|
| `skeleton_dataset.py` | 数据集：候选 `S_m` / 稠密 `Γ_m` / `shape4_gt` / `shape_valid` / lazy GT soft mask |

## src/diffusion

| 文件 | 作用 |
|---|---|
| `schedule.py` | `NoiseSchedule`（`squaredcos_cap_v2`，T=16） |
| `sampler.py` | `sample`：DDIM 反向过程 + 可选 ALM 引导 |
| `alm_guidance.py` | `alm_correct`：凸区域约束下的推理期一阶修正 |

## src/losses

| 文件 | 作用 |
|---|---|
| `losses.py` | `trajectory_x0_loss` / `trajectory_smoothness_loss` / `topology_ce` / `center_alignment_loss` / `ellipse_shape_loss` / `ellipse_iou_loss` / `ellipse_safety_loss` |

## src/geometry

| 文件 | 作用 |
|---|---|
| `skeleton_graph.py` / `skeleton_paths.py` | 骨架图构建、候选路径生成、`Γ_m` 稠密曲线 |
| `thinning.py` / `topology.py` | Guo-Hall 细化与拓扑工具 |
| `ellipse_shape.py` | 稳定椭圆参数化 `raw_to_shape4` / `shape4_to_abtheta`（共享） |
| `ellipse_raster.py` | 可微软椭圆栅格化与场景栅格坐标 |
| `convex_region.py` / `convex_region.py` | `EllipseRegionBuilder` 与凸区域顶点工具 |

## src/utils

| 文件 | 作用 |
|---|---|
| `config.py` | `load_config` |
| `checkpoint.py` | `save_checkpoint` / `load_checkpoint` |
| `seed.py` | `set_seed` |

---

## 入口

- 训练：`train.py --config configs/config.yaml`
- 采样：`sample.py --config ... --ckpt outputs/ckpt/best.pt`
- 评估：`evaluate.py --config ... --ckpt outputs/ckpt/best.pt`
- Dashboard：`diffusion-dashboard/backend.py` + `diffusion-dashboard/app/page.tsx`
