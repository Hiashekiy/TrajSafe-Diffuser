# src 代码说明

本项目是 **Maze2D 安全扩散规划器**。默认把三张迷宫（umaze / medium / large）的
数据**混合**在一起做 train / val / test（不再区分“单场景/联合”）。训练入口在根目录
`train.py`，采样/评估在 `sample.py` / `evaluate.py`，数据与可视化脚本在 `scripts/`。

`src/` 是算法与数据组件，按职责分 6 个包：`datasets / geometry / diffusion / models / losses / utils`。

---

## src/datasets —— 数据读取与归一化
| 文件 | 作用 |
|---|---|
| `normalization.py` | `LimitsNormalizer`：把 6 维状态 `[ax,ay,x,y,vx,vy]` 各维归一化到 `[-1,1]`，并保存/读取 `normalization.json`。 |
| `mixed_dataset.py` | `MixedDataset` + `make_loader`/`make_collate`：**训练用**。从 `data/processed/mixed` 按 sample 级混合三迷宫读取轨迹/条件/椭圆标签，并返回 per-sample 的 `map_tensor`/`sdf_tensor`（80×80）。`EXTENT=(0,8,0,8)` 定义统一世界范围。 |
| `data_io.py` | **采样/评估/绘图共用小工具**：`load_maze` 按 `maze_id` 从 mixed 数据取某迷宫的条件+地图+SDF；`sdf_metrics` 用 SDF 算碰撞/间距；`unnorm_positions` 归一化→世界坐标。`DATA_BASE="data/processed/mixed"`。 |

## src/geometry —— 地图、坐标、IRIS 椭圆
| 文件 | 作用 |
|---|---|
| `d4rl_coordinates.py` | **坐标/几何唯一来源**。定义 obs 帧与粒子体偏移、墙中心 `get_wall_centers_qpos`、box-SDF（`box_sdf`/`distance_to_wall`/`particle_clearance`）、迷宫行解析。 |
| `d4rl_geometry.py` | `build_occupancy_grid`：由墙中心生成全局 `occ` + SDF 网格（`01/04/05/06` 用）。 |
| `maze2d_env.py` | 只保留三张迷宫布局 `MAZES`（供几何/墙心推导用；原仿真类已删）。 |
| `maze_occupancy.py` | 全局 occ + 局部 128×128 patch 裁剪（IRIS 标签生成用）。 |
| `iris_solver.py` | **离线 IRIS MVIE 求解器**：在 patch 内凸优化出最大内接椭圆（04 生成 GT 椭圆标签核心）。 |
| `offline_iris_wrapper.py` | `OfflineIrisWrapper`：带位置缓存地调用 `iris_solver`，把像素椭圆换算回世界坐标（04 用）。 |
| `sdf_utils.py` | `sample_sdf_torch`：可微双线性采样 SDF（碰撞损失/指标用）。 |
| `ellipse_utils.py` | `patch_Q_to_world`：把 patch 像素的椭圆二次型换算到世界坐标。 |

## src/diffusion —— 扩散训练与采样
| 文件 | 作用 |
|---|---|
| `schedule.py` | `NoiseSchedule`：squaredcos_cap_v2 噪声调度（`betas/alphas/q_sample`）。 |
| `sampler.py` | `sample`：reverse 扩散生成轨迹，含端点重定（start/goal 写回）。 |
| `conditioning.py` | `apply_endpoint_condition`：**端点 inpainting**（Janner 式），把起点/终点位置写回。 |

## src/models —— 网络结构
模型 = 场景编码 + 轨迹编码 + 二者交互 + 安全融合 + 椭圆头 + 轨迹重建。
| 文件 | 作用 |
|---|---|
| `planner.py` | `Planner` 总入口，`forward(x_t,t,map,cond,extent,state_norm)` 拼装所有组件。 |
| `scene_encoder.py` | CNN 把 80×80 地图编码成 40×40 特征网格。 |
| `position_encoding.py` | `Sinusoidal2DPositionEmbedding`（场景坐标）、`SinusoidalTimestepEmbedding`（时间步）。 |
| `trajectory_encoder.py` | 编码**噪声轨迹**（无条件去噪器骨干）。 |
| `local_scene_sampler.py` | 对每个轨迹点在其局部窗口采样场景特征。 |
| `point_scene_attention.py` | 轨迹 token 与局部场景特征做注意力。 |
| `safety_fusion.py` | 融合轨迹特征与场景注意力成安全特征。 |
| `ellipse_head.py` | 从安全特征预测每点椭圆 `[center,r1,r2,theta]`。 |
| `trajectory_decoder.py` | 特征 + 全局场景 memory 解码。 |
| `trajectory_head.py` | 输出 `x0_pred`（clean 轨迹 `[B,H,6]`）。 |

## src/losses —— 目标函数
| 文件 | 作用 |
|---|---|
| `trajectory_loss.py` | `l_diff`（含时间一致性项 `lambda_var`/`lambda_var_vel`）、`l_collision`（SDF 采样）、`l_smooth`。 |
| `ellipse_loss.py` | `ellipse_loss` = 椭圆参数损失 + IoU + 椭圆碰撞（`L_ecol`）+ anchor 损失。 |
| `total_loss.py` | `total_loss` 组合上述子损失，含 **warmup** 分支（前几轮只训 `L_diff + L_smooth`）。 |

## src/utils —— 基础设施
| 文件 | 作用 |
|---|---|
| `config.py` | `load_config` 读取 `configs/config.yaml`（唯一配置）。 |
| `checkpoint.py` | `save_checkpoint` / `load_checkpoint`。 |
| `logger.py` | `Logger` 写训练日志到文件。 |
| `seed.py` | `set_seed` 固定随机种子。 |
| `metrics.py` | `interp_trajectory`：把 128 个节点稠密插值到 1017 点，算碰撞用。 |
| `visualization.py` | `plot_map_traj_ellipses`：地图+轨迹+椭圆绘图（仅 `scripts/data/06` 用）。 |

---

## 数据流
```
datasets 读数据 → geometry 建地图/算椭圆/碰撞 → models 网络 → diffusion 训练/采样
→ losses 监督 → utils 配置/日志/指标
```

## 相关入口
- 训练：根目录 `train.py`（读 `configs/config.yaml` + `data/processed/mixed`）
- 采样：根目录 `sample.py --maze <maze> --ckpt outputs/ckpt/best.pt`
- 评估：根目录 `evaluate.py --maze <maze> --ckpt outputs/ckpt/best.pt`
- 数据准备/绘图：`scripts/data`、`scripts/plot`（见 `scripts/README.md`）