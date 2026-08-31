# DSH 实施指南：Maze2D 安全扩散规划器（自包含可运行版）

> 版本 v2.1。目标：按确定的最终框架实现**可训练 / 可验证 / 可复现**的 PyTorch 实现。
> 实现分三阶段：**数据准备 → 网络构建 → 训练与测试**。
>
> **环境**：统一使用 GGMPC conda 环境，解释器
>   E:/CondaEnvData/envs/GGMPC/python.exe
> **所有超参数 / 路径 / 环境**都放在**一个配置文件**：
>   configs/config.yaml
> 后端代码只读取这个配置文件，严禁把数值硬编码进源码。

---
## 0. 当前工作区现状（以下是真实存在的，直接基于它写代码）

> **▶ 开工第一步（新对话从这里开始）**：按 §7.15 把 `_archive/integration/` 里的
> `d4rl_geometry.py`、`maze_occupancy.py`、`neural_iris_adapter.py`、`maze2d_env.py` 拷进
> `src/geometry/`，并把坐标统一到 `src/geometry/d4rl_coordinates.py`（同时
> 更新 `d4rl_geometry.py` 里的 `from integration.maze2d_env import MAZES` 为同目录相对导入）。
> 之后再按 §9 顺序跑数据准备 → 训练 → 采样/评估。

- 原始 D4RL hdf5（已存在）：
  - data/d4rl/maze2d-umaze-sparse-v1.hdf5
  - data/d4rl/maze2d-medium-sparse-v1.hdf5
  - data/d4rl/maze2d-large-sparse-v1.hdf5
- 地图/几何（旧代码在 **_archive/integration/**，其中**可复用**、且**无 gym/mujoco 依赖**）:
  - _archive/integration/d4rl_geometry.py   （坐标映射 / AABB-SDF / occupancy）
  - _archive/integration/maze_occupancy.py  （128×128 局部 patch）
  - _archive/integration/neural_iris_adapter.py （IRIS 推理 wrapper）
  - _archive/integration/maze2d_env.py      （MAZES 迷宫字符串，d4rl_geometry 依赖它）
  按 §7.15 把这些拷进新的 src/geometry/，并统一坐标到 src/geometry/d4rl_coordinates.py。
- IRIS 标签：离线 IRIS MVIE 求解器（`src/geometry/iris_solver.py`）；Neural-IRIS 网络已删除，不再使用。
- 预训练 base：原 RGG 对照权重（`third_party/leekwoon_maze2d_*`）已删除；本网络不依赖第三方预训练权重。
  - 归属备注：leekwoon/maze2d-* 是 RGG NeurIPS 2023 转换的 Diffuser 权重，非 Janner 官方（仅供历史参考）。
- 环境约束：GGMPC 里 gym / d4rl / mujoco / mujoco_py **均装不上** → 地图必须**离线重建**；
  所有 collision / clearance / SDF 指标在日志注明：
    evaluated against reconstructed D4RL geometry

---
## 1. 目标目录（新实现按此搭，老代码已归档到 _archive/）

    Neural-IRISDiffuser/
    ├── configs/config.yaml              # 唯一配置（已提供）
    ├── data/d4rl/*.hdf5                 # 原始数据（已存在，== data/raw）
    ├── data/processed/maze2d_umaze/     # 数据准备产物
    ├── scripts/{data, plot, debug}/   # 数据准备 / 绘图 / 调试
    ├── src/
    │   ├── datasets/{normalization.py, mixed_dataset.py, data_io.py}
    │   ├── models/{scene_encoder, position_encoding, trajectory_encoder,
    │   │          local_scene_sampler, point_scene_attention, safety_fusion,
    │   │          ellipse_head, trajectory_decoder, trajectory_head, planner}.py
    │   ├── diffusion/{schedule.py, sampler.py}
    │   ├── losses/{ellipse_loss.py, trajectory_loss.py, total_loss.py}
    │   ├── geometry/{d4rl_coordinates.py, d4rl_geometry.py, maze2d_env.py,
    │   │            maze_occupancy.py, sdf_utils.py, ellipse_utils.py,
    │   │            iris_solver.py, offline_iris_wrapper.py}
    │   └── utils/{checkpoint.py, logger.py, seed.py, metrics.py, visualization.py}
    ├── train.py / evaluate.py / sample.py
    └── README.md

---
## 2. 配置：configs/config.yaml（唯一来源）

已提供，包含全部超参数。顶层键：
    env( python, device, seed )
    data( maze, raw_hdf5, raw_dir, processed_dir, horizon, stride, max_episodes, train_val_test )  # umaze 旧默认，兼容单场景
    geometry( coord_module, maze_name, obs_offset, wall_half, particle_radius, cell_map,
              extent, global_res, inflate_particle, occupancy_threshold, local_patch_size, local_res )
    mazes( name/raw_hdf5/processed_dir/extent/global_res/local_res ×3 )   # 离线 IRIS + mixed 构建用
    model( horizon, state_dim, d_model, num_heads, trajectory_encoder_layers,
           trajectory_decoder_layers, ffn_dim, local_window, dropout )
    diffusion( timesteps, prediction_type, beta_schedule, beta_start, beta_end )
    train( batch_size, epochs, lr, weight_decay, grad_clip, num_workers, save_every,
           warmup_epochs, ckpt_dir, ckpt_dir )
    loss( lambda_diff, lambda_var, lambda_var_vel, lambda_e, lambda_col, lambda_s, lambda_param,
          lambda_iou, lambda_ecol, lambda_anchor, lambda_c, lambda_r, lambda_theta,
          ellipse_valid_min_sdf, collision_margin, collision_sigma, ellipse_margin )
    iris( patch_size, cache_resolution )   # 离线 IRIS 求解器用

要求：写一个 src/utils/config.py::load_config(path="configs/config.yaml")，
返回 dict；所有模块从它取参数。三张迷宫的几何（extent/global_res/processed_dir）在
config.yaml 的 mazes: 列表里统一管理（不再需要 experiment_*.yaml）。

---
## 3. 第一阶段：数据准备（全部提前离线生成，训练阶段只做动态部分）

### 3.1 scripts/data/01_prepare_map.py
输出到 processed_dir/map/：occupancy.npy, obstacle_mask.npy, sdf.npy, map_xy.npy, meta.json。
- 用 _archive/integration 拷来的 d4rl_geometry 重建地图（qpos/observation 帧）。
- **强制 Gate**：地图生成后立即做真实观测碰撞率 sanity check，输出
  observation_collision_rate / mean_clearance / clearance_p05 到 outputs/data_checks/map_validation.json。
  umaze 正常应约 **0.0011**；若 > 0.01 直接判失败，禁止继续。

### 3.2 scripts/data/02_prepare_trajectories.py
- 读 raw_hdf5；用 **terminals | timeouts** 切 episode（禁止跨 episode 拼窗 / 只看 terminals）。
- 生成序列 [N, H, 6]（state 通道序固定 [ax, ay, x, y, vx, vy]），stride 用 config。
- train/val/test 在 sequence 层面固定划分；保存 normalization.json（round-trip 误差≈0）。
- 输出 processed_dir/{train,val,test}/trajectories.npy + conditions.npy。
- Gate：window_cross_episode_count == 0。

### 3.3 scripts/data/04_generate_ellipse_labels.py（03_benchmark_iris.py 已归档）
- benchmark_n=[100,1000,10000]，统计 latency / success / invalid rate → outputs/iris_benchmark.json+csv。
- GT 标签：ellipse_params [N,H,5]=[cx,cy,r1,r2,theta]；ellipse_Q [N,H,2,2]；ellipse_valid [N,H]。
- **必须位置缓存**：ix=round(x/cache_resolution), iy=round(y/cache_resolution)，相同 key 只跑一次 IRIS。
- Q = R(theta) diag(1/r1^2, 1/r2^2) R(theta)^T；r1>=r2；u1·u2≈0（用图像核验长轴沿走廊）。
- **Gate**：SDF(p_gt)>0 才给有效标签，否则 ellipse_valid=0。

### 3.4 scripts/data/05_validate_processed_data.py / scripts/data/06_visualize_processed_sample.py
- 检查 NaN/Inf/shape/normalization round-trip/连续性/start-goal/ellipse 有效及是否穿墙。
- 输出 processed_dataset_report.json + 若干可视化（map + waypoint + ellipse 叠加）。

---
## 4. 最终模型定义（不可改动约束，数值全读 config）

输入：noisy trajectory x_t, diffusion timestep t, fixed map M
输出：x0_pred, ellipse_pred
主链路与各模块（数值全读 config，见 §2）：

**Scene Encoder**：输入 map [1,1,Hm,Wm]（U-Maze 固定，同 batch 只编码一次）；
输出 scene_map [B,C,Hs,Ws] + scene_tokens [B,Ns,C]，Ns=Hs*Ws。
scene_map 用于局部 grid_sample；scene_tokens 用于 Trajectory Decoder 全局 Cross-Attention。

**Scene Position Encoding**：生成 scene_xy [Ns,2]，一次性
scene_tokens + map_position_embedding(scene_xy) = scene_memory [B,Ns,C]。后续不再重复加位置编码。

**Trajectory Token Encoder**：输入 x_t [B,H,6]（[ax,ay,x,y,vx,vy]）+ t [B]。
拆 motion=[ax,ay,vx,vy]->[B,H,4]、pos=[x,y]->[B,H,2]。
Token = motion_embedding + xy_position_embedding + trajectory_index_embedding + timestep_embedding。
输出 H0 [B,H,C]，再若干层 Trajectory Self-Attention -> F_traj [B,H,C]。

**Local Scene Sampler**：输入 scene_map [B,C,Hs,Ws]、p_t [B,H,2]；用
torch.nn.functional.grid_sample 批量采样 window x window（默认 5）-> local_scene [B,H,Nl,C]。
p_t 只用于构造 grid，不再做位置编码；采样坐标由世界坐标转到 [-1,1]。

**Point-Scene Cross-Attention**：Q=[B,H,1,C]、K=V=[B,H,Nl,C]；合并 B,H，用
scaled_dot_product_attention -> 重排 [B,H,C]。用**局部** scene feature；禁用 PE_rel(m_j-p_t,k)。

**Safety Feature Fusion**：h = cat([F_traj, A_t], -1)；S_t = fusion_mlp(layer_norm(h)) -> [B,H,C]。

**Ellipse Head**：输入 S_t，输出 raw [B,H,6]=[cx,cy,rho1,rho2,dir_x,dir_y]。
r_raw = softplus(rho)+eps；r1=max(r_raw1,r_raw2)、r2=min(...)；v=normalize([dir_x,dir_y])；
方向用 [cos(2theta), sin(2theta)]（不直接回归 theta）。

**Trajectory Decoder**：T0=S_t；每层 = PreNorm -> Traj Self-Attn -> Residual ->
PreNorm -> Global Scene Cross-Attn(K=V=完整 scene_memory) -> Residual -> PreNorm -> FFN -> Residual。
num_layers / d_model / num_heads / ffn_dim 读 config。

**Trajectory Head**：输入 [B,H,C]，输出 [B,H,6] = x0_pred。

**Planner 统一接口**：x0_pred, ellipse_pred = model(x_t, t, map_tensor)；
forward 返回 dict {"x0_pred","ellipse_center","ellipse_radii","ellipse_dir","shared_feature"（debug 可选）}。
    E:/CondaEnvData/envs/GGMPC/python.exe evaluate.py --config configs/config.yaml