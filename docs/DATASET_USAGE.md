# D4RL Maze2D 数据集使用总结（PNGD 项目）

本文档总结本项目如何读取 D4RL Maze2D 数据集、如何重建环境/占据地图、
轨迹与环境的对应关系怎么确立，以及前面踩过的各种坑。

## 0. 一句话概括

- **数据**：D4RL `maze2d-{umaze,medium,large}-sparse-v1.hdf5`（从 HF 镜像 `imone/D4RL` 下载）。
- **模型**：`leekwoon/maze2d-*` HuggingFace Diffuser 权重（**RGG NeurIPS 2023 转换的**，不是 Janner 官方发布）。
- **环境**：**离线重建**，由 D4RL 源码里的固定迷宫网格 + 坐标映射推出；**没有可运行的 gym/d4rl/mujoco 环境**。
- **任务**：Maze2D 规划——给定起点+终点，扩散模型生成整条轨迹，Neural-IRIS 做几何引导。

## 1. 数据文件与字段

每个迷宫一个 hdf5（路径 `data/d4rl/maze2d-*-sparse-v1.hdf5`）：

```
observations : (N, 4) float   # [x, y, vx, vy]  (qpos 系)
actions      : (N, 2) float   # [ax, ay]，范围 [-1,1]
rewards      : (N,)   float
terminals    : (N,)   bool
timeouts     : (N,)   bool
infos/group  : (episodes 分组)
```

**数量/范围（实测）**：
- umaze：1,000,000 次转移；obs x∈[0.39,3.22] y∈[0.63,3.22]，速度 ±5.226。
- medium：2,000,000；x∈[0.67,6.22]。
- large：4,000,000；x∈[0.40,7.22] y∈[0.44,10.22]。
- 无 episode 数固定；用 `terminals | timeouts` 划分 episode（umaze ~2754 个长度足够的、medium ~1981、large ~2214）。

**关键点**：hdf5 里**没有迷宫布局、没有墙坐标、没有序列化的环境对象**。
这是离线 RL 数据集的常态——环境定义在外部源码里，由**数据集名** `maze2d-{maze}-v1` 对应到 D4RL 注册的固定环境。

## 2. 轨迹 ↔ 环境对应关系怎么确立

这是本项目最容易踩坑、也最关键的地方。分三步：

### 2.1 源码推导坐标映射

从 D4RL `maze_model.py`/MuJoCo 定义推出（`src/geometry/d4rl_geometry.py`）：

```
obs = 滑块的 slide-joint qpos（不是世界坐标）
世界坐标 = obs + (1.2, 1.2)                 # 粒子 body 固定偏移
墙格 (row,col) 的世界盒中心 = (row+1, col+1)
墙格在 qpos 系中心 = (row-0.2, col-0.2)
格点映射 = round(x), round(y)             # 官方用 round，不是 floor
碰撞 = 粒子球(r=0.1) vs AABB(半宽0.5)，用 AABB SDF 减 r
```

### 2.2 用“数据本身”验证

因为真实观测都是智能体在自由空间到达的位置，所以用上面的几何模型去判数据集观测，
**应该几乎都不在墙里**。实测（全量观测）：

```
迷宫   碰撞率    平均净距   5%分位净距
umaze  0.0011   0.251      0.105
medium 0.0009   0.302      0.120
large  0.00065  0.296      0.146
```

（更正前的错误模型曾得到 66%~82% 的“碰撞率”，见第 6 节坑 1。）

### 2.3 交叉验证

- 观测范围 `x∈[0.39,3.22]` 与推导出的自由空间边界（左墙 obs≈0.4、右墙 obs≈3.2）几乎吻合。
- 官方 Diffuser 轨迹反归一化后平均最近距离到真实观测 ≈0.016，100% 在 0.25 内。

**诚实说明**：这是“源码推导 + 数据一致性验证”，不是“真环境验证”。
因为 GGMPC 环境里 `gym / d4rl / mujoco / mujoco_py` **全部装不上**，所以只能离线重建；
碰撞/安全指标是**对着重建几何算的**，不是对着 MuJoCo 算的。要 100% 坐实需在 Linux/WSL 装 d4rl+gym+mujoco。

## 3. 地图 / Occupancy 构建

- `src/geometry/d4rl_geometry.build_occupancy_grid(maze, extent, global_res=20.0)`
  - 返回 `(occ 1=墙/0=自由, distance_field, global_res)`，`occ` 形状 `(ny, nx)`，坐标是 qpos 系。
  - `inflate_particle=True` 会把墙按粒子半径膨胀 0.1（碰撞意义上的墙更厚）。
  - 像素中心映射：`px = x * global_res, py = y * global_res`（origin=0）。
- （旧 `build_occupancy` 已移除；统一用 `build_occupancy_grid`。）
- 局部 patch：`src/geometry/maze_occupancy.crop_local_patch(global_occ, anchor, global_res, local_res=20.0)`
  - 以 anchor 为中心裁 **128×128**，返回 `(patch, patch_to_world, world_to_patch, local_res)`。
  - patch→world 是**均匀缩放+平移**：`world = anchor + (px + 0.5 - half)/local_res`，无旋转。所以椭圆方向在 patch 和 world 里一致，半径乘 `1/local_res`。

## 4. 轨迹怎么用（window + conditions + normalization）

- 数据预处理改为 `scripts/data/01_prepare_map.py` 与 `02_prepare_trajectories.py`（h5py 直接读）；`data/d4rl_loader.py` 已删除。
- 条件（start/goal）由 `scripts/data/02` 生成并存为 `conditions.npy`：
  ```
  cond = {0: start_obs[None,:], horizon-1: goal_obs[None,:]}   # 世界坐标
  ```
- 条件化：`src/diffusion/conditioning.py::apply_endpoint_condition` 在训练/采样时把第 0 帧钉成起点、第 H-1 帧钉成终点（inpainting）。
- 轨迹张量通道序：`[a_x, a_y, x, y, vx, vy]`（action 在前 2 维，obs 在后 4 维），长度 = horizon。
  - 三迷宫统一 H=128、T=64（`data/processed/mixed`，[0,8]^2）。

## 5. 模型与 reverse diffusion 流程

- 模型：本项目自研 `src/models/planner.py`（Scene Encoder + Trajectory Encoder + Point-Scene Attention + Ellipse Head，见 `METHOD.md`）；扩散用 `src/diffusion/schedule.py`（`prediction_type='sample'`，预测 x0）。
- 原每迷宫一套的 RGG 对照权重（`third_party/leekwoon_maze2d_*`）已删除；当前仅使用本项目自训权重。
- reverse loop 在 `src/diffusion/sampler.py`；不再使用 RGG/Neural-IRIS 引导。
- （原 Neural-IRIS 引导 `iris_guidance.correct_clean` 已废弃；现仅用离线 IRIS MVIE 标签。）
  ```
  x_t -> UNet -> clean estimate x_hat0
       -> 取 x/y 位置 p_hat_1..H
       -> 每隔 subsample 个点裁局部 patch
       -> Neural-IRIS -> {center, Q(=P), A, b}
       -> 每点归属最近锚点几何（转 world）
       -> 解全程联合 QP（Axistrajectory / euclidean / q_aware）
       -> 写回 x/y -> DDPMScheduler.step -> x_{t-1} -> 重新钉起点/终点
  ```
- 几何只在锚点稀疏算，但**优化是整条轨迹一起做**（关键修正，避免锯齿）。

## 6. 踩过的坑（按严重程度）

**坑 1：坐标映射错（最致命）**。一开始用 floor 且把 qpos 当世界坐标，导致 66%~82% 的真实观测被判成“在墙里”。后来改成 `obs + (1.2,1.2)` + `round()` + sphere-vs-AABB SDF，碰撞率降到 ~0.1%。
**坑 2：build_occupancy_grid 网格索引错**。`meshgrid(indexing='ij')` + reshape 把 occupancy 弄乱了，一度造成“Full 30 碰撞点”。改成 `indexing='xy'` + reshape(ny,nx) 后对齐。
**坑 3：`rng.randint` on np.Generator**。numpy 新版没 `randint`，要用 `rng.integers`。
**坑 4：旧依赖兼容**。`collections.Mapping`→`collections.abc.Mapping`；diffusers 要 `from diffusers import UNet1DModel`，并打补丁去掉 gym/mujoco/skvideo 依赖。
**坑 5：GeometryAdapter 的 bias 没归零**，导致“零 adapter”基线非零，出现假的 `-30%`。基线应该是 `geo_adapter=None`（真无 GeoCond）。
**坑 6：单 window 的 projection magnitude 极噪**（std≈0.025）。之前单次窗口得出 `+15.7%/+17.2%` 是错的；必须多窗口（N=20）平均才可信。
**坑 7：粗 subsample 导致“只在每 4 个点修 1 个点”**，中间点不动 → 锯齿。正确做法：几何稀疏采样，但**优化全程一起做**（`project_trajectory_anchored` / `project_trajectory_axis`）。
**坑 8：coarse subsample=8 造成 Full 失败的假设被推翻**。subsample=1 下 medium 仍 30 碰撞、large 仍 52——**问题是 GeoCond（几何注入进 UNet）本身在 harder maze 上过度修正**，不是 subsample。
**坑 9：椭圆长轴/半径/坐标系要验证**。`P=Q` 是精度矩阵，长短轴=`1/eig(P)`、方向=特征向量；center 是 patch 像素坐标；必须转 world 并画图验证（长轴贴走廊、`u1·u2=0`、`r1≥r2`）。
**坑 10：Neural-IRIS 假设锚点在自由空间**。当 clean estimate 穿墙时，椭圆/长轴不可信，轴吸引会把最坏点拉得更深（large：minClr -0.569 / max_corr 1.20）。加了**碰撞排斥**（检测到点在障碍里→禁用轴吸引、沿 SDF 梯度往外推）后：max_corr 1.20→0.93、minClr -0.569→-0.436，但仍未超过 Base。
**坑 11：guidance 注入时机**。默认 `gs_frac=0.4` 偏晚（轨迹已成形）。umaze 扫描：0.4→minClr 0.133、**0.6→0.141（最好）**、0.8→0.139 且 IterProj 冒碰撞（太早，几何太噪）。默认已改 0.6。
**坑 12：环境装不上**。`gym/d4rl/mujoco/mujoco_py` 在 GGMPC 里全缺失 → 只能离线重建环境；指标是“对重建几何”而非 MuJoCo 真环境。
**坑 13：checkpoint 归属**。`leekwoon/maze2d-*` 是 **RGG 转换**的 Diffuser 权重，**不是 Janner 官方**发布的 Maze2D HF checkpoint。每个迷宫一套，评估用各自权重，不跨迷宫复用。

## 7. 关键文件

- 数据读入：`scripts/data/01_prepare_map.py`、`scripts/data/02_prepare_trajectories.py`（h5py）
- 几何：`src/geometry/d4rl_geometry.py`、`src/geometry/maze_occupancy.py`
- 椭圆标签：`src/geometry/iris_solver.py`、`src/geometry/offline_iris_wrapper.py`
- 训练/采样/评估：`train.py`、`sample.py`、`evaluate.py`
- 绘图：`scripts/plot/*`（见 `scripts/README.md`）

## 8. 一句话给下一步

先用 `evaluation/evaluate_axis.py --maze umaze --gs_frac 0.6` 对照 AxisTrajectory 与 IterQProj；
medium 可多场景；large 因 QP 太贵，先单方法+粗锚点跑，重点看“更早注入 + 碰撞排斥”能否把 minClr 拉回 Base 之上。