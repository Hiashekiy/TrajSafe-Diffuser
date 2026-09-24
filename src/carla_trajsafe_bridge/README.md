# TrajSafe-Diffuser -> CARLA 闭环桥接包

把 TrajSafe-Diffuser 的规划结果接到 CARLA 0.9.16 上，做**停车规划一次 + 闭环跟踪到终点**的
演示，并输出 guided 安全轨迹的俯视视频和“规划 vs 实车”轨迹对比图。

两个入口均使用 160 m 新模型 `A_oneshot:best_task`：固定 test_0056 场景使用完整
**16 步去噪**，长距离滚动规划入口使用固定 **4 步去噪**。

本包是**增量代码**：模型仓库原有的训练 / 推理 / 几何代码一行都没有改动，
只新增桥接层，并复用仓库里 dashboard 用的同一个推理引擎。

---

## 1. 依赖

| 依赖 | 说明 | 本机路径 |
|---|---|---|
| CARLA 0.9.16 | 服务端（打包版），必须用 Town03 / Town03_Opt | `E:/CARLA_0.9.16/CarlaUE4.exe` |
| CARLA Python binding | 与 0.9.16 匹配的 py3.10 版本 | `E:/CarDataSample/tools/carla/_vendor` |
| TrajSafe-Diffuser 仓库 | 提供模型 / 几何 / 采样器 / dashboard 推理引擎 | `E:/CarDataSample/Neural-IRISDiffuser` |
| 模型权重 | `best_task.pt` 等 | `<repo>/outputs/bspline_carla/ckpt/` |
| 数据集快照 | processed 占据图 / 条件 / B 样条 knots | `<repo>/data/carla_processed`、`data/carla_v1` |
| Python 环境 | torch+cu126、numpy、opencv、matplotlib、pyyaml | `E:/CondaEnvData/envs/GGMPC/python.exe` |
| （可选）车道栅格化 | 只用于 live occupancy 交叉校验，缺失会自动降级 | `<workspace>/tools/carla/map_cache.py` |

桥接包本身只额外依赖 `numpy / opencv / matplotlib / pyyaml / torch`。

---

## 2. 安装

### 方式 A：自动安装到仓库

```bash
E:/CondaEnvData/envs/GGMPC/python.exe install.py --repo E:/CarDataSample/Neural-IRISDiffuser
```

会把 `src/carla_bridge/`、`scripts/carla_*.py`、`configs/carla_demo.yaml`、`docs/CARLA_DEMO_*`
复制进仓库对应位置（已存在的同名文件会先备份为 `.bak`）。

### 方式 B：手工复制

```
carla_trajsafe_bridge/src/carla_bridge/   ->  <repo>/src/carla_bridge/
carla_trajsafe_bridge/scripts/*.py        ->  <repo>/scripts/
carla_trajsafe_bridge/configs/*.yaml      ->  <repo>/configs/
carla_trajsafe_bridge/docs/*.md           ->  <repo>/docs/
```

脚本用 `ROOT = <repo>` 加入 `sys.path`，因此必须保持上面这个目录层级。

---

## 3. 快速开始

```bash
# 0) 启动 CARLA（必须用 WMI 启动才能跨命令存活；脚本已备好）
powershell -File carla/launch_carla.ps1

# 1) 只规划一次并冻结结果（含 10 项硬验收）
python scripts/carla_trajsafe_demo.py --mode plan

# 2) 用冻结的 plan 在 CARLA 里闭环行驶（不会重新规划）
python scripts/carla_trajsafe_demo.py --mode drive

# 3) 一次跑完（推荐）
python scripts/carla_trajsafe_demo.py --mode all

# 4) 只校验 plan 是否压在真实车道上，不开车
python scripts/carla_verify_plan.py
```

固定场景的 Guided 运行输出 `trajsafe_guided.mp4` 和 `plan_vs_executed_guided.png`。
RAW 仅用于生成 `trajectory_guided_vs_raw_16step.png` 轨迹对比图，不生成右侧视频面板。

### 长距离连续规划（160 m / 四步去噪）

```bash
# 启动 Town03_Opt 后，复现已验证的 spawn 0 -> 200（201.8 m）路线
python scripts/carla_trajsafe_continuous.py

# 显式指定长距离起终点
python scripts/carla_trajsafe_continuous.py --start-spawn 12 --goal-spawn 87
```

连续入口使用 `configs/carla_continuous.yaml`，关键约束如下：

- `planner.reverse_steps` 必须为 `4`，其他值会直接拒绝启动；
- `A_oneshot:best_task` + `config_160k8p.yaml`；
- `carla.python_agents` 指向 CARLA 安装包的 `PythonAPI/carla`（其中包含 `agents`）；
- CARLA 全局 route 提供导航意图，模型每轮规划约 90 m；
- 控制器继续以 20 Hz 跟踪，模型在后台单线程异步规划；
- 默认在旧轨迹前方 15 m 设置接管点，过期或航向跳变过大的计划会丢弃；
- 没有可接管的新计划时，车辆在当前局部轨迹末端受控停车。
- 默认启用 45 m 跟随式俯视相机并输出 1024x1024、10 FPS 的 MP4；
- 录像叠加全局 route（青）、当前计划（洋红）、待接管计划（黄）、已执行轨迹（白）
  和接管点（橙），HUD 显示活动/候选 plan 编号及 `TRACKING/PLANNING/READY` 状态。

连续模式输出 `continuous_manifest.json`、`continuous_trajectory.csv` 和
`trajsafe_continuous.mp4`，每个 plan 均记录 `reverse_steps: 4`、推理延迟、ALM 状态、
验收结果和 route 偏差。录像参数可在 `configs/carla_continuous.yaml` 的 `video` 段调整。

在线部署使用未腐蚀的 CARLA 真实道路边界，并对完整车身四角做二次验收；训练缓存
使用的 k=8 obstacle erosion 只用于训练分布，不作为实车安全边界。四步计划采用
2 次 warmup + 2 次 stronger-ALM guided forward，总网络前向仍严格为 4 次。

停止 CARLA：`taskkill /F /IM CarlaUE4.exe`。

---

## 4. 目录结构

```
carla_trajsafe_bridge/
├─ README.md                     本文件
├─ install.py                    安装到模型仓库
├─ configs/carla_demo.yaml       全部参数（唯一配置入口）
├─ carla/launch_carla.ps1        用 WMI 启动 CARLA 服务端
├─ carla/run_carla.bat           服务端启动的批处理外壳
├─ scripts/
│  ├─ carla_trajsafe_demo.py     命令行入口（plan / drive / all，--ablation none|raw）
│  └─ carla_verify_plan.py       plan 对 CARLA 真值地图的校验
├─ src/carla_bridge/
│  ├─ frame.py                   world / local / scene / grid 坐标变换（唯一来源）
│  ├─ occupancy.py               占据图：数据集快照、实时栅格化、障碍烧录、车身越界度量
│  ├─ planner_adapter.py         推理引擎包装 + 走廊裕度覆盖 + 结果换算 + 硬验收
│  ├─ path_profile.py            弧长重采样、航向、曲率、速度剖面
│  ├─ controller.py              Pure Pursuit + PID -> Ackermann
│  ├─ scenario.py                CARLA 连接、自车、障碍车、转向探针、俯视相机录像
│  ├─ overlay.py                 俯视投影 + 几何 / 可行驶区 / 走廊叠加
│  ├─ demo.py                    plan / drive 主流程
│  └─ visualize.py               离线出图
├─ docs/
│  ├─ CARLA_DEMO_TEST0056.md     实现说明、坐标约定、踩过的坑、指标
│  └─ CARLA_DEMO_TEST0056_SLIDE_TEXT.md   PPT 文案（配置块 / 效果块 / 口播稿）
└─ examples/                     一次真实运行的结果样例
   ├─ trajectory_both_on_occupancy.png   guided vs raw 画在同一张占据图上
   ├─ trajectory_on_occupancy.png        同上，左右分幅
   ├─ plan_test_0056.json / _raw.json    冻结的规划指标
   └─ run_manifest_guided.json / _raw.json  闭环指标
```

---

## 5. 模块职责与关键函数

### frame.py — 坐标系统（最重要）

三个坐标系之间的所有换算只在这里实现一次，任何调用点都不允许自己写 yaw 旋转或上下翻转。

```
world   CARLA 世界 XY（米）
local   以数据集锚点自车为原点的米制坐标系
        x_local = (world - anchor) . forward_xy    前方为正
        y_local = (world - anchor) . right_xy      右侧为正
scene   模型使用的 [-1,1]^2，80 m 窗口
        x_scene = 2*(x_local + 10)/80 - 1
        y_scene = y_local / 40
grid    canonical 256x256 图像
        col = (x_scene + 1) * 0.5 * (res - 1)
        row = (y_scene + 1) * 0.5 * (res - 1)
```

关键函数：`LocalFrame.from_episode()`、`to_local/to_world`、`local_to_scene/scene_to_local`、
`to_scene/world_from_scene`、`scene_to_grid/grid_to_scene`、`raw_occupancy_to_canonical()`。

### occupancy.py — 占据图

- `load_dataset_occupancy()`：读 processed 快照（**已经是 canonical，不要再翻**）；
- `build_canonical_occupancy()`：从 CARLA 车道 quad 实时栅格化，内部只做一次 flipud；
- `add_box_obstacles()`：把场景里的停放车按 yaw 烧进占据图（Minkowski 膨胀）；
- `inflate_obstacles()`：整体腐蚀可行驶区（C-space 近似，本路口未采用，保留接口）；
- `body_free_rate()`：沿轨迹推自车矩形，算车身越界比例与最深穿透。

### planner_adapter.py — 规划与验收

- `PlannerAdapter(corridor_region_override=...)`：构造 dashboard 的同一个 Engine，
  并可覆盖凸走廊安全裕度（本包默认 1.20 m = 自车半宽 0.995 + 0.20）；
- `PlannerAdapter.plan()`：一次 `Engine.generate()`，把结果换算成 world 几何并返回 `PlanResult`；
- `obstacle_body_gap()`：真实车身间隙；
- `check_acceptance()`：10 项硬验收。

### path_profile.py — 轨迹后处理

模型输出的 128 个 scene 采样点弧长不均匀，先按 0.20 m 重采样，再算航向、曲率与速度剖面
（曲率限速 + 终点制动限速 + 反向可达性回扫）。

### controller.py — 控制

`PurePursuitPID.command()`：前视距离 `Ld = clip(1.8 + 0.45*v, 2.0, 4.0)`，
`delta = atan2(2*W*sin(alpha), Ld) + 0.25*heading_error`，纵向 PID 出加速度，
终点前 0.6 m 切显式全刹车。转向符号由 `scenario.probe_steer_sign()` 实测（本车为 +1）。

### scenario.py — CARLA 侧

- `CarlaSession.connect()`：连接、必要时 `load_world` / `reload_world`、切同步模式、设天气；
- `spawn_ego()`：按数据集记录位姿生成自车；
- `spawn_static_vehicle()`：停放障碍车（拉手刹，之后不再动）；
- `probe_steer_sign()`：用 throttle + steer 实测转向符号（Ackermann 从静止起步不动，故不用它）；
- `CameraRecorder`：固定俯视 RGB 相机录像，支持把几何叠加到每一帧。

### demo.py — 主流程

- `build_sample_context()`：样本溯源 -> 局部坐标系 -> 占据图 -> 障碍烧录；
- `run_plan()`：**唯一一次**规划 + 车身越界度量；
- `save_plan()/load_plan()`：把规划结果（曲线、走廊、椭圆、候选、占据图）冻结成 npz/json；
- `run_drive()`：闭环主循环（tick -> 读位姿 -> 控制 -> `apply_ackermann_control` -> 录像 -> 判停）。

### overlay.py — 画面叠加

相机是固定世界朝向的俯视相机，投影为纯相似变换，无透视标定：

```
u = cx + s * (P.y - C.y)      s = f / h,  f = (W/2) / tan(fov/2)
v = cy - s * (P.x - C.x)      相机 right = world +Y, up = world +X
```

静态几何只投影一次，每帧只加平移偏移。占据图用 `cv2.warpAffine` 直接仿射进画面，
所以"轨迹压到非行驶区"是像素级对齐的。

---

## 6. 配置说明（configs/carla_demo.yaml）

| 段 | 关键项 | 含义 |
|---|---|---|
| `carla_bridge` | `dataset_root` / `processed_root` | 数据集与 processed 快照根目录 |
| `carla` | `town` | 用 **Town03_Opt**（非优化版 Town03 会让 CARLA 着色器崩溃） |
| | `fixed_dt` | 0.05 s，即 20 Hz 同步仿真 |
| | `reload_world` | 每次运行前重载世界，保证可复现 |
| | `ego_blueprint` / `spawn_z_offset_m` | 自车车型与离地高度 |
| `sample` | `condition_scene` | 起终点（scene 坐标），当前终点 = 前方 35.4 m / 右侧 33.9 m |
| `scenario` | `obstacle_margin_m` / `obstacles` | 停放车位置与膨胀量 |
| | `ego_length_m` / `ego_width_m` | 车身尺寸，用于越界验收 |
| `planner` | `corridor_region.safety_margin` | **凸走廊安全裕度，0.030 scene = 1.20 m** |
| | `diffusion_seed` / `model_id` / `alm_enabled` | 采样种子、权重、ALM 开关 |
| | `occupancy_source` | `dataset`（默认）或 `live` |
| `controller` | `wheelbase_m` / `max_steer_rad` | **2.641 m / 1.2217 rad（70 度）**，实测值 |
| | `max_speed_mps` / `a_lat_max_mps2` / `a_brake_mps2` | 速度剖面约束 |
| | `steer_sign` | 0 = 运行时用探针实测 |
| `video` | `width/height` | 1024x1024 方形，只框动作区 |
| | `camera_height_m` / `fov` / `center_local_m` | 可见 56 x 56 m，18.29 px/m |
| | `show_occupancy/show_corridor/show_ellipses` | 叠加开关（椭圆默认关闭） |
| `output` | `dir` | 产物目录 |

---

## 7. 数据与坐标约定（踩过的坑）

1. `data/carla_processed/<split>/occupancy.npy` **已经是 canonical（y 已翻转）**，
   与 CARLA 车道 quad 原始栅格化相比 `flipud(raw)` 有 99.33% 像素一致。翻转只能做一次。
2. `y_scene > 0` 是自车**右侧**。
3. `trajectory_raw.npy` 第 4 列（yaw）单位是**角度**，不是弧度。
4. 画图时底图与叠加必须**一起翻转**；只翻叠加会把曲线镜像到黑色区域。
5. 判断车身越界时行号必须用 canonical 映射 `row = (y_scene+1)*127.5`；
   写成 `(40-y_local)*3.1875` 会拿到上下镜像的占用图，结论是假阴性。
6. `WheelPhysicsControl.max_steer_angle` 的单位是**度**（本车 70.0），不要再做一次角度换算。

---

## 8. 验收项（10 项，全过才允许发车）

| 验收 | 阈值 |
|---|---|
| `guided` | True |
| `alm_status_guided` | guided |
| `final_collision_false` | 稠密 512 点无碰撞 |
| `endpoint_error` | <= 1e-3 scene |
| `max_constraint_violation` | <= 5e-3 scene |
| `corridor_membership` | >= 0.99 |
| `dense_curve_free` | True |
| `curvature_within_vehicle` | <= tan(max_steer)/wheelbase = 1.040 1/m |
| `obstacle_body_gap` | >= 0.30 m |
| `body_in_drivable_area` | >= 0.99（自车四角在未腐蚀可行驶区内） |

任一失败：`--mode all` 不会启动车辆，只保存 plan 与失败原因。

---

## 9. 复现结果（examples/ 里的那一次运行）

样本 test_0056 = processed #56 -> episode 70 / Town03 / 第 104 帧；
起点 (132.085, 62.487) yaw -0.149 deg；终点前方 35.4 m / 右侧 33.9 m；
支路停放一辆 `vehicle.mini.cooper_s`。

| 指标 | Guided (ALM on) | Raw (ALM off) |
|---|---:|---:|
| 规划长度 | 69.16 m | 59.19 m |
| 轨迹越界 | 0 / 128 | 53 / 128 (41.4%) |
| 车身越出可行驶区最深 | 0.63 m | 4.70 m |
| 对障碍车车身间隙 | 2.07 m | 0.00 m |
| 最大曲率（极限 1.040） | 0.652 | 0.113 |
| 实车结果 | arrived，误差 0.435 m，0 碰撞 | collision，撞停放车 6 次，停在终点前 17.8 m |
| 横向误差 RMS | 0.276 m | 0.189 m |

---

## 10. 已知限制

1. **规划约 1.8 s / 次**，本包采用"停车规划一次 + 闭环跟踪"，不做在线重规划。
2. **Ackermann 低速特性**：静止起步约 1 s 后齿轮才啮合；速度环低速会超调，
   因此终点前 0.6 m 用显式全刹车。
3. **长驻服务端会漂移**：跑久了 `try_spawn_actor` 会在原本能生成的位置失败，
   所以每次运行都 `reload_world()`。
4. **CARLA 非优化版 Town03 会崩溃**（`Shader compilation failures are Fatal`），
   必须用 `Town03_Opt`（路网相同）。
5. **CARLA 进程会被工具 shell 连带杀掉**，需用 WMI 启动（见 `carla/launch_carla.ps1`）。
6. 走廊裕度只有横向语义：急弯处车头/车尾角点仍可能轻微越界（当前 0.35%）。

---

## 11. 产物清单（运行后生成在 configs 里的 output.dir）

```
plan_<sample>.npz / .json          冻结的 guided 规划结果
plan_<sample>_raw.npz / .json      冻结的 raw 规划结果
run_manifest.json / _guided.json / _raw.json   闭环指标与验收
trajectory_guided.csv / _raw.csv   逐帧状态、控制量、横向误差
trajsafe_guided.mp4 / _raw.mp4     1024x1024 俯视录像（含几何叠加）
trajsafe_comparison.mp4            2048x1024 左右对比
trajectory_both_on_occupancy.png   两条轨迹画在同一张占据图上
trajectory_on_occupancy.png        同一底图左右分幅
plan_vs_executed_guided.png / _raw.png  各模式规划 vs 实车轨迹
```
