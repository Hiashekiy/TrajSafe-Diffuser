# test_0056 接入 CARLA：闭环跑通 + guided/raw 消融

> 实现 `CARLA_CLOSED_LOOP_INTEGRATION_PLAN.md` 第一版：固定场景、静态障碍、
> 停车规划一次、闭环跟踪到终点；轨迹 / 凸走廊 / 可行驶区直接画在俯视相机画面上。

## 1. 结果

场景：起点取数据集记录位姿，终点取操作者在画面上圈出的区域
（local **35.4 m 前方、33.9 m 右侧** = scene `[0.1353, 0.8473]`），
支路里停一辆 `vehicle.mini.cooper_s`。

同 checkpoint、同 seed、同 occupancy、同控制器，只切 `alm_enabled`：

| 指标 | Guided (ALM on) | Raw (ALM off) |
|---|---:|---:|
| `guided` / `alm_status` | True / guided | False / disabled |
| `dense_curve_free` | **True** | **False** |
| 规划中心线越界 | **0 / 128** | **52 / 128 (41%)** |
| **车身在可行驶区内** | **99.35%** | **57.77%** |
| **车身最大越界深度** | 0.63 m | **4.70 m** |
| 对障碍车车身间隙 | **2.07 m** | **0.00 m（贴上）** |
| 规划长度 | 69.16 m | 59.19 m |
| 最大曲率（自车极限 1.040） | 0.652 | 0.113 |
| **实车结果** | **arrived**，终点误差 0.445 m | **collision**，撞停放车 6 次，停在终点前 17.8 m |
| 横向误差 RMS | 0.278 m | 0.189 m |
| CARLA 碰撞 | **0** | **6**（全部为 `vehicle.mini.cooper_s`） |

10 项硬验收 guided 全过；raw 按预期挂掉 8 项
（guided / alm_status / final_collision / max_constraint_violation /
corridor_membership / dense_curve_free / obstacle_body_gap / body_in_drivable_area）。

## 2. 这一轮修的东西

### 2.1 终点挪到圈出的区域

原来终点是 dataset 自带的 `[0.1911, 0.5339]`（local 37.6, 21.3），
路线短、弯不急，raw 只切角 1.13 m，看不出差别。
现在终点挪到 local (35.4, 33.9)，路线从 52 m 拉到 69 m，
raw 的越界直接涨到 4.7 m、并真的撞上停放的车。

### 2.2 修正自车转向极限（重要）

之前把 `WheelPhysicsControl.max_steer_angle` 读成"度 x 100 = 40.1 度"，
于是把曲率上限压到 `tan(0.70)/2.642 = 0.319`。
实测该字段就是**度**：

```
wheel x positions 13332.5 / 13068.4 cm  ->  wheelbase 2.641 m
max_steer_angle = 70.0 deg = 1.2217 rad
min turning radius = 2.641 / tan(70 deg) = 0.961 m
curvature limit    = 1.040 1/m
```

旧上限把一批本来合法的规划误判成"曲率超限"（正是它挡住了终点外移）。
现在 `controller.wheelbase_m = 2.641`、`max_steer_rad = 1.2217`，
控制器和验收共用同一组真值。

### 2.3 凸走廊安全裕度 0.02 -> 0.030

规划器把轨迹当**质点**，所以"中心线在可行驶区内"不等于"车在可行驶区内"。
决定曲线离路缘多远的是 `corridor.region.safety_margin`（scene 单位，1 = 40 m），
默认 **0.02 = 0.80 m**，比自车半宽 0.995 m 还小。

现在由 bridge 覆盖为 **0.030 = 1.20 m = 半宽 0.995 + 车身余量 0.20**，
放在 `configs/carla_demo.yaml` 的 `planner.corridor_region.safety_margin`，
经 `PlannerAdapter(corridor_region_override=...)` 注入，**不改模型仓库的 `configs/config.yaml`**。

| safety_margin | 车宽 | 车身在区内 | 最大曲率 |
|---:|---:|---:|---:|
| 0.020 | 0.80 m | 96.8% | 0.232 |
| 0.026 | 1.04 m | 98.6% | 0.295 |
| **0.030** | **1.20 m** | **99.8%** | **0.270** |
| 0.040 | 1.60 m | 100% | 0.435 |

> 另一条路（对 occupancy 整体做半宽 Minkowski 腐蚀）在这个路口走不通：
> 腐蚀 0.6–0.94 m 把转弯半径压到 1.4 m，腐蚀 1.25 m 以上凸区域直接构建失败。
> 只推走廊半空间才是对的杠杆。

### 2.4 新增硬验收 `body_in_drivable_area`

把自车 4.18 x 1.99 m 的矩形沿弧长重采样后的曲线推一遍，四角必须落在**未腐蚀的**
可行驶区内，比例 >= 0.99。实现在 `occupancy.body_free_rate()`。
这样规划器再怎么"自欺"也过不了这一关。

### 2.5 不再画椭圆

`video.show_ellipses: false`（`OverlayGeometry.draw` 默认也改成 False）。

## 3. 相机与叠加

固定不跟随的俯视 RGB 相机，**方形 1024 x 1024**，高度 28 m、fov 90°，
中心在 ego-local (20, 19) 处。可见 **56 x 56 m、18.29 px/m**，
只框住动作区域（local x ∈ [-8, 48]，local y ∈ [-9, 47]），方便直接放进 PPT。

图像横轴 = ego-local +y，纵轴 = ego-local +x（该相机姿态下 right = world +Y、
up = world +X）。
相机是固定世界朝向（pitch -90, yaw 0, roll 0）俯视，投影为纯相似变换：

```
u = cx + s * (P.y - C.y)
v = cy - s * (P.x - C.x)        s = f / h,  f = (W/2) / tan(fov/2)
```

* **可行驶区着色**：绿=可行驶、红=路缘/草地/建筑。canonical 256² 栅格用
  `cv2.warpAffine` 仿射进相机画面，所以"raw 压到红色区域"是像素级对齐的；
* **凸走廊**绿色半透明、**guided B 样条**洋红、**raw 预测**橙虚线；
* 实车轨迹白色、障碍车红框、起终点圆点；
* HUD：模式/ALM 状态、t/v/弧长、横向与航向误差、规划越界比例、车身间隙、碰撞数。

## 4. 样本溯源

```
test_0056 -> processed index 56 -> sample_id 1252 -> episode 70 -> Town03 -> anchor_raw_index 104
```

起点 = episode 70 第 104 帧自车位姿 world (132.0854, 62.4873)，yaw -0.1489 deg；
局部基向量取自同一帧 `ego_frame_vectors.npy`。
起点落在 CARLA 真值 road 52 / lane 1（Driving），离车道中心 0.000 m，航向点积 1.000000。

## 5. 坐标约定（关键坑）

```
world --(记录的自车基向量)--> local(米) --线性映射--> scene [-1,1]^2 --> canonical 256²
x_local = (world - anchor) . forward_xy     x_scene = 2*(x_local+10)/80 - 1
y_local = (world - anchor) . right_xy       y_scene = y_local / 40
```

* `occupancy.npy` **已经是 canonical（y 已翻转）**：与 CARLA 车道 quad 原始栅格化相比
  `flipud(raw)` 有 99.33% 像素一致，翻转只能做一次；
* `y_scene > 0` 是自车**右侧**；`trajectory_raw.npy` 第 4 列 yaw 是**角度**；
* **画图时底图和叠加必须一起翻转**（只翻叠加会把曲线镜像到黑色区域）；
* 检查车身越界行号必须用 canonical 映射 `row=(y_scene+1)*127.5`，
  用 `(40-y_local)*3.1875` 会得到上下镜像的占用图，结果是假阴性（这个错犯过）。

## 6. 环境中的坑

1. **非优化版 `Town03` 会崩**（`Shader compilation failures are Fatal`）；
   `Town03_Opt` 路网完全相同且正常，故 `carla.town: Town03_Opt`。
2. **Ackermann 静止起步不动**：静止时下发有 throttle 但位移严格为 0，约 1 s 后啮合；
   速度环低速超调，终点前 0.6 m 改为显式 `brake=1.0`。转向探针改用 throttle + steer。
3. **长驻 server 会漂移**：跑久了 `try_spawn_actor` 会在原本能生成的位置失败，
   每次 run 开头 `reload_world()`。
4. **CARLA 进程会被工具 shell 连带杀掉**：用 WMI `Win32_Process.Create` 启动
   （`launch_carla.ps1`）才能常驻；停止：`taskkill /F /IM CarlaUE4.exe`。

## 7. 复现

```bash
powershell -File outputs/carla_t0056/launch_carla.ps1

E:/CondaEnvData/envs/GGMPC/python.exe scripts/carla_trajsafe_demo.py --mode all
E:/CondaEnvData/envs/GGMPC/python.exe scripts/carla_trajsafe_demo.py --mode all --ablation raw
E:/CondaEnvData/envs/GGMPC/python.exe scripts/carla_verify_plan.py
```

## 8. 代码结构

```
src/carla_bridge/frame.py             world/local/scene/grid 唯一变换源
src/carla_bridge/occupancy.py         快照 / 实时栅格化 / 障碍烧录 / 腐蚀 / 车身越界度量
src/carla_bridge/planner_adapter.py   Engine 包装 + 走廊裕度覆盖 + 换算 + 硬验收
src/carla_bridge/path_profile.py      弧长重采样、航向、曲率、速度剖面
src/carla_bridge/controller.py        Pure Pursuit + PID -> Ackermann
src/carla_bridge/scenario.py          Town 复现、自车、障碍车、转向探针、俯视相机
src/carla_bridge/overlay.py           俯视投影 + 几何/可行驶区/走廊叠加
src/carla_bridge/demo.py              plan / drive 主流程
src/carla_bridge/visualize.py         离线图（occupancy 上的 plan / 实车轨迹）
scripts/carla_trajsafe_demo.py        入口（--mode plan|drive|all，--ablation none|raw）
scripts/carla_verify_plan.py          plan 对 CARLA 真值地图的校验
configs/carla_demo.yaml               全部参数
```

## 9. 产物

```
trajsafe_comparison.mp4     左右对比（2048x1024，左 GUIDED 右 RAW）
trajsafe_guided.mp4         guided 单独（1024x1024，含橙色虚线的 raw 预测）
trajsafe_raw.mp4            raw 单独（1024x1024，能看到切进广场并撞车）
trajectory_both_on_occupancy.png **两条轨迹画在同一张 256² 占据地图上（主展示图）**
trajectory_on_occupancy.png 同一张占据地图、左右两幅分开画
plan_over_occupancy.png     规划 + 走廊 + 候选画在占据地图上
plan_vs_executed_guided.png / _raw.png   各模式规划 vs 实车轨迹
trajectory_guided.csv / trajectory_raw.csv  各模式逐帧状态
plan_test_0056*.npz/.json   冻结的规划结果
run_manifest.json / _guided.json / _raw.json  指标与验收
```

展示图：

* `trajectory_both_on_occupancy.png` —— **一张底图两条曲线**：洋红 = guided（0/128 越界，
  全程贴着白色可行驶区），橙色 = raw（53/128 越界，越界处用红点高亮，能看出它斜穿黑色区域），
  另附起点/终点/停放车与实际车迹；
* `trajectory_on_occupancy.png` —— 同一张底图左右分幅，各自带实车轨迹，
  适合并排讲"规划 vs 实际"。

## 10. 尚未做

* 相机画面里没有画椭圆（按要求去掉）；
* 动态障碍、在线重规划、tracking MPC（计划文档第二阶段）；
* 该场景已满足"raw 必撞 / guided 必过"，如要更严格可再跑一遍多 seed 场景搜索固化。
