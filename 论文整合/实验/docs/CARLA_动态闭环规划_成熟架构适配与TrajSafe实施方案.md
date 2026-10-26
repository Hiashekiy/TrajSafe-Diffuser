# TrajSafe-Diffuser × CARLA 0.9.16：动态交通在线闭环规划实施方案

> **性质**：论文中的单场景仿真应用，不开展其他规划器对比或新的消融实验。  
> **核心要求**：复用成熟自动驾驶架构中的路线管理、行为安全批准、动态目标预测、速度规划及轨迹管理；**以 TrajSafe-Diffuser 加速版替换局部几何轨迹生成器**。  
> **实施平台**：CARLA 0.9.16，Python，项目现有 `diffusion-dashboard/engine_carla.py::Engine` 等接口以实际仓库核对结果为准。  
> **本文件为可执行开发规格，不代表代码已完成或数值已实测**。旧集成方案仅用作 CARLA 接口、坐标、场景和跟踪控制的参考；当前方案采用**行驶中持续规划**，不采用“停下规划一次再行驶”的旧演示流程。

---

## 0. 一页结论：选择哪套成熟方法？

**最终选择：以 Autoware 的分层路径规划思想为主线，搭配 CARLA 原生 GlobalRoutePlanner 和 ScenarioRunner。**

- **实际复用的现成代码**：CARLA 0.9.16 的全局导航/waypoint API、Actor/BoundingBox/VehicleControl API；可选直接使用与 CARLA 0.9.16 匹配的 ScenarioRunner 0.9.16 固定 NPC 行为和触发条件。
- **借鉴、轻量实现而不整套移植的成熟机制**：Autoware Behavior Path Planner 的模块化 `Request → Ready → Plan/Execute`、候选路径执行前/执行中安全检查、Obstacle Cruise/Stop 的纵向避障思想、Velocity Smoother 的速度/加速度/曲率约束。
- **明确替换的算法环节**：原本行为模块可能使用中心线平移、采样或优化生成的**局部几何路径**，统一改为 `TrajSafe-Diffuser(O, S, G) → 三次 B 样条 48 控制点 → 安全修正后路径`。不要再额外实现第二个 A*/Frenet/路径优化器来偷偷替代生成结果。
- **保留的非生成模块**：导航目标与合法车道、行为状态、动态预测、时空碰撞检查、停走/速度剖面、路径采纳管理、Pure Pursuit + PID 跟踪、安全停车。这些不是本文规划网络的创新，论文应明确区分。

### 0.1 调研结论与真实引用

| 参考体系 | 查证到的成熟做法 | 本项目怎么用 |
|---|---|---|
| CARLA Navigation Agents / Waypoints | 使用道路 waypoint 与全局路由维护从起点到目的地的路线；局部控制沿路线运行 | 实际导入 `GlobalRoutePlanner`，提取全局路线、目标车道和可通行车道 |
| Autoware Behavior Path Planner | 按场景激活 Lane Following、Static Obstacle Avoidance、Avoidance by Lane Change、Lane Change 等模块；候选路径与已批准路径分别检查 | 用轻量状态机实现本场景的 `FOLLOW / PREPARE / WAIT_GAP / CHANGE / PASS / RETURN` |
| Autoware Lane Change | 要求请求条件、合法换道车道、候选路径有效、安全检查通过；执行中持续安全检查与完成/终止判断 | 不因 Diffusion 每次结果变化而立即换拓扑；换道需要安全批准和状态锁定 |
| Autoware Obstacle Cruise Planner | 在已给定路径上进行停止、跟车/巡航、减速处理，独立于几何路径生成 | 用简单纵向速度规划与安全制动管理动目标，不要求 Diffusion 预测时间 |
| Autoware Velocity Smoother | 根据限速、曲率、加速度、加加速度和停车点生成可执行速度剖面 | 在 B 样条路径后生成 `v_ref(s)`；第一版简化为带限幅的前后向速度扫描 |
| CARLA Traffic Manager / ScenarioRunner | TM 可以生成背景交通；ScenarioRunner 的 0.9.16 版本与 CARLA 0.9.16 匹配，适合固定场景 | **主角干扰车辆用确定性的脚本/ScenarioRunner控制**；非关键背景车辆可用 TM |

**官方或原始资料（开发时优先查这些，而非搜索摘要）：**

1. [Autoware Behavior Path Planner](https://autowarefoundation.github.io/autoware_universe/main/planning/behavior_path_planner/autoware_behavior_path_planner/)
2. [Autoware Lane Change Design](https://autowarefoundation.github.io/autoware_universe/main/planning/behavior_path_planner/autoware_behavior_path_lane_change_module/)
3. [Autoware Obstacle Cruise Planner](https://autowarefoundation.github.io/autoware.universe_planning/latest/planning/obstacle_cruise_planner/)
4. [Autoware Velocity Smoother](https://autowarefoundation.github.io/autoware_core/pr-689/planning/autoware_velocity_smoother/)
5. [CARLA 0.9.16 Traffic Manager](https://carla.readthedocs.io/en/0.9.16/adv_traffic_manager/)
6. [CARLA ScenarioRunner 项目及版本说明](https://github.com/carla-simulator/scenario_runner)；[v0.9.16 release](https://github.com/carla-simulator/scenario_runner/releases/tag/v0.9.16)
7. [CARLA 官方源码的导航 agents 目录](https://github.com/carla-simulator/carla/tree/0.9.16/PythonAPI/carla/agents/navigation)

**重要范围说明**：这不是“直接安装 Autoware 再替换一个函数”——Autoware 的 ROS 2、Lanelet2、规划消息和执行管理与现有 Python/CARLA 网络不直接兼容。为控制论文实验工作量，采用**CARLA 原生接口 + Autoware 已验证过的管理设计**。如果后续确实需要完整 Autoware 节点级集成，应另立工程任务，不在本实验实施范围。

---

## 1. 最终仿真应用的定义

### 1.1 场景名称

**前方低速车停车 + 左侧来车通过 + 自车等待后安全换道绕行**（同向多车道道路）。

选择 **至少两条同向合法行驶车道、足够长且无复杂路口影响的道路段**。首选 Town05 中经过地图查询验证的道路段；若 Town05 没有满足地图尺度、变道许可、障碍膨胀和路线长度的路段，则从现有 CARLA Town 中自动筛选其他道路段。**不得在没有查地图的情况下硬写 Town05 某条道路/坐标一定可用。**

**参与车辆：**

- `ego`：初速约 5 m/s，从原车道沿既定全局路线行驶；由本系统控制，绝不交给 Traffic Manager 自动驾驶。
- `lead`：初始在 ego 前方约 30–40 m、速度约 2 m/s；经过触发位置后以限定减速度停车，占用 ego 车道。
- `adjacent`：在目标车道、ego 后方约 15–25 m，初速约 7–9 m/s；从后方接近并超过 ego，形成换道冲突。
- 可选 `background`：一辆远离关键冲突区域的背景车，只用于画面氛围，**第一版默认关闭**。

车辆蓝图、车道具体坐标、速度和初始距离先由地图筛选和简单仿真检查确定，参数属于**工程初始化候选值**，不是已验证可行的 CARLA 场景配置。所有运动脚本按仿真时间/空间触发，不按实际墙钟时间触发。

### 1.2 希望自然发生的流程（不是预先给 ego 写动作剧本）

1. Ego 跟随全局路线；前车尚能低速行驶，系统执行跟车速度规划。
2. 前车开始减速并最终停止；它被识别为持续低速/准静态障碍，进入用于几何规划的障碍栅格。
3. 行为管理请求绕行/换道；TrajSafe-Diffuser 持续生成局部参考路径；左后方快速来车使换道安全检查暂时不通过。
4. Ego **保持原车道及其已批准路径，减速到障碍物前停车等待**；不能执行尚未被批准的换道轨迹。
5. 邻道车辆通过、安全检查满足后，批准一条 TrajSafe-Diffuser 生成的换道路径；ego 开始执行。
6. 绕过停靠车辆，满足后方纵向距离和目标车道状态约束后，更新行为目标，返回原车道；继续沿全局路线到达最终目的地。

**禁止**通过脚本强制 ego 在某时刻换道或把其 transform 直接瞬移到某车道；自车的轨迹必须来自模型的实际输出。

### 1.3 仿真范围

- 一次完整闭环演示；**不做多方法对比、不重新训练、不把结果伪装成动态安全性的理论证明**。
- CARLA 真值输入：道路几何、车身包围盒、车辆位姿与速度；不增加感知网络。
- 不加行人、交通信号复杂交互、交叉路口博弈或多模态车辆行为预测。
- 目标是**几何轨迹生成与连续重规划的工程应用**，动态冲突由外部安全速度模块保障。

---

## 2. 系统总架构与替换位置

```text
                         CARLA 0.9.16 (20 Hz)
     world map ──┐             ego state ───┐        other actor states
                  ▼                          ▼                   ▼
            RouteManager                EgoPose              ObjectTracker
                  │                          │                   │
             fixed global route         current S         current pose/speed/box
                  │                          │                   │
                  ├──────► BehaviorManager ◄─┴───────┬───────────┘
                  │       FOLLOW / WAIT / CHANGE...   │
                  │                                   ▼
                  └─► GoalManager              OccupancyBuilder
                      local G + allowed lanes   road+quasi-static obstacles
                                │                     │
                                └─────────┬───────────┘
                                          ▼
                              ReplanManager (5→10 Hz)
                                          │
                               TrajSafe-Diffuser
                         4 DDIM steps + region sharing + ALM
                                          │
                               B-spline Q[48,2]
                                          ▼
                                  PathProcessor
                           arc-length/heading/curvature
                                          ▼
                                TrajectoryManager
                              static/vehicle/continuity
                                  validity gating
                                          │
                           approved geometric path only
                                          ▼
                                  VelocityPlanner ◄──── ObjectPredictor
                            speed limits/stop/follow/gap  5–7 s prediction
                                          ▼
                            Controller (20 Hz PP+PID)
                                          ▼
                               carla.VehicleControl
                                          │
                          Car motion → state feedback

        SafetySupervisor (20 Hz): independent collision monitor + safe stop
```

**所有环境输入有 snapshot_id 和 sim_timestamp；下游规划结果携带用于计算时的快照编号。**

**和原 Autoware 的职责分工保持一致，但几何路径仅由 TrajSafe-Diffuser 生成。** 如果行为管理要求换道，是它决定“允许使用哪些车道、局部终点应落在哪一条车道”，而不是另行生成换道曲线。

---

## 3. RouteManager：全局终点与局部目标

这是两个最重要问题中的**第一个**。不能每次把随机的 50 m 前方位置交给模型，否则终点跳动会造成轨迹跳变。

### 3.1 建立一次全局路线

使用 CARLA 0.9.16 `agents.navigation.global_route_planner.GlobalRoutePlanner` 及 `trace_route(start, destination)` 获得路线 Waypoint 列表（需要对本地包安装版本的构造方式进行核对）。

- `global_goal_world`：整段演示唯一的固定最终目标，距离起点足够远以涵盖跟车、换道、避让、回归车道，建议约 130–180 m，按道路合法性与地图验证调整。
- `R(s)`：由全局路线插值得到的按弧长排列的世界坐标曲线/Waypoints。
- 每帧寻找最邻近且**路线进度单调合理**的点，得到 `s_ego`；限制单帧路线进度不合理回跳，避免相邻平行车道投影跳变。
- 全局路线优先表达**最终期望的原始行驶车道**；绕行是局部行为，不改变最终任务终点。

### 3.2 行为目标与普通局部目标区分

局部目标由 `GoalManager` 根据行为状态和全局路线生成，不直接由 NPC 坐标简单平移。

| 行为状态 | 目标 G 的选择规则 | 备注 |
|---|---|---|
| `FOLLOW` | `R(s_ego + Llook)` 上的原车道位置 | 正常路线跟随 |
| `PREPARE` / `WAIT_GAP` | 可以保持最近一次有效 `G`；或按需求准备**目标车道**的候选目标，但**未批准换道不得执行** | 不把障碍前的停车点错误当作 Diffusion 目标；停车由速度层完成 |
| `CHANGE` | 在目标车道同向 waypoint 上选前方 `Llook` 的位置 | 允许跨越的车道集合为原车道+目标车道 |
| `PASS` | 目标车道向前的 waypoint | 确认完全超过停靠前车 |
| `RETURN` | 原路线车道的前方 waypoint | 验证原车道有足够空间 |
| `DONE` | `global_goal_world` | 终点减速停车 |

`Llook = clip(40 + 3·v_ego, 45, 65) m` 为**工程初始值**；受当前 `160 × 160 m` 地图边界、车道连接和道路终点约束，若有效道路不足则缩短；没有有效目标则返回 `GOAL_INVALID` 并安全停车。

### 3.3 更新局部 G 的规则（解决目标抖动）

- 仅当 **行为状态/目标车道改变**，或沿路线前进 `≥5 m`，或局部目标到 ego 的剩余路线距离 `<20 m`，或目标超出当前可用局部地图时重新选择。
- 在其他 5–10 Hz 重规划调用中保留相同的 `G_world`；每次只把它投影进最新的局部 scene 坐标。
- `G_world` 不因为 NPC 的每帧位置变化而移动；NPC 变化触发的是地图/安全与速度更新。
- 检查 `G` 的自由栅格、允许道路集合、坐标转换、距离和可达性；**不可通过任意移动到最近 free pixel 来掩盖导航目标不合法**。
- 换道过程中锁定目标车道，直到安全中止或完成，不能在两个车道间高频反复改变。

---

## 4. BehaviorManager：复用 Autoware 的 Request / Ready / Execute 思想

这不是“另一个轨迹生成器”，它只负责是否允许进入某种交通行为。建议轻量有限状态机，而不是引入整套 ROS 2 行为树。

### 4.1 状态定义

```text
FOLLOW
  ├─ front obstacle slows/stops ──> PREPARE
  └─ arrived ─────────────────────> DONE
PREPARE
  ├─ valid candidate + gap safe ──> CHANGE
  ├─ gap unsafe ──────────────────> WAIT_GAP
  └─ obstacle clears ─────────────> FOLLOW
WAIT_GAP
  ├─ gap safe stable + valid plan ─> CHANGE
  ├─ blocker clears ──────────────> FOLLOW
  └─ otherwise ───────────────────> WAIT_GAP (follow/stop)
CHANGE
  ├─ lane change completed ───────> PASS
  └─ safety becomes critical ─────> emergency slow/stop or safe abort
PASS
  ├─ passed obstacle with margin ─> RETURN
  └─ otherwise ───────────────────> PASS
RETURN
  ├─ original lane regained ──────> FOLLOW
  └─ unsafe ──────────────────────> wait/continue in target lane
```

### 4.2 三种判据必须分开

1. **Request（是否有换道需求）**：前车变慢且与自车距离/预期路线存在冲突；邻道存在合法同向道路；正常跟车无法按预设任务进度通过。可以设 `v_lead < 0.5 m/s 持续 ≥1 s` 作为准静态前车识别条件，同时支持前车更早减速触发减速跟随。
2. **Candidate（几何是否可行）**：TrajSafe-Diffuser 返回了完整轨迹、与当前车辆航向可衔接、静态/准静态占据物安全、合法车道、满足曲率及最小制动余量；不满足则拒绝。
3. **Ready（动态是否可以执行）**：目标车道前后交通在预计完整换道时间区间内满足车辆包围盒时空检查与纵向距离约束；安全条件稳定超过 `0.3–0.5 s` 才允许开始；**风险出现时撤回许可不需要同等等待**。

**不可只判断左车前方**；必须包括左后方快速来车、相对速度、换道持续时间、车身长度/宽度以及预计到达相交区域的时刻。

### 4.3 何时完成换道与何时返回

- `CHANGE → PASS`：车身中心进入目标车道并保持，横向距车道中心 `<0.5 m`、航向误差 `<10°`、车辆整体占用区域不再跨越原车道界线（阈值按车辆尺寸验证）。
- `PASS → RETURN`：ego 后保险杠超过停靠车辆前保险杠至少 `5–8 m`，原车道预测无新风险，且不存在禁止换道标线。
- `RETURN → FOLLOW`：完整车辆 footprint 回到原车道并稳定，避免单凭位置投影小幅跨线就认定完成。
- 执行中的换道不可因为某次候选路线更短而重新选另一侧；确有危险时交由安全主管判断可停、可安全中止的方向。

---

## 5. 环境地图与移动障碍：不能混成一张“未来占据图”

### 5.1 CARLA 真值信息

复用旧文档的 `frame.py`、`occupancy.py` 思路：

- `world.get_map()` + waypoint/车道宽度生成道路允许区域；车道合法性依据 lane marking、lane change 权限、同向 waypoint 关系。
- `world.get_actors().filter('vehicle.*')` 获取所有 NPC actor；读取 `get_transform()`、`get_velocity()`、`bounding_box.get_world_vertices()`。
- ego 当前状态、其他车辆状态均以同一个 CARLA world tick 的 snapshot 构成，避免一辆车是上帧位置另一辆车是下帧位置。
- 模型输入从 world 转 local meter，再转 scene，然后生成 canonical occupancy。**只由一个接口实现翻转；严禁重复 flipud。**

### 5.2 建议维护三类信息

| 名称 | 含义 | 用途 |
|---|---|---|
| `O_static` | 道路边界、不可行驶车道、固定障碍、持续停车的前车 | 输入 TrajSafe-Diffuser 的几何占据图 |
| `DynamicObjects` | 正在运动的邻道车辆、低速但尚未分类为停车的前车 | 独立预测与时空安全检查；跟车与制动 |
| `AllowedLanes` | 当前行为允许行驶的车道或道路面域 | 限制生成轨迹不可穿越实线或进入不允许道路 |

对停止车辆采用带滞回的准静态转换：如 `|v|<0.5 m/s` 持续 1 s 才进入静态路径障碍，恢复 `|v|>1.2 m/s` 持续 1 s 才移出；速度阈值需根据场景调试。

**不建议把快速邻道车未来数秒走过的整条道路“永久涂黑”给 Diffusion。** 这会把本来可以等待后通过的交通间隙视为永久封闭。移动障碍应通过时序预测和速度规划处理。

### 5.3 安全距离与现有训练尺度（硬检查）

**旧集成方案写的是 80 m × 80 m、`scene 单位 =40 m`，但当前论文统一设置是 160 m × 160 m、256²、0.625 m/grid、障碍膨胀 4 grid、控制点 `C=48`。**

在首次调用当前 checkpoint 前，检查训练 metadata、占据预处理、模型内部 scene 范围、原始 CARLA 数据的像素变换、ALM 安全裕度，建立**唯一真实的坐标约定**。若确认为 160m 且 scene `[-1,1]^2`，则局部物理坐标 `x_local/80, y_local/80`；不要沿用旧文件中的 `/40`。地图窗口中心可以在车辆前方偏移，但需要确认训练分布容许 ego 在图内偏心位置。

**非常重要**：障碍膨胀 4 栅格约 2.5m，会使普通城市车道上的一辆停车汽车“侵占”邻道可用走廊。先在拟选场景用实际包围盒栅格化做可达性检查：

- 若即使允许合法换道，仍无有效通路，不要悄悄调小 4 栅格；优先换用更宽车道/更合理的障碍停靠位置，并如实记录。
- 若确因模型几何安全域与车身/道路约束不匹配，则标记 `SCENE_GEOMETRY_INFEASIBLE`，这属于当前方法的真实限制。
- 道路边界和车身膨胀不能混淆；仅中心点无碰撞不等于实际车辆 footprint 无碰撞。

### 5.4 坐标回环测试

在允许 ego 启动车辆前：world→local→scene→grid→scene→world 的数值转换（忽略栅格量化时）误差 `<0.01 m`；grid round-trip 允许量化误差；左转方向不镜像；起终点均在地图与 free cell；B 样条曲线落点与 CARLA debug 绘图重合；椭圆半轴、角度变换正确。

---

## 6. TrajSafe-Diffuser 作为唯一局部几何路径生成模块

### 6.1 加速推理设置

使用已实现的：

- **4 步 DDIM**；
- **凸区域共享**（历史效率对照里的加速配置）；
- **C=48 的三次 B 样条控制点**；
- **历史反馈、ALM、端点补偿保持正常开启**。

用户报告单次规划约 **20 ms**，将其视为待重新测量的**实际单次端到端延迟假设**。旧实验表中的约 27 ms 为批量推理折算指标，二者不可以互相代替。正式测量至少拆分：地图与骨架构造、模型/采样、ALM、输出解码、静态验收与动态检查。

### 6.2 实际调用契约（接口以仓库核实为准）

```python
@dataclass
class PlanningRequest:
    snapshot_id: int
    sim_time: float
    ego_pose_world: object
    ego_speed_mps: float
    occupancy_canonical: np.ndarray     # [256,256], 0=free, 1=occupied
    start_scene: np.ndarray             # [2]
    goal_scene: np.ndarray              # [2]
    allowed_lane_mask: np.ndarray       # [256,256]
    behavior_state: str
    target_lane_id: str
    model_seed: int

@dataclass
class PlanningCandidate:
    source_snapshot_id: int
    source_sim_time: float
    control_points_scene: np.ndarray   # [48,2] (当前配置)
    curve_world: np.ndarray
    selected_topology: object
    alm_status: str
    final_validation: dict
    latency_ms: float
    used_scene_frame: object             # 生成时 world-local 仿射变换快照
```

**调用旧接口之前需要验证**：`Engine.generate()` 当前是否还支持 `occupancy=`、`condition=`、`seed=`、`model_id=` 等参数；`x0_history` 代表每一步解码轨迹还是控制点；当前 checkpoint 名称和各字段 shapes。不得把旧文档中已过时的 `[32,2]` 控制点当成事实。

### 6.3 输出后处理

1. 将模型控制点/轨迹从 scene 变换回 world（使用生成请求携带的 frame 变换，**不要用新的 ego 坐标系反算旧结果**）。
2. B 样条进行约 0.2 m 弧长采样，计算切线方向、曲率及车辆 footprint。
3. 验收端点误差、ALM 成功与否、512 点静态碰撞、栅格安全、合法车道、曲率与车辆当前航向的可连接性。
4. 生成 `candidate`，但**不立即让车辆执行**；交给 `TrajectoryManager` 和动态安全层。

轨迹前端航向需与当前 ego yaw 接近，例如 `<12°`，横向初始投影偏差 `<0.4 m`，曲率满足 `|κ| ≤ tan(δmax)/wheelbase`；这些是初始验收阈值，可调整并记录。若连续若干候选因为起点航向不匹配而失败，应报告模型缺少航向条件的局限，而非引入未说明的几何重优化来掩盖问题。

---

## 7. ReplanManager：什么时候请求新轨迹（核心二）

### 7.1 调度与线程模型

| 任务 | 初期频率 | 说明 |
|---|---|---|
| `world.tick()`、ego 控制、碰撞/制动监测 | 20 Hz（dt = 0.05 s） | **仅一个主循环拥有 tick 权限** |
| 世界状态采集、动态预测 | 20 Hz | 每帧统一快照 |
| TrajSafe 新路径计算 | 先 5 Hz，再视实际时延提高至 10 Hz | 非阻塞 worker，不能冻结 CARLA 仿真时间 |
| Route 局部 G 维护 | 随路线进度与行为事件触发 | 不等于每次重规划都更新 G |
| 安全检查与速度更新 | 20 Hz | 跟随最新 NPC 状态执行 |

一个规划 worker 即可，最多 1 个正在执行的推理 + 1 个最新等待请求。若新快照到来，覆盖旧的等待请求，不叠加过时任务。模型对象单线程使用，避免并发调用导致 CUDA/缓存不安全。

`world.tick()` 由主 CARLA client 统一推动；**worker 中禁止 tick**。主循环应持续推进仿真时钟，否则得到的 20 ms“规划实时性”可能只是将整个世界冻结后的离线运算。

### 7.2 周期与事件触发

**周期触发**：先每 0.2 s 发一次请求，通过延迟和稳定性测试后可每 0.1 s。

**事件触发优先级**：

1. 车道/行为切换：`FOLLOW→PREPARE`、`WAIT_GAP→CHANGE`、`PASS→RETURN`；
2. 已执行轨迹剩余有效长度 `<15 m` 或不足当前速度下的制动余量；
3. 新的准静态障碍进入规划区域或阻塞障碍消失；
4. 执行轨迹偏差超过阈值，如横向误差 `>0.7 m`；
5. 新轨迹/旧轨迹存在动态碰撞预测风险；**立即进入减速/安全检查，不等规划器返回才制动**。

周期和事件通过相同请求接口，限频合并；避免危险出现时疯狂生成重复请求。

### 7.3 何时必须丢弃计算完成的结果

- `source_snapshot_id` 或时间戳太旧：初值规划结果年龄 `>0.20 s`；
- 生成期间 ego 已移动导致轨迹起点投影/航向不匹配；
- 行为状态或目标车道在计算期间已切换；
- 目标点或允许车道政策已变化；
- 最新时刻动态安全检查不通过；
- 当前模型 ALM 失败或路径无效。

注意这里的年龄阈值是工程初值：不同车辆速度下，还应检查`v×age`造成的位姿差而非只看时间。发生排队积压时应丢弃旧请求，而不是依次执行多条陈旧候选。

---

## 8. TrajectoryManager：什么时候采纳或继续执行路径

**原则：每次重规划 ≠ 每次替换执行轨迹。**

### 8.1 每次候选需要满足的接纳条件

```text
model_output_valid
AND static_footprint_safe
AND allowed_lane_safe
AND endpoint_and_curvature_valid
AND ego_pose_heading_connectable
AND dynamic_safety_at_activation_time
AND behavior_state_compatible
AND no_illegal_topology_flip
```

如果当前轨迹仍安全，新的候选即使通过验收，也可以保留旧轨迹，特别是在换道动作已经执行时。可设最小有效切换间隔 `0.3–0.5 s` 或路径剩余进度触发；若旧轨迹存在真实风险则不受这个限频约束、优先安全停车。

### 8.2 三种场景下如何决策

- **FOLLOW**：采用沿全局路由的最新有效局部路径，或保留旧路径；速度层负责低速前车的跟随。
- **WAIT_GAP**：可以持续生成 CHANGE 候选以准备，但**执行路径保持原车道中已批准的路径**；速度层在停车障碍前建立停止点。不得“跟着未批准的换道候选慢慢压线”。
- **CHANGE/PASS/RETURN**：锁定目标车道/绕障拓扑，不因相邻两次 Diffusion 输出轻微差异而来回变道；新路径必须与正在执行的曲线相容。中途检测危险时，优先安全减速和可行的中止方案，禁止直接按新轨迹急转。

### 8.3 保留旧路径的安全边界

旧路径仅在**当前状态下仍可执行且安全**时保留：

- 旧路径受新出现障碍/动态预测威胁：降低速度或停车；不可“旧路径还没走完”就盲目继续。
- 若旧路径只剩不足制动距离，而新路径全部失败：提前减速停止。
- 若停车后没有任何安全轨迹：保持停止状态并记录 `PLANNING_UNAVAILABLE`，而不是回退为 unconstrained raw 曲线。
- 不加入隐式 Frenet/样条再优化输出替换模型轨迹。必要的最近点投影、重采样和停止点速度控制不会改变路径几何；若后续必须拼接几何轨迹，应单独说明并记录结果不再是纯网络输出。

---

## 9. DynamicSafety + VelocityPlanner：解决“几何安全不等于动态安全”

### 9.1 预测模型

对移动 NPC 使用真值当前状态进行 **恒速沿车道或恒速加速度简化预测**，获取未来 `0–6 s`、间隔 `0.1–0.2 s` 的包围盒序列：

\[
\hat B_j(t_k)=\operatorname{Box}(\hat p_j(t_k),\hat\psi_j(t_k),\ell_j,w_j)\oplus m_{\mathrm{safe}}
\]

这里 `m_safe` 为适度安全裕度，按预测时间增长可增加不确定性膨胀。对于确定性 ScenarioRunner NPC，**不得直接读取未来脚本轨迹作为自车规划器的先验**；实际控制侧只使用当前位姿、速度、道路信息进行预测。真实未来脚本只能在离线评分中使用。

候选 ego 轨迹由弧长与速度曲线获得同时间戳预测 `B_ego(t_k)`，检测：

\[
\operatorname{Intersect}(B_{ego}(t_k),\hat B_j(t_k))=\text{False}
\]

还需检查目标车道后向来车的跟车/制动间距：后车高速靠近时，不能只因为当前框未重叠就批准换道。可参考 Autoware 的 RSS-inspired 安全制动距离逻辑，但本应用只做轻量可验证版本。

**局限**：恒速预测在急刹、变道时可能失真；安全主管必须滚动重估，在危险时执行紧急制动。不能宣称对所有动态交互具有保证。

### 9.2 速度规划（必须存在，不能让横向 Pure Pursuit 代替避撞）

对路径按弧长采样、计算曲率，初值：

```text
v_limit(s) = min(
    scenario_speed_limit,
    sqrt(a_lat_max / max(|curvature(s)|, eps)),
    lead_vehicle_follow_limit(s),
    lane_change_gap_limit(s),
    stop_point_limit(s),
    goal_stop_limit(s)
)
```

然后做前向/后向速度扫描，满足最大加速/减速度约束；必要时施加加加速度限幅。停靠障碍物前的目标停止点由**车身 footprint 与障碍前沿距离、纵向速度及最大舒适制动能力**计算，不是在几何规划器中随意改变局部 `G`。

例如初值 `v_max≈5 m/s`、`a_acc≈1.5 m/s²`、舒适 `a_brake≈2.5 m/s²`、应急制动能力根据车辆实测；停车距离应至少覆盖 `v·(latency+reaction)+v²/(2a_brake)+safety_margin`。**若剩余距离不足以舒适制动，启动独立应急制动逻辑**；不要假定修改下一个速度点就能让车辆瞬停。

### 9.3 紧急安全优先级

1. 已发生碰撞传感器事件：记录并终止。
2. 未来短时间出现高置信碰撞风险：优先制动，不受 Diffusion 任务调度影响。
3. 候选几何无效/还没过批准：保留安全旧路径并跟车/停靠。
4. 只有路径有效、动态安全、行为允许时，才加速或执行换道。

`SafetySupervisor` 不负责重新生成几何路径，只负责判断与保守制动；若外部安全层让所有方法都能避免碰撞，论文需按完整系统应用说明，不能宣称这由 Diffusion 单独保证。

---

## 10. 控制器、仿真执行与更新时序

**控制器先继续使用旧方案的 Pure Pursuit（横向）+ PID（纵向）**。因为论文不研究 MPC，初版不引入 MPC 额外求解器；若 20 Hz 下实际误差不合格，再考虑控制器升级，并记录为什么。

- 弧长采样约 `0.2 m`；`Pure Pursuit` 前视距离随速度变化，例如 `clip(1.8+0.45v,2.0,4.0)m` 作为初值。
- 使用车辆模型获得真实轴距、转向极限；将曲率转为控制器可实现的转角；CARLA 方向符号做低速单元测试。
- 纵向速度目标必须来自 `VelocityPlanner` 而非单一 `v=5m/s` 常数。
- 可用 `carla.VehicleAckermannControl`，但 CARLA 0.9.16 中该接口参数与效果以本机 Python API 实测为准；或用 `carla.VehicleControl` 手动映射 throttle/brake/steer。**同一时刻只能使用一种控制入口。**
- 主循环每 tick 应先收齐对应 snapshot，再执行安全监测、速度与控制，之后推进下一次 CARLA tick；传感器监听与 collision event 按 `frame` 对齐。

### 10.1 线程与消息模型示例（伪代码，不是项目实际函数签名）

```python
# 一个线程负责推进 CARLA；另一个串行 worker 负责模型推理。
while not done:
    snapshot = capture_world_snapshot(world)
    ego_state, objects = extract_state(snapshot)

    route.update_progress(ego_state)
    predictions = predictor.predict(objects)
    behavior.update_request(ego_state, objects, route)
    goal = goal_manager.get_stable_goal(behavior, route, ego_state)

    if scheduler.periodic_due() or scheduler.event_due():
        request = build_planning_request(snapshot, behavior, goal)
        planner_worker.submit_latest(request)  # 覆盖尚未开始的旧请求

    candidate = planner_worker.poll_completed()
    if candidate is not None:
        trajectory_manager.evaluate_and_maybe_accept(
            candidate, ego_state, objects, predictions, behavior)

    approved = trajectory_manager.get_approved_path()
    safety = supervisor.check(approved, ego_state, objects, predictions)
    speed_profile = velocity_planner.compute(approved, ego_state, predictions, safety)
    command = controller.step(ego_state, approved, speed_profile, safety)
    vehicle.apply_control(command)

    log_snapshot_and_decisions(...)
    world.tick()  # 一个唯一的 tick owner
```

上面为了表达线程关系，`capture_world_snapshot(world)` 在实际同步 CARLA 客户端中应按真实 tick/snapshot 生命周期放置，不要误以为该伪代码能直接复制运行。**失败路径必须避免给 `VelocityPlanner` 传 `None` 后继续加速：若尚无 approved path，则输出安全制动。**

---

## 11. 确定性 CARLA 场景实现：优先 ScenarioRunner 0.9.16

### 11.1 推荐的 NPC 控制方式

**关键两辆车（前车与邻道车）**建议使用 ScenarioRunner 0.9.16 的 Python 场景，或实现相同时间/路程触发的确定性车辆脚本：

- 前车：沿自己的车道行驶；ego 到达指定道路距离/仿真时间后，前车匀减速停车并持续停靠。
- 邻道车：保持允许范围内的定速从 ego 左后方驶过，禁止干扰车临时自动跨车道。
- 全部 NPC 使用真实车辆 Actor 和物理控制，**禁止每帧通过 `set_transform` 瞬移**，除非是初始 spawn/reset。
- 使用 `seed` 固定 CARLA 随机性；重要车辆起始姿态及速度写入场景配置并保存。

**TM 只适合背景车流**：CARLA 官方 TM 文档说明其交通车辆通常会随机选择路口方向，并不默认朝固定全局目的地导航；如果把本场景两辆关键车全部交给 TM，可能无法复现“前车停车—邻车通过”的明确冲突过程。因此关键两车不要无条件用 TM 代替控制脚本。

ScenarioRunner v0.9.16 需要搭配 CARLA 0.9.16，不要随意使用 master 或较旧 release。初期也可以只用简单的 `scenario.py` 实现确定性交互，验证成功后再纳入 ScenarioRunner 正式管理。

### 11.2 道路段自动筛选

先写离线筛选脚本，从 CARLA 地图中筛选：

- 同向两车道且有合法允许换道的标线；
- 路口、行人横道、交通灯等复杂设施不在本次换道主要区间；
- 可提供足够的直线或缓弯道路长度（建议至少 120–150m）；
- ego、lead、adjacent 车身初始无重叠且不会在 spawn 时与建筑物相交；
- 使用当前 4 栅格膨胀、160m 窗口及选择的合法道路 mask 后，停靠前车旁仍存在连续几何通路；
- 在既定约束下调用模型能产生至少一条静态可行轨迹；无法满足的道路筛选为不适用，并记录原因。

使用经过审核的道路段后，再固定 `road_id/lane_id/s` 或 world transform 到配置，形成可复现演示场景。**不能在正式运行中自动遍历种子寻找最漂亮的结果，然后只保留成功录像而不记录失败。**

---

## 12. 工程目录（在既有代码上最小增量实现）

```text
scripts/
  carla_online_closed_loop.py            # 一键运行入口，单 tick owner
  inspect_carla_lane_candidates.py       # 地图与合法车道/膨胀可行性筛选
configs/
  carla_online_closed_loop.yaml          # 场景、模型、调度及安全配置
src/carla_bridge/
  frame.py                               # 复用旧 world/local/scene/grid 变换并修订尺度
  occupancy.py                           # 道路/停止车辆的动态栅格构建
  planner_adapter.py                     # Engine 的在线推理封装，C=48
  scenario.py                            # NPC/道路段构建；可适配 ScenarioRunner
  path_profile.py                        # 弧长采样、航向、曲率
  controller.py                          # Pure Pursuit + PID
  route_manager.py                       # 新增：CARLA 全局路线与进度
  goal_manager.py                        # 新增：稳定局部 G 与目标车道
  behavior_manager.py                    # 新增：Request/Ready/Execute 状态机
  dynamic_objects.py                     # 新增：NPC 状态、准静态分类、短期预测
  replan_manager.py                      # 新增：非阻塞 5→10Hz /事件触发
  trajectory_manager.py                  # 新增：路径验收、复用、锁定与切换
  velocity_planner.py                    # 新增：跟车/停车/曲率/加减速约束
  safety_supervisor.py                   # 新增：每 tick 风险检查与安全制动
  recorder.py                            # 复用并扩展：MP4 + 同步日志
  visualization.py                       # 复用并扩展：轨迹/NPC/风险/状态
outputs/closed_loop/
  scenario_manifest.json
  planning_requests.csv
  plan_candidates.csv
  trajectory_decisions.csv
  ego_state.csv
  npc_state.csv
  control.csv
  events.jsonl
  trajectory_visualization.mp4
  summary.json
```

**这是推荐文件结构，不要求无条件新建全部文件。** 若当前仓库已有近似接口，应复用并实现最小必要扩展，避免产生两套重复的坐标转换/碰撞检查逻辑。

---

## 13. 初始配置示例（调试初值，不是经过优化的默认最优值）

```yaml
carla:
  host: 127.0.0.1
  port: 2000
  tm_port: 8000
  version_expected: "0.9.16"
  town: "Town05"         # 地图筛选失败时允许替换，并在 manifest 记录
  fixed_dt: 0.05
  synchronous: true
  traffic_seed: 43

scenario:
  name: "stopped_lead_with_adjacent_fast_car"
  ego_speed_mps: 5.0
  lead_initial_distance_m: 35.0
  lead_initial_speed_mps: 2.0
  adjacent_initial_behind_m: 20.0
  adjacent_initial_speed_mps: 8.0
  max_sim_duration_s: 80.0
  scripted_actor_control: true
  fixed_global_goal_distance_m: 150.0

planner:
  model: "TrajSafe-Diffuser"
  checkpoint: "REPLACE_WITH_VERIFIED_CURRENT_CHECKPOINT"
  control_points: 48
  ddim_steps: 4
  use_shared_regions: true
  use_history_feedback: true
  use_alm: true
  grid_hw: [256, 256]
  map_size_m: 160.0   # 必须在当前训练 checkpoint 上核实
  obstacles_inflation_cells: 4
  forward_horizon_min_m: 45.0
  forward_horizon_max_m: 65.0

routing:
  goal_update_progress_m: 5.0
  goal_near_distance_m: 20.0
  legal_lane_changes_only: true

replanning:
  initial_hz: 5.0
  target_hz_if_validated: 10.0
  max_waiting_requests: 1
  max_candidate_age_s: 0.20
  min_switch_interval_s: 0.4

behavior:
  quasi_static_enter_speed_mps: 0.5
  quasi_static_enter_duration_s: 1.0
  quasi_static_exit_speed_mps: 1.2
  safe_gap_hold_s: 0.4
  pass_clearance_m: 6.0

safety:
  predict_horizon_s: 6.0
  predict_dt_s: 0.1
  footprint_margin_m: 0.25
  curvature_check: true
  connect_lateral_tolerance_m: 0.4
  connect_yaw_tolerance_deg: 12.0
  max_cross_track_m: 0.8
  stop_margin_m: 3.0

control:
  frequency_hz: 20.0
  controller_type: "pure_pursuit_pid"
  max_speed_mps: 5.0
  max_accel_mps2: 1.5
  comfort_decel_mps2: 2.5
  max_lateral_accel_mps2: 1.5

record:
  save_video: true
  save_all_candidate_logs: true
  save_world_snapshots: true
```

注：初始 5–10 Hz 的目标是**非阻塞持续规划**。如果实测整个规划/验收链路 P95 > 规划周期，降为 5 Hz 或更低，不能只凭模型核单次 20ms 就认定系统满足 10 Hz。

---

## 14. 最小实施阶段与明确验收门槛

### P0：审查当前仓库与数据契约（必须先完成）

- [ ] 找到当前 Engine 真实推理入口、checkpoint 与 4-step/shared-region 配置。
- [ ] 核实实际 map_size/scene scale、占据图方向、障碍膨胀、网络输入输出 shape、48 控制点。
- [ ] 测量**单样本**完整规划时间 P50/P95，不用 batch 摊销时间冒充在线耗时。
- [ ] 检查当前代码是否已有非阻塞推理接口、地图快照和完整轨迹评估。

**交付**：一份 `INTEGRATION_AUDIT.md`，列出已存在的可复用接口与不一致项。

### P1：RouteManager 与 CARLA 地图（尚不控制车辆）

- [ ] 找到满足条件的两同向车道路段并固定场景配置。
- [ ] world/local/scene/grid 坐标测试通过；road mask 与合法跨线规则可视化正确。
- [ ] GlobalRoutePlanner 生成真实导航路线，局部 G 能随路线进度稳定前进，不因每帧 NPC 位置乱跳。
- [ ] 停靠车完成栅格化后仍存在合法、足够宽的可行驶通路。

**交付**：俯视 PNG、路线/局部 G/准静态占据地图叠加图、坐标单元测试。

### P2：完全在线规划但不启动自车油门

- [ ] CARLA 持续 20 Hz tick 时，规划 worker 以 5 Hz 输出不同时刻的真实网络轨迹。
- [ ] NPC 按脚本运动，车辆世界坐标位置与输出轨迹坐标对齐。
- [ ] 记录规划请求快照、规划完成时间、模型/ALM 成功、候选验收结果。
- [ ] 模拟超时、错误和旧快照拒绝，车辆依旧能正常 tick。

**交付**：逐步变化的轨迹俯视图/视频、延迟统计 CSV。

### P3：低速车辆闭环执行（先单前车，后加邻道来车）

- [ ] Ego 采用模型轨迹 + Pure Pursuit/PID 跟踪，且**持续重规划、动态替换有效路径**。
- [ ] 减速跟随前车，不发生明显超调；前车静止时可在安全位置制动。
- [ ] 前车成为几何阻塞后，产生换道候选；目标车道交通不可用时不压线。
- [ ] 邻道车辆驶过且动态安全检查通过后，自主执行 TrajSafe 换道轨迹。
- [ ] 绕障并回到原车道，安全到达全局目标。

**交付**：同一个 run 的状态日志和闭环视频，不需要另跑基线方法。

### P4：正式应用录像与论文素材

- [ ] 记录 `ego world path / approved reference / candidate paths / NPC boxes / behavior state / local goal`。
- [ ] 输出 1 段完整视频，包含俯视世界视角、规划轨迹与行为/安全原因叠加。
- [ ] 从同一次运行中抽取 3–4 个关键帧：前车减速、邻道来车导致等待、换道执行、绕行回归。
- [ ] 生成 `summary.json`：到达与否、CARLA collision event 数、运行时长、请求/接受/拒绝轨迹次数、停车等待时间、跟踪误差 RMS/最大值、单次规划时间 P50/P95/最大值及重规划周期丢帧。
- [ ] 所有日志带仿真时钟/帧号、提交版本、checkpoint hash、场景 seed 和地图版本，可重复回放。

**验收定义**：主要演示目标为合法、无碰撞、无手动强制的完整交通过程；若由于当前模型输入限制导致不能换道，保留真实失败日志并报告，不允许通过预写换道曲线替代模型输出。

---

## 15. 异常处理矩阵（必须实现）

| 异常 | 行为 | 日志原因 |
|---|---|---|
| 无合法全局路线/局部目标 | 不开始演示或安全停车 | `ROUTE_OR_GOAL_INVALID` |
| 场景障碍膨胀后不连通 | 更换场景候选，运行中停止而非取消安全约束 | `SCENE_GEOMETRY_INFEASIBLE` |
| 当前模型 ALM 失败 | 丢弃候选，保留安全旧轨迹或停车 | `ALM_FAILURE` |
| 新候选生成太慢/过时 | 丢弃结果，不覆盖当前 approved path | `STALE_CANDIDATE` |
| 轨迹与当前车姿不连续 | 拒绝并保留安全路径，记录不匹配角度 | `START_POSE_MISMATCH` |
| 目标车道左后车辆太快 | 保持原车道并减速等待，不执行换道候选 | `GAP_UNSAFE` |
| 当前路径变得动态不安全 | 即刻限速/制动，必要时解除轨迹批准 | `DYNAMIC_RISK` |
| 所有候选失败，旧路径也不安全 | 稳定制动停车 | `NO_SAFE_PATH` |
| 碰撞事件/车辆异常 | 记录并终止仿真 | `COLLISION` / `CONTROLLER_FAILURE` |

---

## 16. 论文写作口径

论文中把该实验称为：**Dynamic CARLA Closed-Loop Application / CARLA Dynamic Closed-Loop Demonstration**。

建议只占一个紧凑段落与一张时间序列可视化图：说明 CARLA 真值状态构造占据地图，TrajSafe-Diffuser 加速版周期生成安全几何参考轨迹，成熟的行为管理/动态安全/速度控制模块保证车辆执行；展示车辆对前车停车和邻道来车的响应。若有实际数字则报告模型推理延迟、全链路延迟、重规划次数、最终完成与碰撞事件，避免冗长的实验对比表。

**切勿把以下两种性质混为一谈**：

- TrajSafe 本身的凸区域/ALM 修正：对**给定静态占据图**中的几何安全进行处理；
- 外部行为/预测/速度/制动：根据其他车辆随时间移动的情况判断何时能换道、需不需要等待。

最后结果只能支持“基于 TrajSafe 轨迹生成器的动态 CARLA 闭环系统可进行某类交互演示”，而不是“TrajSafe 网络本身已经具备时空动态障碍预测和安全保证”。

---

## 17. 给 DSH 的交付要求

请先审查实际仓库代码，更新旧集成文件里所有过时假设；在真实验证路径和模型接口后按照 P0→P4 实施。

**必须做到**：

1. CARLA 始终运行、NPC 真实运动、自车持续 5–10 Hz 重规划；不可在停车状态只规划一次。
2. 使用 CARLA 的 `GlobalRoutePlanner` 管理全局路线，采用稳定局部 G 管理，而不是每次随机给目标。
3. 借鉴 Autoware 的 Request/Ready/Execute、执行中安全检查、速度与障碍物分层；**只把几何局部路径生成换成 TrajSafe**。
4. 所有车辆位姿、轨迹输出和障碍占据地图使用经验证的唯一坐标约定；重点核对 80m/160m 与 C=32/48 的旧文档冲突。
5. 运动 NPC 以确定性脚本或 ScenarioRunner 0.9.16 为主，非关键车辆可用 TM；ego 不交给 autopilot。
6. 若 20 ms 是 batch 摊销值或推理不含骨架与安全检查，必须重新报告真实 P50/P95 并适当降规划频率。
7. 所有候选经过静态安全、真实 footprint、可控曲率、动态时空风险和当前行为允许性验证；不安全则拒绝，车辆可减速停车。
8. 生成完整场景视频、所有控制/规划/行为日志、运行配置、提交版本与最终论文用时间序列截图。
9. 如遇模型输入限制/场景不可行，报告根因，不偷偷修改安全约束、手绘换道路径或把外部动态安全模块算作模型贡献。

**项目实施范围至此为止：单动态交互场景的仿真应用，不新增训练、不新增规划器比较、不引入完整 Autoware ROS 2 系统。**
