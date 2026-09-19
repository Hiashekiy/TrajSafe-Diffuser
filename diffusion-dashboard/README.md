# Diffusion Lens 使用与启动说明

Diffusion Lens 是 TrajSafe-Diffuser 的本地扩散过程观测面板。它使用项目中的真实
Maze2D 数据、occupancy map 和 checkpoint，在本地 GPU 上运行报告一致的骨架引导
反向扩散，并保存每个 timestep 的中间状态供前端回放。

> 本面板只服务当前模型（每步重选骨架的 TrajSafe-Diffuser）。

## 1. 功能

- 选择 U-Maze、Medium、Large 数据集样本。
- 在 `best` / `latest` 两个 checkpoint 间切换。
- 设置随机种子。
- 在地图上点击修改起点、终点和障碍物。
- 在线重建骨架图与候选搜索路径，并运行真实的 16-step DDIM 采样。
- 回放 `Pₜ [128,2]` 以及每步的 `x̂₀` / 椭圆预测。
- 切换查看 noisy state、每一步的 `x₀ prediction`，或将两者叠加比较。
- 暂停、拖动时间轴，并以 0.25×、0.5×、1×、2×、4× 播放。
- 分别显示或隐藏轨迹、椭圆、waypoint、最终结果、Ground Truth、occupancy map、
  候选搜索路径、verified convex region。
- 按参数缓存生成结果。

## 2. 目录关系

```text
Neural-IRISDiffuser/
├─ configs/
│  └─ config.yaml     模型 / 采样 / 候选生成参数 + 推理期 ALM 段
├─ data/
│  └─ scenes/         test 样本 + 三张地图
├─ outputs/
│  └─ ckpt/           best.pt / latest.pt
├─ src/
└─ diffusion-dashboard/
   ├─ app/                 前端页面
   ├─ cache/               扩散序列缓存
   ├─ lib/                 前端数据目录
   ├─ backend.py           本地推理服务（HTTP + 缓存）
   ├─ engine.py            推理引擎（在线骨架/候选 + 采样器 + ALM）
   └─ package.json
```

推理服务读取：

- `data/scenes/test/positions.npy`、`conditions.npy`、`maze_id.npy`
- `data/scenes/maps/{umaze,medium,large}.npy`
- `configs/config.yaml`（含推理期 `alm` 段）
- `outputs/ckpt/{best,latest}.pt`

## 3. 环境要求

- Windows PowerShell。
- Node.js 22.13 或更高版本；首次使用需运行 `npm install`。
- 可正常运行主项目的 Python 环境：

```powershell
E:\CondaEnvData\envs\GGMPC\python.exe
```

- 推荐 CUDA GPU；没有 CUDA 时回退 CPU，生成明显更慢。

## 4. 启动方法

前端与推理服务分别启动，建议开两个 PowerShell 窗口。

### 4.1 启动模型推理服务

```powershell
cd D:\ProjectDirectory\Neural-IRISDiffuser\diffusion-dashboard
& 'E:\CondaEnvData\envs\GGMPC\python.exe' backend.py
```

启动成功后会显示：

```text
TrajSafe dashboard API on http://localhost:8765 (cuda)
```

检查服务：

```powershell
Invoke-RestMethod http://localhost:8765/health
```

返回 `status: ready` 表示推理服务正常。

### 4.2 启动前端

```powershell
cd D:\ProjectDirectory\Neural-IRISDiffuser\diffusion-dashboard
npm run dev
```

访问 `http://localhost:3000/`。页面右上角显示 `GPU SERVICE READY` 时表示已连接推理服务。

### 4.3 停止服务

分别在两个 PowerShell 窗口中按 `Ctrl+C`。

## 5. 使用流程

### 5.1 选择基础参数

左侧面板依次设置：数据集样本、模型 checkpoint、随机种子。任何参数变化都会使当前
扩散序列失效，需要重新生成。

### 5.2 编辑地图条件

- **起点 / 终点**：点击画布设置新的起终点。
- **添加障碍**：点击添加圆形障碍，可连续添加。
- **擦除障碍**：点击已添加的自定义障碍附近将其删除。
- **重置**：恢复样本原始起终点并清除全部自定义障碍。

自定义障碍只叠加在原始 occupancy map 的副本上，不会修改数据集文件。

### 5.3 生成扩散序列

点击 **生成扩散序列**。推理服务将：

1. 按当前 occupancy（含自定义障碍）重建骨架图并哈希缓存；
2. 在线生成候选搜索路径（与离线预处理同一个生成器）；
3. 加载所选 checkpoint；
4. 运行完整的 16-step DDIM；
5. 保存每个 timestep 的 noisy state（`state_history`）、`x̂₀`（`x0_history`）、
   选中拓扑 `m`、`π(m)` 与椭圆预测（`ellipse_history`）；
6. 结果写入 `cache/`。

首次加载某个 checkpoint 通常比后续生成稍慢。

### 5.4 回放

底部控制栏支持播放/暂停、拖动时间轴、回到 `t=15`、切换播放速度。左侧“展示状态”支持：

- **每步 x₀ prediction**：模型在当前 timestep 直接预测的 clean 轨迹与椭圆；
- **当前 noisy state**：该步送入模型的 `Pₜ`；
- **两者叠加比较**。

这里显示的是整条 128-point 轨迹在不同扩散 timestep 的状态，不是物体沿轨迹移动的动画。

### 5.5 图层控制

- 当前扩散状态 `Pₜ`；对应椭圆状态；128 个 waypoint；模型最终输出 `P₀`；
- 数据集 Ground Truth；Occupancy map；
- **选中骨架拓扑**：浅蓝虚线 = 全部候选搜索路径，深蓝实线 = 当前选中的 `m`；
- **轨迹主干 x̂₀ / verified convex region**：见第 7 节。

为避免画面过密，椭圆默认每隔 8 个显示一个。

## 6. 缓存规则

缓存键包含样本、checkpoint、随机种子、自定义起终点与自定义障碍。每次点击生成都会
先清除旧的 JSON 缓存再重新运行完整扩散，因此右侧通常显示 `SAVED`。缓存文件位于
`diffusion-dashboard/cache/`，是派生数据；需要清理时删除其中的 `.json`（保留 `.gitkeep`）。

## 7. 模型：TrajSafe-Diffuser（每步重选动态骨架）

模型下拉：

- `TrajSafe · Best (ep 19)` → `outputs/ckpt/best.pt`
- `TrajSafe · Latest (ep 100)` → `outputs/ckpt/latest.pt`

### 7.1 链路

```
P_t -> H_traj -> {R_m} -> m = argmax(pi) -> H_prog -> s
    -> c = Gamma_m(s) -> H_ell -> H_clean -> P0_hat -> DDIM
```

只有 `P_t` 是扩散状态；椭圆圆心不是网络回归量，而是选中骨架曲线上的点
`c_i = Γ_m(s_i)`。每个 reverse timestep 都重新对候选打分并取 `argmax(pi)`
（没有 commit timestep）。

### 7.2 候选搜索路径的配套设计

模型的输入就是骨架图上的搜索路径，所以 dashboard 与离线预处理使用**同一个生成器**：

1. `Engine.get_graph` 按当前 occupancy（含用户画的自定义障碍）哈希缓存骨架图；
2. `generate_candidates(graph, start, goal, CandidateConfig)` 生成候选，参数与
   `configs/config.yaml` 的 `topology` 段一致；
3. `Engine._pack_candidates` 把 `coords`（128 点 `S_m`）与 `geometry`（dense `Γ_m`）
   打包成模型输入；
4. 返回 `topology.candidate_paths` / `topology.candidate_mask` / `topology.topology_path`，
   前端把全部候选画成浅蓝虚线，把选中的 `m` 画成深蓝实线。

修改起终点或障碍后候选会重新生成；若障碍截断通路，接口返回
`该起终点在骨架图上没有合法候选路径…`。

### 7.3 图层含义

- **选中骨架拓扑**：浅蓝虚线 = 当前 occupancy/起终点下的全部候选搜索路径；
  深蓝实线 = 每步 `argmax(pi)` 选中的 `m`。
- **轨迹主干 x̂₀ / 修正前 x̂₀**：`Head_P(H_traj)` 的 coarse 分支，粉色虚线。
- **对应椭圆状态**：payload 的 `ellipse_history` 直接给出每步的 `center = Γ(s)`（绝对
  坐标）与 `shape4`（`[log a, log b, cos 2t, sin 2t]`）。

### 7.4 右侧诊断面板

右栏显示 `安全诊断`：每步重选 `m`、当前 `π(m)`、候选数 valid/total、
CenterFree 中心安全率、中心最小余量（格）、轨迹碰撞、椭圆点碰撞率、进度单调性、
每步选择抖动 `stepJitter`、是否发生拓扑切换、骨架 nodes/branches。

### 7.5 已知限制

个别 OD 上仍可能预测出贴墙/切角轨迹（例如 `large-855` 的 waypoint 17–25），
dashboard 会在「轨迹碰撞」中明确标红；这是模型效果检查的一部分，不是渲染错误。

## 8. 凸区域 + ALM 修正（推理期，可选）

开关「凸区域 + ALM 修正（较慢）」打开后，每个 `t <= start_t` 的 reverse step 会：

1. 取模型预测椭圆 `(center = Γ(s), shape4)`；
2. 用 `EllipseRegionBuilder` 为每个椭圆生成 verified convex region；
3. 用 `alm_correct` 修正模型刚输出的 `x0`（保持端点、correction 平滑）；
4. 修正后的 `x0` 才进入 DDIM，物理椭圆中心保持在 `Γ(s)`。

右侧显示每帧区域数、`violation before → after`、修正量、λ、平滑度等；关闭开关时
仍是纯 DDIM，粉色虚线显示 coarse 分支。

该功能只发生在推理期，网络权重与训练损失不变；参数在
`configs/config.yaml` 的 `alm` 段（`start_t`、`rho`、`step_size` 等）。

## 9. 常见问题

### 页面显示 `SERVICE OFFLINE`

确认 `backend.py` 正在运行，并检查 `Invoke-RestMethod http://localhost:8765/health`。

### 端口 8765 被占用

```powershell
netstat -ano | Select-String ':8765'
```

关闭重复启动的旧推理服务后重新运行 `backend.py`。

### 页面无法打开

访问 `http://localhost:3000/`；若 3000 被占用，请使用终端实际打印的 Local URL，
不要用 `file://` 打开。

### 生成很慢

查看 `/health` 的 `device`：`cuda` 表示使用 GPU，`cpu` 会明显变慢。

### 修改参数后旧序列消失

预期行为：旧序列对应旧参数，点击“生成扩散序列”重新生成。

### 自定义障碍是否会修改原始地图

不会，后端只在 occupancy map 的副本上叠加障碍。

## 10. 构建检查

```powershell
cd D:\ProjectDirectory\Neural-IRISdiffuser\diffusion-dashboard
npm run build
```

该命令只检查前端构建，不会启动推理服务。
