# Diffusion Lens 使用与启动说明

Diffusion Lens 是 Neural-IRISDiffuser 的本地扩散过程观测面板。它使用项目中的真实 Maze2D 数据、occupancy map 和模型 checkpoint，在本地 GPU 上运行联合轨迹–椭圆反向扩散，并保存每个 timestep 的中间状态供前端回放。

## 1. 功能

- 选择 U-Maze、Medium、Large 数据集样本。
- 选择不同训练阶段的模型 checkpoint。
- 设置随机种子。
- 在地图上点击修改起点、终点和障碍物。
- 运行真实的 16-step 联合扩散采样。
- 回放 `Pₜ [128,2]` 与 `Eₜ [128,6]` 的完整变化过程。
- 切换查看 noisy state、每一步的 clean `x₀ prediction`，或将两者叠加比较。
- 暂停、拖动时间轴，并以 0.25×、0.5×、1×、2×、4× 播放。
- 分别显示或隐藏当前轨迹、椭圆、waypoint、最终结果、Ground Truth 和 occupancy map。
- 按参数缓存生成结果，相同配置无需重复运行模型。

## 2. 目录关系

面板位于主项目的 `diffusion-dashboard` 目录：

```text
Neural-IRISDiffuser/
├─ configs/
├─ data/
├─ outputs/
├─ src/
└─ diffusion-dashboard/
   ├─ app/                 前端页面
   ├─ cache/               扩散序列缓存
   ├─ lib/                 前端数据目录
   ├─ scripts/             数据导出脚本
   ├─ backend.py           本地模型推理服务
   └─ package.json
```

推理服务会读取以下主项目资源：

- `data/processed_scene_v1/test/`
- `data/processed_scene_v1/maps/`
- `configs/config_v1_continue.yaml`
- `outputs/ckpt_v1_smooth_iou_free_cvar_center_balanced/epoch_100.pt`
- `outputs/ckpt_v1_smooth_iou_free_cvar_center_balanced_continue100/best.pt`
- `outputs/ckpt_v1_smooth_iou_free_cvar_center_balanced_continue200/best.pt`
- `outputs/ckpt_v1_center_safe_isolated/best.pt`

## 3. 环境要求

- Windows PowerShell。
- Node.js 22.13 或更高版本。
- 已安装前端依赖；首次使用时需运行 `npm install`。
- 可正常运行 Neural-IRISDiffuser 的 Python 环境。
- 推荐使用 CUDA GPU；没有 CUDA 时服务会回退到 CPU，但生成会更慢。

当前项目配置使用的 Python 为：

```powershell
E:\CondaEnvData\envs\GGMPC\python.exe
```

如果本机路径不同，请将后续命令中的 Python 路径替换为实际环境路径。

## 4. 启动方法

前端和推理服务需要分别启动。建议打开两个 PowerShell 窗口。

### 4.1 启动模型推理服务

在第一个 PowerShell 窗口中执行：

```powershell
cd D:\ProjectDirectory\Neural-IRISDiffuser\diffusion-dashboard
& 'E:\CondaEnvData\envs\GGMPC\python.exe' backend.py
```

启动成功后会显示：

```text
Diffusion dashboard API on http://localhost:8765 (cuda)
```

可用以下命令检查服务：

```powershell
Invoke-RestMethod http://localhost:8765/health
```

返回 `status: ready` 表示推理服务正常。

### 4.2 启动前端

在第二个 PowerShell 窗口中执行：

```powershell
cd D:\ProjectDirectory\Neural-IRISDiffuser\diffusion-dashboard
npm run dev
```

启动成功后访问：

```text
http://localhost:3000/
```

页面右上角显示 `GPU SERVICE READY` 时，前端已连接推理服务。

### 4.3 停止服务

分别在两个 PowerShell 窗口中按 `Ctrl+C`。

## 5. 使用流程

### 5.1 选择基础参数

在左侧面板依次设置：

1. 数据集样本。
2. 模型 checkpoint。
3. 随机种子。

任何参数变化都会使当前扩散序列失效，需要重新生成。

### 5.2 编辑地图条件

画布上方提供以下编辑工具：

- **起点**：选择后，在地图上点击新的起点。
- **终点**：选择后，在地图上点击新的终点。
- **添加障碍**：选择后，在地图上点击添加圆形障碍；可连续添加。
- **擦除障碍**：点击已添加的自定义障碍附近将其删除。
- **重置**：恢复当前数据集样本的原始起终点，并清除全部自定义障碍。

自定义障碍只叠加在原始 occupancy map 上，不会修改数据集文件。

### 5.3 生成扩散序列

点击左侧或画布中央的 **生成扩散序列**。

推理服务将：

1. 加载所选 checkpoint。
2. 应用所选数据集地图。
3. 应用自定义起点、终点和障碍物。
4. 使用指定 seed 初始化高斯噪声。
5. 运行完整的 16-step sampler。
6. 保存 `t=15` 到 `t=0` 的模型输入状态以及最终 `x₀`。
7. 同时保存每一步模型输出的 `P̂₀⁽ᵗ⁾ / Ê₀⁽ᵗ⁾`。
8. 将结果写入 `cache/`。

首次加载某个 checkpoint 通常比后续生成稍慢。

### 5.4 回放

生成完成后可使用底部控制栏：

- 播放/暂停按钮。
- 拖动时间轴查看任意 timestep。
- 回到 `t=15`。
- 切换播放速度。

左侧“展示状态”支持：

- **每步 x₀ prediction**：显示模型在当前 timestep 直接预测的 clean trajectory 和 ellipse，可用于判断模型从第几步开始形成稳定结果。
- **当前 noisy state Pₜ / Eₜ**：显示该步送入模型的带噪联合状态。
- **x₀ prediction + noisy state**：将两者叠加，青色细线为 noisy state，亮绿色线为当前 clean prediction。

这里显示的是整条 128-point 轨迹与全部椭圆变量在不同扩散 timestep 的状态，不是物体沿轨迹移动的动画。

### 5.5 图层控制

左侧的“显示元素”可控制：

- 当前扩散状态 `Pₜ`。
- 当前椭圆状态 `Eₜ`。
- 128 个 waypoint。
- 模型最终输出 `P₀`。
- 数据集 Ground Truth。
- Occupancy map。

为了避免画面过密，椭圆默认从 128 个状态中每隔 8 个显示一个，共显示 16 个。

## 6. 缓存规则

缓存键包含：

- 数据集样本。
- 模型 checkpoint。
- 随机种子。
- 自定义起点和终点。
- 自定义障碍物的位置与半径。

每次点击生成都会先清除旧的 JSON 缓存，再重新运行完整扩散；相同参数也
不会复用旧结果。右侧“缓存状态”正常显示 `SAVED`，表示本次结果由模型
重新生成并写入缓存。该缓存只保留到下一次生成。

缓存文件保存在：

```text
diffusion-dashboard/cache/
```

缓存是派生数据，不会修改 checkpoint 或数据集。需要清理时，可在前后端停止后删除 `cache` 目录中的 `.json` 文件；保留 `.gitkeep` 即可。

## 7. 数据与坐标约定

- Occupancy map 分辨率：`256 × 256`。
- 场景坐标范围：`[-1,1]²`。
- `occupancy = 1` 表示障碍，`occupancy = 0` 表示自由区域。
- 轨迹长度：128 个 waypoint。
- 联合扩散变量：
  - `Pₜ [128,2]`：轨迹位置。
  - `Eₜ [128,6]`：椭圆中心偏移、对数半轴和方向表示。
- 起点和终点在每个反向扩散步骤后都会重新施加硬约束。

## 8. 常见问题

### 页面显示 `SERVICE OFFLINE`

确认 `backend.py` 正在运行，并检查：

```powershell
Invoke-RestMethod http://localhost:8765/health
```

### 端口 8765 被占用

检查占用进程：

```powershell
netstat -ano | Select-String ':8765'
```

关闭重复启动的旧推理服务后重新运行 `backend.py`。

### 页面无法打开

确认前端仍在运行，并访问 `http://localhost:3000/`。如果 3000 端口被占用，开发服务器可能会打印另一个地址，应使用终端实际显示的 Local URL。

### 生成很慢

查看 `/health` 返回的 `device`：

- `cuda`：正在使用 GPU。
- `cpu`：CUDA 不可用，生成速度会明显下降。

### 修改参数后旧序列消失

这是预期行为。旧序列对应旧参数；点击“生成扩散序列”生成或加载新参数对应的缓存。

### 自定义障碍是否会修改原始地图

不会。后端会复制当前 occupancy map，并在副本上叠加障碍，然后将副本送入模型。

## 9. 构建检查

如需检查前端能否正常构建：

```powershell
cd D:\ProjectDirectory\Neural-IRISDiffuser\diffusion-dashboard
npm run build
```

该命令只检查前端构建，不会启动本地 GPU 推理服务。

---

## 9. V2 模型：Skeleton-Topology-Grounded Trajectory Diffusion

模型下拉里新增两项（后端由 `backend_v2.py` 提供）：

- `V2 Skeleton-Grounded · Best (ep 69)` → `outputs/ckpt_v2_skeleton/best.pt`
- `V2 Skeleton-Grounded · Latest (ep 88)` → `outputs/ckpt_v2_skeleton/latest.pt`

### 9.1 与 V1 的差异

| | V1 | V2 |
|---|---|---|
| diffusion state | `P_t` 与 `E_t` 同时加噪 | **只有 `P_t`**（不再有椭圆扩散状态） |
| 椭圆中心 | `c_i = p_i + delta_c_i`（网络预测偏移） | **`c_i = gamma_m(s_i)`**：选中骨架拓扑上的点，几何映射，不可能进障碍 |
| 椭圆表示 | 6 维 `[dx,dy,log a,log b,cos2t,sin2t]` | 4 维 shape（中心另给） |
| topology | 无 | 在 `t=7` 对若干条**已保证安全连通**的候选路径打分，`m ~ Cat(pi)` 采样后**锁定不再变** |
| 后半程 | 无 | 椭圆序列反过来 condition **同一条** trajectory diffusion |

前端回放无需区分：后端把 V2 椭圆换算回 V1 的 6 维格式（偏移相对同一步的轨迹点），
所以「对应椭圆状态」图层照常工作。

### 9.2 图层含义变化（选 V2 时自动改名）

- **未条件化 x̂₀（纯轨迹主干）**（原 `ALM 修正前 x̂₀`）：不做椭圆条件化时主干自己预测的 `x0`，粉色虚线。
- **验证过的凸区域**（原 `ALM 凸区域`）：由预测椭圆生成、并经过**独立逐格验证 + 整格内推修复**的凸区域。
- **选中骨架拓扑**（新增）：蓝色点线，即 `m` 对应的那条候选路径。

### 9.3 开关与耗时

V2 模式下「启用 ALM 引导」开关变为 **验证凸区域（较慢）**：

- 打开：每一步对约 16 个椭圆生成并验证凸区域，单次生成约 **3.0 s**；
- 关闭：只跑轨迹与椭圆，单次生成约 **0.4 s**（骨架图按 occupancy 哈希缓存）。

### 9.4 自定义起终点 / 自定义障碍

V2 的候选路径是**在线**生成的，与离线预处理用的是同一个函数：

1. 按当前 occupancy（含你画的障碍）重建骨架图；
2. 对当前 start/goal 跑 Yen K 短路 + 长度/Jaccard 过滤；
3. 再做拓扑打分与 categorical 采样。

所以画布上的编辑全部可用。若编辑后没有合法候选（例如障碍把通路截断），
接口返回明确错误：`该起终点在 V2 骨架图上没有合法候选路径…`。

### 9.5 右侧诊断面板

选 V2 时右栏显示 `V2 安全诊断`：commit timestep、选中的 `m`、
`pi(m)`、候选数、**CenterFree 中心安全率**、中心最小余量（格）、轨迹碰撞、
椭圆点碰撞率、本帧验证凸区域数、进度单调性、骨架 nodes/branches。

---

## 10. V3 模型：TrajSafe-Diffuser（每步重选动态骨架）

模型下拉新增两项（后端由 `backend_v3.py` 提供）：

- `V3 TrajSafe · Best (ep 19)` → `outputs/ckpt_v3_skeleton/best.pt`
- `V3 TrajSafe · Latest (ep 90)` → `outputs/ckpt_v3_skeleton/latest.pt`

### 10.1 与 V2 的关键差异

| | V2 | V3 |
|---|---|---|
| diffusion state | 只有 `P_t` | 只有 `P_t` |
| 椭圆中心 | `c_i = gamma_m(s_i)` | `c_i = gamma_m(s_i)` |
| topology | `t=commit_t` 采样一次后锁定 | **每个 reverse timestep 重新 argmax(pi)** |
| topology feature | selector 的 handcrafted 特征 | **一次共享 MatchBlock 输出的 `R_m`** |
| progress | 来自 selected path | `ProgressHead -> s -> c = Gamma(s)` |
| 椭圆参数 | 4 维 shape | 4 维 shape，`L_shape` 只监督 shape head |
| 中心监督 | 无独立中心回归 | `L_align = SmoothL1(Gamma(s), p_GT)`（不经过 GT 投影） |
| 候选搜索路径 | 在线生成 | **在线生成，且把全部候选 `Gamma_m` 返回给 UI** |

### 10.2 候选搜索路径的配套设计

V3 的输入就是骨架图上的搜索路径，所以 dashboard 必须和离线预处理使用**同一个生成器**：

1. `V3Engine.get_graph` 按当前 occupancy（含用户画的自定义障碍）哈希缓存骨架图；
2. `generate_candidates(graph, start, goal, CandidateConfig)` 生成候选，参数与 `configs/config_v3_skeleton.yaml` 的 `topology` 段一致；
3. `V3Engine._pack_candidates` 把 `coords`（128 点 `S_m`）和 `geometry`（dense `Gamma_m`）打包成 V3 模型需要的张量；
4. 返回 `v2.candidatePaths`、`v2.candidateMask`、`v2.topologyPath`，前端在「选中骨架拓扑」图层里把**所有候选搜索路径画成浅蓝虚线**，选中的那条画成深蓝实线。

因此画布上修改起终点、添加/擦除障碍后，候选搜索路径会重新生成；如果障碍截断通路，接口返回：
`该起终点在骨架图上没有合法候选路径…`。

### 10.3 图层含义（选 V3 时）

- **选中骨架拓扑**：浅蓝虚线 = 当前 occupancy/起终点下的全部候选搜索路径；深蓝实线 = 每步 `argmax(pi)` 选中的 `m`。
- **轨迹主干 x̂₀（未椭圆条件化）**（原 `ALM 修正前 x̂₀`）：`Head_P(H_traj)` 的 coarse 分支，粉色虚线。
- **对应椭圆状态**：V3 预测的 `center = Gamma(s)` / `shape4`，后端换算回 V1 的 6 维 `[dx,dy,loga,logb,cos2t,sin2t]`，所以回放器无需改动。

### 10.4 右侧诊断面板

选 V3 时右栏显示 `V3 安全诊断`：

- 每步重选 `m`（不是 commit timestep）、当前 `pi(m)`；
- 候选数 valid / total；
- CenterFree 中心安全率、中心最小余量（格）；
- 轨迹碰撞、椭圆点碰撞率；
- 进度单调性、每步选择抖动 `stepJitter`、是否发生过拓扑切换；
- 骨架 nodes / branches。

### 10.5 已知限制

V3 当前权重下仍可能在个别 OD 上预测出贴墙/切角的轨迹（例如 `large-855` 的 waypoint 17–25），
dashboard 会在「轨迹碰撞」里明确标红；这是模型效果检查的一部分，不是渲染错误。

### 10.6 凸区域 + ALM 修正（推理期，可选）

V3 开关「凸区域 + ALM 修正（较慢）」打开后，每个 `t <= start_t` 的 reverse step 会：

1. 取 V3 预测椭圆 `(center = Gamma(s), shape4)`，构造成 V1 的 6 维形式
   `[dx, dy, log a, log b, cos2t, sin2t]`（`dx,dy = center - x0`）；
2. 用 `EllipseRegionBuilder` 为每个椭圆生成 verified convex region；
3. 用 `alm_correct` 修正模型刚输出的 `x0`（保持端点，correction 平滑）；
4. 修正后的 `x0` 才进入 DDIM，物理椭圆中心保持在 `Gamma(s)`。

右侧会显示每帧区域数、`violation before → after`、修正量、λ、平滑度等；
关闭开关时仍是纯报告版 DDIM，粉色虚线显示 `Head_P(H_traj)` 的 coarse 分支。

该功能只发生在推理期，网络权重、训练损失和报告结构都不变；ALM 参数复用
`configs/config_v1_alm.yaml` 的 `alm` 段（`start_t`、`rho`、`step_size` 等）。


