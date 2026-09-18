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
