# PITFALLS — Maze2D 安全扩散规划器踩坑记录

本文档记录实现过程中踩过的坑与最终处理方式，供后续排查/复现参考。

## 1. 数据集/几何

### 1.1 地图坐标轴转置导致“轨迹穿墙”
- **现象**：画图时轨迹看起来穿墙、椭圆错位。
- **原因**：`build_occupancy_grid` 返回 `occ[ny,nx]`（行=y），画图时误用 `occ.T` 且 `origin=lower`，把 x/y 对调。
- **解决**：画图用 `ax.imshow(occ, origin='lower', extent=(x0,x1,y0,y1))`，不要 `.T`。
- **验证**：修正后轨迹 collision_rate=0；之前 0.98 是视觉化/几何对齐错误。

### 1.2 碰撞检测只查 128 节点，漏检线段
- **现象**：报告 0 碰撞，但线段其实可能穿墙。
- **原因**：只对 waypoint 节点做 point-in-wall。
- **解决**：`src/utils/metrics.py::interp_trajectory` 每段插值 8 点（128→1017 点），在稠密点云上算碰撞/clearance。

### 1.3 Neural-IRIS 网络椭圆太小/太碎、Q 值过大
- **现象**：椭圆 `r1≈0.25, r2≈0.13`，`L_te≈1400`，训练很不稳、椭圆沿走廊但碎。
- **原因**：用 Neural-IRIS 网络预测椭圆；其 `a,b` 语义与二次型转换有歧义，且预测偏保守。
- **解决**：改用**真正离线 IRIS MVIE 求解器**（`src/geometry/iris_solver.py`，自原 IRIS 数据集生成器移植）。新椭圆 `r1≈1.14, r2≈0.33`，`L_te≈20`，更贴走廊。

### 1.4 数据集重新生成后旧 checkpoint 失效
- 换了标签生成方式/模型结构后，旧 `best.pt`/`epoch_*.pt` 的 state_dict 与新版不匹配。
- **解决**：每次大改动后 `rm -rf outputs/ckpt` 重训。

## 2. 模型/条件化

### 2.1 “广播相加”条件注入无效（偏离参考）
- **尝试**：把 start/goal 位置 embedding 广播加到每个 token。
- **结果**：1 epoch 测试仍聚团，无效。
- **原因**：Janner Diffuser / RGG 参考实现是**无条件网络**，条件是采样时 `apply_conditioning` 钉住首尾；网络里 `cond` 根本没被使用。
- **解决**：去掉广播相加，改用 `apply_endpoint_condition`：训练时对 `x_t` 和 `x0_pred` 都钉住首尾，采样时每步钉住。

### 2.2 Scene Encoder 缓存导致 `backward a second time`
- **现象**：多 batch 训练报“backward through the graph a second time”。
- **原因**：`Planner` 用 `id(map_tensor)` 缓存 `scene_map`；训练时每批传新的 `expand` 视图，Python 复用 `id`，命中上一批已释放的图。
- **解决**：训练时（`torch.is_grad_enabled()`）**不缓存**、每批重新算；推理时（`no_grad`）才缓存。

### 2.3 局部场景特征把 waypoint 差异抹平
- **现象**：`F_traj` waypoint std≈1.12，但 `A_t`（point-scene）骤降到 0.09，`local_scene` waypoint-std≈0.096，最终 `x0_pred` 位置 std≈0.04（几乎一个点）。
- **判断**：不是主干坏了——只保留 `L_diff` 做 32 样本 overfit 时，位置 std 能长到 0.79（GT 0.87）。所以主要是**损失/安全项从第 0 步把轨迹压塌**，局部场景差异是结果而非主因。
- **后续可选**：若仍需要更强局部区分，可增大 `local_window`、提高 `scene_map` 分辨率，或给 point-scene 的 K/V 加相对位置。

## 3. 损失/训练策略

### 3.1 `L_diff` 纯 MSE 导致均值坍缩
- **现象**：模型预测整条轨迹≈一个点（数据集均值），reverse diffusion 全被吸引到该点。
- **解决**：`L_diff` 加“时变惩罚”：`MSE(Δx) + MSE(Δ位置速度)`，常数轨迹 `Δ=0` 被惩罚。

### 3.2 安全/几何损失从第 0 步压制轨迹展开
- **现象**：即便有端点 inpainting，训练 1 epoch 仍“聚团+直冲”。
- **诊断**：只训 `L_diff` 时轨迹正常展开；说明塌缩由安全/几何损失导致。
- **解决**：**warmup**：前 `warmup_epochs=3` 个 epoch 只训 `L_diff`（+小 `L_smooth`），之后才开 `L_ellipse/L_col`。这是最关键的修复。

### 3.3 `L_te` 权重为 0 仍出现在日志里
- 虽然 `lambda_te=0`，但 `total_loss` 仍计算/返回 `L_te`，日志打印了原始值，看起来像没关掉。
- **解决**：从 `total_loss.py` 完全删除 `L_te` 计算与返回值，并从 config 移除 `lambda_te`。

### 3.4 `L_ecol` 数值天然很大
- `L_ecol = Σ M̂·O` 对每椭圆 21×21≈441 点求和，原始值几十。
- **解决**：日志显示**归一化**值（除以 grid_n²，~0.02–0.07）；训练仍用原始值（`L_ecol_raw`）保证语义。

## 4. 训练稳定性

### 4.1 长时间训练 CUDA OOM（第 41 epoch）
- **现象**：跑到 epoch 40 后，epoch 41 开始 `torch.OutOfMemoryError`（GPU 12 GiB，分配 15.07 GiB）。
- **原因**：多 epoch 显存碎片/泄漏累积。
- **解决**：每 epoch `torch.cuda.empty_cache()`；`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（本机不支持，实际靠 empty_cache）。若仍 OOM，把 `train.batch_size` 降到 32。

### 4.2 手滑删掉关键 checkpoint
- 排查时误删 `epoch_40.pt`，只能从 `best.pt`（epoch 38）续训。
- 教训：**不要随意 `rm` 训练产物**；保留 `best.pt` 作为断点。

## 5. 其他

### 5.1 `from src/geometry.maze2d_env import MAZES`
- 迷宫布局集中在 `src/geometry/maze2d_env.py`（`MAZES`），由 `d4rl_coordinates.py` 使用同目录相对导入。

### 5.2 Neural-IRIS 网络已删除
- 原 `neural_iris` 子包与仓库 `src` 的冲突问题已不存在（Neural-IRIS 目录已移除）。
- 现仅用离线 IRIS MVIE（`src/geometry/iris_solver.py`），无神经网络依赖。

### 5.3 DataLoader 只设 pin_memory，不搬到设备
- `make_loader(device=...)` 只设 `pin_memory`，batch 仍是 CPU tensor；训练需 `.to(device)`。
- **解决**：train/val 循环里 `batch = {k: v.to(device) ...}`。

### 5.4 早期 reverse 点超出地图范围
- `x_T~N(0,I)` 归一化位置映射回 world 会超界（如 t≈63 时 42/128 点出界）。
- **这不是 bug**：扩散早期本就是噪声；到 t≤18 出界归 0。真正的问题是模型 `x0_pred` 坍缩成点。

### 5.5 `torch.as_tensor(numpy)` 默认 float64 导致 grid_sample dtype 不符
- `state_norm.mins` 是 float64 numpy，`torch.as_tensor` 变成 double，和 float32 模型 tensor 运算后 grid_sample 报 dtype 不匹配。
- **解决**：用 `torch.as_tensor(..., dtype=p_norm.dtype)` 转成与输入一致。