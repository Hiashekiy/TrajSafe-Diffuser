# DSH 夜间执行任务：TrajSafe-Diffuser 切换到 CARLA + B-spline 控制点扩散

> **目标：今晚直接完成代码升级、数据清洗/预处理、冒烟测试、过拟合测试，并启动正式训练。明早应至少有可用的 `best.pt` / `latest.pt` 和训练日志。**
>
> 不要只给方案、不要停在“代码已修改”。请实际执行、验证、训练，并在最后输出结果摘要。

当前工程根目录：

`D:\ProjectDirectory\Neural-IRISDiffuser`

当前 CARLA 数据目录：

`D:\ProjectDirectory\Neural-IRISDiffuser\data\carla_v1`

数据集仍可能处于继续采集状态，**不要假设已经有 20,000 条**。以磁盘上当前完整、可成功读取并通过校验的 sample 为准。

---

## 0. 先读取本地真实状态，不要按旧版本猜

开始修改前，先在工程根目录执行并记录：

```powershell
cd D:\ProjectDirectory\Neural-IRISDiffuser
git status
git branch --show-current
git log -5 --oneline
```

要求：

1. **本地代码是唯一 source of truth**，不要强行 checkout 或 reset 到远程版本。
2. 如果有未提交修改，先记录，不要覆盖用户已有改动。
3. 重点阅读当前这些文件的真实内容：
   - `configs/config.yaml`
   - `src/datasets/skeleton_dataset.py`
   - `src/models/trajsafe/planner.py`
   - `src/models/trajsafe/encoders.py`
   - `src/models/trajsafe/heads.py`
   - `src/models/trajsafe/ellipse.py`
   - `src/models/trajsafe/fusion.py`
   - `src/models/trajsafe/geometry.py`
   - `src/losses/losses.py`
   - `src/diffusion/sampler.py`
   - `src/diffusion/schedule.py`
   - `train.py`
   - `sample.py`
   - `evaluate.py`
   - `src/geometry/skeleton_graph.py`
   - `src/geometry/skeleton_paths.py`
   - `src/geometry/convex_region.py`
4. 读取数据目录下的说明文件，尤其是：
   - `data\carla_v1\DATASET_REPORT.md`
   - `data\carla_v1\dataset_config.json`
   - `data\carla_v1\dataset_state.json`
   - `data\carla_v1\samples.jsonl`
   - `data\carla_v1\episodes.jsonl`
5. 不要按照以前 Maze2D 的数据规模、路径或字段写死逻辑。

---

# 1. 本次升级最终定义

本次只完成 **训练主链升级**，不要把 ALM 一起重写。

## 1.1 Diffusion state 改为 32 个 cubic B-spline 控制点

旧：

\[
P_t\in\mathbb R^{B\times128\times2}
\]

新：

\[
\boxed{Q_t\in\mathbb R^{B\times32\times2}}
\]

其中：

- cubic B-spline，degree = 3
- controls = 32
- knots = 36
- curve decode points = 128
- knots 使用数据集根目录：`data\carla_v1\bspline_knots.npy`

**真正加噪、DDIM 反推的状态必须是 `Q_t [B,32,2]`。**

## 1.2 网络内部仍然保持 128 个 trajectory tokens

不要把现有 Transformer / Skeleton / Ellipse 主干改成 32 token。

新链路：

```text
Q_t [B,32,2]                   <- diffusion state
    ↓
Fixed B-spline Decoder
    ↓
P_t [B,128,2]
    ↓
TrajectoryEncoder
    ↓
Trajectory Backbone
    ↓
Skeleton Match / Topology
    ↓
固定 128 个等弧长 Skeleton centers
    ↓
128 Ellipse Geometry / Shape
    ↓
Fusion
    ↓
Final Denoiser
    ↓
raw curve [B,128,2]
    ↓
Trajectory→Control Head
    ↓
Q0_hat [B,32,2]
    ↓
B-spline Decoder
    ↓
P0_hat [B,128,2]
```

即：

\[
\boxed{\text{Control-space diffusion + 128-point curve-space reasoning}}
\]

---

# 2. Progress 设计：取消“预测进度”

当前旧逻辑类似：

```text
R_use -> MLP_prog -> H_prog -> Head_prog -> softplus+cumsum -> predicted s
```

本次要求删除：

- `Head_prog`
- softplus gaps
- cumulative predicted progress
- 所有 learned progress
- `L_align`
- `center_alignment_loss`
- 任何 `progress_gt`
- 任何 `ellipse_center_gt`

原 `MLP_prog` 的 feature transform 功能可以保留，但改名，例如：

```python
PathFeatureHead
```

输入输出：

```text
R_use [B,128,D]
    ↓
PathFeatureHead
    ↓
H_path [B,128,D]
```

在 planner 中注册固定 buffer：

```python
fixed_progress = torch.linspace(0.0, 1.0, 128)
```

因此：

\[
\boxed{s_i=i/127,\quad i=0,\ldots,127}
\]

然后仍然使用当前 dense Skeleton curve decoder：

\[
c_i=\Gamma(s_i)
\]

这里必须是 **dense Skeleton curve 的归一化弧长插值**，不是 array index 直接采样。

最终 128 个 ellipse center 应沿选中的 Skeleton **等弧长均匀分布**。

---

# 3. Ellipse 仍然保持 128 个

本次不要压成 29 个。

保持：

```text
ellipse center    [B,128,2]
ellipse shape4    [B,128,4]
a,b,theta         [B,128]
```

保留现有：

- `EllipseGeometry`
- `EllipseShapeHead`
- fine geometry attention
- `L_shape`
- `L_iou`
- `L_safe`

但 center 不再来自预测 progress，而是：

\[
\boxed{c_i=\Gamma(i/127)}
\]

本轮先不做 corridor/ALM 重写。

---

# 4. 新增统一 B-spline 模块

新增：

```text
src/geometry/bspline.py
```

建议提供：

```python
class BSplineCodec(nn.Module):
    ...
```

至少包括：

```python
decode_controls(q) -> p128
fit_curve_to_controls(p128, start, goal) -> q32
```

## 4.1 Control → Curve

预计算：

\[
B_{128}\in\mathbb R^{128\times32}
\]

并注册为 buffer。

解码：

\[
P=B_{128}Q
\]

PyTorch 可直接：

```python
p = torch.einsum("hk,bkd->bhd", basis_128, q)
```

要求：

```text
q : [B,32,2]
p : [B,128,2]
```

必须可微。

## 4.2 Curve → Control

用户要求“加一个轨迹到控制点之间的转换头”。

第一版请不要做可学习 MLP，优先实现成：

```python
class TrajectoryToControlHead(nn.Module)
```

但内部是 **固定、可微、endpoint-constrained least-squares projection**。

原因：

- B-spline degree / knots / K 全固定；
- 不需要额外学习映射；
- 输出有严格 B-spline control 语义；
- 可通过 autograd 回传；
- 改动更稳定。

端点约束必须：

\[
Q_0=start,\qquad Q_{31}=goal
\]

只拟合中间 30 个 control points。

设：

\[
B=[B_0,\ B_I,\ B_{31}]
\]

其中：

\[
B_I\in\mathbb R^{128\times30}
\]

则：

\[
Y=P-B_0start-B_{31}goal
\]

\[
Q_I=B_I^\dagger Y
\]

最后：

```text
Q = [start, Q_I, goal]
```

pseudoinverse 在 `__init__` 预计算并 `register_buffer()`，不要每个 batch 重算。

---

# 5. 起终点硬条件

以后真正 diffusion state 是 Q，所以硬条件作用在：

```python
q[:, 0]  = cond[:, 0]
q[:, -1] = cond[:, 1]
```

clamped cubic B-spline 天然保证：

\[
C(0)=Q_0,\qquad C(1)=Q_{31}
\]

新增类似：

```python
hard_control_endpoints(q, cond)
```

不要让“diffusion state 是 Q，但 endpoint 只在 P 上硬覆盖”的混合语义存在。

---

# 6. planner.py 目标结构

`TrajSafePlanner.forward_all()` 改为接收 `q_t`，而不是 `p_t`。

核心逻辑应接近：

```python
q_t = self.hard_control_endpoints(q_t, cond)
p_t = self.bspline.decode_controls(q_t)

c_g, c_e = self.scene_tokens(occ)
h_t = self.time_pe(t)

h_traj = self.encode_trajectory(p_t, c_g, h_t)

coarse_raw = self.head_p(h_traj)
q_coarse = self.traj_to_control(coarse_raw, cond)
coarse = self.bspline.decode_controls(q_coarse)

h_s = self.skeleton_encoder(candidate_xy)
r = self.match_block(h_traj, h_s, h_t)
topo = self.topology_head(r, candidate_mask)

r_use = r[batch_index, idx]

h_path = self.path_feature_head(r_use)

s = fixed_progress[None].expand(B, -1)

gamma = geometry[batch_index, idx]
gamma_len = geometry_lengths[batch_index, idx]
center = self.curve_decoder(gamma, gamma_len, s)

h_ell, geo_attn = self.ellipse_geometry(
    h_path, center, c_e, h_t, ab
)
shape = self.ellipse_shape_head(h_ell, h_t)

f = self.fusion_mlp(h_traj, h_path, h_ell)
h_clean = self.final_denoiser(f, c_g, h_t)

raw_final = self.head_p(h_clean)
q_final = self.traj_to_control(raw_final, cond)
final = self.bspline.decode_controls(q_final)
```

输出字典至少保留：

```python
{
    "input_control": q_t,
    "input_curve": p_t,
    "q_coarse": q_coarse,
    "coarse_raw": coarse_raw,
    "coarse": coarse,
    "control": q_final,
    "raw_curve": raw_final,
    "final": final,
    "H_traj": ...,
    "H_path": ...,
    "topo": ...,
    "selected_idx": ...,
    "ellipse": {
        "progress": fixed_progress,
        "center": center,
        "shape4": ...,
        "a": ...,
        "b": ...,
        "theta": ...,
    }
}
```

可以保留 `"progress"` key 兼容展示，但它现在必须是固定值，不是网络预测值。

---

# 7. heads.py / fusion.py / ellipse.py

## heads.py

删除旧 ProgressHead 的预测部分，替换为：

```python
class PathFeatureHead(nn.Module):
    def __init__(...):
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, r_use):
        return self.net(r_use)
```

新增/导出 `TrajectoryToControlHead`。

## fusion.py / ellipse.py

不改数学结构，只把 `h_prog` 重命名成 `h_path`。

不要继续在注释中写 learned progress feature。

---

# 8. 新 CARLA Dataset：不要硬塞进旧 Maze2D Dataset

新增：

```text
src/datasets/carla_spline_dataset.py
```

数据根目录：

```text
D:\ProjectDirectory\Neural-IRISDiffuser\data\carla_v1
```

每条 `.npz` 原始字段以本地 `DATASET_REPORT.md` 为准，不要猜。

至少会有：

```text
occupancy               [256,256]
trajectory_128          [128,2]
bspline_controls        [32,2]
bspline_dense_512       [512,2]
reference_route_local
trajectory_raw_local
start                    [2]
goal                     [2]
episode_id
raw_start_idx
raw_end_idx
```

---

# 9. 数据仍未采完：必须自己扫描和清洗

不要把 `target=20000` 当实际数据量。

实现 robust manifest loader，优先读 `samples.jsonl`，但每条必须同时检查：

1. 对应 sample `.npz` 文件真实存在；
2. `np.load()` 成功；
3. 关键 shape 正确；
4. 所有 float finite；
5. occupancy `[256,256]`；
6. trajectory_128 `[128,2]`；
7. start / goal 合法；
8. episode_id 合法；
9. split 与 sample 实际路径一致；
10. 不包含 validator 明确异常的样本。

遇到缺文件、半写入、损坏 npz、manifest 已写但文件未落盘：**跳过并记录 reason，不要崩整个程序。**

训练开始前做一次完整 scan，写：

```text
data/carla_processed/clean_manifest.jsonl
data/carla_processed/cleaning_report.json
```

之后本次训练使用这个固定 snapshot。不要训练中途动态把新采集的样本塞入 DataLoader。

---

# 10. CARLA occupancy 坐标必须修

CARLA 数据说明中的图像栅格：

```text
row 0   -> y_local = +40 m
row 255 -> y_local = -40 m
```

而当前仓库 Skeleton/ellipse 几何约定：

```text
row 0 -> scene y = -1
```

因此 CARLA 数据进入当前网络/几何代码之前统一：

```python
occupancy = np.flipud(occupancy).copy()
```

必须保证 preprocessing 和训练使用同一种 canonical occupancy。

不要只在 Dataset 训练时 flip，而 Skeleton preprocessing 使用未 flip map。

---

# 11. Control GT：不要直接相信原 bspline_controls

数据报告说明：

- `goal` 来自 Global Route；
- `bspline_controls` 来自真实 executed trajectory 的 B-spline fit。

因此原 `bspline_controls[-1]` 不保证严格等于 condition goal。

而新模型要求：

\[
Q_0=start,\qquad Q_{31}=goal
\]

所以从：

```text
trajectory_128
start
goal
```

重新生成训练专用：

```text
control_gt [32,2]
```

采用和模型完全一致的 endpoint-constrained projection。

优先不要改原始 `.npz`，在 processed cache 中保存新 control_gt。

统计：

```text
fit_rmse_scene
fit_rmse_meter
fit_max_error_meter
goal_to_executed_endpoint_meter
```

如果 endpoint 强制 goal 后拟合误差异常大，应剔除或标记，不要静默使用。

---

# 12. CARLA Skeleton candidate 预处理

不要破坏旧 Maze2D 脚本。

新增：

```text
scripts/data/carla/
    00_clean_dataset.py
    01_build_candidates.py
    02_build_ellipse_labels.py
    03_validate_processed.py
```

对每条 clean sample：

```text
canonical occupancy
    ↓
build_skeleton_graph()
    ↓
generate_candidates(start, goal)
    ↓
candidate_xy [M,128,2]
candidate_mask [M]
candidate_geometry
candidate_geometry_lengths
```

当前 candidate generator、dense geometry、no-corner-cut 逻辑尽量直接复用。

`topology_best` 仍定义：

\[
m^*=\arg\min_m nDTW(P^{GT}_{128},S_m)
\]

GT trajectory 只用于 topology label，不用于 progress label。

---

# 13. Ellipse GT 标签必须重新生成

旧标签如果以 `GT trajectory waypoint` 为 ellipse center，已经不符合新定义。

新定义：训练 routing 使用 `topology_best=m*`。

在 best Skeleton dense geometry 上固定：

\[
s_i=i/127
\]

得到：

\[
c_i=\Gamma_{m^*}(s_i)
\]

然后在这些固定 center 上生成/查询 safe ellipse shape：

```text
ellipse_shape4_gt [128,4]
shape_valid       [128]
```

不要生成：

```text
progress_gt
ellipse_center_gt
```

ellipse center 没有监督，它由 best Skeleton + fixed progress 决定。

---

# 14. GT ellipse mask 不再围绕 GT trajectory 构造

如果旧 Dataset 有：

```python
ellipse_mask = _ellipse_mask(pos, shape_gt, ...)
```

必须移除这种语义。

推荐 Dataset 只返回：

```text
ellipse_shape4_gt
shape_valid
```

训练 forward 已经得到 `ell["center"]`，且 training routing 使用 best candidate。

训练时临时生成 GT mask：

```python
a_gt, b_gt, theta_gt = shape4_to_abtheta(shape_gt)

gt_mask = ellipse_soft_mask(
    ell["center"].detach(),
    a_gt,
    b_gt,
    theta_gt,
    ...
)
```

这样 center 定义唯一。

---

# 15. Loss 修改

删除：

```text
Lalign
center_alignment_loss
lambda_align
```

新总损失：

\[
L=
\lambda_{traj}L_{traj}
+\lambda_{ctrl}L_{ctrl}
+\lambda_{coarse}L_{coarse}
+\lambda_{smooth}L_{smooth}
+\lambda_{topo}L_{topo}
+\lambda_{shape}L_{shape}
+\lambda_{iou}L_{iou}
+\lambda_{safe}L_{safe}
\]

## 15.1 Ltraj

作用在 decode 后曲线上：

\[
\boxed{L_{traj}=MSE(P_0^{pred},P_{128}^{GT})}
\]

## 15.2 Lctrl

新增：

\[
\boxed{L_{ctrl}=MSE(Q_{1:30}^{pred},Q_{1:30}^{GT})}
\]

endpoint 已 hard condition，不计算。

新增 `control_x0_loss()`。

## 15.3 其他

- `Lcoarse` 保留，比较 `coarse` 与 curve GT；
- `Lsmooth` 保留，作用于 decode 后 128 点曲线；
- `Ltopo` 保留；
- `Lshape/Liou/Lsafe` 保留，但使用新的 fixed-Skeleton-center ellipse GT 语义。

---

# 16. train.py：diffusion target 必须改为 Q

旧：

```python
p0 = batch["pos"]
p_t = add_noise(p0, ...)
```

新：

```python
q0 = batch["control_gt"]
p_gt = batch["curve_gt"]
```

随机 timestep：

```python
q_t, _ = add_noise(q0, t, schedule)
q_t = model.hard_control_endpoints(q_t, cond)

out = model.forward_all(q_t, ...)
```

Loss：

```python
l_traj   = trajectory_x0_loss(out["final"], p_gt)
l_ctrl   = control_x0_loss(out["control"], q0)
l_coarse = trajectory_x0_loss(out["coarse"], p_gt)
l_smooth = trajectory_smoothness_loss(out["final"], p_gt)
l_topo   = topology_ce(...)
l_shape  = ...
l_iou    = ...
l_safe   = ...
```

删除所有 `l_align`。

---

# 17. 修复 sel_best_rate 统计

训练时 `select_index=best`，所以：

```python
out["selected_idx"] == best
```

不是预测准确率。

改成：

```python
pred_idx = out["topo"]["pi"].argmax(dim=-1)

sel_best_rate = (
    pred_idx[has_cand] == best[has_cand]
).float().mean()
```

---

# 18. sampler.py：DDIM 真正切换到 32 control state

旧初始化：

```python
p = torch.randn(B, H, 2)
```

新：

```python
q = torch.randn(B, 32, 2)
q[:, 0] = start
q[:, -1] = goal
```

每个 reverse timestep：

```python
out = model.forward_all(q, ...)
q0 = out["control"]
```

DDIM 公式全部作用于：

\[
q_t,\ q_0
\]

最后：

```python
p = model.bspline.decode_controls(q)
```

返回：

```python
{
    "control": q,
    "p": p,
    ...
}
```

trace 同时保存 q/p/q0/final curve/ellipse/topology。

---

# 19. ALM 本次不要重写

当前 ALM 是 waypoint-based，新 diffusion state 已改 control space。

本次 config 默认：

```yaml
alm:
  enabled: false
```

如果 dashboard/CLI 主动开启旧 ALM，明确阻止或报提示：

```text
B-spline control-space ALM 尚未迁移，本版本禁用旧 waypoint ALM
```

不要让旧 ALM 静默作用在错误语义上。

本次目标是先得到训练好的 B-spline 模型。

---

# 20. config 建议

不要机械覆盖本地已有配置，按真实字段兼容修改。

推荐至少：

```yaml
data:
  dataset: "carla_v1"
  root: "D:/ProjectDirectory/Neural-IRISDiffuser/data/carla_v1"
  processed_root: "D:/ProjectDirectory/Neural-IRISDiffuser/data/carla_processed"
  batch_size: 16
  num_workers: 0

model:
  horizon: 128
  num_controls: 32
  d_model: 128
  num_heads: 4
  traj_blocks: 8
  skeleton_blocks: 2
  final_blocks: 3

bspline:
  degree: 3
  num_controls: 32
  curve_points: 128
  knots: "D:/ProjectDirectory/Neural-IRISDiffuser/data/carla_v1/bspline_knots.npy"
  endpoint_constrained: true

loss:
  lambda_traj: 1.0
  lambda_control: 0.2
  lambda_coarse: 0.5
  lambda_smooth: 0.08
  lambda_topology: 0.25
  lambda_shape: 0.08
  lambda_iou: 0.25
  lambda_safe: 0.15

alm:
  enabled: false
```

`lambda_control=0.2` 只是初始值。

正式长训前跑 20~50 batch 看 weighted loss contribution；只允许做必要的一次权重修正，不要整夜耗在调参。

---

# 21. 训练前必须完成的测试

至少验证：

### Test 1：B-spline decode

```text
[B,32,2] -> [B,128,2]
```

### Test 2：hard endpoint

```text
decoded[:,0]  == start
decoded[:,-1] == goal
```

允许浮点小容差。

### Test 3：curve → control → curve

```text
trajectory_128
 -> endpoint-constrained fit
 -> q32
 -> decode
 -> p128
```

打印 RMSE/max error。

### Test 4：fixed progress

必须：

```text
s[0] = 0
s[-1] = 1
diff(s) == 1/127
```

### Test 5：ellipse shape

```text
center [B,128,2]
shape4 [B,128,4]
```

### Test 6：diffusion state

实际噪声 tensor 必须是：

```text
[B,32,2]
```

### Test 7：gradient

至少一批：

```text
Ltraj + Lctrl + ... -> backward()
```

所有核心模块 grad finite。

### Test 8：CARLA y-axis

人工占据位置验证 flip 后 map / Skeleton / scene coordinates 一致。

### Test 9：无旧 learned progress

全仓库 grep，确保训练主链不存在：

```text
progress_gt
ellipse_center_gt
lambda_align
center_alignment_loss
Head_prog
```

兼容展示字段 `progress` 可以存在，但必须来自 fixed buffer。

---

# 22. 今晚执行流程：不要停在代码完成

## 阶段 A：数据扫描/清洗

运行 clean script。

最终打印：

```text
manifest entries
existing npz
valid samples
invalid/missing samples
train / val / test counts
unique episodes
control refit RMSE p50/p95/max
goal-vs-executed-end distance p50/p95/max
```

坏样本：记录到 `cleaning_report.json`，跳过，不要因少数坏样本停止任务。

## 阶段 B：CARLA preprocessing

为当前 clean snapshot 生成 candidate cache、topology_best、fixed-center ellipse shape labels。

要求支持 resume。

如果个别 sample Skeleton 无候选：

- 标记 `has_candidate=false`
- 统计
- 不要让整个 preprocessing 崩掉

如果无候选比例异常高（例如 >10%），先检查 occupancy flip / start-goal / dilation，不要直接进入训练。

## 阶段 C：完整 validator

运行 `03_validate_processed.py`。

至少检查：shape、finite、candidate validity、best index、fixed centers、ellipse label validity、control fit、endpoint、split leakage。

## 阶段 D：1 batch forward/backward

检查所有 loss finite、total finite、gradient finite、显存正常。

## 阶段 E：32 样本过拟合

必须实际 overfit 32 samples。

要求：

```text
Ltraj 明显下降
Lctrl 明显下降
total 明显下降
```

保存一张 overfit 可视化：

```text
occupancy
GT curve
pred curve
GT controls
pred controls
selected Skeleton
128 ellipse centers
若方便可画部分 ellipses
```

如果 32 个样本都过拟合不了，不允许直接开启长训，先定位问题。

---

# 23. 今晚正式训练策略

只要前面全部通过，立即开始正式训练。

数据仍在采集，所以训练使用 clean snapshot，不等 20k。

如果当前有效数据 >= 1000 就直接训练；如果已有 3k/5k/10k，也全部使用当前 snapshot。

默认：

```text
batch_size = 根据显存选择稳定值，优先 16
lr = 2e-4
AdamW
grad_clip = 1.0
```

目标是明早有尽可能好的 checkpoint，而不是机械追求 100 epochs。

建议：

```text
max_hours = 6~7 小时
```

或运行至 epochs 完成，以先到者为准。

每个 epoch 保存：

```text
outputs/bspline_carla/ckpt/latest.pt
```

有 val 时保存：

```text
outputs/bspline_carla/ckpt/best.pt
```

训练日志：

```text
outputs/bspline_carla/train.log
```

训练 summary：

```text
outputs/bspline_carla/training_summary.json
```

---

# 24. Validation 不要只看总 loss

至少记录：

```text
Ltraj
Lctrl
Lcoarse
Lsmooth
Ltopo
Lshape
Liou
Lsafe
```

并尽量记录：

```text
pred topology best rate
B-spline curve RMSE
goal error
collision rate
ellipse center free rate
ellipse unsafe / collision metric
```

不要再记录假的 training `selected_idx == best` 作为 topology accuracy。

---

# 25. 明早必须留下的结果

在：

```text
D:\ProjectDirectory\Neural-IRISDiffuser\outputs\bspline_carla\
```

至少留下：

```text
ckpt\best.pt
ckpt\latest.pt
train.log
training_summary.json
cleaning_report.json 或其路径说明
preprocess_report.json
overfit_preview.png
sample_preview.png
NIGHT_RUN_REPORT.md
```

如果 best.pt 因验证尚未触发而不存在，主动做一次 validation 并保存 best checkpoint，不允许最终只有 latest。

---

# 26. NIGHT_RUN_REPORT.md 必须写什么

生成：

```text
D:\ProjectDirectory\Neural-IRISDiffuser\outputs\bspline_carla\NIGHT_RUN_REPORT.md
```

内容必须包括：

```text
1. 当前 git commit / branch / 工作区状态
2. 实际读到的数据量
3. 数据清洗掉了多少、原因
4. train/val/test 数量
5. preprocessing 成功率
6. candidate empty rate
7. ellipse label valid rate
8. B-spline endpoint-constrained fit RMSE/max
9. 32-sample overfit 是否通过
10. 正式训练开始时间/结束时间
11. epochs / steps
12. best val epoch
13. best checkpoint 路径
14. latest checkpoint 路径
15. 最终各项 loss
16. 主要评估指标
17. sample preview 路径
18. 修改了哪些文件
19. 当前仍未做的事项
20. 明确写：ALM 本轮未迁移，仍关闭
```

---

# 27. 代码保护要求

1. 不要删除旧 Maze2D 数据和原始 CARLA 数据。
2. 不要覆盖 `data\carla_v1` 原始 `.npz`。
3. 新 processed 数据放独立目录。
4. 新 output 放 `outputs\bspline_carla`。
5. 不要改变 80m scene 定义。
6. 不要改变 128 ellipse 设计。
7. 不要重新引入 predicted progress。
8. 不要做 GT trajectory → Skeleton projection。
9. 不要生成 `progress_gt`。
10. 不要生成 `ellipse_center_gt`。
11. 不要把 ALM 混入本次训练升级。
12. 不要为了兼容旧 dashboard 破坏新训练主链。
13. dashboard 如需大改，先保证 train/sample/evaluate CLI 工作。
14. basis、pseudoinverse 等固定矩阵使用 `register_buffer()`。
15. debug/assert 模式下检查 tensor shape。
16. Dataset 必须容忍未完全采样的数据目录。
17. 单条坏 sample 跳过并报告，不要让整夜任务中断。

---

# 28. 本轮不要做的事情

全部推迟：

```text
128 convex regions -> 整体 safety corridor
B-spline convex hull ALM
Bezier extraction
control-space ALM
CARLA online closed-loop deployment
NPC/dynamic obstacles
vehicle dynamics constraints
```

本轮唯一目标：

\[
\boxed{
\text{CARLA dataset}
+
\text{32-control B-spline diffusion}
+
\text{128-token existing network}
+
\text{fixed 128 Skeleton progress}
+
\text{128 ellipse}
}
\]

并且：

\[
\boxed{\text{明早留下一个实际训练好的模型}}
\]

---

# 29. 最终验收标准

只有同时满足以下条件才算完成：

- [ ] 当前 CARLA 数据成功扫描并建立 clean snapshot
- [ ] corrupted/incomplete samples 被跳过并有报告
- [ ] occupancy 坐标方向验证正确
- [ ] 32-control endpoint-constrained B-spline codec 测试通过
- [ ] diffusion state 真实为 `[B,32,2]`
- [ ] 网络内部仍为 128 trajectory tokens
- [ ] learned ProgressHead 已从主链移除
- [ ] fixed progress 为 `i/127`
- [ ] 128 ellipse 保留
- [ ] `L_align` 删除
- [ ] `L_control` 新增
- [ ] ellipse GT 改为 fixed Skeleton center 语义
- [ ] 1 batch forward/backward 成功
- [ ] 32 samples overfit 成功
- [ ] 正式训练实际启动并运行
- [ ] `best.pt` 生成
- [ ] `latest.pt` 生成
- [ ] `NIGHT_RUN_REPORT.md` 生成
- [ ] ALM 保持关闭且没有错误复用旧 waypoint ALM

如果本地代码与本说明存在小的文件名/接口差异，以**实现上述数学定义和数据契约**为最高优先级，不要为了逐字遵守文件名而做错误修改。

最重要的是：**不要停在分析或代码编写阶段。今晚必须实际把 pipeline 跑通，并让正式训练跑起来。**
