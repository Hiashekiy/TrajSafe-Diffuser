# TrajSafe-Diffuser 项目报告

> **一句话**：在 CARLA 占据地图上，用「骨架拓扑先验 + 控制点空间扩散 + 推理期安全走廊 ALM」
> 生成满足端点硬约束、可行驶、并附带沿路径安全椭圆的 B 样条轨迹。
>
> | 项 | 值 |
> |---|---|
> | 仓库 | `D:\ProjectDirectory\Neural-IRISDiffuser` |
> | 分支 / HEAD | `v3-skeleton-dynamic` / `b94c631`（2026-09-20） |
> | Python 环境 | `E:/CondaEnvData/envs/GGMPC/python.exe`（Python 3.10 + torch 2.9.1+cu126） |
> | 数据 | CARLA v1 清洗快照，5098 样本（train 4265 / val 398 / test 435） |
> | 主 checkpoint | `outputs/bspline_carla/ckpt/best_task.pt`（epoch 449） |
> | 模型规模 | `TrajSafePlanner-controlspace-32`，7.31 M 参数 |
> | 测试 | `python -m pytest tests -q` → **113 passed**（23 s） |
>
> 权威规格：`docs/CONTROL_SPACE_REFACTOR.md`（当前实现）＞ `docs/ARCHITECTURE.md`（实现说明）
> ＞ `docs/TrajSafe-Diffuser_网络结构设计报告_实现细化版.md`（重构**前**的 128 曲线点设计，
> 与当前实现冲突时以控制点空间重构说明为准）。

---

## 目录

1. [项目背景与问题定义](#1-项目背景与问题定义)
2. [系统总览](#2-系统总览)
3. [形式化与符号](#3-形式化与符号)
4. [方法](#4-方法)
5. [数据流水线](#5-数据流水线)
6. [工程实现](#6-工程实现)
7. [实验与结果](#7-实验与结果)
8. [已知局限](#8-已知局限)
9. [复现指南](#9-复现指南)
10. [后续工作](#10-后续工作)
11. [附录：代码地图与术语](#11-附录代码地图与术语)

---

## 1. 项目背景与问题定义

### 1.1 任务

给定一张占据地图（occupancy grid）、起点 `S` 与终点 `G`，输出：

1. 一条**从 `S` 到 `G`、无碰撞、平滑**的轨迹；
2. 沿轨迹的**安全椭圆序列**（每个采样点处可容纳的最大安全椭圆），用于下游
   控制/MPC 的可行域或安全校验。

评估场景来自 CARLA 采集的 80 m × 80 m 局部地图快照，栅格分辨率 256 × 256。

### 1.2 为什么用扩散模型

- 轨迹规划本质是**多模态分布**建模：同一对起终点存在多条合理走廊，回归模型会输出
  均值轨迹（常常撞墙），扩散模型可以采样出多个模式。
- 反向扩散天然支持**推理期约束注入**：不需要重训即可在每个去噪步对 clean 预测做投影/
  修正（本项目的 ALM 正是建立在这一性质上）。

### 1.3 三个关键设计选择

| 选择 | 替代方案 | 本项目的原因 |
|---|---|---|
| **骨架（Skeleton）拓扑作为几何先验** | 纯 occupancy 条件 | 骨架把自由空间压缩成拓扑图，模型只需在 M 条候选走廊里选择，而不是从像素里"猜"可行通道 |
| **B 样条控制点作为唯一 diffusion state** | 128 个轨迹点 token | 控制点维度低（C=32 vs H=128）、天然平滑、端点可硬约束；网络内部不再出现 128 点张量 |
| **推理期凸走廊 + ALM 修正** | 训练期安全损失 / 采样后滤波 | 训练期安全损失只鼓励"期望安全"，无法给硬保证；ALM 在每步把预测拉进冻结走廊，且不改网络/checkpoint |

---

## 2. 系统总览

```
                       ┌──────────────────────── 离线（数据流水线）────────────────────────┐
data/carla_v1  ──00_clean──►  clean_manifest + 控制点 GT ──01_candidates──►  候选 S_m / Γ_m
   (.npz)                    (control_gt.npy)                                  (candidate_*.npy)
                             └──02_ellipse_labels──►  ellipse_shape4_gt.npy ──03_validate──► 报告
                       └───────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
   data/carla_processed/{train,val,test}   ──►  CarlaSplineDataset  ──►  train.py
                                                                          │  (8 项损失)
                                                                          ▼
                                                                  outputs/bspline_carla/ckpt/
                                                                   best.pt / best_task.pt / latest.pt
                                              │
                       ┌──────────────────────┴───────────────────────┐
                       ▼                                              ▼
              sample.py / evaluate.py                     diffusion-dashboard/
        (DDIM + 走廊 + ALM，出图 + 指标)              (backend 8765 + Next 前端 3000)
                                                      在线骨架/候选重建 + 逐步回放
```

- **训练期**：只训练网络（拓扑选择、控制点去噪、椭圆形状），`alm_enabled = False`。
- **推理期**：`sample()` 内部跑三阶段状态机（WARMUP → TRY_ACTIVATE → GUIDED），
  走廊与约束包构建一次后**冻结**，只刷新网络的 clean 预测。ALM 是纯推理期扩展，
  不改变网络结构、checkpoint 与训练损失。

---

## 3. 形式化与符号

| 记号 | 含义 | 配置项 | 默认 |
|---|---|---|---:|
| `C` | B 样条控制点数（= 轨迹 token 数，唯一轨迹表示） | `model.num_controls` = `bspline.num_controls` | 32 |
| `Q` | 骨架 / 椭圆安全 query 数 | `model.num_safety_queries` = `topology.candidate_points` | 128 |
| `H` | 末端解码曲线采样点数 | `bspline.curve_points`（别名 `model.horizon`） | 128 |
| `M` | 候选骨架拓扑条数 | `topology.num_candidates` | 4 |
| `L` | 候选折线输入长度 | `topology.candidate_points` | 128 |
| `T` | 扩散步数 | `diffusion.timesteps` | 16 |
| `B_128` | 曲线解码矩阵 `[H, C]` | `src/geometry/bspline.py` | — |

**扩散状态只有一个**：控制多边形 `Q_t ∈ R^{B×C×2}`。
`C` 与 `Q` 语义彻底分离——`C` 是轨迹变量，`Q` 只服务骨架匹配与椭圆几何；
两处 `num_controls` 必须相等，否则 `TrajSafePlanner.__init__` 直接报错（防止"改了配置没改数据"）。

**坐标**：场景坐标 `scene ∈ [-1, 1]²`，80 m ↔ 2.0，即 **1 scene = 40 m**。
CARLA 原始 occupancy 的 row 0 对应 `y_local = +40 m`，进入几何/网络前必须
`np.flipud(occupancy).copy()`；仓库只保存/使用 flip 后的 canonical 栅格。

---

## 4. 方法

### 4.1 端点硬约束：固定、零参数的 Boundary Decoder

网络自由输出 `Q~`，随后施加**常量修正 profile**：

```
e_s = S - Q~_0          e_g = G - Q~_{C-1}
w^s = [1, 0.75, 0.5, 0.25, 0, ...]        (前 k = max(1, min(span, C//2)) 个控制点)
w^g = [..., 0, 0.25, 0.5, 0.75, 1]        (末端镜像)

Q*_i = Q~_i + w^s_i·e_s + w^g_i·e_g
```

- 对**任意**预测严格有 `Q*_0 = S`、`Q*_{C-1} = G`；clamped 节点向量保证这等价于
  曲线端点条件 `C(0)=S`、`C(1)=G`。
- 相邻控制点**协调移动**，而不是被瞬间拉到端点上（避免端点附近折角）。
- 模块**零参数**，`profile` 是 `persistent=False` 的 buffer，因此不出现在
  `state_dict()` 里，不破坏旧 checkpoint 的 `strict=True` 加载。
- `C` 很小（如 2/4）时自动退化为纯端点硬条件。

### 4.2 网络结构（`TrajSafePlanner.forward_controls`）

```
Q_t [B,C,2]                       扩散状态；端点硬条件
  ├─ ControlEncoder                                                  -> H_ctrl [B,C,D]
  ├─ ControlBackbone（N_T=8 个 TrajBlock，AdaLN）                     -> H_ctrl [B,C,D]
  ├─ head_p                                                          -> Q~_coarse [B,C,2]
  ├─ BoundaryDecoder（固定）                                          -> Q_coarse  [B,C,2]（仅绘图/回退）
  ├─ SkeletonEncoder(candidate_xy)（N_S=2）                           -> H_S [B,M,L,D]
  ├─ MatchBlock(H_ctrl, H_S, h_t)                                    -> R [B,M,C,D]
  ├─ TopologyHead(R)                                                 -> π；m = m*(训练) / argmax π(推理)
  ├─ PathFeatureHead(R[m])                                           -> H_path [B,C,D]
  ├─ SafetyQueryHead(H_S[m])                                         -> H_safety [B,Q,D]
  ├─ c_i = Γ_m(i/(Q-1))（固定 buffer，无 center head / 无 progress head）
  ├─ EllipseGeometry + EllipseShapeHead                              -> shape4 [B,Q,4]
  ├─ SafetyControlFusion（cross attention：控制 ← 安全）               -> A_safe [B,C,D]
  ├─ FusionMLP([H_ctrl, H_path, A_safe])                             -> F [B,C,D]
  ├─ FinalDenoiser（N_F=3）                                          -> H_clean [B,C,D]
  ├─ head_p                                                          -> Q~_final [B,C,2]
  └─ BoundaryDecoder（固定）                                          -> Q_final  [B,C,2]
                                                                        └─ B_128·Q_final -> 曲线（仅绘图/指标/控制器）
```

关键点：

- 粗解码头**直接输出控制点**，不再走 `128 点 → 最小二乘拟合 → 32 控制点`。
- 网络内部**没有任何 128 点轨迹张量**；`B_128` 解码只在链路最末端出现一次。
- 训练链是**纯控制点的**：没有任何损失读取解码曲线；两个诊断解码
  （`input_curve` / `raw_curve`）在 `torch.no_grad()` 下计算，不进 autograd 图。
- 椭圆分支运行在**被选中骨架的 Q 个 token** 上，`shape4` 仍是 `[B,128,4]`，
  与离线标签逐位对应。
- 条件编码：`SpatialPE` / `PE_1D` / sinusoidal timestep + 2 层 MLP（AdaLN）；
  场景编码 `C_G`（16×16 全局）与 `C_E`（32×32 细粒度）。
- 模型超参：`d_model=128`、`num_heads=4`、`ffn_dim=512`、`traj_blocks=8`、
  `skeleton_blocks=2`、`final_blocks=3`、dropout=0。

### 4.3 骨架拓扑先验

1. `build_skeleton_graph(occ, safety_dilation_cells=1, thinning_backend="auto")`：
   对自由空间做膨胀 + Guo-Hall 细化，得到骨架图（nodes / branches）。
2. `generate_candidates(graph, start, goal, CandidateConfig)`：在骨架图上搜索 M=4 条
   候选路径，去重后打包成网络的 `candidate_xy [B,M,L,2]`（`S_m`）与 dense 几何
   `Γ_m`（`candidate_geometry`，1280 点预算）。
3. 训练标签 `topology_best = argmin_m nDTW(curve_gt, S_m)`；
   损失 `L_topo = CE(π, m*)`。
4. 推理时 `m = argmax(π)`，**每个反向步都重新打分**（没有 commit timestep）。

### 4.4 椭圆安全场

- 椭圆**不是**扩散状态，也**不是** head 输出：圆心由选中骨架的**固定进度**给出
  `c_i = Γ_m(i/(Q-1))`，`s = linspace(0,1,128)`。
- **没有 center head、没有 progress head、没有 `L_align`**；`L_shape` 只作用于
  `EllipseShapeHead`，不含圆心。
- 形状参数化 `shape4 = [log a, log b, cos 2θ, sin 2θ]`，`a ≥ b > 0`（稳定参数化，
  避免 π 周期歧义与长轴/短轴翻转）。
- 离线标签：在 canonical occupancy 上用 36 个朝向 / `local_radius=0.25` /
  `boundary=48` / 2×12 内部采样 / 8 步二分，求每个固定 center 处的最大安全椭圆。

### 4.5 扩散与采样

- `NoiseSchedule`：`squaredcos_cap_v2`，`T = 16`，`prediction_type = "x0"`。
- DDIM 反向：`ε_t = (q_t − √ᾱ_t·Q̂_0)/√(1−ᾱ_t)`，`q_{t−1}` 由 `Q̂_0` 与 `ε_t` 重参数化；
  只有 `Q_final` 进入 DDIM，`t=0` 之后**不再做第二次 final projection**。
- 支持 `--steps` 子采样（少于 16 步的 DDIM）与 `--arch {auto,control,legacy}`。

### 4.6 训练损失（8 项）

```
L = 0.20·L_ctrl + 0.50·L_coarse + 0.08·L_smooth + 0.20·L_boundary
  + 0.25·L_topo + 0.08·L_shape + 0.25·L_iou + 0.15·L_safe
```

| 项 | 权重 | 定义与设计动机 |
|---|---:|---|
| `L_ctrl` | 0.20 | `MSE(Q~_final[1:-1], Q_GT[1:-1])`，用 Boundary Decoder **之前**的 raw 控制点；端点 control 故意排除（否则会把 `Q_1..Q_3` 一起拽向起点） |
| `L_coarse` | 0.50 | `MSE(Q~_coarse[1:-1], Q_GT[1:-1])`，约束粗解码头 |
| `L_smooth` | 0.08 | 控制多边形二阶/三阶差分（`α=0.25, β=1.0`），按 detached GT 平均步长归一化；**不解码任何轨迹点** |
| `L_boundary` | 0.20 | `L_b(Q~_final) + 0.5·L_b(Q~_coarse)`；目标是把 GT 控制多边形刚体平移到本样本端点：`T^s_i = S + (Q^GT_i − Q^GT_0)`、`T^g_j = G + (Q^GT_j − Q^GT_{C-1})` |
| `L_topo` | 0.25 | `CE(π, m*)`，`m*` 由 nDTW 与 GT 轨迹确定 |
| `L_shape` | 0.08 | 椭圆 `shape4` 监督 |
| `L_iou` | 0.25 | 椭圆软 mask IoU（lazy GT soft mask） |
| `L_safe` | 0.15 | 椭圆 unsafe fraction + 逐轨迹 CVaR（`cvar_fraction=0.2`） |

**已删除**：`L_traj`、`L_align`，以及 `trajectory_x0_loss` / `trajectory_smoothness_loss`。
超级版设计里的 `progress_gt` / `ellipse_center_gt` / GT 骨架投影均已不存在。

> ⚠️ **踩过的坑（有测试守着）**：`L_boundary` 必须**对 batch 取平均**
> （先对每个样本求和、再 `.mean()`）。曾经的实现"整个 batch 求和 + 分母只是权重和"
> 等于把 `lambda_boundary` 乘上了 batch size（batch=16 时等效权重 3.2 而不是 0.2）。
> 见 `tests/test_control_space.py::test_control_losses_are_batch_size_invariant`
> 与提交 `b94c631`。

### 4.7 推理期：安全走廊 + 控制空间 ALM

三阶段状态机（`src/diffusion/sampler.py`）：

```
Q_T
 └─ WARMUP       前 warmup_reverse_steps(=3) 次 reverse forward，纯 DDIM
 └─ TRY_ACTIVATE 此后每次 reverse forward 尝试一次，按 argmax(π) 起逐个候选（最多 4 个）
      forward_all(select_index=m)
        → center = Γ_m(i/127)，shape4 → 128 个凸区域
        → 相邻区域 overlap ratio（面积交 / 较小面积）
        → overlap < min_overlap_ratio(0.10) ⇒ 在 Skeleton 进度中点 Γ((s_i+s_{i+1})/2)
          生成 identity-metric point-seeded gap region（每 gap 最多 1 个，不递归）
        → 冻结 corridor + BSplineConstraintPack
 └─ GUIDED       走廊/约束冻结，每个 reverse step：
      Q0_raw  = network(q_t)
      Q0_safe, λ = bspline_alm_correct(Q0_raw, pack, λ)   # λ 跨步 warm-start
      DDIM(ε_t = (q_t − √ᾱ_t·Q0_safe)/√(1−ᾱ_t)) → q_{t−1}
 └─ 512 点 dense validation：collision / endpoint / pack violation / corridor membership
```

关键不变量（全部有单测覆盖）：

- activation 后 topology / corridor / bridge / region / constraint pack **完全冻结**；
- 每步的 proximity reference 都是**当前步**网络输出的 `Q0_raw`；
- primal 变量只有 32 个控制点，`Q_0 = S`、`Q_31 = G` 恒成立；
- 约束是 `[piece × 4 个 Bézier 控制点 × 全部有效 face]`，来自**精确 Bézier 抽取**
  `β = E·Q`，靠凸包性质给出**连续**安全证书（200 点采样最大误差 < 1e-5）；
- `max_curve_step_scene` 限制的是真实曲线位移 `max_i ‖B·δQ_i‖`，而不是参数增量。

---

## 5. 数据流水线

### 5.1 流程与命令

```bash
python scripts/data/carla/00_clean_dataset.py        --config configs/config.yaml
python scripts/data/carla/01_build_candidates.py     --config configs/config.yaml
python scripts/data/carla/02_build_ellipse_labels.py --config configs/config.yaml
python scripts/data/carla/03_validate_processed.py   --config configs/config.yaml
```

- 原始数据 `data/carla_v1/`（**只读**）：`samples.jsonl` / `episodes.jsonl` /
  `bspline_knots.npy` / `samples/<split>/*.npz`（`occupancy[256,256] u8`、
  `trajectory_128[128,2]`、`bspline_controls[32,2]`、`start`、`goal`、`episode_id`）。
- 处理结果 `data/carla_processed/{train,val,test}/`，接口契约见
  `docs/CARLA_BSPLINE_PIPELINE_SPEC.md`（改字段/形状前必须先改该文）。

### 5.2 处理快照的数组契约

| 文件 | shape | dtype | 含义 |
|---|---|---|---|
| `conditions.npy` | `[N,2,2]` | f32 | `[start, goal]`（scene） |
| `control_gt.npy` | `[N,32,2]` | f32 | 端点约束重拟合控制点，`q[0]=start, q[-1]=goal` |
| `curve_gt.npy` | `[N,128,2]` | f32 | 原始 `trajectory_128` |
| `occupancy.npy` | `[N,256,256]` | u8 | **canonical（已 flipud）** 占据栅格 |
| `episode_id.npy` | `[N]` | i64 | episode 划分单位（禁止跨 split 泄漏） |
| `candidate_xy.npy` | `[N,4,128,2]` | f32 | 网络输入 `S_m` |
| `candidate_mask.npy` | `[N,4]` | bool | 候选是否有效 |
| `candidate_geometry*.npy` | flat + offsets | i16/i64/i32 | dense `Γ_m` cell 索引（`(px+0.5)·cell−1` 还原 scene） |
| `topology_best.npy` | `[N]` | i64 | `argmin_m nDTW(curve_gt, S_m)` |
| `ellipse_shape4_gt.npy` | `[N,128,4]` | f32 | `[log a, log b, cos2θ, sin2θ]` |
| `shape_valid.npy` | `[N,128]` | bool | 该 center 是否有安全椭圆标签 |

**禁止**出现 `progress_gt.npy` / `ellipse_center_gt.npy`（语义已删除）。

### 5.3 实际数据统计（`NIGHT_RUN_REPORT.md`）

| 指标 | 值 |
|---|---|
| 扫描 / 有效样本 | 5098 / 5098（清洗无效 0，缺失 npz 0） |
| split | train 4265（215 episodes）/ val 398（19）/ test 435（21） |
| split 泄漏 | 无 |
| 候选为空率 | 0.39 %（train 0.40 % / val 0.75 % / test 0 %） |
| 平均有效候选数 | 3.43 / 3.62 / 3.59（M=4） |
| 椭圆标签有效率 | 99.19 %（test 99.52 %） |
| B 样条端点约束拟合 RMSE | mean 0.031 m / p95 0.111 m / max 0.240 m |
| 数据 GT 终点 vs 执行终点 | mean 0.335 m（原始轨迹本身不精确到点，是数据属性） |

---

## 6. 工程实现

### 6.1 配置驱动，无硬编码

`configs/config.yaml` 是唯一数值来源：数据路径、`C` / `Q` / `H` / `M`、网络超参、
8 项损失权重、DDIM 步数、`corridor` / `alm` 全部在此。

**修改控制点个数 `C` 的完整步骤**：

1. 同时改 `model.num_controls` 与 `bspline.num_controls`；`knots` 可写 `"auto"`
   （按 `(C, degree)` 生成 clamped 均匀节点向量），或让代码检测到长度不匹配时自动切换并告警。
2. 重跑 4 个数据脚本（`control_gt` 是**离线数据**，必须与 `C` 对齐）。
3. 训练 / 采样照旧。若 `C` 与快照不一致，`CarlaSplineDataset` **直接报错**而不是静默用错标签。

`Q` 同理可改，但必须等于 `topology.candidate_points` 与 `ellipse_shape4_gt` 的行数。

### 6.2 旧 checkpoint 兼容（重构的关键约束）

需求原文要求"已经训练好的模型修改代码后仍可使用（要用于展示，重训成本高）"。

- `src/utils/checkpoint.py`：
  - `detect_architecture(state)`：出现 `safety_query_head.*` / `safety_cross_attention.*`
    判为 `control_space`，否则判为 `legacy_curve`；
  - `load_model(cfg, ckpt, arch="auto"|"control"|"legacy")`；
  - `load_state_dict_flexible(...)`：容忍 `rel_bias_len` 变化（截断/补零），并报告
    `mapped / adapted / missing / unexpected`。
- `TrajSafePlanner.forward_curve_tokens(...)` **完整保留**重构前的 128 曲线 token 链路。
- 新增模块在加载旧权重时随机初始化，但 `SafetyControlFusion.out` 是**零初始化 + 残差**，
  初始等价于恒等映射，不会扰动已加载的网络。
- **已验证**：`best_task.pt`（epoch 449）在旧代码与新代码 `--arch legacy` 下，
  曲线 / 控制点 / 椭圆中心 / `shape4` / `π` 的 **max |diff| = 0.0**。

### 6.3 交互式可视化（Diffusion Lens）

- 后端 `diffusion-dashboard/backend_carla.py`（`backend.py` 转发到它）
  （HTTP：`GET /health`、`GET /splits`、`GET /sample`、`POST /generate`，端口 8765；
  payload format = 6）。
- 前端 Next.js（3000 端口），`app/page.tsx`。
- 功能：选**处理缓存** / 数据划分 / 样本 / checkpoint / seed；**画布上点选起终点、增删圆形障碍**；
  在线重建骨架图与候选路径；跑真实 DDIM；逐步回放 `P_t`、`x̂₀`、椭圆、控制点；图层可切
  （候选拓扑 / 选中 m / 冻结走廊 128 区域 + bridge / raw x̂₀ 洋红 vs safe x̂₀ 青色 /
  GT / occupancy）；右侧诊断面板显示 `π(m)`、候选数、CenterFree、ALM 违约 before→after、
  λ、内迭代、稠密验证结果。
- **处理缓存下拉**（`engine_carla.DATASETS`）：同一批 420 个 test 样本可在两套缓存间切换 ——
  `160k8p`（k=8，障碍各让 5 m，自由面积 0.357，train/val/test）与 `160k4p`（k=4，各让
  2.5 m，自由面积 0.279，**仅 test**；见 `docs/CAMPAIGN_160K8P_RESULTS.md` 第 9 节）。
  同一 `test_0167` 在两者上是两张不同的地图，所以切换缓存会作废当前样本与已生成序列；
  `dataset` 已加入 HTTP 缓存键（format 6），切换必然 `cache_hit=false`，不会回放另一张
  地图的旧结果。划分下拉按该缓存实际存在的目录生成，选 `160k4p` 时只提供 test。
- 一键启动：`start_dashboard.cmd`（清 8765 端口 → 起后端 → 必要时起前端 → 打开页面）。
- 缓存：`diffusion-dashboard/cache-carla/*.json`（派生数据，键含 dataset/样本/ckpt/seed/
  条件/障碍/ALM 开关/步数）。
- 注意：`diffusion-dashboard/README.md` 仍是 Maze2D 版本（见第 8 节遗留问题）。

### 6.4 测试

```
$ python -m pytest tests -q
113 passed in 23.08s
```

覆盖重点：

| 测试文件 | 关注点 |
|---|---|
| `test_bspline_constraints.py` | 精确 Bézier 抽取（<1e-5）、4 控制点凸包 ⇒ 连续安全、责任区间 |
| `test_safety_corridor.py` | overlap ratio 手算一致、bridge 用 Skeleton 进度中点而非欧氏中点、不可连通 gap 失败不递归 |
| `test_bspline_alm.py` | 已可行 ⇒ 精确不动点且 λ=0、violation 0.12→<1e-3、端点严格保持、对偶 warm-start |
| `test_alm_qp_reference.py` | 与 cvxpy/OSQP 参考解对照；矛盾 pack 时 QP infeasible 且 ALM 不假装成功 |
| `test_alm_state_machine.py` | WARMUP/TRY_ACTIVATE/GUIDED、走廊冻结、DDIM 不做二次 final projection |
| `test_control_space.py` | 控制点链路、损失 batch-size 不变性、配置一致性校验 |
| `test_model.py` / `test_dataset.py` / `test_geometry.py` / `test_thinning.py` / `test_convex_region.py` | 网络形状、数据集校验、几何与细化 |

---

## 7. 实验与结果

### 7.1 训练

| 项 | 值 |
|---|---|
| 训练脚本 | `scripts/night_train.py`（可设 deadline / 断点续训 / 自动重启，覆盖 config 的 epochs 与 batch_size） |
| epochs | 请求 700，实际 **469**（run1 到 272 按 deadline 停，从 `latest.pt` 续训 run2） |
| batch size | 8（由夜间脚本传入；**当前 config 写的是 16**，见局限 §8.6） |
| 优化器 | AdamW，`lr=2e-4`，`weight_decay=1e-4`，`grad_clip=1.0` |
| 参数量 | 7.31 M |
| 训练时长 | run2 12 658 s（≈ 3.5 h） |
| 选点 | `best.pt` = val total 最优（epoch 420，0.1901）；**`best_task.pt` = `val(curve_rmse_m + 40·collision_rate)` 最优（epoch 449，score 0.6934）** |
| 训练期 ALM | 关闭（`alm_enabled = False`，符合"推理期扩展"设计） |

选 `best_task.pt` 的原因：val total 被 topology CE 主导（该 head 早期过拟合），
不能代表规划质量，因此额外用任务指标选点。

最后一个 epoch 的验证指标（epoch 468）：

```
curve_rmse_m = 0.3813   pred_topo_best_rate = 0.885
collision_rate = 0.0217 ellipse_center_free_rate = 0.9997
ellipse_free_frac = 0.9892  ellipse_area_mean = 0.0623
```

### 7.2 test split 定量结果

`outputs/bspline_carla/eval_test.json`（`best_task.pt`，epoch 449，`--num-batches 2 --runs 2`，M=4）：

| 指标 | best_task | latest（epoch 469） | 说明 |
|---|---:|---:|---|
| `recall@M` | **1.000** | 1.000 | 候选池中总有一条与 GT 接近（τ=0.15），拓扑先验的下限保证 |
| `goal_dist_m` | **0.000** | 0.000 | 端点硬约束 |
| `progress_violations` | 0 | 0 | 沿路径进度单调 |
| `center_free` | 1.000 | 1.000 | 椭圆圆心从不落在障碍上 |
| `center_min_clearance` | 5.239 | 5.234 | 圆心最小余量（栅格） |
| `ellipse_collision` | 0.0347 | 0.0347 | 椭圆点碰撞率 |
| `curve_rmse_m` | 0.544 | 0.531 | 解码曲线 vs GT |
| `ctrl_rmse_m` | 0.559 | 0.542 | 控制点 vs GT |
| `smooth` | 0.0123 | 0.0108 | 越小越平滑 |
| `traj_collision` | 0.156 | 0.156 | ⚠️ 见下方口径说明 |
| `sel_best_rate` / `pred_topo_best_rate` | 0.813 / 0.813 | 0.781 / 0.781 | 选中候选即最优候选的比例 |
| `topo_entropy` | 0.0144 | 0.0373 | π 分布很尖（见局限 §8.5） |
| `step_jitter` / `step_switch_rate` | 0.0 / 0.0 | 0.0 / 0.0 | 16 步中拓扑选择不抖动 |
| `topo_diversity` / `traj_diversity` | 0 / 0.0026 | 0 / 0.0028 | 跨 seed 多样性 |

**⚠️ 结果口径说明（重要）**：上表 `traj_collision = 15.6 %` 由 13:40 的
`evaluate.py` 运行产生（`eval.log` 中该命令未带 `--ablation`），而 16:14 之后的消融批次
（`ckpt/eval_test_M4_ablA..D.json`）给出 raw = 6.25 % → guided = 0 %。
两组数字的**代码版本与运行配置不同**（ALM 接入 `evaluate.py` 是在前者之后），
因此**不能跨表比较**，也不能直接说"ALM 让碰撞从 15.6 % 降到 0"。
正式对外展示前应使用同一命令、同一 `num-batches / runs` 重跑 baseline 与 A/B/C/D。

### 7.3 消融实验（同一批 16 个 test 样本，1 run，`best_task.pt`）

| | A. Raw（无 ALM） | B. 仅最后一步修 | C. Warmup+冻结走廊+逐步 ALM | D. C 且关闭 bridge |
|---|---:|---:|---:|---:|
| 轨迹碰撞率 | 6.25 % | **25.0 %** | **0 %** | 0 % |
| 终点误差 (m) | 0 | 0 | 0 | 0 |
| 曲线 RMSE vs GT (m) | 0.69 | 1.71 | 1.52 | 1.52 |
| 平滑度 | 0.0143 | 0.0397 | 0.0357 | 0.0357 |
| selected nDTW | 0.0425 | 0.0445 | 0.0445 | 0.0445 |
| bridge 区域数 | — | 0 | 0 | 0 |
| ALM 曲线修正 (m) | — | 2.58 | 2.27 | 2.27 |
| 最终 max violation | — | −0.0035 | −0.0109 | −0.0109 |
| 走廊归属率 | — | 0.949 | 0.995 | 0.995 |
| 运行时 / 样本 (ms) | **77** | 560 | 530 | 552 |

**结论**：

1. **C 明显优于 A 与 B**：raw 有 6.25 % 碰撞；"只在最后修一次"反而升到 25 %——
   逐样本检查显示 B 的碰撞样本走廊归属率只有 0.81–0.84、violation 仍是 0.03 量级，
   即最终步预算内没收敛，且缺少 DDIM 反馈把中间状态逐步拉回。
2. **D 与 C 完全相同**：该批次 `overlap_min = 0.866 ≫ 0.10`，没有任何 gap 需要 bridge，
   所以 bridge 分支在真实数据上几乎不触发；其正确性由 Test C/D 单测与人为窄场景覆盖。
3. **安全的代价是偏离 GT**：曲线 RMSE 0.69 m → 1.52 m。ALM 把网络原始预测（离骨架
   3–7 m）拉回走廊，这既是安全性来源，也是"轨迹不再像 GT"的原因。
4. **推理开销可接受**：guided 约 0.53 s/样本（GPU），约为 raw 的 7 倍。

### 7.4 ALM 数值细节

| 指标 | 值 |
|---|---|
| activation 实际 reverse step | 3（即 `t=12→11` 那一步，按 forward 计数） |
| activation 成功率 / fallback 率 | 93.75 % / 6.25 % |
| 候选尝试次数均值 | 1.25 |
| base region / bridge region | 128 / 0 |
| overlap min / mean | 0.866 / 0.975 |
| region face 总数均值 | 933.8 |
| constraint piece 数 | 156（= 129 责任边界 ∪ 28 内部 knot 去重 − 1） |
| 有效不等式数 | 17 064（batch-of-16，单样本约 1 073–4 292） |
| `max_violation` before → after | 0.0445 → **0.00319**（scene；1 scene = 40 m） |
| `mean_positive_violation` before → after | 1.67e-3 → **1.49e-5** |
| `constraint_feasible_rate` | 0.936 |
| 实际内迭代 / λ max | 5.77 / 3.02 |
| 稠密验证（512 点） | `final_collision` 0/16、`final_corridor_membership_rate` 0.995、`endpoint_error` 0 |

### 7.5 定性结果（现有素材）

| 文件 | 内容 |
|---|---|
| `outputs/bspline_carla/sample_trace.png` / `trace_test_*_*.png` | 16 步反向扩散逐步回放（4×4 面板，橙=coarse、绿=final，标题带每步 `m` 与 `π(m)`） |
| `outputs/bspline_carla/sample_preview.png` / `samples_test_0_s0_{0,1,2}.png` | 单样本叠加图：occupancy + GT + 预测曲线 + 椭圆 + 候选拓扑 |
| `outputs/bspline_carla/samples_test_0_s0_ablC*.png` | 开启 ALM 的同一样本（洋红 raw vs 青色 safe + 走廊） |
| `outputs/bspline_carla/preview_test2/goal_change_compare.png` | 改终点后的对比 |
| `outputs/bspline_carla/inspect/inspect_test_59.png` | 单样本详细诊断 |
| `outputs/bspline_carla/overfit_preview.png` | 32 样本过拟合 sanity check |

---

## 8. 已知局限

1. **`max_curve_step_scene` 需要按 checkpoint 重标定。** 方案建议 0.01（0.4 m/步），
   实测 0.03 + inner 6 才能收敛（见下）。根因不是 ALM 实现，而是当前 checkpoint 的 raw
   预测与走廊所在骨架相距 3–7 m，0.4 m/步 × 3 次内迭代在一个 reverse step 内不够用。

   | 配置 | 最终 max violation（mean/max） | 走廊归属率 | 碰撞 |
   |---|---|---|---|
   | step 0.01 + inner 3（方案建议） | 0.041 / 0.130 | 0.862 | 1/4 |
   | step 0.02 + inner 4 | 0.019 / 0.087 | 0.904 | 2/4 |
   | **step 0.03 + inner 6（当前默认）** | **−0.011 / 0.004** | **0.998** | **0/4** |
   | step 0.05 + inner 6 | −0.014 / 0.000 | 1.000 | 0/4 |

2. **`u ↔ s` 对应仍是假设。** `progress_alignment_*` 诊断（用真实曲线 `P_i = C(i/127)`
   对比 `c_i = Γ(i/127)`）显示 RMSE ≈ 2.6–3.8 m、max ≈ 4.7–7.3 m，即"曲线参数 = 骨架弧长进度"
   在这批数据上只是近似。V1 明确只记录不改映射。
3. **局部凸区域不是凹障碍的硬证书。** 区域由障碍边界点的 halfspace 贪心交得到，对 U 形凹
   障碍仍可能跨过凹口。ALM 保证的是"落在冻结走廊内"，独立的碰撞判据仍是 dense validator。
4. **bridge 在真实数据上几乎不触发**（overlap 均值 0.975 / 最小 0.866 ≫ 阈值 0.10），
   消融 C 与 D 无差异。要验证需提高 `min_overlap_ratio` 或缩小 `obstacle_window_half`。
5. **拓扑头高度确定（`topo_entropy` 0.014，单样本 `π(m)=1.000`）。** 意味着模型几乎不做
   多模态拓扑选择，`recall@M=1.0` 更多是候选池的性质而非模型能力。对外展示时应同时给出
   `pred_topo_best_rate`（0.81）与 π 分布，避免只展示"看起来全对"的样本。
6. **配置与训练记录存在不一致**：`configs/config.yaml` 现为 `batch_size=16`，
   而 `training_summary.json` 记录训练时 `batch_size=8`（由 `scripts/night_train.py` 传入）；
   同理 `train.epochs=200` 与实际请求的 700 不同。复现历史结果时需明确以哪一份为准。
7. **严格碰撞判据下模型仍高于 GT 基线**：GT 轨迹自身碰撞率 1.8 %、逐点 0.018 %；
   raw DDIM 逐点 0.7 %–3 %。这正是引入 ALM 的动机。
8. **`diffusion-dashboard/README.md` 仍是 Maze2D 版本**，与实际 `backend.py`（CARLA 链路）
   不符；`diffusion-dashboard/engine.py`（旧 Maze2D 面板）调用了已删除的 API，不可运行。
9. **`src/README.md` 的模型表仍列着已删除的 `ProgressHead`**，与当前实现不符。

---

## 9. 复现指南

```bash
# 环境
E:/CondaEnvData/envs/GGMPC/python.exe --version      # Python 3.10 + torch 2.9.1+cu126

# 0) 测试
python -m pytest tests -q                            # 113 passed

# 1) 数据（已有 data/carla_processed 时可跳过）
python scripts/data/carla/00_clean_dataset.py        --config configs/config.yaml
python scripts/data/carla/01_build_candidates.py     --config configs/config.yaml
python scripts/data/carla/02_build_ellipse_labels.py --config configs/config.yaml
python scripts/data/carla/03_validate_processed.py   --config configs/config.yaml

# 2) 训练（夜间脚本带 deadline / 自动重启 / 续训）
python scripts/night_train.py --config configs/config.yaml --ckpt-dir outputs/bspline_carla/ckpt
#    或直接： python train.py --config configs/config.yaml

# 3) 采样出图 / 单样本诊断
python sample.py   --config configs/config.yaml --ckpt outputs/bspline_carla/ckpt/best_task.pt --split test --num 6
python scripts/preview_samples.py --config configs/config.yaml --ckpt outputs/bspline_carla/ckpt/best_task.pt
python scripts/inspect_sample.py  --config configs/config.yaml --ckpt outputs/bspline_carla/ckpt/best_task.pt --index 59

# 4) 评估 / 消融（--ablation A|B|C|D，--compare-raw 同时记录 raw 与 safe）
python evaluate.py --config configs/config.yaml \
  --ckpt outputs/bspline_carla/ckpt/best_task.pt --split test \
  --num-batches 28 --runs 4 --out outputs/bspline_carla/eval_test.json
python evaluate.py --config configs/config.yaml \
  --ckpt outputs/bspline_carla/ckpt/best_task.pt --split test \
  --num-batches 1 --runs 1 --ablation C --compare-raw

# 5) 交互面板
start_dashboard.cmd            # backend 8765 + UI 3000；/health 的 format 应为 4
```

> `--arch auto`（默认）会按 checkpoint 自动选择控制点链路或 legacy 128 曲线 token 链路；
> `--arch control` / `--arch legacy` 可强制指定。

---

## 10. 后续工作

1. **用新链路完整重训**：`L_smooth`（控制多边形）与 `L_boundary`（端点邻域）的量级需要
   在真实训练中重新标定；当前权重是首版经验值。
2. **让轨迹靠近骨架**：raw 预测离骨架 3–7 m 是 ALM 需要 0.03 步长的根因。
   可考虑在训练期加入骨架吸引力（避 opening the door to `L_align` 的回归）或更好的
   进度对齐（DP / 动态时间匹配）代替 `u ↔ s` 假设。
3. **给凹障碍加硬证书**：把局部凸区域换成完整的 free-space 分解（IRIS / 安全走廊图），
   或引入可见性检查。
4. **bridge 的真实覆盖**：构造窄走廊数据集或调 `min_overlap_ratio`，让 C/D 消融可分。
5. **闭环部署**：CARLA 在线闭环 + 车辆动力学约束 + 动态障碍（当前只做离线快照规划）。
6. **一致性收尾**：统一评估口径（baseline 与 A/B/C/D 同批同参）、更新
   `diffusion-dashboard/README.md` 与 `src/README.md`、补 `scripts/plot` 一键出图。

---

## 11. 附录：代码地图与术语

### 11.1 代码地图（入口/模型/几何/损失/测试合计约 1.04 万行）

| 路径 | 职责 |
|---|---|
| `train.py` / `sample.py` / `evaluate.py` | 训练 / 采样出图 / 指标评估（含 `--ablation`、`--compare-raw`、`--arch`） |
| `scripts/night_train.py` | 带 deadline、续训、自动重启的夜间训练驱动 |
| `scripts/data/carla/{00..03}_*.py` | CARLA 数据流水线四步 |
| `scripts/preview_samples.py` / `inspect_sample.py` / `rank_dashboard_samples.py` | 批量预览 / 单样本诊断 / dashboard 样本筛选 |
| `src/models/trajsafe/planner.py` | `TrajSafePlanner`：`forward_controls`（默认）/ `forward_curve_tokens`（legacy）/ `forward_all` |
| `src/models/trajsafe/blocks.py` | `TrajBlock` / `MatchBlock` / `TrajSelfAttention` / `CrossAttention`（AdaLN） |
| `src/models/trajsafe/encoders.py` | `TrajectoryEncoder`（控制点）/ `SkeletonEncoder` / `CoordMLP` |
| `src/models/trajsafe/boundary.py` | 固定 Boundary Decoder + `boundary_targets` |
| `src/models/trajsafe/ellipse.py` / `heads.py` / `fusion.py` / `geometry.py` | `Γ_m` 可微插值 / 三个 head / `FusionMLP`+`FinalDenoiser`+`SafetyControlFusion` / `CurveDecoder`+`dense_arclength` |
| `src/models/common/{blocks,scene_cnn}.py` | 共享 AdaLN / MHA；`SceneCNN`（`C_G` 16²、`C_E` 32²） |
| `src/diffusion/schedule.py` / `sampler.py` / `bspline_alm.py` | `NoiseSchedule`（T=16）/ DDIM + 三阶段状态机 / 控制空间 ALM |
| `src/diffusion/alm_guidance.py` | **LEGACY** waypoint 语义 ALM，保留但 `sample()` 已拒绝 |
| `src/geometry/skeleton_*.py` / `thinning.py` / `topology.py` | 骨架图、候选路径、`Γ_m`、Guo-Hall 细化 |
| `src/geometry/bspline.py` / `bspline_constraints.py` | B 样条编解码（`knots: auto`）/ 责任区间、精确 Bézier 抽取、约束包 |
| `src/geometry/convex_region.py` / `safety_corridor.py` | `EllipseRegionBuilder`（绝对圆心语义）/ `SafetyCorridor` + bridge |
| `src/geometry/ellipse_shape.py` / `ellipse_raster.py` | 稳定椭圆参数化 / 可微软栅格化 |
| `src/losses/losses.py` | 8 项损失（无 `L_traj` / `L_align`） |
| `src/datasets/carla_spline_dataset.py` | 当前训练数据集（校验 `C` / `Q` 与快照一致） |
| `src/utils/{config,checkpoint,seed}.py` | 配置读取 / 架构判定 + 柔性加载 / 随机种子 |
| `diffusion-dashboard/` | 交互面板（`backend_carla.py` + `engine_carla.py` + Next 前端） |

### 11.2 术语表

| 术语 | 含义 |
|---|---|
| **Skeleton / 骨架** | 自由空间细化得到的中轴图，网络的拓扑先验 |
| **候选（candidate）** | 骨架图上 start→goal 的一条搜索路径 `S_m` 及其 dense 版本 `Γ_m` |
| **控制多边形 / 控制点** | B 样条的 `C` 个控制点 `Q`，本项目唯一的轨迹表示 |
| **coarse / final** | 网络的两个 clean 预测：粗解码头输出与最终去噪输出 |
| **Boundary Decoder** | 固定、零参数的端点修正模块，保证 `Q_0=S`、`Q_{C-1}=G` |
| **走廊（corridor）** | 由 128 个凸区域 + 可选 bridge 区域构成的冻结安全域 |
| **bridge** | 相邻区域重叠不足时，在骨架进度中点插入的补隙区域 |
| **ALM** | 增广拉格朗日法；本项目在控制点上做一阶修正，`λ` 跨反向步 warm-start |
| **nDTW** | 归一化动态时间规整距离，用于确定最优候选 `m*` 与评估 |
| **DDIM / x₀ 参数化** | 确定性反向采样；网络直接预测 clean 控制点 |

---

*报告依据仓库当前工作区（分支 `v3-skeleton-dynamic`，HEAD `b94c631`）与
`outputs/bspline_carla/` 下的运行产物撰写；数据引用时已标注来源与口径差异。*
