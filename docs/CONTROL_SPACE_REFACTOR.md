# 控制点空间重构说明（Control-Space Refactor）

> 本文是**当前实现**的权威说明。`docs/TrajSafe-Diffuser_网络结构设计报告_实现细化版.md`
> 描述的是重构前的 "128 轨迹点 token" 版本；两者冲突时以本文为准。
>
> 触发这次重构的需求（`docs/扩散路径安全建议 (5).md`）：
> 1. 网络内部**不再出现 128 轨迹点**，轨迹主变量从头到尾只有 B 样条控制点；
> 2. 控制点个数**不写死**，由配置文件参数控制，方便修改；
> 3. 已经训练好的模型**修改代码后仍可使用**（要用于展示，重训成本高）。

---

## 1. 一句话总结

* 扩散状态仍然是控制多边形 `Q_t ∈ R^{B×C×2}`，但**网络直接在控制点上推理**：
  `C` 个控制点 token + `Q` 个骨架/椭圆安全 query，二者语义彻底分离。
* `C`（控制点数，默认 32）**只来自配置**，代码里不再有硬编码的 `32`；
  `Q`（骨架/椭圆 query 数，默认 128，`model.num_safety_queries`）**只**用于
  Skeleton / 椭圆安全场。
* 删除 `L_traj`，新增 `L_boundary`，损失固定为 8 项。
* 端点由**固定（不训练）的 Boundary Decoder** 保证，`L_boundary` 负责让网络自己
  也学会端点附近的控制多边形形状。
* 重构前的 checkpoint（例如 `outputs/bspline_carla/ckpt/best_task.pt`，epoch 449）
  **无需任何转换**即可继续使用：`--arch auto` 会自动识别并按旧链路原样回放。

---

## 2. 新前向链路

```
Q_t [B,C,2]                      扩散状态；端点硬条件（clamped knot ⇒ 曲线端点）
  │  hard_control_endpoints
  ├─ ControlEncoder        (planner.traj_encoder)          -> H_ctrl [B,C,D]
  ├─ ControlBackbone       (planner.traj_backbone, N_T)    -> H_ctrl [B,C,D]
  ├─ head_p                                               -> Q~_coarse [B,C,2]
  ├─ BoundaryDecoder（固定）                                -> Q_coarse  [B,C,2]
  │                                                            └─ 解码仅用于绘图/回退
  ├─ SkeletonEncoder(candidate_xy)                        -> H_S [B,M,L,D]
  ├─ MatchBlock(H_ctrl, H_S, h_t)                         -> R [B,M,C,D]
  ├─ TopologyHead(R) -> pi ; m = m*(训练) / argmax(pi)(推理)
  ├─ PathFeatureHead(R[m])                                -> H_path [B,C,D]
  │
  ├─ SafetyQueryHead(H_S[m])                              -> H_safety [B,Q,D]
  ├─ c_i = Γ_m(s_i),  s_i = i/(Q-1)   （固定 buffer，无 head）
  ├─ EllipseGeometry(H_safety, c, C_E, h_t, ab)           -> H_ell [B,Q,D]
  ├─ EllipseShapeHead(H_ell, h_t)                         -> shape4 [B,Q,4]
  │
  ├─ SafetyControlFusion: cross attention 控制 token ← 安全 token
  │                                                       -> A_safe [B,C,D]
  ├─ FusionMLP([H_ctrl, H_path, A_safe])                  -> F [B,C,D]
  ├─ FinalDenoiser(F, C_G, h_t)                           -> H_clean [B,C,D]
  ├─ head_p                                               -> Q~_final [B,C,2]
  └─ BoundaryDecoder（固定）                                -> Q_final  [B,C,2]
                                                             └─ B_128·Q_final -> 曲线（仅绘图/指标/控制器）
```

要点：

* 粗解码头（`Q~_coarse`）**直接输出控制点**，不再 `128 点 → LS 拟合 → 32 控制点`。
* 网络内部**没有任何 128 点轨迹张量**；`B_128` 解码只在最末端出现一次，
  用于画线、碰撞评估和控制执行。
* 椭圆分支运行在**被选中骨架的 Q 个 token** 上，因此 `shape4` 仍是 `[B,128,4]`，
  与离线标签（`ellipse_shape4_gt.npy`）逐位对应，这一支的语义没有变化。

### 2.1 `C` 与 `Q` 的语义分离

| 记号 | 含义 | 配置项 | 默认 |
|---|---|---|---:|
| `C` | 轨迹表示：B 样条控制点数（token 数） | `model.num_controls` = `bspline.num_controls` | 32 |
| `Q` | Skeleton / 椭圆安全 query 数 | `model.num_safety_queries`（= `topology.candidate_points`） | 128 |
| `H` | 解码曲线采样点数（仅末端） | `bspline.curve_points`（`model.horizon` 为其别名） | 128 |
| — | 自注意力相对位置偏置表长度 | `model.rel_bias_len`（默认 `max(C,Q,H)`） | 128 |

`model.num_controls` 与 `bspline.num_controls` 同时存在时**必须相等**，
否则 `TrajSafePlanner.__init__` 直接报错（避免"配置改了、数据没改"的静默错配）。

---

## 3. 固定 Boundary Decoder（`src/models/trajsafe/boundary.py`）

网络自由输出 `Q~`，随后施加**常量修正 profile**：

```
e_s = S - Q~_0                e_g = G - Q~_{C-1}
w^s = [1, 0.75, 0.5, 0.25, 0, ...]      （前 k = max(1, min(span, C//2)) 个控制点）
w^g = [..., 0, 0.25, 0.5, 0.75, 1]      （末端镜像）

Q*_i = Q~_i + w^s_i · e_s + w^g_i · e_g
```

于是对**任意**预测都严格有 `Q*_0 = S`、`Q*_{C-1} = G`；因为节点向量是 clamped 的，
这也等价于曲线端点条件 `C(0)=S`、`C(1)=G`。相邻控制点会随端点**协调移动**，
而不是被瞬间拉过去。

* profile 来自配置 `model.boundary_decoder.{span,profile}`，是**唯一来源**：
  损失函数从 `model.boundary_decoder` 读取，不在 YAML / 代码里重复定义。
* 该模块**零参数**，`profile` 是 `persistent=False` 的 buffer，因此
  `model.state_dict()` 里没有它的任何条目（不破坏旧 checkpoint 的严格加载）。
* `C` 很小时（例如 `C=2/4`）自动退化为纯端点硬条件，不会出现窗口重叠错误。

---

## 4. 损失函数（8 项）

```
L = λ_ctrl·L_ctrl + λ_coarse·L_coarse + λ_smooth·L_smooth + λ_bound·L_boundary
  + λ_topo·L_topo + λ_shape·L_shape + λ_iou·L_iou + λ_safe·L_safe
```

| 项 | 权重 | 定义 |
|---|---:|---|
| `L_ctrl` | 0.2 | `MSE(Q~_final[1:-1], Q_GT[1:-1])`，用 **Boundary Decoder 之前**的 raw 控制点 |
| `L_coarse` | 0.5 | `MSE(Q~_coarse[1:-1], Q_GT[1:-1])`，coarse head 直接输出控制点 |
| `L_smooth` | 0.08 | 控制多边形的二阶/三阶差分（`Δ²Q`、`Δ³Q`），按 GT 控制点平均步长 detach 归一化，`α=0.25, β=1.0`；**不解码任何轨迹点** |
| `L_boundary` | 0.2 | `L_b(Q~_final) + 0.5·L_b(Q~_coarse)` |
| `L_topo` | 0.25 | `CE(π, m*)`（不变） |
| `L_shape` | 0.08 | 椭圆 shape4 监督（不变） |
| `L_iou` | 0.25 | 椭圆软 mask IoU（不变） |
| `L_safe` | 0.15 | 椭圆 unsafe fraction + CVaR（不变） |

`L_boundary` 的局部目标把 GT 控制多边形**刚体平移到本样本的起终点**：

```
T^s_i = S + (Q^GT_i - Q^GT_0)
T^g_j = G + (Q^GT_j - Q^GT_{C-1})

L_b(Q~) = [ Σ_i w^s_i ‖Q~_i - T^s_i‖² + Σ_i w^g_i ‖Q~_{C-1-i} - T^g_{C-1-i}‖² ]
          / (2 Σ_i w_i)
```

监督的是**局部控制多边形形状**（而不是把 `Q_1..Q_3` 都拽到 `S` 上），
这样网络自己就能生成合理的端点邻域，不必依赖 decoder。

**明确删除**：`L_traj`、`L_align`，以及 `trajectory_x0_loss` /
`trajectory_smoothness_loss` 两个函数（`src/losses/losses.py` 中已不存在）。

---

## 5. 配置项

```yaml
model:
  num_controls: 32            # C：控制点个数（与 bspline.num_controls 必须一致）
  control_space: true         # true = 控制点 token 链路（默认）；false = 旧 128 曲线 token 链路
  num_safety_queries: 128     # Q：Skeleton / 椭圆安全 query 数
  rel_bias_len: 128           # 自注意力相对位置偏置表长度
  boundary_decoder:
    span: 4
    profile: [1.0, 0.75, 0.5, 0.25]

bspline:
  degree: 3
  num_controls: 32            # 与 model.num_controls 保持一致
  curve_points: 128
  knots: "...bspline_knots.npy"   # 也可写 "auto"：按 (C, degree) 生成 clamped 均匀节点向量

loss:
  lambda_control: 0.2
  lambda_coarse: 0.5
  lambda_smooth: 0.08
  lambda_boundary: 0.2
  boundary_coarse_weight: 0.5
  smooth_acc_weight: 0.25
  smooth_jerk_weight: 1.0
  lambda_topology: 0.25
  lambda_shape: 0.08
  lambda_iou: 0.25
  lambda_safe: 0.15
```

### 5.1 修改控制点个数 `C` 的完整步骤

1. 改 `configs/config.yaml`：`model.num_controls` 与 `bspline.num_controls` 同时改成新值；
   如沿用旧的 `bspline_knots.npy`，代码检测到长度不匹配会**自动改成 clamped 均匀节点向量**
   （并打印 warning）；也可直接写 `knots: "auto"`。
2. 重新生成离线控制点标签（`control_gt` 是**离线数据**，必须与 `C` 对齐）：

   ```bash
   python scripts/data/carla/00_clean_dataset.py --config configs/config.yaml
   python scripts/data/carla/01_build_candidates.py --config configs/config.yaml
   python scripts/data/carla/02_build_ellipse_labels.py --config configs/config.yaml
   python scripts/data/carla/03_validate_processed.py --config configs/config.yaml
   ```

   `00_clean_dataset.py` 现在从配置读 `bspline.num_controls` / `bspline.curve_points`
   （也可用 `--num-controls` / `--curve-points` 覆盖），不再是常量。
3. 训练 / 采样照旧。若 `C` 与数据快照不一致，`CarlaSplineDataset` 会**直接报错并提示重跑数据**，
   不会静默用错标签。

> `Q` 也可以改（`model.num_safety_queries`），但它必须等于 `topology.candidate_points`
> 和 `ellipse_shape4_gt` 的行数，所以同样要重跑数据流水线。

---

## 6. 旧 checkpoint 继续可用（重点）

### 6.1 机制

* `src/utils/checkpoint.py`
  * `detect_architecture(state)`：checkpoint 里若出现 `safety_query_head.*` /
    `safety_cross_attention.*`，判为 `control_space`，否则判为 `legacy_curve`。
  * `load_model(cfg, ckpt, arch="auto"|"control"|"legacy", device=...)`：
    `auto` 按上面的判定选择 `model.control_space`，再构建模型并加载权重。
  * `load_state_dict_flexible(...)`：允许少量张量形状变化（例如 `rel_bias_len` 不同时
    对 `b_horizon` 做截断/补零），并报告 `mapped / adapted / missing / unexpected`。
* `TrajSafePlanner.forward_curve_tokens(...)` 完整保留重构前的 128 曲线 token 链路
  （`decode_controls → TrajectoryEncoder → … → 128 点曲线 → LS 拟合控制点`），
  旧 checkpoint 走这条链路时**数值与重构前逐位一致**。
* 新增模块（`safety_query_head`、`safety_cross_attention`）在加载旧权重时保持随机初始化；
  其中 `SafetyControlFusion.out` 是**零初始化 + 残差**，初始等价于恒等映射，
  不会扰动已加载的网络。
* `sample.py` / `evaluate.py` / `diffusion-dashboard/engine*.py` /
  `scripts/{preview_samples,inspect_sample}.py` 全部默认 `--arch auto`。

### 6.2 已验证

对 `outputs/bspline_carla/ckpt/best_task.pt`（epoch 449）：

* 旧代码（refactor 前的 `HEAD`）与新代码（`--arch legacy`）在相同输入下的
  曲线 / 控制点 / 椭圆中心 / `shape4` / `pi` **最大差 = 0.0**；
* `python sample.py --ckpt outputs/bspline_carla/ckpt/best_task.pt --split test
  --num 2 --steps 8` 仍然给出 `alm=guided`、`final_collision=False`、
  `endpoint_error=0.0` 的结果，与重构前展示效果一致。

### 6.3 用旧权重跑新链路

`--arch control` 可以强制用新链路加载旧权重（结构可跑，但分布不匹配，
椭圆/曲线质量不保证）。若要真正把旧模型迁到控制点空间，建议
`python train.py --config configs/config.yaml --resume <old ckpt>`：
旧权重被柔性加载，新增模块随机初始化后继续训练。

---

## 7. 代码地图

| 文件 | 职责 |
|---|---|
| `src/models/trajsafe/planner.py` | 两条链路：`forward_controls`（默认）/ `forward_curve_tokens`（legacy），`forward_all` 分发 |
| `src/models/trajsafe/boundary.py` | 固定 Boundary Decoder + `boundary_targets` |
| `src/models/trajsafe/fusion.py` | `SafetyControlFusion`（安全 → 控制 cross attention） |
| `src/models/trajsafe/blocks.py` | `TrajSelfAttention` 的 `rel_bias_len` 与动态相对距离索引（不再有 `_kd` buffer） |
| `src/models/trajsafe/encoders.py` | 控制点编码器（索引 PE 动态生成，无 token 数相关 buffer） |
| `src/losses/losses.py` | 8 项损失；`L_smooth` 作用于控制点，`L_boundary` 作用于 raw 控制点 |
| `src/utils/config.py` | `num_controls` / `curve_points` / `num_safety_queries` 三个取值入口 |
| `src/utils/checkpoint.py` | 架构判定 + 柔性加载 |
| `src/geometry/bspline.py` | `default_knots` / `resolve_knots`（`knots: auto`） |
| `src/datasets/carla_spline_dataset.py` | `num_controls` / `num_safety_queries` 与快照一致性校验 |
| `scripts/data/carla/00_clean_dataset.py` | 离线控制点标签，`C` 来自配置 |
| `scripts/data/carla/03_validate_processed.py` | 用配置推导期望形状做校验 |

---

## 8. 后续可做的事（未在本次改动内）

* 用新链路**完整重训**：现在 `L_smooth` 约束的是控制多边形，`L_boundary` 约束端点邻域，
  两者的量级需要在真实训练中重新标定（首版权重见第 4 节）。
* `SafetyControlFusion` 的零初始化在长训练后是否仍然合适（可考虑按层 `LayerScale`）。
* 若把 `Q` 改成不等于 `H`（解码点数），需要同时更新 `topology.candidate_points`
  与椭圆标签流水线；当前默认 `Q = H = 128` 保持不变。
