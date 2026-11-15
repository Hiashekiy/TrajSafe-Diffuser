# 端点朝向约束设计（TrajSafe-Diffuser 边界条件）

**状态：** 设计定稿，**代码未改动**（确认后再实施）  
**范围：** 只改扩散模型/采样的**端点边界条件**，不改网络主干、不改损失以外的训练流程、不改
`src/grid_search/`（栅格搜索只吃解码折线，天然兼容）。  
**相关文档：** [`PDG_Focal_Astar_S0_接口核查.md`](PDG_Focal_Astar_S0_接口核查.md)
（项目接口/坐标系）、[`PDG_Focal_Astar_实验报告.md`](PDG_Focal_Astar_实验报告.md)（栅格搜索实验）。

---

## 1. 要做什么（一句话）

把现在的"**只钉起点/终点位置**"扩展成"**钉起点位置 + 起点朝向，钉终点位置 + 终点朝向（可选）**"。
**不引入速度**（原因见 §3）。

```
现在：  C(0) = S ,  C(1) = G                      仅位置
目标：  C(0) = S ,  C'(0) ∥ t̂₀                    位置 + 起点朝向
        C(1) = G ,  C'(1) ∥ t̂₁（可选）             位置 + 终点朝向
```

---

## 2. 现状（已核查）

| 层 | 位置 | 行为 |
|---|---|---|
| 硬赋值 | `src/geometry/bspline.py:354` `hard_control_endpoints` | `q[:,0]=S`、`q[:,-1]=G` |
| 固定 Boundary Decoder | `src/models/trajsafe/boundary.py` | `Q*_i = Q~_i + w^s_i(S−Q~_0) + w^g_i(G−Q~_{C−1})`，`w^s=[1,.75,.5,.25,0,…]`、`w^g` 镜像；零参数、不训练 |
| 调用点 | `planner.py:456/478/529`、`sampler.py:775/811`、`sdf_guidance.py:221/416/473` | 进网络前、出网络后、每次 DDIM 步后各一次 |
| 曲线端点 | clamped knot（`bspline.knots: "auto"`） | 曲线首末点 = `Q_0` / `Q_{C−1}` ⇒ 控制点端点条件 ⇔ 曲线端点条件；实测 `endpoint_err = 0.0` |
| 模型输入 | `cond [B,2,2]` = (start, goal) | **只有位置** |
| 数据 | `processed/.../conditions.npy` | 无朝向/速度/时间字段 |

关键结论：端点位置已经"由构造精确"，本次要加的是**方向**。

---

## 3. 为什么不加"速度"

clamped 三次 B 样条的端导数只由端点附近两个控制点决定：

```
C'(0) ∝ (Q₁ − Q₀)          C'(1) ∝ (Q_{C−1} − Q_{C−2})
```

* 给"起点速度" ⇔ 把 `Q₁` 沿 `t̂₀` 放到 `S` 之外（长度 ∝ 速度 / 参数化尺度）。
* 给"终点速度 = 0" ⇔ `Q_{C−2} = Q_{C−1} = G`，即**最后一段曲线塌成一个点**，切向量为 0。

后者会让几何路径末端退化（下游控制器看到"最后一段几乎不动"），而且"停下来"本质是
**时间维度**的约束，不是几何约束。因此：

> 几何路径只负责形状（位置 + 朝向），**速度剖面交给执行层**（控制器 / MPC / 梯形速度律）。

这也是为什么不把 `cond` 扩成 `[S, v0, G, v1]`：那会把时间语义硬塞进几何模型。

---

## 4. 数学定义（朝向约束怎么写）

记 `C` = `model.num_controls`（默认 32/48），`ℓ₀`、`ℓ₁` = 起/终点切线段长度（米，或 scene 单位）。

```
起点：Q₀ = S,            Q₁ = S + ℓ₀ · t̂₀
终点：Q_{C−1} = G,       Q_{C−2} = G − ℓ₁ · t̂₁        （t̂₁ = 期望到达方向）
```

其中 `t̂₀`、`t̂₁` 是单位向量（scene 坐标）。`ℓ` 的取法建议**按路径长度自适应并夹紧**：

```
ℓ = clip(0.05 · |G − S|,  2 · min_control_step,  0.15 · |G − S|)
```

理由：`ℓ` 太小 → 朝向约束几乎无效；`ℓ` 太大 → 起点附近形状被锁死（曲线"拐不过来"），
碰撞率会上升。

**与现有 profile 修正的关系（重要）**：现有 `w^s=[1,.75,.5,.25]` 的作用是把 `e_s`
**平移式**分摊到前 4 个控制点，目的是"端点精确但局部形状连续"。朝向约束应复用同一
套 profile 施加在**切向偏差**上，而不是再写一套硬赋值：

```
d₀ = (S + ℓ₀·t̂₀) − Q₁          d₁ = (G − ℓ₁·t̂₁) − Q_{C−2}
Q_i += v^s_i · d₀ + v^g_i · d₁        v^s = [.,1,.75,.5,.25,…]（从第 2 个控制点起）
```

这样 `Q₀`/`Q_{C−1}` 的位置条件保持不变，`Q₁`/`Q_{C−2}` 被推向目标切线位置，`Q₂…`
平滑跟随。

---

## 5. 数据从哪来

| 来源 | 可用性 | 用途 |
|---|---|---|
| `data/raw/80m/samples/*.npz`：`trajectory_raw_local`（≈215 点）+ `raw_start_idx/raw_end_idx` | ✅ 已有 | 用相邻点差分得切线方向 |
| 采集端时间步 | ✅ `fixed_delta_seconds = 0.05`（20 Hz） | 需要**速度**时才用；本次不需要 |
| `data/raw/160m-256px/tasks.jsonl` | 只有 `path_length_m`、曲率等，无朝向字段 | 需回采集缓存确认是否保留 raw 轨迹 |
| `processed/.../conditions.npy` | 仅 `[N,2,2]` 位置 | 需新增 `heading.npy`（见下） |

新增标签（重跑一次标签流水线即可，无需重采）：

```
heading.npy   [N, 2, 2]   heading[:,0] = 起点单位切向 t̂₀
                          heading[:,1] = 终点单位切向 t̂₁
heading_valid [N, 2]      bool（端点太短/退化时置 False）
```

推导：对 `trajectory_raw_local[raw_start_idx : raw_end_idx]` 取首/末 k 个点做加权差分
（或对首末 2–3 个点做最小二乘直线拟合），再归一化。`heading_valid=False` 的样本
走"无朝向"的老分支。

---

## 6. 接口设计（不实现，只约定）

**原则：`cond [B,2,2]` 布局一律不动**（`cond[:,0]/cond[:,1]` 在 planner / loss / ALM /
SDF / eval / dashboard / carla_bridge 有 20 余处引用）。朝向用**独立张量**传入，
默认 `None` ⇒ 行为与今天逐位一致（旧 ckpt 可用）。

| 层 | 变更 |
|---|---|
| `CarlaSplineDataset` | 多返回 `heading [2,2]`、`heading_valid [2]`（可选，缺文件时返回 None/False） |
| `BoundaryDecoder.forward(q_raw, cond, heading=None, heading_valid=None, ell=None)` | 现有位置修正之后，追加 §4 的切向修正；`heading is None` 时完全不进入 |
| `planner.hard_control_endpoints(q, cond, heading=None, …)` | 同上（用于每次 DDIM 步后的状态重钉） |
| `planner.forward_all(..., heading=None, heading_valid=None)` | 透传给上面的两处；`out` 字段不变 |
| 训练 | 新增 `L_tangent`：端点切线方向误差（cos 距离）+ 切线段长度误差；建议初值权重 0.1，需实测 |
| 采样/评估 | `sample.py` / `evaluate.py` / dashboard / carla_bridge 把新张量往下传（默认 None） |

**一个必须确认的设计选择：`Q₁` 该不该在每个 DDIM 步后都硬钉？**

* 只在**网络输出端**（BoundaryDecoder）钉：网络保留完整自由度，末端形状由输出端统一保证；
  推理与训练一致，实现最省事。 ← **建议**
* 每个 DDIM 步后都钉（输入端也钉）：状态始终满足朝向，但会压缩网络的搜索空间，
  中间步的噪声状态被强制对齐，可能拖慢收敛/降低多样性。

建议先做"仅输出端"，训练后看端点朝向误差；若仍偏大，再把状态重钉打开（做成配置开关）。

---

## 7. 标定与验收

| 指标 | 目标 | 备注 |
|---|---|---|
| 起点朝向误差 | < 5° | `acos(t̂₀ · C'(0)/|C'(0)|)` |
| 终点朝向误差（若启用） | < 5° | 同上 |
| 端点曲率 | 不显著高于 GT 95 分位 | 防止 `ℓ` 过大造成折角 |
| 末段位移 | > 0.05 m | 防止末端退化 |
| 路径长度 / `traj_collision` / `center_free` | 不退化 | 复用 `evaluate.py` 现有指标 |
| 回归 | `heading=None` 时与现 checkpoint 逐位一致 | 与现有 `--arch auto` 兼容性同级别要求 |

---

## 8. 风险与回退

1. **`ℓ` 过大会锁死起点形状** → 按 `§4` 自适应夹紧，并在训练日志里记录 `ℓ`。
2. **终点朝向不该无条件强加**（有些任务允许任意到达方向）→ 配置项 `heading.goal: on/off`。
3. **数据无朝向的样本** → `heading_valid=False` 自动走老分支，保证混合 batch 不炸。
4. **必须重训**：旧 `best_task.pt` 无法凭空获得朝向能力；但**旧 ckpt 在 `heading=None`
   下仍可直接调用**（向后兼容）。
5. **回退开关**：`model.boundary.heading.enabled: false` 一键回到今天的行为。

---

## 9. 实施顺序（确认后执行）

| 步骤 | 内容 | 产出 |
|---|---|---|
| S1 | 生成 `heading.npy` / `heading_valid.npy`；`CarlaSplineDataset` 返回新字段 | 数据就绪，旧代码不受影响 |
| S2 | `BoundaryDecoder` + `hard_control_endpoints` 增加可选朝向参数（默认 None） | 单测：None 分支逐位一致；朝向分支端点朝向误差 < 5° |
| S3 | `L_tangent` + 训练脚本透传；短程训练验证 | 端点朝向误差、碰撞指标 |
| S4 | `sample.py` / `evaluate.py` / dashboard / carla_bridge 透传；出图核对 | 可视化 + 指标回归 |

工作量估计：S1 半天、S2 一天、S3 一天（不含训练时间）、S4 半天。

---

## 10. 明确不做

* 不做速度 / 时间参数化（见 §3）；速度剖面留在控制器/MPC。
* 不改 `cond` 的 `[B,2,2]` 布局。
* 不改 `src/grid_search/`（PDG-Focal A* 只消费解码折线；端点朝向变好会让参考更平滑，
  无需任何改动）。
* 不改 ALM / 安全走廊 / 凸区域 / 椭圆分支。
