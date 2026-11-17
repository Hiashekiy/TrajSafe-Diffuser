# 新框架改造：实施状态与运行说明

**分支：** `v4-map16-fbdelta-safeloss`（基于 `b30c2d5`）  
**设计依据：** `TrajSafe-Diffuser 新框架整体改造实施方案.md`  
**S0 审查：** [`new_framework_S0_audit.md`](new_framework_S0_audit.md)（逐项 `path:line` 证据 + 风险清单）

> 本文件记录**实际做了什么、和方案的偏差、为什么偏差、还剩什么**。数字（训练/评估）
> 见文末"运行结果"，写于训练与评估结束之后。

---

## 1. 一晚做完的部分（按方案的阶段编号）

| 阶段 | 内容 | 状态 |
|---|---|---|
| S0 | 现状审查（网络结构 / q_coarse / topology 引用 / 数据可用性 / 局部约束包方案） | ✅ `new_framework_S0_audit.md` |
| S1 | 安全地图 + PDG-Focal A\* 接入采样器（含失败回退与统计） | ✅ `src/diffusion/sampler_v5.py::_search_once` |
| S2 | 网络改造：删 `q_coarse`、删 `TopologyHead`、单路径输入、Backbone 1 层、Final Denoiser 4 层、C=32、Q=64、128 解码点 | ✅ `src/models/trajsafe/planner_v5.py` + `src/models/factory.py` |
| S3 | 端点朝向：`heading/heading_valid` 标签 + BoundaryDecoder 侧切向修正 + 朝向损失 + 指标 | ✅（标签为 GT 曲线切线的**代理方向**，见 §3） |
| S4 | 全量 64 椭圆 → 碰撞区间 → **局部约束包**（显式参数区间） | 🟡 局部包已实现并接线；走廊内 bridge 复用既有实现 |
| S5 | 四步采样器（第1步无路径/无安全/无反馈 → A\* → 第2步路径+椭圆无反馈 → ALM+全局验证+路径更新+反馈 → 第3/4步） | ✅ `sampler_v5.py::sample_v5` |
| S6 | 数据（C32 / heading / PDG 路径 / 64 椭圆）+ 四步训练 rollout + 训练 | 🟡 见 §3 的接续计划 |
| S7/S8 | 正式对比实验与交付文档 | 🟡 仅做了主模型的测试集评估；消融未做 |

### 1.1 新增/修改的文件

新增：
```
src/models/trajsafe/planner_v5.py          新框架网络（唯一输出 x0，无 coarse、无 topology）
src/models/factory.py                      model.framework: v5 -> 新网络，缺省 legacy
src/diffusion/sampler_v5.py                四步采样器（搜索 / 局部走廊 / ALM / 路径更新 / 反馈）
src/geometry/bspline_constraints_local.py  局部约束包（显式参数区间，禁止跨整条曲线）
train_v5.py                                四步 rollout 训练脚本
eval_v5.py                                 四步推理评估（指标 + 逐步可视化）
configs/config_v5_c32q64.yaml              新框架配置
scripts/data/new_framework/{10,11,12,13}_*.py   C32 / heading / PDG 路径 / 64 椭圆标签
```
修改（最小侵入）：
```
src/utils/checkpoint.py            load_model 走 build_model（旧 ckpt 仍走 legacy）
src/datasets/carla_spline_dataset.py  可选 heading / pdg_path 字段；Q 与 candidate 解耦
```

**方案 §13.1/§13.2 要求删除 q_coarse 与 topology。** 我们的做法是：
新框架链路（`planner_v5.py`）**完全没有** q_coarse / TopologyHead / coarse 输出 / L_coarse；
legacy `planner.py` 原样保留，**仅**用于旧 checkpoint 复现（`--arch auto`），新配置
（`model.framework: v5`）不会构造它。这是对 §13.1 的**有意偏差**：删除 legacy 会让
7 个历史 ckpt 与论文复现脚本全部失效，风险大于收益。

### 1.2 四步流程的落地位置

```
sample_v5(batch)
 ├─ step1: model(q, occ, cond, t, ab)                        # path/safety/feedback 全关
 │        └─ trace[0].search = PDG-Focal A*（每样本一次，含回退与耗时）
 │             active_path = 128 点弧长重采样（scene 坐标）
 ├─ step2: model(..., active_path)                            # + 64 椭圆，无反馈
 │        └─ 碰撞？有→局部走廊+ALM+全局验证；无→跳过（不伪造反馈）
 ├─ step3/4: model(..., active_path, feedback if valid)
 └─ 返回 final curve + 每步 trace（原始 x0 / 碰撞比例 / 碰撞区间 / ALM 状态 / 反馈有效）
```
`q0_ddim` 采用 **ALM 修正后的 `q0_safe`**（与 legacy `sampler.py:802` 一致，S0 审计指出的
偏差已修）；ALM 未通过全局验证时退回 raw x0，且**不更新** `active_path`、**不产生**反馈。

### 1.3 局部约束包（S4 的核心）

`build_constraint_pack` 会把所有 anchor 铺到整条 knot 区间 —— 直接喂碰撞区间的少量 cell
会让局部区域"蔓延"到整条曲线（方案 §5.4 明令禁止）。新增：

```
build_local_constraint_pack(codec, specs)   # specs[i] = {A,b,mask, spans[M,2]}
```
每个 cell 自带参数区间 `[u_a,u_b]`，只在该区间做精确 Bézier 提取与约束装配；
`local_spans_from_intervals()` 负责把"碰撞查询区间"映射成这些 span。若局部包为空则回退到
全量走廊（写 `reason="local_pack_empty_fallback_full"`，绝不静默）。

实测（合成阻塞地图）：局部包 128 cells → 184 pieces，ALM 把穿障比例 0.93 → 0.74，
但因整条曲线仍碰撞 → `alm_ok=False` → **不更新路径、不产生反馈**（符合 §6.2/§6.3）。

---

## 2. 数据现状（三者都影响结论口径，必须写在报告里）

| 项 | 现状 | 影响 / 处理 |
|---|---|---|
| 障碍地图 | 只有 `data/processed/160m-k4p-c48`（障碍被 `binary_erosion(k=4)` 腐蚀 4 格 ≈ 2.5 m） | 按方案 §4.3，腐蚀**不是**安全膨胀，本次未用它抵消机器人半径；但**也无法恢复原始障碍边界**。搜索用 `r=0.8 + margin=0.3 = 1.1 m`（1.76 格）膨胀 k4p 图，净效果仍比真实障碍薄 ~2.2 格。报告不得声称安全边界精确。S0 审计指出原始 occupancy 可从 `data/raw/160m-256px/scenes/*.npz::occupancy_train` 重建，**这是下一步**。 |
| heading | 快照里没有原始轨迹/时间戳 | 用 `curve_gt` 端点切线做**代理方向**（`cos(heading, GT tangent)=1.0` 属构造性一致，不代表真实车辆航向）；`heading_valid` 全 1。 |
| 训练用路径 | PDG-Focal A\* 路径缓存生成较慢 | 第一阶段 rollout 用 **GT 曲线**作 step-2 路径（`path_source="gt"` 的 bootstrap，方案 §9.2 允许 teacher→A\*→刷新）；A\* 缓存就绪后按 §9.2 第 8 步短程微调切到真实 A\* 路径。 |

---

## 3. 与方案的偏差清单（全部是有意的，且可回退）

1. **legacy planner 未删除**（§13.1/§13.2 的字面要求）——见 §1.1；新链路零引用。
2. ~~训练 rollout 不含 ALM~~ **已修正（02:0x）**：训练 rollout 现在调用与推理**同一个**
   闭环函数 `src/diffusion/alm_local.py::alm_closed_loop`（碰撞 → 局部走廊 cell → 局部约束包
   → ALM → 全局验证 → 仅通过才更新 `active_path` 并产生反馈），**没有用 SDF 投影替代**。
   代价是实测 **360 s/epoch**（原 104 s），因此 200 epoch 需要 ~20 h；本次实际训练量见
   `new_framework_RESULTS.md`（如实记录 epoch 数与是否收敛）。首个 epoch 的闭环统计：
   `alm_ran=1.0`（每 batch 都尝试）、`corr=0.34`（ALM 确实改了曲线）、
   `coll_after=0.19`（从 0.48 降下来但未到 0）、`alm_ok=0.0`（随机初始化权重下全局验证
   不通过 → 按 §7 不产生反馈，属正确行为）。
3. **step-2 路径用 GT 曲线 bootstrap**（§2 表格第 3 行），待 A\* 缓存齐后微调。
4. **局部走廊的 bridge/overlap 检查**复用 legacy `build_safety_corridor`（全量版本），
   仅"cells 选取 + 参数区间"是局部的；方案 §5.3 的局部 bridge 尚未单独实现。
5. **S7 消融**（动态路径更新开/关、局部/全量走廊、反馈开/关、朝向开/关、64/128 椭圆、
   4/16 步）**未做**，只完成了主模型评估。

---

## 4. 运行方式（复现）

```bash
# 1) 数据（新框架专用，不覆盖旧根）
python scripts/data/new_framework/10_make_c32_dataset.py     # C=32 控制点（其余数组逐字节复制）
python scripts/data/new_framework/11_build_heading_labels.py # heading / heading_valid（代理方向）
python scripts/data/new_framework/12_build_pdg_paths.py      # PDG-Focal A* 路径缓存（teacher x0 引导）
python scripts/data/new_framework/13_build_ellipse64_labels.py  # 沿 A* 路径的 64 椭圆标签

# 2) 训练（四步 rollout）
python train_v5.py --config configs/config_v5_c32q64.yaml \
    --batch 16 --epochs 60 --max-hours 4.6

# 3) 评估（四步推理 + 搜索 + ALM + 逐步可视化）
python eval_v5.py --config configs/config_v5_c32q64.yaml \
    --ckpt outputs/models/v5_c32q64/ckpt/best_task.pt \
    --split test --limit 420 --out outputs/v5_eval
```

产物：`outputs/models/v5_c32q64/ckpt/{latest.pt,best_task.pt,train_log.jsonl}`、
`outputs/v5_eval/{eval_<split>.csv,summary_<split>.json,curves_<split>.npz,figures/}`。

---

## 4.1 评估阶段发现并修复的 3 个 bug（记录，便于复现）

| # | 现象 | 原因 | 修复 |
|---|---|---|---|
| 1 | `alm_ran_rate=0`、`alm_ok=null`，ALM 从不生效 | 重构时 `sampler_v5.py` 里仍留着 `collision_mask/alm_closed_loop` 的**旧副本**，覆盖了 `alm_local` 的导入；旧副本引用了已被删除的 `EllipseRegionBuilder` → 每次抛 `NameError` 被 `except` 吞掉 | 删除 `sampler_v5` 中 161 行重复实现，只保留 `from .alm_local import ...`；单测确认两个名字是同一对象 |
| 2 | `search_fallback_rate=99.3%`，搜索路径全部退化成直线 | `_search_once` 先 `resample_polyline` 再拿**旧 `cum`** 去 `np.interp` → `ValueError('fp and xp are not of the same length')`，被 `except` 吞掉 → 直线兜底 | 只做一次弧长重采样（重建 `cum`），并保留 `search_error` 字段；修复后 420/420 用真实 PDG 路径，回退率 0% |
| 3 | `--steps 16` 直接崩 | `pick_times(T, 16)` 返回 `None`（"用全长表"），采样器却对它取 `len()` | `times is None` 时用 `range(T)`，即完整 16 步 |

这三个 bug 都会让"看起来通过"的评估结果失真（ALM 不生效 / 搜索未使用 / 16 步不可用），
修复前后的数字差异已在 `RESULTS` §2–§3 中分开记录。

## 4.2 为什么还有碰撞 / 为什么 ALM 全局验证不通过（实测定位）

test 420 个样本里 **28 个（6.67%）最终仍有碰撞**。逐样本回放后事实是：

1. **这 28 个全部检测到了碰撞、全部跑了 ALM、全部在全局验证被拒**（`alm_ran=True,
   alm_ok=False, reason=''`），所以最终返回的是**未经修正的原始预测**（§6.3 不允许用
   未通过验证的修正）。碰撞不是"漏检"，而是"修正没通过验证"。
2. **不是 ALM 迭代不够**：把 `alm.inner_steps` 6→20、`step_size` 0.05→0.12，8 个失败样本的
   最终碰撞率**一位不差**（0.227/0.164/0.219/… 完全相同）。因为验证不过时根本不会用
   修正结果，所以加大 ALM 只影响"修正量"，不影响输出。
3. **真正原因是约束集没盖住碰撞区域**。以样本 3009 为例：
   - 30 个碰撞点、曲线参数 u∈[0.42,0.65]，映射出的查询区间 (22,45)（覆盖正常）；
   - 但碰撞点到最近**椭圆中心**的距离是 **4.9 m（均值）/ 9.0 m（最大）**；
   - 而这一步预测出的椭圆半轴只有 **a≈15.9 m, b≈2.3 m**，标签是 **a≈20.8 m, b≈8.2 m**；
   - 也就是说椭圆被预测得**短轴只有标签的 1/3.5**，局部走廊退化成一条细管；
     模型曲线本身又离 A\* 路径（椭圆中心所在的折线）**3.5 m 均值 / 8.8 m 最大**；
   - ALM 只对"单元半空间内"负责，碰撞点根本在单元外 → 内部看"无违规"、外部仍在障碍里
     → 全局验证必然失败。

**结论**：碰撞的直接来源是**椭圆形状头对短轴严重低估（b 2.3 m vs 8.2 m）**，加上
"椭圆中心挂在 A\* 路径上、模型曲线偏离该路径"这一组合，使局部 corridor 无法覆盖真正
的碰撞区。ALM 与验证逻辑本身是对的（宁可不改也不放行）。

**待做（未实施）**：
(a) 局部 cell 改为按**碰撞点几何最近查询**选取，并把 `corridor.region.safety_margin`
    从 0.02 scene（1.6 m）放大到能覆盖曲线-路径偏离量（≥0.12 scene ≈ 9.6 m）；
(b) 修形状头：本次训练 val 的椭圆标签一直是 128 列（进程启动时 64 列还没落盘），
    `val_shape` 从未参与选模；需在 64 列标签上重训/微调并提高 `lambda_shape`(0.08→0.2 量级)。

## 4.3 ALM 修正能力实测：优化器不是瓶颈，约束集才是

对 7 个"最终仍碰撞"的样本，固定同一次预测与同一组碰撞区间，只改 ALM 预算：

| 样本 | 修正前 | 6 步/0.20 | 12 步/0.20 | 24 步/0.20 | 24 步/0.60 | 48 步/0.60 |
|---|---|---|---|---|---|---|
| 3009 | 0.227 | 0.133 | 0.117 | 0.117 | 0.219 | 0.211 |
| 3011 | 0.250 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 9013 | 0.156 | 0.039 | 0.000 | 0.000 | 0.000 | 0.000 |
| 47013 | 0.219 | 0.164 | 0.148 | 0.180 | 0.102 | 0.156 |
| 54018 | 0.156 | 0.109 | 0.141 | 0.141 | 0.195 | 0.195 |
| 79010 | 0.062 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 54045 | 0.148 | 0.000 | 0.000 | 0.000 | 0.180 | 0.125 |
| **均值** | — | **0.064** | 0.058 | 0.062 | 0.099 | 0.098 |

结论：
1. **加迭代几乎没用**（0.064→0.058→0.062），**放大步长反而更差**（0.099）——优化器/预算不是瓶颈。
2. 7 个样本里 **4 个能被 ALM 完全清掉碰撞**（3011/9013/79010/54045 → 0.000），
   另外 3 个（3009/47013/54018）无论怎么加预算都留 0.10–0.22 的残余。
3. 原因同 §4.2：这 3 个样本的椭圆短轴被预测得过窄，**走廊单元里根本没有"自由出口"**，
   投影只能落在障碍内 → 全局验证必然失败。此时**增加 ALM 预算在数学上不可能改善**。

**因此当前实现无法"保证"安全性**，缺的是设计里的第三层。要保证只能补齐：

| 层 | 机制 | 现状 |
|---|---|---|
| ① 模型自身少碰障碍 | 安全损失 + 椭圆形状准确 | 弱：train 软碰撞 0.006 但 test 离散碰撞 6.67%；椭圆短轴 b 只有标签的 1/3.5 |
| ② ALM + 全局验证 | 只接受验证通过的修正 | 已实现；对 57% 的碰撞样本有效，其余无效 |
| ③ 验证失败的兜底 | 用"已验证无碰撞的 A\* 路径"作为输出 | **缺失** ← 这就是为什么最终输出里还有碰撞 |

补齐 ③ 之后，输出要么是"模型预测 + 验证通过的 ALM 修正"，要么是"A\* 路径"，
两者的碰撞都由同一个离散检查保证 → 才能对外说"输出轨迹安全"。代价：兜底样本退化成
纯搜索路径，必须作为 `fallback_rate` 单独统计，不能算模型能力。

## 4.4 安全凸区域到底建了没有？（三次实测，纠正 §4.2/§4.3 的说法）

**建了，而且单元本身确实是"安全"的。** 但局部装配方式错了，且约束集对样条类而言
可能根本不可行。三次测量的结果如下。

### 测量 1：单元内部是否真的无碰撞 → 是

对失败样本重建 ALM 实际使用的 cell（`EllipseRegionBuilder.build_from_ellipse`），
在每个 cell 的半空间内拒绝采样 4000 点，再到**同一张 occupancy** 上查：

| 样本 | 选中的 cell 数 | 单元内部占用率 | 占用>5% 的 cell |
|---|---|---|---|
| 3009 | 52 | mean 0.000 / median 0.000 | **0** |
| 47013 | 47 | 0.000 | **0** |
| 54018 | 36 | 0.000 | **0** |

→ `build_from_metric` 确实用 `_greedy_faces` 把椭圆按窗口内的障碍点切开，
`region_complete` 全为 1.0。**"走廊压在障碍上"这个说法是错的，我撤回。**

### 测量 2：ALM 输出是否满足它自己的约束 → 不满足

对 ALM 输出的每个 Bézier piece，用 pack 给它的半空间算 `A x - b`：

| 样本 | constrained pieces | 违反量 mean / max (scene) | 违反 > 1e-3 的 piece | 残余碰撞点落在受约束 span 内 |
|---|---|---|---|---|
| 3009 | 37/37 | 0.0006 / **0.067** | 19/37 | 16/17 |
| 47013 | 36/36 | 0.0018 / 0.072 | 14/36 | 4/5 |
| 54018 | 43/43 | 0.0088 / **0.081** | 22/43 | 4/5 |

max 违反 0.067–0.081 scene ≈ **5.4–6.5 m**，而且残余碰撞点就在"已加约束"的区间里。

### 测量 3：加预算能否收敛 → 不能（§4.3）

iteration 6→48、step 0.2→0.6 后残余碰撞 0.064→0.058→0.098，**不降反升**。

### 结论（有证据的三条）

1. **凸区域是安全的**（测量 1），ALM 也不是没跑（测量 2 的 constrained 全覆盖）。
2. **不可行**：32 个控制点、C² 光滑的三次 B 样条无法被塞进一串**短轴只有 2.3 m
   的细长 cell**（标签是 8.2 m），逐片约束联立起来对样条类无解 → ALM 只能停在
   违反 ~5 m 的妥协点 → 全局验证拒绝 → 输出保留碰撞。**这是形状头低估短轴造成的，
   不是优化器，也不是单元不安全。**
3. **我的局部装配方式比设计更差**：§5.3 要求"相邻 cell 求重叠 + 间隙桥接"的
   **走廊链**（`build_safety_corridor` 就是干这个的，含 `convex_polygon_intersection`
   / `overlap_ratio` / `build_bridge_region`），而我在局部路径里**绕过了它**，
   改成"每个 Bézier piece 只绑一个 cell"。这条路径本来就更难可行。
   （实测：把 cell 裁到碰撞区间再走 `build_safety_corridor` 时 `valid=False`、
   pieces=0 —— 因为间隙桥接需要 `gamma`（骨架点），我传了 None；全量走廊能建出来
   是因为 64 个 cell 重叠足够，不需要桥。）

**修正方向（按收益排序）**
1. 形状头：用 64 列标签重训、提高 `lambda_shape`，把短轴 b 从 2.3 m 拉回 8 m 量级
   —— 单元变"胖"，约束才可行（**根治**）。
2. 局部走廊：改用 `build_safety_corridor` + `gamma=active_path` 做 chained corridor，
   而不是单 cell 绑定（符合 §5.3）。
3. 输出兜底：验证失败回退到已验证无碰撞的 A\* 路径，并单列 `fallback_rate`
   —— 只保证"输出安全"，不提升模型能力。

## 4.5 更正 §4.4 的措辞 + 走廊链为什么建不出来

### cell 到底是什么

`build_from_ellipse -> build_from_metric` 里，`_greedy_faces(c, quadratic,
border_points)` 用**椭圆的切平面**生成半空间（实测每 cell 11–14 个面），再用
`real_points`（窗口内障碍点）切一遍。所以：

* 它**确实是凸区域**（半空间交），不是"只存了椭圆参数"；
* 但它的形状**就是那个椭圆**（预测半轴 15.9 m × 2.3 m），不是沿走廊拉长的多面体；
* 「走廊」是 §5.3 里把相邻 cell 求重叠、必要时插桥得到的**链**，不是一个 cell。

我 §4.4 那句"光滑样条塞不进细长 cell"是**措辞错误**：4.6 m 宽的管对样条毫无难度。
真正的问题在下一条。

### 为什么约束不可行（实测，不是推测）

把 ALM 的 `proximity_weight` 从 1.0 压到 0.001、`inner_steps` 加到 100，
对这三张 pack 的**最大违反量几乎不动**：

| 样本 | default | prox=0.1 | prox=0.01, inner=30 | prox=0.001, inner=100 |
|---|---|---|---|---|
| 3009 | +0.0671 | +0.0631 | +0.0557 | +0.0601 |
| 47013 | +0.0724 | +0.0362 | +0.0852 | +0.0860 |
| 54018 | +0.0811 | +0.0913 | +0.0897 | +0.0606 |

（单位 scene，0.06 ≈ 4.8 m）→ **不是优化器没收敛，是给它的约束集本身不可行**。

### 为什么不可行：我的 piece→cell 绑定

`build_local_constraint_pack` 把每个 Bézier piece 绑到**它自己查询下标那一个 cell**。
而 ellipse 中心在**搜索路径**上（`ellipse.center = resample(active_path)`），
模型曲线偏离该路径 5–9 m，短轴只有 2.3 m → 这个 piece 被要求"待在那个远处的椭圆里"，
联立起来对 32 控制点的样条无解。

### 改成 §5.3 的走廊链也建不出来

用 `build_safety_corridor`（重叠检查 + 间隙桥接）在碰撞区间上重建走廊：

| 样本 | cell 数 | corridor.valid | failure_reason | pieces |
|---|---|---|---|---|
| 3009 | 37 | False | `bridge_failed:9` | 0 |
| 47013 | 36 | False | `bridge_failed:15` | 0 |
| 54018 | 36 | False | `bridge_failed:8` | 0 |

即**相邻预测椭圆连重叠都不够**，需要桥接，而桥接也失败 → 走廊闭合不了、pack 为空。
（全量 64 cell 时恰好能闭合，所以"全量走廊"消融跑得起来——但那是靠数量凑上的。）

### 根因与两条合法修法

根因：**形状头把短轴预测成 2.3 m（标签 8.2 m，差 3.5 倍）**，导致
① 相邻 cell 不重叠/桥接失败；② cell 太窄而曲线偏离 5–9 m，绑定不可行。

| 修法 | 内容 | 性质 |
|---|---|---|
| A（根治） | 64 列标签重训 + `lambda_shape` 0.08→0.2；并给短轴加**下限** `b ≥ 机器人半径 + margin`（≈5 m） | 符合"椭圆描述局部安全集"的定义 |
| B（兜底） | ALM 验证失败 → 输出已验证无碰撞的 A\* 路径，单列 `fallback_rate` | 只保证输出安全，不提升能力 |

## 4.6 已按"直接复用旧代码"重写（自定义局部装配全部删除）

**改动（只做减法）**
* 删除 `src/geometry/bspline_constraints_local.py`（我自己写的局部 pack，连同测试一起删）。
* `src/diffusion/alm_local.py::alm_closed_loop` 现在**逐行复用** `src/diffusion/sampler.py`
  的既有链路，参数名与调用顺序完全一致：

  ```
  EllipseRegionBuilder.build_from_ellipse(center, shape4)     # 每查询凸单元
  build_safety_corridor(..., anchors=i/(Q-1),                 # 重叠检查 + 间隙桥接
                        gamma=active_path, gamma_lengths=len(path))
  build_constraint_pack(bspline, corridors, margin=...)       # 贝塞尔提取
  bspline_alm_correct(q0, pack, bspline, lam, alm_cfg, inner_steps=...)
  dense_validation(..., num_points=512)                       # 仅验证，不投影
  ```

  只加了两处**防崩**保护（不含任何几何）：`build_constraint_pack` 抛异常 → 该次视为无 pack
  （记录 `pack_failed`）；`lam` 的形状与 pack 不一致 → 丢弃热启动（记录 `lam_reset`）。
* `tests/test_v5_framework.py` 改为断言"闭环必须是 legacy 链路"，不再测我自己的 pack。

**在 test 420 上实测（ep138，4 步，旧链路）**

| 指标 | 值 |
|---|---|
| 最终碰撞（均值 / 有碰撞样本） | **2.34% / 17.1%（72/420）** |
| ALM 实际执行率 | 67.6% |
| **ALM 全局验证通过率** | **0.6%（8 次 / 828 次执行）** |
| `active_path` 更新率 | 1.9% |
| 单样本耗时 | 1791 ms |

**为什么通过率这么低（逐样本审计 20 个碰撞样本，batch=1 排除掩码干扰）**

| 项 | 次数 |
|---|---|
| ALM 调用 | 50 |
| **corridor 有效** | **4（8%）** |
| corridor 无效 | 46（92%） |
| 通过验证 | 4（全部是 corridor 有效的那些） |

失败原因两类：
1. `invalid_base_region:N` —— 基础凸单元本身无效（预测椭圆退化，`region_complete=False`）；
2. `bridge_gamma_failed:N` —— 相邻椭圆重叠不足、间隙桥接失败。

**只要 corridor 能闭合，ALM 就把碰撞清到 0**（例：3018 三步 `ok=True`，
`viol=-0.012`、`membership=1.00`；4016 第 3 步同样 ok）。

结论：**旧链路本身没问题，是我的椭圆形状头预测太差导致走廊闭合不了**（短轴 2.3 m
vs 标签 8.2 m，且部分椭圆退化）。所以修复必须在训练侧（形状头 + 短轴物理下界），
而不是再动几何代码。

## 4.7 新增：搜索路径居中（按你的判断，已实测有效）

**动机（你的判断）**：A* 得到的是栅格折线，会贴着障碍拐角走；椭圆中心挂在这条
贴墙线上，于是局部凸单元退化（`invalid_base_region`）或相邻单元不再重叠
（`bridge_gamma_failed`），旧链路的走廊闭不上 → ALM 根本没用上。

**实现**：新增 `src/grid_search/path_centering.py::center_path`
* 复用仓库既有 `grid_search.grid_map.clearance_field`（同一个 EDT）；沿 clearance 梯度
  **上升**（远离墙），按局部 clearance 缩放步长，沿路径 1-2-1 平滑，端点钉死，
  落点若进障碍就吸附到最近自由栅格；
* 只动**内点**，起终点精确不变；仍是 128 点弧长采样，下游（椭圆中心、`gamma`、路径特征）不变；
* 单条 10 ms；配置 `grid_search.center_path: {enabled, iters, step_cells, smooth}`，
  在 `_search_once` 里搜索完成后调用，统计写 `search[].clearance_before/after`。

**实测（test 420，ep138，4 步，旧链路 + 居中）**

| 指标 | 居中前（旧链路） | **居中后** |
|---|---|---|
| 有碰撞样本 | 72 / 420（17.1%） | **39 / 420（9.3%）** |
| 碰撞率均值 | 2.34% | **1.01%** |
| ALM 执行率 | 67.6% | 59.0% |
| **ALM 全局验证通过率** | 0.6% | **38.7%** |
| `active_path` 更新率 | 1.9% | 22.9% |
| 曲线 RMSE | 0.0898 | 0.0871 |
| 单样本耗时 | 1791 ms | 1889 ms |
| 路径 clearance（均值） | 6.1–6.5 m | **7.1–7.3 m** |

逐样本审计（20 个碰撞样本，batch=1）：

| | 居中前 | 居中后 |
|---|---|---|
| ALM 调用次数 | 50 | 33 |
| corridor 有效 | 4（8%） | **16（48%）** |
| 有效时通过验证 | 4/4（100%） | **16/16（100%）** |

**结论**：你的判断是对的——居中把 corridor 闭合率从 8% 提到 48%，只要走廊能闭合，
ALM 就 100% 通过并把碰撞清到 0。剩余 39 个碰撞样本的失败原因**已经不再是走廊无效**，
而是 ALM 投影后仍有残余碰撞被 dense 验证拒绝（`rejected` 25 个、`ok/rejected` 混合 11 个、
`no_collision→rejected` 3 个）。

**仍未做**：`pdg_path_points.npy` 训练缓存还是**未居中**的旧 A* 路径，与现在的推理不一致；
需要重跑 12 号脚本（加居中）并短程微调，训练/推理才完全对齐。

## 4.8 新增：走廊间隙插点（复用前一个椭圆）——你的第二个建议

**做法**：`src/geometry/ellipse_densify.py::densify_ellipses`。相邻两个椭圆单元的
重叠率低于 `min_overlap`（默认 0.10，和走廊构建器同一判据，用
`halfspaces_to_vertices` + `overlap_ratio`）时，在两者**中点插入一个新点**，
**直接把左边（前一个）椭圆的 shape4 复制过去**，每个 gap 最多插 2 个、总数上限 256；
插完仍交给 `build_safety_corridor` 做它原本的重叠/桥接。形状头完全不动，没有新几何。

**实测**

| | 居中 | 居中 + 插点 |
|---|---|---|
| 逐样本审计：corridor 有效 | 16/33（48%） | **21/33（64%）** |
| 有效时通过验证 | 16/16 | **21/21（100%）** |
| test 420：ALM 通过率 | 38.7% | **41.9%** |
| test 420：有碰撞样本 | 39（9.29%） | **40（9.52%）** |
| test 420：碰撞率均值 | 1.01% | 1.03% |
| 单样本耗时 | 1889 ms | **2563 ms（+36%）** |

**结论**：插点确实提高了走廊闭合率（48%→64%，审计子集 +6 个），但**端到端碰撞没有下降**
（39→40），代价是耗时 +36%。原因是最后一两步仍会被 dense 验证拒绝，而拒绝后输出的是
未修正的预测——真正的瓶颈已经变成"最后一步失败时用什么输出"，而不是走廊闭合。

（配置：`corridor.densify.{enabled,min_overlap,max_insert_per_gap,max_total}`，
默认开启；关掉即回到 4.7 的居中版。）

## 4.9 输出选择：`last_verified`（按你的确认实现）

`sample_v5` 现在明确区分"模型预测"和"已通过 dense 验证的轨迹"：

```
每步：ALM 通过 → 记 last_verified = 该步修正后的曲线（active_path 同步更新）
结束：final_curve = last_verified            # 有验证过的就用它（source=last_verified）
      否则若预测仍碰撞 → 居中的搜索路径      # 它是无碰撞的（source=search_path）
      否则 → 模型预测                        # source=model
```

`summary_test.json` 新增 `final_source_counts`，CSV 新增 `final_source` 列
（`model / last_verified / search_path`），评估输出 `final_collision_frac` 用的是**最终
输出的那条曲线**，不再是"最后一步的 raw x0"。这样"输出安全"这句话就有同一个离散检查背书。

## 4.10 重新训练（Q=128，按你的要求）

* `configs/config_v5_c32q128.yaml`：`num_safety_queries: 128`、`lambda_shape 0.08→0.20`、
  `epochs 200`、`max_hours 10`（其余与 c32q64 相同；模型 4.05 M 参数不变）。
* 椭圆标签：`13_build_ellipse64_labels.py --num-queries 128` 重新生成 train/val/test，
  旧的 64 查询标签已备份为 `ellipse_shape4_gt_q64.npy` / `shape_valid_q64.npy`。
  （该脚本串行 2.5–3.6 s/样本，沙箱禁止 multiprocessing，所以用 8 个分片进程并行。）

**注意**：这一轮训练是用**未居中的 A\* 路径缓存**（`pdg_path_points.npy`）作为路径条件，
而推理侧已经居中——训练/推理仍有这一处不一致，需要在下一轮把 12 号脚本也加上 `center_path`。

## 4.11 Q=128 重训的完整准备（两个纠正 + 已开训）

### (a) 训练路径现在也居中了（之前确实没有）

`scripts/data/new_framework/14_center_pdg_paths.py` 原地重写 `pdg_path_points.npy`
（原始备份 `pdg_path_points_raw.npy`），端点钉死不漂移：

| split | clearance 均值 | clearance 最小 | 端点漂移 |
|---|---|---|---|
| train 3850 | 6.25 → **8.82 m** | 1.64 → **4.82 m** | 0 |
| val 1550 | 7.28 → **10.68 m** | 1.93 → **6.33 m** | 0 |
| test 420 | 5.26 → **7.72 m** | 1.96 → **6.64 m** | 0 |

报告：`data/processed/160m-k4p-c32q64/center_paths_report.json`。
训练/推理的路径分布从此一致（推理侧 `_search_once` 也是同一函数 `center_path`）。

### (b) Q=128 的椭圆标签沿**居中后**的路径重算

`13_build_ellipse64_labels.py --num-queries 128`（8 分片并行，串行 3.5 s/样本）：

| split | shape4 | 有效率 | 半轴均值 |
|---|---|---|---|
| train | (3850, 128, 4) | 0.996 | a 18.9 m / b 7.6 m |
| val | (1550, 128, 4) | 0.971 | a 20.3 m / b 11.0 m |
| test | (420, 128, 4) | 0.994 | a 19.2 m / b 6.7 m |

旧 64 查询标签保留为 `*_q64.npy`。**插点已关**（`corridor.densify.enabled: false`）：
128 个椭圆足够密，插点是为 64 个补空隙用的。

### (c) 训练已启动

`configs/config_v5_c32q128.yaml`（C=32 / Q=128 / path 128 点 / 1 层 backbone / 4 层
final / 4 步闭环 rollout / `lambda_shape 0.20` / `epochs 200` / `max_hours 11`）。

实测：**2.56 s/step（B=16）→ 每 epoch 241 步 ≈ 10.3 min**；随着模型少撞障碍，闭环 ALM
的调用变少、epoch 会更快。11 h 预算 → 预计 **60–70 epoch**（不会跑满 200）。

## 5. 还没做完（按优先级）

1. **A\* 路径缓存 → 微调**（方案 §9.2 第 7–8 步）：把 step-2 路径从 GT 换成真实 A\* 路径并短程
   微调，这是训练/推理一致性的最后一块（§13.18）。
2. **原始 occupancy 重建**（S0 审计 §6）：从 `raw/160m-256px/scenes/*.npz` 还原未腐蚀地图，
   重跑路径缓存与安全栅格，才能声称真实安全边界。
3. **训练 rollout 内的 ALM + 动态局部走廊**（§9.3、§13.14）：需要 lam 随 pack 变化的重置/映射，
   目前只有全局 `inherit_dual` 布尔。
4. **S7 消融**与论文级对比表。
5. 局部 bridge（§5.3）与 `queries_collision` 诊断字段。
