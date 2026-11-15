# 七种官方安全修正方法对比：总结报告（2026-11-14）

> 本文是本轮工作的**唯一权威汇总**。其余文档（`official_vs_ours_diff.md`、`official_frozen20_table.md`、
> `official_test420_table.md`、`pd_why_no_effect.md`、`official_paths/`）是过程记录与证据，遇到冲突以本文为准。

---

## 0. 一页摘要

**做了什么**：把三种安全扩散方法族的**修正部分**全部换成官方原文（vendored），删掉此前自己发明的
机制（高噪声门控、离散证书、dt 减半重试、warm start、双 β、hinge…），模型、数据、场景、评价器保持我们自己的。

**最终结果**（test 全量 420 场景，七行统一 1.25 m 余量）：

| 方法 | 撞墙率 | 成功 | 净距 | 违规占比 | 长度比 |
|---|---:|---:|---:|---:|---:|
| 无修正基线 | 49.5% | 212 | −0.63 m | 13.75% | 1.09 |
| RoS | 14.8% | 358 | +1.18 m | 1.85% | 1.10 |
| ReS | 15.0% | 357 | +1.23 m | 1.59% | 1.09 |
| TVS | 11.7% | 371 | +1.67 m | 1.35% | 1.11 |
| **Projected** | **8.3%** | 385 | +1.01 m | 2.47% | 1.31 |
| Primal-Dual | 41.0% | 248 | +0.17 m | 7.57% | 1.22 |
| ALM | 23.6% | 321 | +0.86 m | 2.22% | 1.68 |
| **PDM（段约束）** | **0.0%**（337/420 可行） | 337 | +1.62 m | 0.00% | 2.79 |

**四句话结论**：

1. **余量比算法更有用**：给 1.25 m 余量，RoS/ReS/Projected 的碰撞率各降 2.6–3.9 倍；
   而它们彼此的算法差异在零余量时只有 6–16 个百分点。
2. **TVS 原先的"最强"是余量买来的**：它自带的时变项等于 10 m 余量，对齐到 1.25 m 后从 3.6% 掉到 11.7%，
   与 RoS/ReS 同档。**这一条改写了此前所有以 TVS 为冠军的表。**
3. **机制上有效的是"投影"，不是"乘子"**：PDM（每级精确投影）0.0%、Projected（每步投回可行集边界）8.3%；
   而乘子法 PD 41.0% / ALM 23.6% 即便给了同样的余量也救不动。
4. **二值指标会吃掉真实效果**：按弧长加权，基线有 13.75% 的路径长度压在障碍里，RoS 只有 1.85%。
   只看"某场景是否撞"这一列，会得出"六个方法几乎没区别"的错误印象。

---

## 1. 这次换掉了什么

新增 `algorithms/official/`（每个函数标注官方文件:行号）：

| 文件 | 内容 | 官方来源 |
|---|---|---|
| `safediffuser.py` | RoS/ReS/TVS 的 QP、`h=Lfb+k·b`、slack 权重 100、TVS 的 sigmoid 水平 | `SafeDiffuser/diffuser/models/diffusion.py:451-512, 588-658, 895-956` |
| `constrained_diffusers.py` | PD/ALM 的乘子步（逐行逐粒子）、投影 | `NeurIPS-2025-…/code/umaze-large/diffuser/models/diffusion.py:188-224, 306-380, 382-466` |
| `posterior.py` | 后验系数、官方级阶梯 `reversed(range(-200,T))`、t=1 不加噪 | `diffusion.py:117-131, 503-510` |
| `pdm.py` | `eps·(σ/σ_min)²` 步长、`idx>5` 门控、失败补 15 步重投 | `Projected-Diffusion/…/scoremodel.py:509-531, 557-627` |

**删掉的（我们自己发明的，不是论文的）**：连续时间 CBF 速度场积分与 `cbf_dt` 子步、离散证书与 dt 减半
重试、点速度上限、`legal_gamma`、高噪声残差门控、`cd_beta` 双 β、对粒子求违反量均值、hinge、
PDM 的 warm start 与 `allow_partial`、`kernels.py` 四个旧核、14 个配置死键、两个只量已删代码的审计脚本。

**必须声明的两处替换**（不是论文内容）：障碍几何（官方的 maze2d 椭圆 → 我们的占用栅格）、
QP 后端（`qpth` → 同一问题陈述的闭式解，已用 SLSQP 逐点比对）。

---

## 2. 实验协议

| 项 | 值 |
|---|---|
| 数据 | `data/processed/160m-k4p-c48`，**test 全部 420 场景**（另有 val 1550） |
| 基座 | `results/safety_points_t256/ckpt/best.pt`，5.3958 M，checkpoint 记录 epoch 41 |
| 状态 | 128 点原生轨迹，两端点由任务固定（126 个自由点，D=252） |
| 阶梯 | 七行共用 `reversed(range(-64,256))`：SD 256 步、CD 320 步、PDM 200 步 |
| 初始噪声 | `initial_seed(scene, seed)` 配对协议，七行完全相同 |
| 评价器 | `adapters/unified_curve_eval.py`，512 点曲线 + 弧长均匀违规占比 |
| **余量** | 七行统一 **2 格 = 0.015625 场景单位 = 1.25 m**（1 格 = 2.0/256） |
| PD 的 η | `6.1035e-09`（官等效值的 1/10，见 §4.3） |
| 种子 | 0（单种子） |

余量落到四个配置字段，对七行同时生效：`radius_scene`（点约束）、`segment_circle_margin_scene`（PDM 圆半径）、
`sd_tvs_level_scene`（TVS 时变项，否则它会多拿 10 m）、`dual_eta`（PD）。
基线不受余量影响，复用同批 420 场景结果。

---

## 3. 四个发现（含证据与反证）

### 3.1 余量是比算法更强的杠杆

同一批 420 场景，零余量 → 1.25 m：

| 方法 | 零余量 | 1.25 m | 倍数 |
|---|---:|---:|---:|
| RoS | 39.0% | 14.8% | 2.6× |
| ReS | 39.5% | 15.0% | 2.6× |
| Projected | 32.6% | **8.3%** | 3.9× |
| ALM | 30.0% | 23.6% | 1.3× |
| Primal-Dual | 49.3% | 41.0% | 1.2× |
| PDM | 1.2% | **0.0%** | — |

余量不能无限加：60 场景的子集上，RoS 在 8 格（5 m）时到 **2.7%**，但到 16 格（10 m）时
**37 个场景因端点被占直接不可行**、其余也变形（长度比 3.16）。**可用的余量上限由端点的净距决定**；
实现上已改为"端点用未膨胀的地图判定、只对内部点施加余量"。

### 3.2 TVS 的冠军是余量买的（重要更正）

零余量版里 TVS 3.6%，远好于 RoS 39.0%，看起来"TVS 算法更强"。但 TVS 的时变水平项
`sd_tvs_level_scene` 原先设为 **0.125 = 16 格 = 10 m** —— 它并非没有余量，而是余量最大。

| TVS 的余量 | 撞墙率 | 长度比 |
|---|---:|---:|
| 10 m（原设置） | **3.6%** | 1.187 |
| 1.25 m（与其它行对齐） | **11.7%** | 1.109 |

余量敏感性（同 420 场景，只改这一个值）：0.05 → 11.7%、0.125 → 3.6%、0.25 → 10.5%，
**非单调且两侧都差 3 倍**，说明它是"与地图/障碍尺度绑定的量"，不是可随便调的旋钮。

**结论**：任何以"TVS 3.6%"为冠军的表述都必须同时给出它的余量设置，否则是把余量记在算法头上。

### 3.3 机制上有效的是投影，不是乘子

| 类型 | 方法 | 撞墙率（1.25 m） | 机制 |
|---|---|---:|---|
| 投影到可行集 | PDM（段 + 每级精确最近点） | **0.0%** | 残差必须 ≤1e-6，否则判失败 |
| 投影到可行集 | Projected（每步投回边界） | **8.3%** | 一步落到边界，交付即可行 |
| 乘子软约束 | ALM | 23.6% | 加在后验均值上，λ 累积 |
| 乘子软约束 | Primal-Dual | 41.0% | 同上，且 η 量纲需标定 |
| 单侧 QP（CBF） | RoS / ReS / TVS | 11.7–15.0% | 每点 2 变量、沿 ∇h 推一步 |

乘子法的问题有量化证据（`pd_why_no_effect.md`）：PD 确实把它优化的量降下去了
（逐场景顶点违反 49→5、14→0），但**它停在"几乎干净"**，而指标是逐场景二值的 ——
49 个坏点降到 5 个一分不加。PD 还叠加一个量纲问题：η 按官方等价式换算后，
在高噪声段（此时状态是随机点云、`|g|` 可达 173 m）把乘子推到饱和；η 缩小 10 倍后
碰撞率 49.3% → 42.1%、长度比 2.815 → 1.222。

### 3.4 二值指标会吃掉真实效果

| 方法 | 二值撞墙率 | 弧长违规占比 | 违规长度 |
|---|---:|---:|---:|
| 基线 | 49.5% | **13.75%** | 16.14 m |
| RoS | 14.8% | 1.85% | — |
| Projected | 8.3% | 2.47% | — |
| Primal-Dual | 41.0% | **7.57%** | — |
| PDM | 0.0% | 0.00% | 0.00 m |

基线与 RoS 在违规占比上差 7 倍，而二值列只是 49.5% 与 14.8%。**报论文时，两列必须同时给**；
只给二值列会低估单侧 QP 类方法，只给连续列会高估 PD（它违规占比不差，但路径拧成 1.22–3.24 倍）。

---

## 4. 已知问题与未完成

1. **PDM 的可行率是 337/420**：83 个场景段约束残差收不到 1e-6（官方会无限重投，我们设了上限并判失败）。
   它的 0.0% 只能标"在 337 个可行样本上"，不能与 420 并列。
2. **PD / ALM 的长度比仍偏高**（1.22 / 1.68）：乘子法"收敛到边界附近"必然带形状代价。
3. **单种子、单基座**：没有跑多种子/多基座，也没跑 val 全量（1550）与 test 的交叉验证。
4. **PD 的 η 标定未进入配置文件**：`dual_eta` 仍是原值，所有 PD 相关数字都通过 `--override` 复现；
   统一到 η/10 需要重跑所有涉及 PD 的历史结果。
5. **余量上限**：>5 m 会让大量场景端点不可行；"端点也留余量"在本任务里无意义（端点由数据固定）。
6. `lambda_smooth 0.08→0.5` 的基座重训**未做**（属基座标定，与方法对比无关）。

---

## 5. 文件索引

### 结论与表格
| 文件 | 内容 |
|---|---|
| **`docs/official_methods_report.md`** | ← 本文 |
| `docs/official_final_table.md` | 最终表（1.25 m 余量）+ 余量前后对照 |
| `docs/official_test420_table.md` | 零余量版 420 场景表 + TVS 余量敏感性 |
| `docs/official_frozen20_table.md` | val-20 自检（已标注被 420 场景推翻） |
| `docs/pd_why_no_effect.md` | PD 的机制诊断：接入正确性四条证据、连续指标、权衡曲线 |
| `docs/official_paths/` | 两个 test 场景 × 八行的路径图（grid + overlay） |

### 代码
| 路径 | 作用 |
|---|---|
| `algorithms/official/` | 官方修正（vendored，含出处行号） |
| `algorithms/{safediffuser,pdm,constrained_diffusers}/runner.py` | 三个官方循环（只写胶水） |
| `adapters/native_points.py` | 128 点状态、点约束、投影（支持余量）、端点判定 |
| `adapters/segment_points.py` + `paper_segment_constraints.py` | PDM App.C.3 段约束（46k 行，稀疏） |
| `adapters/unified_curve_eval.py` | 统一评价器（含连续违规指标） |

### 脚本
`run_seven_safety_points.py`（主跑批，支持 `--split/--override`）、`plot_seven_safety_paths.py`（出图）、
`analyze_pd_why.py`（顶点/曲线分解）、`diag_pd_step.py`（单步解剖）、`diag_primal_dual.py`（多场景 trace）。

---

## 6. 复现

```powershell
$py='E:/CondaEnvData/envs/GGMPC/python.exe'
$env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; $env:MPLBACKEND="Agg"

# 最终表（三段并行，各 140 场景）
& $py -B -W ignore experiments/safety_baselines/scripts/run_seven_safety_points.py `
  --split test --indices ((0..139) -join ',') `
  --methods RoS-diffuser,ReS-diffuser,TVS-diffuser,Projected,Primal-Dual,ALM,PDM `
  --override radius_scene=0.015625 --override segment_circle_margin_scene=0.015625 `
  --override sd_tvs_level_scene=0.015625 --override dual_eta=6.1035e-09 `
  --out-dir experiments/safety_baselines/results/final_m2/b0

# 测试
& $py -B -m pytest -q -p no:cacheprovider experiments/safety_baselines/tests/   # 131 passed
```

`results/` 被 `.gitignore` 忽略（6 GB+），关键结果已复制到 `docs/`。

---

## 7. 写论文时的口径红线

1. 报撞墙率必须同时给**长度比**与**违规占比**，否则读者无法判断代价。
2. **TVS 必须标注它的余量**；把 10 m 余量下的 3.6% 当成算法优势是错的。
3. PDM 只能写"**337/420 可行，其中 0 撞墙**"，不能写"0/420 撞墙"。
4. 六种点约束方法与 PDM 的段约束**不是同一件事**，表格必须分列并标注 "task-adapted segment variant"。
5. 所有数字是 **val/test 的 420 场景、单种子、单基座**，不能写成"test 集结论"以外的更强表述。
6. 官方修正 + 我们的场景 = "vendored official correction on our base and scenes"，
   不能说"复现了论文结果"；两处替换（几何、QP 后端）必须在方法节声明。
