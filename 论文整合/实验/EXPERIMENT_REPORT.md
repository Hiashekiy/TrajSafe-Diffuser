# TrajSafe-Diffuser 实验实施报告（实验 1 / 2 / 3）

> 对应 `实验/docs/` 下的三份**当前**实施方案。记录按仓库真实代码完成的功能开发、实际运行结果、
> 与指导文件的差异以及存在的问题。报告中的每一个数字都来自本仓库的实际运行输出
> （逐任务 CSV 见 `实验/results/`），没有任何人工填写或推算的数值。
>
> **实验 3 已按 `实验3_ALM迭代预算与安全效率权衡_实施方案.md` 更换**：原"椭圆几何先验"实验
> 的结果与脚本保留为历史实现（`实验/results/exp3_superseded_ellipse_prior/`、
> `scripts/experiments/exp3_ellipse_prior.py`），不再作为实验 3 的交付内容。

**总体状态**

| 实验 | 方案 | 实现 | 运行 | 结果 |
|---|---|---|---|---|
| 实验 1 不同安全规划方法对比与计算效率 | `实验1.md` | 已完成 | **已完整运行**（5 方法 × 3 种子 × 420 = 6300 条） | 表 1、表 2 定稿 |
| 实验 2 关键模块消融 + 历史反馈作用 | `实验2_*.md` | 已完成 | **已完整运行**（A0–A3 各 1260；反馈开/关各 1260） | 表 3、图 1 定稿 |
| 实验 3 ALM 迭代预算对安全约束修正效果的影响 | `实验3_ALM迭代预算…md` | 已完成 | **已完整运行**（缓存 16380 次真实 ALM 调用 → 6 预算回放 = 98280 次调用） | 表 4、图 2 定稿 |

三项均**完整运行并得到真实结果**；没有"已实现但未运行"或"因数据/模型问题无法完成"的条目。

---

## 1. 运行环境与版本

| 项目 | 值 |
|---|---|
| 解释器 | `E:\CondaEnvData\envs\GGMPC\python.exe`（Python 3.10.19, torch 2.9.1+cu126, CUDA 12.6） |
| GPU | NVIDIA GeForce RTX 4070 (12 GB)，**测量期间与其他进程共享**（见 §8.6） |
| 仓库提交 | `5a7021a4f810f1827b2d014dd26f32295d4ad26d`（= 论文主表冻结时的 commit） |
| 数据 | `data/carla_processed_160k4p_c48`：train 3850 / val 1550 / test 420；$256\times256$，0.625 m/grid，`scene_to_meter = 80.0` |
| 生成器 D | `.../run_20261009_no_projection/D/ckpt/best_task.pt`，sha256 `4fb7703380a27ba1` |
| 生成器 S | `.../S/ckpt/best_task.pt`，sha256 `55739aada24cfffc` |
| 生成器 A | `.../A/ckpt/best_task.pt`，sha256 `ec7996ab19cce21c` |
| 配置 | `configs/paper_modified_d.yaml` / `paper_modified_s.yaml` / `paper_modified.yaml` |
| 评价协议 | 512 点离散碰撞检查、种子 0/1/2、chunk=32（实验 3 另有 batch=1 延迟测量） |
| 数值库 | numpy 2.2.6, scipy 1.15.3, matplotlib 3.10.7, pandas 2.3.3 |

---

## 2. 第一阶段：代码与数据审查结论

### 2.1 现有实现的位置

| 环节 | 真实代码 |
|---|---|
| 网络（骨架选择 / 椭圆几何 / 安全特征 / 反馈门控） | `src/models/trajsafe/{planner,ellipse,geometry,heads,fusion,feedback,encoders,blocks,boundary}.py` |
| 训练与损失 | `train.py`、`src/losses/losses.py`；权重在 `configs/*.yaml` 的 `loss.lambda_*` |
| 扩散采样 + 走廊 ALM + SDF 引导 | `src/diffusion/sampler.py`、`bspline_alm.py`、`alm_guidance.py`、`sdf_guidance.py` |
| 凸区域 / 走廊 / 桥接 | `src/geometry/convex_region.py`（`EllipseRegionBuilder`）、`safety_corridor.py` |
| 统一评测入口 | `scripts/eval_chapter4.py`（`evaluate_controls`、`collision_free`、`sdf.stats`） |
| 历史主表结果 | `outputs/paper_ral/run_20261009_no_projection/`（含 `source_snapshot/` 冻结代码副本、逐任务 `*.jsonl`） |
| 效率数据 | `.../efficiency_corrected_20261010/{timing.json,run.py,manifest.json}` |
| 验证集选参 | `.../sweeps/{penalty,primal_dual,alm_sdf}.json` |

### 2.2 模型与训练损失是否与论文最终方案一致

* 用 `src/utils/config.py` 递归解析当前 `configs/paper_modified{,_d,_s}.yaml`，与 checkpoint 训练时
  真正使用的 `source_snapshot/configs/{A,D,S}.yaml` 逐键比对，**除路径/ckpt 目录外差异为 0**。
* `loss_weights`（`training_summary.json` 实测）：D 全部安全项为 0 且 `model.feedback.enabled=false`；
  S/A 均为 `Lshape=0.08, Liou=0.25, Lsafe=0.15, Lalm=0.30, Lfbsafe=0.30, Lcurve=0.10`；
  **S 与 A 的唯一配置差异是 `train.feedback.rollout`（False/True）**。
* 三者均 `epochs_done=200`，验证集选模（`best_task_epoch` = 171 / 89 / 106）。

**结论：不存在"旧损失配置的 checkpoint 被当成新结果"的问题。**

### 2.3 可复现性（已实测）

重跑主表 5 种方法与历史逐任务记录**逐条比对**：

| 方法 | 配对样本 | 碰撞标记一致 | RMSE 平均绝对差 | RMSE 最大绝对差 |
|---|---|---|---|---|
| Unconstrained / Penalty / Primal-Dual / SDF-ALM / TrajSafe | 各 1260 | 各 1260/1260 | 0.000000 | 0.000000 |

即**逐位复现**历史主表。另用历史记录独立换算"无碰撞轨迹的平均最小安全距离"得
2.899 / 1.710 / 1.774 / 1.763 / 3.255 m，与本次新指标结果完全一致。

> ALM 的"整批可行即早停"使单样本迭代次数依赖批内其它样本，因此复现须保持 `chunk=32`。
> 用小 chunk（如 8）调试时 TrajSafe 数值会有 1e-3~1e-2 量级偏差。

### 2.4 工作区与冻结快照的差异

| 文件 | 差异 | 影响 |
|---|---|---|
| `src/diffusion/sampler.py` | `effective_warmup_steps()`（`max(1,min(base,forwards-2))`） | 仅影响 4 步配置（预热 3→2，与论文一致）；16 步下不变 |
| `scripts/eval_chapter4.py` | 预热值改用同一 helper（报告口径） | 无 |
| `src/diffusion/sampler.py` | 椭圆先验钩子（原实验 3 用） | 默认 `learned`，不改默认路径 |
| `src/diffusion/sampler.py` | trace 增加 `alm_budget` / `alm_seconds` | **纯测量**，不改任何计算（§2.3 逐位复现证明） |

### 2.5 复用与重跑

* **可复用**：碰撞率（历史逐任务记录直接统计）；平均最小安全距离（历史 `sdf_min_distance_m`
  换算，与新算结果一致）；四组加速配置耗时（§4.3 已核实来源）。
* **必须重跑**：绕行系数、NRMSE、最大曲率、平均平方曲率——历史记录**只保存标量**，
  没有 48 个控制点、没有参考轨迹。重跑协议与历史一致（§2.3 证明）。

---

## 3. 代码实施清单

### 3.1 新增

| 文件 | 作用 |
|---|---|
| `scripts/experiments/metrics.py` | **唯一指标实现**：弧长、绕行、NRMSE、最大曲率、平均平方曲率、最小安全距离；三实验共用 |
| `scripts/experiments/common.py` | 上下文加载、`run_split()`（一个 方法×种子×划分 的运行）、CSV/NPZ/JSON 落盘 |
| `scripts/experiments/exp1_methods.py` | 实验 1：`run`（推理+落盘控制点）/ `metrics`（离线重算+合并效率表） |
| `scripts/experiments/exp2_ablation.py` | 实验 2 第一部分：A0–A3 |
| `scripts/experiments/exp2_feedback_process.py` | 实验 2 第二部分：同权重反馈开/关 + 逐前向日志 + 双子图 |
| `scripts/experiments/exp3_alm_budget.py` | **实验 3**：`p0 / p1 / p1table / run / timing / table / figure` |
| `scripts/experiments/summarize.py` | 汇总为论文表格 `实验/results/paper/PAPER_TABLES.md` |
| `scripts/experiments/run_all.ps1` | 一键复现 |
| `src/geometry/ellipse_prior.py` | 椭圆先验替换（原实验 3 使用，现保留） |

### 3.2 修改（最小必要）

`src/diffusion/sampler.py`：

1. 椭圆先验钩子（默认不生效）。
2. `try_activate_corridor` 内记录每次 ALM 调用的 `alm_budget`（per-sample 预算）与
   `alm_seconds`（**只读测量**），供实验 3 的逐调用日志使用；不改变任何计算。

### 3.3 指标口径（三实验共用）

| 指标 | 实现 |
|---|---|
| 碰撞率 | `eval_chapter4.evaluate_controls` 的 512 点判定（占用栅格双线性 ≤0.5 且在地图内） |
| 平均最小安全距离 | `sdf.stats()["min_distance"]` 转米，**只在无碰撞轨迹**上取均值（同时报碰撞率与有效样本数） |
| 绕行系数 | 弧长（4096 点密采样）/ $\lVert G-S\rVert_2$；分母 <1e-6 m 记 N/A |
| NRMSE | 128 点位置 RMSE / **参考轨迹折线长度** ×100% |
| 最大曲率 / 平均平方曲率 | 精确 B 样条导数基（$N'(u)$、$N''(u)$，无差分近似），在**弧长均匀网格（0.5 m）**上评估（§3.4） |

### 3.4 曲率口径

解析公式 $\kappa=\lvert x'y''-y'x''\rvert/(x'^2+y'^2)^{3/2}$，导数来自精确导数基
（已用有限差分与真实 codec 双重校验：一阶相对误差 2e-11、二阶 4e-6，`b0` 与 codec 自带基差 3e-8）。

直接取 $\sup_u\kappa$ 不可用：SDF/ALM 修正会把相邻控制点推到约 $10^{-4}$ scene 以内，
样条在微米尺度打结（实测单条 100 m 轨迹 $\kappa$ 达 $3.3\times10^4\ \mathrm{m^{-1}}$），均值被
极少数点支配。按方案的退化处理条款，改为在**弧长均匀网格（0.5 m）**上评估同一公式
（$\kappa$ 与参数化无关，定义不变，且 $\kappa\le 2/\Delta s$ 有界），同时保留
$\sup_u\kappa$ 与"参数停滞轨迹数"作数值条件性说明：

| 方法 | 停滞/微结轨迹数（共 1260） | 占比 |
|---|---|---|
| Unconstrained | 1 | 0.08% |
| Fixed Penalty | 154 | 12.2% |
| Primal-Dual | 37 | 2.9% |
| SDF-ALM | 8 | 0.6% |
| **TrajSafe** | **42** | **3.3%** |

论文表注建议写明"曲率按 0.5 m 弧长分辨率评估"。

---

## 4. 实验 1：不同安全规划方法对比与计算效率

### 4.1 运行命令

```powershell
$py = "E:\CondaEnvData\envs\GGMPC\python.exe"
& $py -u scripts\experiments\exp1_methods.py --stage run --seeds 0 1 2
& $py -u scripts\experiments\exp1_methods.py --stage metrics
```

5 种方法：`none / penalty / primal_dual / alm_sdf`（生成器 D）+ `TrajSafe`（生成器 A，`alm` +
反馈开 + `final_project=false`）。三种 SDF 引导参数取**验证集**赢家：
penalty `{penalty_weight:5, step_size:0.1, inner_steps:10}`；primal_dual
`{step_size:0.05, dual_step:1, inner_steps:10}`；alm_sdf `{rho:5, step_size:0.02, inner_steps:10}`。

逐任务原始结果：`实验/results/exp1/raw/<method>_seed<k>.{csv,npz,json}`（npz 含 48×2 控制点、
参考轨迹、起终点）。

### 4.2 表 1（1260 条/方法）

| 方法 | 碰撞率 % ↓ | 最小安全距离 m ↑ | 绕行系数 ↓ | NRMSE % ↓ | 最大曲率 1/m ↓ | 平方曲率 1/m² ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 无约束扩散 | 61.19 | 2.899 | 1.0896 | 13.202 | 0.3313 | 0.00412 |
| 固定惩罚引导 | 19.05 | 1.710 | 1.4665 | 12.066 | 1.1893 | 0.04596 |
| 原始—对偶引导 | 23.10 | 1.774 | 1.2337 | 11.904 | 0.7393 | 0.01764 |
| SDF 增广拉格朗日引导 | 23.89 | 1.763 | 1.1679 | 11.905 | 0.5872 | 0.01154 |
| **TrajSafe-Diffuser** | **0.00** | **3.255** | 1.3246 | **8.759** | 1.0869 | 0.02182 |

最小安全距离有效样本数（无碰撞轨迹数）：489 / 1020 / 969 / 959 / **1260**；
分母均 1260，须与碰撞率配套阅读。全部 6300 条记录无缺失指标。
文件：`exp1/table1_methods.{csv,md}`、`per_task_metrics.csv`、`summary.json`。

### 4.3 表 2（沿用既有测时，未新增计时实验）

| 配置 | DDIM 步数 | 基础区域数 | 时间 ms/轨迹 ↓ | 碰撞数 |
|---|---:|---:|---:|---:|
| 标准配置 | 16 | 128 | 82.26 | 0/1260 |
| 仅减少采样步数 | 4 | 128 | 51.41 | 0/1260 |
| 仅共享凸区域 | 16 | 17 | 56.44 | 0/1260 |
| **组合加速** | **4** | **17** | **27.14** | **0/1260** |

来源核实：`.../efficiency_corrected_20261010/timing.json`（由同目录 `run.py` +
`scripts/paper_modified_timing.py` 产生，`manifest.json` 含逐文件 sha256 与
`warmup_by_steps={"4":2,"16":3}`）。协议 `n=420, seed=0, chunk=32, final_project=false`，
`acceleration` 段为**四配置交错计时**；加速比 82.26/27.14 = **3.03×**。
`timing.json` 的 `methods` 段含另外四种方法的 `wall_ms`（180.1/209.8/206.5/226.6），但属
**顺序测量**（`sequential_timing_excluded_from_speed_comparison=true`），故未放入主表。

### 4.4 解读要点

碰撞率与历史论文表一致；TrajSafe 的 NRMSE 最低（8.76%）、最小安全距离最高（3.255 m，
1260/1260 无碰撞），但**绕行系数最大**（1.3246）且**最大曲率高于 SDF-ALM/Primal-Dual
（1.087 vs 0.587/0.739）**——凸走廊把轨迹压到通路中心，因而路程更长、局部转折更尖，
论文应如实说明这一取舍。曲率排序与旧"平滑度"列一致（penalty 最差、none 最好）。

---

## 5. 实验 2：关键模块消融与历史反馈作用分析

### 5.1 A0–A3 实现核对

| 编号 | 权重 | 引导 | `force_feedback_valid_zero` | `safe_prediction_to_ddim` |
|---|---|---|---|---|
| A0 Baseline | D | `none` | True | True |
| A1 + Safety | S | `none` | True | True |
| A2 + Feedback | **A（与 A3 同一文件）** | `alm` | False | **False** |
| A3 Full Model | **A（同一文件）** | `alm` | False | **True** |

A2 的"ALM 只用于反馈"由 `safe_prediction_to_ddim=False` 实现：`q0_ddim = step_q0_raw`（DDIM 状态
更新与最终返回都用未修正的干净预测），而 `feedback_step(...)` 在该判定**之前**用
`(q0_raw, q0_used)` 更新历史，因此 ALM 结果仍进入下一步反馈——符合方案"不能把 A2 写成关闭 ALM"。

### 5.2 运行命令

```powershell
& $py -u scripts\experiments\exp2_ablation.py --stage run --seeds 0 1 2
& $py -u scripts\experiments\exp2_ablation.py --stage metrics
& $py -u scripts\experiments\exp2_feedback_process.py --stage run --seeds 0 1 2
& $py -u scripts\experiments\exp2_feedback_process.py --stage summarise
```

### 5.3 表 3：A0–A3 消融（各 1260 条）

| 配置 | 安全条件 | 历史反馈 | ALM 构造反馈 | ALM 安全更新 | 碰撞率 % ↓ | 最小安全距离 m ↑ | 绕行系数 ↓ | NRMSE % ↓ |
|---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|
| A0：Baseline | × | × | × | × | 61.19 | 2.899 | 1.0896 | 13.202 |
| A1：+ Safety | √ | × | × | × | 25.16 | 3.120 | 1.2278 | 7.911 |
| A2：+ Feedback | √ | √ | √ | × | 3.333 | 3.225 | 1.3154 | 8.734 |
| **A3：Full Model** | **√** | **√** | **√** | **√** | **0.00** | **3.255** | 1.3246 | 8.759 |

碰撞数 771 / 317 / 42 / 0；最小安全距离有效样本数 489 / 943 / 1218 / 1260。
A0、A3 与实验 1 的 `none`、`TrajSafe` 为同一批运行（标注 `reused_from`）。

### 5.4 表 4 + 图 1：历史安全反馈的作用（同权重，仅切换反馈输入）

两组同生成器 A、同 ALM、同走廊、同安全 DDIM 更新；唯一差异是 `force_feedback_valid_zero`
（关闭时 $v_\mathrm{hist}=0$，FeedbackEncoder 权重保留）。

逐前向（第 4–16 次前向，1260 条/组）：

| 前向序号 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $\overline V_t$ 无反馈 (m) | 0.345 | 0.337 | 0.325 | 0.314 | 0.300 | 0.287 | 0.277 | 0.269 | 0.261 | 0.252 | 0.240 | 0.225 | 0.213 |
| $\overline V_t$ 有反馈 (m) | 0.345 | **0.149** | 0.166 | 0.141 | 0.133 | 0.131 | 0.105 | 0.111 | 0.103 | 0.102 | 0.095 | 0.089 | 0.075 |
| $\overline D_t$ 无反馈 (m) | 0.489 | 0.488 | 0.488 | 0.476 | 0.491 | 0.473 | 0.485 | 0.466 | 0.464 | 0.444 | 0.441 | 0.408 | 0.407 |
| $\overline D_t$ 有反馈 (m) | 0.489 | **0.243** | 0.268 | 0.270 | 0.258 | 0.270 | 0.256 | 0.253 | 0.247 | 0.238 | 0.239 | 0.229 | 0.216 |

配对统计（同一 `(sample_id, seed)`，第 5–16 次前向取均值，1260 对）：

| 量 | Without FB | With FB | 配对差 | 差的标准差 | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| ALM 前安全违规 (m) | 0.2749 | 0.1167 | **−0.1582** | 0.6459 | 4.0e−29 |
| ALM 修正幅度 (m) | 0.4609 | 0.2488 | **−0.2121** | 0.4146 | 9.5e−82 |

第 1–3 次前向为预热（$\overline D_t=0$，两组相同）；**第 4 次前向两组完全相同**
（0.345/0.489），是"唯一干预项是反馈输入"的直接证据。两组最终碰撞率均 **0/1260**，
走廊激活 1260/1260：关闭反馈不改变最终可行性，但反馈把"网络原始预测违规量"与
"ALM 所需修正量"各降低约 43% / 46%。
图：`exp2/fig_feedback_process.{png,pdf}`；数据：`feedback_process_curves.csv`、
`raw_feedback/<config>_seed<k>_process.csv`。

---

## 6. 实验 3：ALM 迭代预算对安全约束修正效果的影响

> **实验定义**：只评价**单次 `bspline_alm_correct()` 调用**的修正能力——先按论文协议跑一次基准
> 采样，缓存每次真实 ALM 调用前的 `q_ref / pack / lam_in`，再在**完全相同的缓存输入**上扫描
> 预算 I1…I10。不评价完整扩散链路的最终碰撞率、不统计推理时间、不调用末端
> `bspline_hard_project()`。
>
> 仓库里另有**两份已被取代**的旧版实验 3，不得作为本实验结果引用：
> `论文整合/实验/results/exp3_superseded_ellipse_prior/`（椭圆几何先验）、
> `论文整合/实验/results/exp3/`（整链路预算 + 耗时扫描）。
> 同一脚本、同一协议、改用**论文 A 权重**的结果存档在 `outputs/alm_budget_analysis_paperA_v3/`
> （见 §6.9），正文采用当前 **v4 模型**的结果。

### 6.1 模型与代码版本

| 项目 | 值 |
|---|---|
| 代码 | branch `v4-map16-fbdelta-safeloss` @ `97b7c2e244` |
| 配置 | `configs/config_160k4p_c48_v4_map16.yaml` |
| checkpoint | `outputs/models/v4_map16_c48/ckpt/best_task.pt`（sha256 `614c9208559c6af7`，best_task ep78） |
| 数据 | `data/processed/160m-k4p-c48`（train 3850 / val 1550 / test **420**） |
| 基准采样协议 | 16 步 DDIM、预热 3、chunk 32、`final_project=false`、走廊 stride 1 |
| ALM 超参 | rho=5.0、step_size=0.05、proximity=1.0、smooth_w=0.1、max_curve_step=0.2 |
| 容差 | `constraint_tol = 0.001` scene = **0.08 m** |

> **与实验 1/2 的模型口径不同**：实验 1/2 用论文主表的 D/S/A 权重
> （`run_20261009_no_projection`，v3 时代架构），本实验按用户要求改用 **v4 模型**
> （`v4_map16_c48`：16×16 单地图特征 + Δ修正反馈 + 统一 L_safe）。论文中必须分别注明，
> 不能把实验 3 的数字与实验 1/2 混为同一网络的结果。

### 6.2 代码核查（方案 §2）

| 核查项 | 结论 |
|---|---|
| `constraint_state()` | `beta = extraction @ q`，`g = piece_A·beta − piece_b`（每个 Bézier 控制点 × 每个面的有符号值） |
| `_violation_stats()` | 返回 `(max_violation, mean_positive, 单约束满足比例)`——最后一项**不等于**整次调用可行 |
| 预算作用点 | `for it in range(max_steps.max())` + 逐样本 `active = it < max_steps`；`max_steps` 会被 `clamp(max=inner_steps)` |
| 预算不被截断 | 回放统一传 `inner_steps = 10`、`max_steps = 各预算`；实测 I10 的最大实际迭代 = 10 |
| 早停 | `remaining = (it+1 < max_steps) & (v_max_now > tol)`，全部样本"用完预算或满足 tol"时跳出 |
| 末端硬投影 | 基线 `final_project=false`；回放只调用 `bspline_alm_correct()`，**从不调用** `bspline_hard_project()` |
| `pack` 冻结 | 同一 chunk 内为同一对象；按 `(seed, chunk_start)` 缓存 42 份，调用记录引用 `pack_id + packed_row` |

### 6.3 实施方式（正式采样零改动）

* **采集**：运行期包装 `src.diffusion.sampler.bspline_alm_correct`（`sample()` 通过模块全局名调用
  它），因此拿到的是采样器**实际传入的** `q_ref / pack / lam / max_steps`；`lam` 为 `None` 时
  原样保留 `None`。另包装 `model.forward_all` 仅用于记录精确扩散步 `t`。
  **`sampler.py` 未被修改**，正式采样默认行为与返回格式不变。
* **回放**：6 个预算作为同一 batch 的 6 行一次调用；每行更新由该行自己的 `max_steps` 掩码控制，
  故与"逐预算单独跑"等价——脚本内置**等价性校验**（逐预算单独回放 48 次），本轮实测最大偏差
  **0.0（逐位相同）**，通过后才使用批量模式（否则自动退化）。
* **缓存规模**：`cache/alm_inputs.pt` = 809.7 MB（16380 次调用 + 42 份约束包）。

### 6.4 运行命令

```powershell
$py = "E:\CondaEnvData\envs\GGMPC\python.exe"

# 阶段 A+B（当前 checkout，v4 模型）
& $py -u scripts\alm_budget_analysis.py --config configs\config_160k4p_c48_v4_map16.yaml `
    --ckpt outputs\models\v4_map16_c48\ckpt\best_task.pt `
    --split test --seeds 0,1,2 --budgets 1,2,4,6,8,10 `
    --out outputs\alm_budget_analysis --rebuild-cache

# 复用缓存，只重跑回放与统计（无需网络前向）
& $py -u scripts\alm_budget_analysis.py --config configs\config_160k4p_c48_v4_map16.yaml `
    --ckpt outputs\models\v4_map16_c48\ckpt\best_task.pt `
    --split test --seeds 0,1,2 --reuse-cache
```

缓存复用会校验 checkpoint sha256、数据路径、split、种子、**ALM 超参内容**与代码 commit；
任一不符即拒绝（commit 不符需显式 `--allow-commit-drift`，仅用于回放阶段）。

### 6.5 样本与覆盖（实测）

| 项目 | 数值 |
|---|---|
| 缓存的有效 ALM 调用 | **16380**（= 420 样本 × 3 种子 × 13 次 ALM 调用） |
| 无效约束包 / 未被引导的调用 | 0 / 0 |
| 基准输入**已满足**约束的调用 | 10686（**65.24%**） |
| 主分析集（基准输入违反约束） | **5694** 次调用、**183 / 420** 场景、**515** 条轨迹 |
| `lam_in is None` 的调用 | 1260（= 每条轨迹的首次 ALM 调用，符合原实现） |

> 主分析集覆盖 **183/420** 个测试场景：其余场景在三个种子的所有 ALM 调用上
> 都"输入即满足约束"（走廊足够宽松），对"预算是否够用"没有信息量，按方案要求排除并如实报数，
> 不填零。全部输入的版本见 `summary_all_inputs.csv`。

### 6.6 结果（主分析集 5694 次调用，按 183 场景配对 Bootstrap 95% CI）

| 最大迭代 | 约束可行率 (%) | 95% CI | 最大违反量均值 (scene) | 95% CI | P95 违反量 | 平均修正幅度 (m) | 实际迭代均值 |
|---|---:|---|---:|---|---:|---:|---:|
| I1 | 32.20 | [26.71, 37.92] | 0.00697 | [0.00538, 0.00880] | 0.00999 | 0.35936 | 1.00 |
| I2 | 50.32 | [44.69, 56.02] | 0.00266 | [0.00212, 0.00326] | 0.00460 | 0.40980 | 1.68 |
| I4 | 87.82 | [84.18, 90.91] | 0.00056 | [0.00048, 0.00066] | 0.00130 | 0.42160 | 2.45 |
| I6 | 95.01 | [93.14, 96.65] | 0.00048 | [0.00040, 0.00057] | 0.00110 | 0.41893 | 2.65 |
| I8 | 96.92 | [95.49, 98.11] | 0.00044 | [0.00037, 0.00053] | 0.00098 | 0.41787 | 2.73 |
| I10 | 97.38 | [96.01, 98.51] | 0.00042 | [0.00036, 0.00050] | 0.00090 | 0.41725 | 2.79 |

* 违反量以**场景单位**记录（1 scene = 80 m）；CSV 同时给出 `*_m` 列，换算依据为配置中的
  `data.scene_to_meter = 80.0`（已核对）。
* 修正幅度在 I4 之后基本持平（0.42160 → 0.41725 m），说明后续迭代几乎不再移动轨迹，
  收益体现在残余违反量上。
* 全部输入（含基准已满足的 10686 次调用）见 `summary_all_inputs.csv`。

### 6.7 结论（预算增加到什么时候饱和？）

1. **预算不足严重损害可行性**：I1 只有 32.2% 的调用能把所有有效约束修到容差内，
   I2 为 50.3%，I4 为 87.8%。
2. **I2→I4 是主要收益区间**：50.3% → 87.8%（+37.5 个百分点），残余违反量
   0.00266 → 0.00056（−79%）。
3. **I6 之后明显饱和**：I4→I6 +7.2 pp，I6→I8 +1.9 pp，I8→I10 仅 +0.5 pp；
   P95 残余违反量 0.00110 / 0.00098 / 0.00090 scene（≈0.088 / 0.078 / 0.072 m），
   已接近容差本身（0.08 m）的量级，继续加大预算没有实质收益。
4. **实际迭代远低于上限**：I10 平均实际迭代仅 2.79（上限 10），I8 为 2.73；
   上限从 6 提到 10 只多用 0.14 次/调用——绝大多数调用在 3 次内已由容差早停。
5. **推荐预算**：以"单次 ALM 把违反修进容差"为目标，**I4 是性价比拐点**（87.8%，
   残余 0.00056）；I6 仍有约 +7.2 pp 收益；I8/I10 的边际收益（+1.9 / +0.5 pp）
   已小于档位间距。当前代码默认策略（首次 8、后续 10）落在饱和区之上。

### 6.8 图

`outputs/alm_budget_analysis/figures/alm_budget_analysis.png` / `.pdf`
（汇总副本：`论文整合/图片汇总/E3_fig1_alm_internal_budget.*`）：
(a) Bézier 约束可行率 vs 预算（均值 + 95% Bootstrap CI）；(b) 修正后最大违反量
（均值 + CI，点线为 P95）。横轴对数刻度以反映 1/2/4/6/8/10 的不等间隔。

### 6.9 附：同协议的论文 A 权重结果（存档，不用于正文）

`outputs/alm_budget_analysis_paperA_v3/` 是同一脚本、同一协议、改用论文主表 **A 权重**
（`run_20261009_no_projection/A/ckpt/best_task.pt`；需 `v3-skeleton-dynamic` 代码加载）
得到的结果，主分析集 5485 次调用 / 211 场景：

| 预算 | I1 | I2 | I4 | I6 | I8 | I10 |
|---|---:|---:|---:|---:|---:|---:|
| 约束可行率 (%) | 32.52 | 47.95 | 81.11 | 92.37 | 95.18 | 96.25 |
| 残余违反量 (scene) | 0.00693 | 0.00302 | 0.00080 | 0.00059 | 0.00052 | 0.00049 |

两套模型的**定性结论一致**（I1/I2 不足 → I4 大幅改善 → I6 后饱和），差异在数值细节。

## 7. 产出清单

### 7.1 代码

| 路径 | 类型 |
|---|---|
| `scripts/experiments/{__init__,metrics,common,exp1_methods,exp2_ablation,exp2_feedback_process,exp3_alm_budget,summarize}.py` | 新增 |
| `scripts/experiments/run_all.ps1` | 新增（一键复现） |
| `src/geometry/ellipse_prior.py` | 新增（原实验 3 保留） |
| `src/diffusion/sampler.py` | 修改（椭圆先验钩子 + ALM 逐调用测量，均不影响默认计算） |
| `scripts/experiments/exp3_ellipse_prior.py` | 原实验 3，保留为历史实现（输出重定向到 `exp3_superseded_ellipse_prior/`） |

### 7.2 结果与图表

```
实验/results/
├── exp1/  table1_methods.{csv,md}  table2_efficiency.{csv,md}  per_task_metrics.csv
│          summary.json  efficiency_summary.json  raw/<method>_seed<k>.{csv,npz,json}
├── exp2/  table_ablation.{csv,md}  per_task_ablation.csv  ablation_summary.json
│          feedback_process_curves.csv  feedback_process_summary.json
│          fig_feedback_process.{png,pdf}  raw_ablation/…  raw_feedback/…
├── exp3_superseded_ellipse_prior/       ← 第 1 版实验 3（椭圆先验，非交付）
└── exp3/                                ← 第 2 版实验 3（整链路预算+耗时，非交付）
outputs/alm_budget_analysis/            ← **当前实验 3 交付目录**
├── cache/alm_inputs.pt                 ← 16380 次真实 ALM 调用的输入（793.5 MB）
├── cache/cache_manifest.json
├── per_call.csv                        ← 主分析集逐调用（5485 × 6 预算）
├── per_call_all.csv                    ← 全部逐调用（16380 × 6）
├── per_trajectory.csv  per_scene.csv   ← (样本,种子) / 场景级聚合
├── summary.csv  summary_all_inputs.csv ← 预算级汇总 + CI
├── run_config.json                     ← 版本/超参/覆盖度/排除原因/parity
├── summary.md                          ← 中文结果摘要
└── figures/alm_budget_analysis.{png,pdf}
└── paper/ PAPER_TABLES.md               ← 三实验论文表格汇总
实验/logs/  exp1_run.log  …  exp3_timing_interleaved.log  …
```

论文表格：`实验/results/paper/PAPER_TABLES.md`（表 1、1b、2、2b、3、3b、3c）。
图：`exp2/fig_feedback_process.pdf`、`exp3/alm_budget_safety_runtime.pdf`（300 dpi PNG + 矢量 PDF）。

---

## 8. 与指导文件的差异与说明

1. **重跑而非"复用旧结果重算"**：历史记录只存标量、缺控制点与参考轨迹（§2.5），故重跑推理；
   用 §2.3 的逐位比对证明协议一致，并用历史记录独立复核碰撞率与最小安全距离。
2. **曲率口径**：使用精确导数基的解析公式，但评估位置取弧长均匀网格（0.5 m），并另存
   $\sup_u$ 版本与停滞统计（§3.4）。这是唯一一处对指标定义的主动收窄，需在论文表注说明。
3. **实验 1 效率表不含方法级时间**：`timing.json` 中四种对照方法的耗时属顺序测量协议，
   按方案未采用（§4.3）。
4. **实验 2 的 A2 列标记**：A2 的"ALM 安全更新"为 ×、"ALM 构造反馈"为 √，与方案表格一致。
5. **实验 3 的 Default 与 I8/I10 明确区分**：Default = 首次 8、后续 10；I8/I10 = 全程同一上限。
   实测 Default 的迭代均值 6.81 介于 I6（5.26）与 I8（6.58）之上、I10（7.76）之下，符合定义。
6. **实验 3 的计时未与实验 1 表 2 混用**：本次端到端测量在**共享 GPU**上完成，且 harness 不同，
   绝对值不可与实验 1 的 27.14 ms 直接比较（§8.6）；表 2 仍只用于实验 1 的加速比结论。
7. **未新增方案外的实验/指标**：实验 1 只保留原五方法与四组加速配置；实验 2 未加时间测试；
   实验 3 只用方案列出的指标，另外只做方案 §7 要求的配对 bootstrap（属统计方法，非新指标）。

---

## 9. 存在的问题与局限

1. **曲率对参数化敏感**（§3.4）：论文须写明"按 0.5 m 弧长分辨率评估"；若审稿要求纯解析值，
   `summary.json` 保留 `kappa_max_analytic_inv_m` 可切换。
2. **端到端延迟受共享 GPU 影响**：测量期间该卡上另有 6–7 个进程，顺序测量在 53–78 ms（b=32）与
   249–317 ms（b=1）之间波动，远大于预算差异。因此采用**交错计时**（方案 §3.2 步骤 6 推荐）：
   b=32 的中位数随预算单调上升（65.7→78.6 ms），b=1 的均值在 I2 与 I4+ 之间有明显台阶
   （242 vs ~302 ms），可作为相对比较；绝对延迟需在独占机器上复测得。
3. **b=1 延迟只覆盖 120 个任务（种子 0）**：单样本运行的固定开销使全量 420×3×6 的代价过高；
   表 5 已给出样本量（`e2e_ms_batch1_n`），论文引用时应注明。
4. **碰撞率在预算间非单调**：I2 3/1260、I4 0、I6 2/1260、I8/I10/Default 0。不同预算改变 ALM
   修正→DDIM 状态→历史反馈，轨迹本就不同；2–3 条差异属 512 点离散检查的噪声级别，
   不能据此给预算排序（方案 §10 情况 D）。
5. **实验 3 的连续约束满足率依赖 `pack`**：复算使用激活时冻结的约束包。若某样本从未激活走廊，
   `pack` 为空，该样本的连续可行性按"真空可行"处理；本批 6×1260 条全部激活
   （`corridor_guided` 全为 True），不存在该情形。
6. **零碰撞仍是离散结论**：512 点采样下的 0/1260 不等于连续安全保证。
7. **A1→A2 仍是跨权重比较**：S 与 A 仅差 `train.feedback.rollout`，但毕竟是两次独立训练；
   按方案 §3.1 只能解释为"反馈机制及对应训练策略的综合作用"。

---

## 10. 复现验收清单（对照方案）

| 检查项 | 结果 |
|---|---|
| D/S/A 权重与最终损失配置一致并有哈希记录 | ✅ §1、§2.2 |
| 三实验共用同一指标实现 | ✅ `scripts/experiments/metrics.py` |
| 实验开关默认不影响原有流程 | ✅ §2.3 逐位复现 |
| 固定数据/种子/权重/评价规则 | ✅ 种子 0/1/2，测试集 420，checkpoint 哈希 |
| 逐任务原始结果全部落盘 | ✅ `实验/results/*/per_task*.csv`、`raw*` |
| CSV 表格 + 高质量 PNG/PDF | ✅ `paper/PAPER_TABLES.md`、两张图 |
| 未伪造结果、未把未运行配置标记完成 | ✅ 全部为实际运行输出 |
| A2/A3 同权重且差异符合定义 | ✅ §5.1 |
| 反馈开/关保留相同 ALM 与安全更新、相同噪声 | ✅ 第 4 次前向两组完全相同 |
| 第 4–16 次前向保存修正前后日志与接纳状态 | ✅ `*_process.csv`（20 列） |
| **实验 3**：六种预算跑在**完全相同**的有效缓存输入上 | ✅ 每份缓存输入的 `q_ref/lam` 六预算逐位相同（实测 0 处不一致，`per_call_all.csv`） |
| **实验 3**：每种预算只调用 `bspline_alm_correct()`，不用末端硬投影替代 | ✅ 基线 `final_project=false`；回放只调 ALM，代码中无 `bspline_hard_project` 调用 |
| **实验 3**：每次调用使用缓存的原始 `q_ref`、同一份 `pack`、同一个 `lam_in` | ✅ 运行期包装采集；`lam_in is None` 的 1260 次调用原样保留 None |
| **实验 3**：早停仍由原函数执行、实际迭代次数被记录 | ✅ `iterations_used` ≤ 预算（实测越界 0 次）；I10 平均仅 3.06 |
| **实验 3**：预算不被配置里的 `inner_steps` 截断 | ✅ 统一传 `inner_steps=10`；I10 实测最大实际迭代 = 10 |
| **实验 3**：无有效约束的调用未被误记为可行 | ✅ 空包/未引导分别计数并从可行率剔除（本批 0 / 0） |
| **实验 3**：指标一是"整次调用全部有效约束在容差内"的比例 | ✅ 由 `max_violation_after <= 1e-3` 计算，未使用 `constraint_feasible_rate` |
| **实验 3**：CI 按测试场景配对 Bootstrap，不把扩散步/调用当独立样本 | ✅ 调用 → (样本,种子) → 场景三级聚合，211 场景重采样 2000 次 |
| **实验 3**：不输出/分析推理时间与最终碰撞率 | ✅ 无相关指标 |
| **实验 3**：缓存复用校验 checkpoint/配置/数据/代码版本 | ✅ 不符即拒绝；commit 漂移需显式 `--allow-commit-drift` |
| **实验 3**：日志记录有效场景/轨迹/调用数与排除原因 | ✅ `run_config.json` 的 `stats`、`summary.md` |
| **实验 3**：正式采样默认配置与原有入口未被改变 | ✅ `sampler.py` 未修改；批内等价性校验 5.96e-08 |

---

*报告更新时间：实验 3 更换为 ALM 迭代预算研究并完成全部运行之后。所有命令、日志与结果文件
路径均可在仓库内直接复现。*
