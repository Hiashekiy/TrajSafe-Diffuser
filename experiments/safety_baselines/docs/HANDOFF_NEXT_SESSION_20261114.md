# 交接：新会话从这里开始（2026-11-14 夜）

> ## ⚑ 更新（2026-11-14 深夜）：本文的 §6.0 / §6.1 已完成，结论以
> ## **`docs/official_methods_report.md`** 为准
>
> 本轮做完了三件事：
> 1. **§6.0 的差异表**：`docs/official_vs_ours_diff.md`（三份官方实现逐条对照，含文件:行号）；
> 2. **七种方法的修正全部换成官方原文**（`algorithms/official/`），删掉此前自创的
>    高噪声门控 / 离散证书 / dt 减半重试 / warm start / 双 β / hinge，测试 131 passed；
> 3. **全量 test 420 场景的最终对比表**：`docs/official_final_table.md`。
>
> 三条会推翻本文正文的说法，请先读主报告再读正文：
> - **TVS 不再是冠军**：它原先的 3.6% 来自自带 10 m 余量；余量对齐到 1.25 m 后是 11.7%，
>   与 RoS 14.8% / ReS 15.0% 同档。冠军是 **PDM（段约束，337/420 可行内 0 撞墙）** 与 **Projected 8.3%**。
> - **"顶点安全、线段切角"在本轮 420 场景上不成立**：线段单独造成的碰撞只占 0–3%，
>   绝大多数是顶点自己落在障碍格里（见 `pd_why_no_effect.md` §3）。
> - **§6.1 的五个假设已逐条有官方答案**，见 `official_vs_ours_diff.md` §3。
>
> 仍未做：多种子、val 全量、`lambda_smooth 0.08→0.5` 重训、PD 的 η 写入配置文件。
>
> ---

> 本文件是给**新对话窗口**的唯一入口。读完这一份就够开工，不需要翻聊天记录。
> 覆盖：现在在哪、已经确定的事实（带数字）、还差什么、下一步怎么做、文件索引、口径红线。

---

## 0. 一分钟摘要

> **本轮的硬规矩（用户明确要求）：先读 `.refs/` 下的官方实现，按官方代码对齐；
> 不要再去自行发明"修复"。** 我上一轮为了让它跑通，自己发明了"高噪声门控""梯度夹出初值"
> "每段一行约简"等做法——这些现在全部降级为**待官方代码核对**的假设，不再作为既定方案。

1. 原来的七种安全扩散基线（RoS/ReS/TVS、Projected/Primal-Dual/ALM、PDM）跑的基座是
   **map-only 128 点 Diffuser**，质量太差（无约束采样 12/20 穿障）。
2. 找回并验证了历史 **S-C48**（B 样条 48 控制点 + 安全几何特征，epoch 65）。它作为生成器明显更好
   （2/20 碰撞、18/20 成功），但**七种算法塞不进控制点空间**（4/7 直接失败）。
3. 于是按你的决定：**回到原生 128 轨迹点**，重训一个"点模型 + S-C48 安全几何分支"的基座
   （`safety_points_t256`，5.3958 M，100 epoch，已完成）。目的是**让七种方法能用各自论文的状态空间**，
   不是"比原来好"。
4. 七种方法现在**全部能跑**，但结果被两个问题污染：
   - **(a) 基座锯齿**（平滑权重照抄了 B 样条口径，`lambda_smooth=0.08` 太小）→ RoS/TVS 发散成乱麻；
   - **(b) 约束对象错配**（方法约束顶点、评价器量线段）→ Projected/PDM 出现"点全在空地、线横穿墙"。
5. (b) 的接线错误已修（`segment_constraints_for` 之前被静默忽略）。但把它硬切到"段×圆全展开"后，
   SafeDiffuser 的 CBF QP 被证明不可行、PD/ALM 稠密 Jacobian 撑爆内存。
   **这两个报错说明我此前对"怎么用段约束"的理解很可能是错的——去官方代码里找答案。**
6. **下一步（新窗口第一件事，也是唯一一件事）：读 `.refs/` 下三份官方实现，逐条对照我们的代码，
   先出一张"官方 vs 我们"的差异表，再决定改什么。** 详见 §6.0 与 §9。

---

## 1. 仓库与环境

| 项 | 值 |
|---|---|
| 仓库 | `D:\ProjectDirectory\Neural-IRISDiffuser` |
| 分支 | `v4-map16-fbdelta-safeloss` |
| HEAD | `e29475a`（线段约束可用） |
| 已推送 | **到 `606ffc1`**（你自己 push 成功了），**本地还领先 9 个提交** |
| 工作树 | `experiments/safety_baselines` 干净；根 `.gitignore` 有一处**你自己的改动**（未提交，勿动） |
| Python | `E:/CondaEnvData/envs/GGMPC/python.exe`，一律 `-B -W ignore` |
| 环境变量 | `$env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; $env:MPLBACKEND="Agg"` |
| 测试 | `& $py -B -m pytest -q -p no:cacheprovider experiments/safety_baselines/tests/` → **119 passed**（3 项 `tmp_path` 是沙箱 ACL，与本项目无关） |
| 推送 | 我这侧 git HTTPS 无法凭据（`SEC_E_NO_CREDENTIALS`），只能你在终端 push |

**注意**：`experiments/safety_baselines/results/` 被 `.gitignore` 忽略（6 GB+），
所以**关键结果都复制了一份到 `docs/`**（见 §7），否则新窗口看不到数据。

---

## 2. 三个基座的质量（同一评价器：512 点曲线评价，20 个冻结 val 场景）

| 基座 | 碰撞 | 成功 | 平均最小净距 | RMSE | 长度比 | 最大曲率 | NFE | 耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧 map-only `best.pt`（256 级） | 7/20 | 13/20 | −0.6 m | 14.3 | 0.99 | 12.5 | 256 | 3.1 s |
| **S-C48**（B 样条，16 级） | **2/20** | **18/20** | **+3.51 m** | 7.59 | 0.970 | 2.3 | 16 | 0.58 s |
| **safety-point `best.pt`**（新，256 级） | 5/20 | 15/20 | +1.42 m | 10.91 | **1.167** | **2557** | 256 | 10.0 s |

结论：新基座**全面优于旧 map-only**，但**弱于 S-C48**，且**锯齿明显**（长度比 1.167、曲率 2557 vs 2.3）。
锯齿的根因已定位（`docs/native_point_retrain_report.md` §3）：`lambda_smooth=0.08` 是从 B 样条口径
照抄的；三次 B 样条自带 C²，自由点必须显著加大。**修法：只改这一个数 0.08 → 0.5 重训 70 min**，
其余（权重/日程/数据/七方法代码）全不动。**你已明确说不需要重训**，所以此项待你改主意。

---

## 3. 七种方法在这三个基座上的可运行性（核心事实）

| 基座 | 可运行 | 说明 |
|---|---|---|
| 早先点模型（`seven_native_points_vertexall_*`） | 7/7 | 顶点约束；`NativePointTask`/`SegmentPointTask` 原样可用 |
| **S-C48（B样条控制点）** | **3/7** | Projected/PDM 0/20 报错；RoS/TVS 只能靠关约束返回；PD/ALM 发散 → `docs/sc48_seven_methods_report.md` |
| **safety-point（新）** | **7/7** | 这是重训的唯一目的，已达成 |

**换回点表示的直接收益**：`NativePointTask`（每点一行 Jacobian、可分离 CBF QP、逐点投影）
与 `SegmentPointTask`（PDM App.C.3 段约束）**零修改可用**，不需要任何控制点→曲线的映射。

---

## 4. 顶点约束 vs 线段约束（这是你亲自发现的问题）

**方法约束的是 128 个顶点；我的评价器量的是整条折线。** 两者对象不同 ⇒ 方法"完全满足自己约束"
却交付穿墙路径。实测（13 场景表中抽 2 场景，`docs/vertex_vs_segment_audit.md`）：

| 场景 | 方法 | 顶点 | 线段 |
|---|---|---|---|
| 61001 | Projected | **0/128 坏点** | **穿墙** |
| 61001 | PDM | **0/128 坏点** | **穿墙** |
| 74014 | Projected / PDM | **0/128 坏点** | **穿墙** |
| 61001/74014 | RoS/TVS/PD/ALM | 顶点本身有坏点 | 穿墙 |

**已修**（提交 `62c2d35` + `e29475a`）：
1. `algorithms/runner.py` 的 `safety_points` 分支以前**无条件**建顶点 task，
   `segment_constraints_for=['PDM']` 被静默忽略 → 现在按配置选 `SafetySegmentPointTask`；
2. `SafetySegmentPointTask` 有三处工程必需项（都在 `adapters/safety_point_task.py`）：
   - **高噪声门控**：目标自身段残差 > `projection_gate_residual=0.05 scene(≈4 m)` 时**跳过该级投影**
     （PDM App.F 允许调整投影频率），`info.status='skipped_over_gate'`；
     依据：t=255 时 σ≈5100 scene，46k 行局部 SQP 必然 `segment_sqp_maxiter` / `no_descent`；
   - **初值**：沿地图梯度逐点夹出（无求解器，确定性），保证每个顶点在自由空间；
   - **约束对象不变**：接受点仍按段残差校验，不达标仍抛错。

验证（scene 61001 / Projected / nfe=320 / 16.7 s）：
**改前** 顶点 0/128 坏但线段穿墙、净距 −0.15 m；**改后** 不穿墙、净距 **+0.78 m**、长度比 2.34。

---

## 5. 你昨晚那次运行的报错（原文 + 归因）

```
61001 Diffuser       path False clear 1.00 len 1.13 5.3s
61001 RoS-diffuser FAILED qp_infeasible_certified:margin=0.643706:dual_residual=8.88e-16
61001 ReS-diffuser FAILED qp_infeasible_certified:margin=0.0349973:dual_residual=3.28e-16
61001 TVS-diffuser FAILED qp_infeasible_certified:margin=0.0025117:dual_residual=8.88e-16
61001 Projected     path False clear 0.78 len 2.34 16.7s
MemoryError: Unable to allocate 1.68 MiB for an array with shape (220752,) and dtype float64
  at algorithms/constrained_diffusers/runner.py:69  J=J*(g>0)[...,None]
```

| 现象 | 归因 | 状态 |
|---|---|---|
| RoS/ReS/TVS `qp_infeasible_certified` | 段约束把 CBF 的 QP 从 **128 行**变成 **220752 行**（段×圆全展开）。每步解这么大的硬 QP 结构性不可行（dual_residual ~1e-16 说明求解器本身没问题，是问题不可行） | **未修**，见 §6 方案 |
| PD/ALM `MemoryError` | `constrained_diffusers/runner.py:69` 的 `J=J*(g>0)[...,None]` 假设 **J 是稠密 (rows, D)**：220752×252 float64 = 445 MB，且稀疏矩阵乘布尔掩码会先物化 | **未修**，见 §6 |
| Projected 正常（16.7 s，不穿墙） | 段 SQP + 门控 + 梯度夹出初值有效 | ✅ 已验证 |

**这两个报错本身就是结论**：把 (段×圆) 全展开地当约束行，对"每步解一个小 QP"的 CBF 和
"稠密 Jᵀλ"的 PD/ALM 都不成立。

---

## 6. 下一步（新窗口照着做）

### 6.0 【必做·第一步】读 `.refs/` 官方实现，先出差异表，不写新代码

三份参考实现**已经在本地**（你自己下载的）：

| 参考实现 | 路径 | 状态 |
|---|---|---|
| **PDM / Projected Diffusion Models**（NeurIPS 2024） | `.refs/Projected-Diffusion/` | 完整仓库 |
| **SafeDiffuser**（ICLR 2025，RoS/ReS/TVS） | `.refs/SafeDiffuser/` | 完整仓库 |
| **Constrained Diffusers**（NeurIPS 2025，Projected/PD/ALM） | `.refs/NeurIPS-2025-constrained-diffusers-for-safe-planning-and-control-Supplemental-Conference/code/` | 官方补充材料代码，按环境分目录：`BallRunning / Hopper / swimmer / umaze-large / umaze-small` |

**已知的关键落点（省你 grep 时间，但请自行确认内容）：**

- PDM：`.refs/Projected-Diffusion/examples/trajectories/mpd/models/diffusion_models/`
  里的 `projection.py`、`sample_functions.py`、`scoremodel.py`、`guides.py`；
  另有一层 `examples/trajectories/projection.py`。**重点核对**：投影对象（x0 还是 x_t）、
  每级投影几次、步长 `eps·(σ/σ_min)²`、T 与 M 的实际取值、约束是"点"还是"段"。
- SafeDiffuser：`.refs/SafeDiffuser/diffuser/models/diffusion.py` 里 **`#CBF` 标记共 12+ 处**
  （行号 255 / 290 / 329 / 348 / 392 / 419 / 465 / 484 / 529 / 547 / 602 / 629 …），
  这是 RoS/ReS/TVS 的真实实现位置。**重点核对**：barrier 的 `h` 是怎么算的（逐点还是逐段）、
  QP 的行数与 slack 权重、`Δτ` 与 sub-step、γ 的取法、是否真的每步解 QP。
- Constrained Diffusers：`.refs/NeurIPS-2025-.../code/<scheme>/` 下与 `diffuser/models/temporal.py`、
  `diffuser/datasets/sequence.py`、`config/*.py` 同级的采样/约束代码。**重点核对**：
  Projected / Primal-Dual / ALM 三者的**约束行构造**与**乘子单位**，以及它们对 `β_t` / 每级步数的处理。

**要求的产出（第一件事就交付这个）**：
一张 `docs/official_vs_ours_diff.md`，逐条列：
`项目 | 官方实现(文件:行) | 我们(文件:行) | 差异 | 结论(按官方改 / 保留并说明 / 待定)`。
**在差异表出来之前，不要再改 `algorithms/` 或 `adapters/` 里的任何约束逻辑。**

### 6.1 待官方代码确认后才动手的候选改动（均为**假设**，不是既定方案）

| 假设 | 需要官方代码回答的问题 |
|---|---|
| 段约束应约简为"每段一行、取最紧的圆"（127 行，而不是 220752 行） | 官方的约束行到底怎么构造？是否本来就有这种约简？ |
| SafeDiffuser 的 CBF 每步该解多大的 QP | 官方 `#CBF` 段里 QP 的变量数/行数是多少？是否用 slack？ |
| PD/ALM 需要稠密 J，所以只能承载少量行 | 官方实现里 PD/ALM 的约束行数量级与 Jacobian 存储方式 |
| 高噪声级不该做投影（我加的 `projection_gate_residual`） | 官方是否也在某些级跳过投影？PDM App.F 的"投影频率"具体怎么实现？ |
| `lambda_smooth` 0.08 需要加大 | 与安全方法无关，属基座自身；官方代码无法回答，属我们自己的标定 |

---

## 6.5 之后才轮到的（都不该现在动）

1. `lambda_smooth` 0.08→0.5 重训（70 min）——只有在你愿意时才做；
2. 13 场景 × 8 行的完整表（按官方对齐后的实现对跑）；
3. `docs/native_point_retrain_report.md` 更新验收结论。

---

## 7. 文件索引（新窗口先看这些）

### 文档（都在 `experiments/safety_baselines/docs/`）
| 文件 | 内容 |
|---|---|
| **`HANDOFF_NEXT_SESSION_20261114.md`** | ← 本文件 |
| `native_point_retrain_plan.md` | 为什么换回点表示、架构 delta、损失口径的六处适配 |
| `native_point_retrain_report.md` | 100 epoch 训练 + 20 场景统一评价 + 锯齿根因（**§3 是关键**） |
| `seven_safety_points_report.md` | 13 场景 × 8 行表（顶点版）+ 归因 |
| `vertex_vs_segment_audit.md` | 顶点/线段错配的实测证据 |
| `sc48_recovery_audit.md` | S-C48 找回与验证（epoch 65、历史代码 `74d6140^`、7.52 M） |
| `sc48_seven_methods_report.md` | S-C48 上七方法的失败谱（4/7 不可运行） |
| `sc48_phase_b_review.md` | 投影器三处 bug、可行域非空证明、按噪声级的可达性 |
| `native_point_results/` | 图与数据：`scene_{61001,74014}_{grid,overlay}.png`、`eval_best.json`、`training_summary.json`、`train.log`、`seven_safety_points_13scenes.jsonl` |

### 代码（`experiments/safety_baselines/`）
| 路径 | 作用 |
|---|---|
| `models/safety_points.py` | **新基座模型**（5.3958 M）：SceneCNN(16×16) + TrajectoryEncoder + 安全分支（SkeletonEncoder/MatchBlock/TopologyHead/EllipseGeometry/EllipseShapeHead/SafetyControlFusion）+ FinalDenoiser(8) |
| `models/safety_point_data.py` | 数据加载（curve_gt + candidate_* + ellipse_shape4_gt + alm_cell_*） |
| `models/safety_point_losses.py` | 九项损失，六处点表示适配（含 `alm_corridor_loss` 的 union-of-cells 语义） |
| `adapters/safety_point_task.py` | **`SafetyPointTask`（顶点）/ `SafetySegmentPointTask`（段，带门控）** |
| `adapters/segment_points.py` | `SegmentPointTask`（PDM App.C.3 段约束基类） |
| `adapters/paper_segment_constraints.py` | 圆-线段 SQP（OSQP + 信赖域 + `finish_feasible`） |
| `adapters/native_points.py` | 点空间基类（每点一行、可分离 `cbf_solve`、逐点投影） |
| `algorithms/{safediffuser,pdm,constrained_diffusers}/runner.py` | 七种算法主循环 |
| `scripts/train_safety_points.py` | 训练（T=256、100 epoch、40.6 s/epoch） |
| `scripts/eval_safety_points.py` | 统一评价（DDIM-256、512 点） |
| `scripts/run_seven_safety_points.py` | 七方法跑批（`--config` 可切顶点/线段） |
| `scripts/plot_seven_safety_paths.py` | **出 8 宫格图**（`--config --indices --out-dir`） |
| `configs/safety_points_t256.json` | 基座训练配置 |
| `configs/seven_safety_points_t256.json` | 七方法（**仅 PDM 段约束**） |
| `configs/seven_safety_points_t256_segments.json` | 七方法（**全部段约束**，600/1200 s 超时） |

### 复现命令
```powershell
$py='E:/CondaEnvData/envs/GGMPC/python.exe'
$env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; $env:MPLBACKEND="Agg"

# 图（2 场景 × 8 行，线段版）
& $py -B -W ignore experiments/safety_baselines/scripts/plot_seven_safety_paths.py `
  --config experiments/safety_baselines/configs/seven_safety_points_t256_segments.json `
  --indices 0,163 --out-dir experiments/safety_baselines/results/seven_safety_paths_segments

# 表（13 场景 × 8 行）
& $py -B -W ignore experiments/safety_baselines/scripts/run_seven_safety_points.py `
  --config experiments/safety_baselines/configs/seven_safety_points_t256_segments.json `
  --out-dir experiments/safety_baselines/results/seven_safety_segments
```
`--indices` 用的是**数据集索引**：`0=61001`、`163=74014`、`407=86008`（映射见
`data/processed/160m-k4p-c48/val/sample_id.npy`）。

---

## 8. 口径红线（写论文前必须遵守）

1. **新基座不能叫 "Diffuser"**：它的 forward 需要 `candidate_xy/mask/geometry/lengths`，
   含**候选拓扑条件**。正确名称：`S-C48-style safety model on native points`。
2. **顶点约束是各论文的原定义**；线段版本必须标 "task-adapted segment variant"。
3. **高噪声门控**与**梯度夹出初值**必须写进实现差异表（PDM App.F 支持前者）。
4. `√2·σ_min` 只叫**终端噪声尺度**，不是 RMSE 下限。
5. 2/20、5/20、13 场景等小样本数字**不能当 test 集结论**（正式表要跑 `--split test`，本会话未跑）。
6. 本轮所有结论都是**val**上、20 或 13 个场景、单种子。

---

## 9. 官方参考实现（**已下载到本地，这是新窗口的起点**）

| 方法族 | 本地路径 | 说明 |
|---|---|---|
| **PDM / Projected Diffusion Models**（NeurIPS 2024） | `.refs/Projected-Diffusion/` | 官方仓库（RAISELab-atUVA）。我们文档里引用的 `projected_diffusion/scoremodel.py` 就是它的谱系 |
| **SafeDiffuser**（ICLR 2025, RoS/ReS/TVS） | `.refs/SafeDiffuser/` | 官方仓库；CBF 实现在 `diffuser/models/diffusion.py` 的 `#CBF` 段（12+ 处） |
| **Constrained Diffusers**（NeurIPS 2025, Projected/PD/ALM） | `.refs/NeurIPS-2025-constrained-diffusers-for-safe-planning-and-control-Supplemental-Conference/code/` | 官方补充材料代码；5 个环境目录 `BallRunning / Hopper / swimmer / umaze-large / umaze-small` |

`git` 在本会话无法 clone（沙箱 TLS `SEC_E_NO_CREDENTIALS`），但你已经手动下载好了，**直接用本地文件即可**。

**待核实的归属问题**：`Projected` 这个基线我们当初归到 "Constrained Diffusers" 家族；
如果 `Projected-Diffusion` 官方仓库里也有 projection 采样器，那 `Projected` 更应该按
**PDM 官方实现**来做（两者都是"每步投影到可行集"，但步长/投影对象可能不同）。
在 §6.0 的差异表里必须给出结论。

---

## 10. 新窗口第一句话可以这样说

> 读 `experiments/safety_baselines/docs/HANDOFF_NEXT_SESSION_20261114.md`。
> 然后**先读 `.refs/` 下的三份官方实现**（Projected-Diffusion、SafeDiffuser、
> NeurIPS-2025-constrained-diffusers `code/`），按 §6.0 出一张 `docs/official_vs_ours_diff.md`
> 差异表；**在差异表出来之前不要改任何约束代码**，也不要再自己发明修复方案。

## 11. 禁止事项（本轮教训）

1. **不许再自行发明"让它跑通"的做法**（高噪声门控、梯度夹出初值、行数约简…）而不先核对官方实现。
   这些做法可以保留为候选，但必须标注"未与官方核对"。
2. **不许拿改了方法的版本和原方法混在同一张表里排名**（段约束版本必须单独一列并说明）。
3. **不许把 13/20 场景的 val 数字说成 test 结论**。
4. **不许为了让指标好看而调 `relax_weight / η / ρ / γ / λ_smooth`**；每次调整必须是一次可复现的
   单变量实验，并写清依据（论文公式 或 官方代码 或 显式的工程理由）。
