# 图片汇总（实验 + 方法）

本目录把论文与实验用到的**全部图片**集中到一处，便于排版时挑选。文件为**复制**而非移动，
原始位置保持不变（论文 md / LaTeX 的引用仍指向原路径，见下表"原始路径"）。

命名规则：`E*` = 实验图，`M*` = 方法图，`E3_superseded_*` = 已被取代的实验图。

汇总时间：2026-10-25（仓库已重组为 `论文整合/` 结构后的最新路径）
总计：19 个文件 / 22 MB

---

## 一、实验图（E）

| 文件 | 尺寸 | 体积 | 内容与用途 | 状态 |
|---|---|---|---|---|
| `E2_fig1_feedback_process.png` | 2130×840 | 144 KB | **实验 2 第二部分核心图**：(a) ALM 修正前安全违规量、(b) ALM 修正幅度随第 4–16 次前向的变化；有/无历史反馈两条曲线 | ✅ 本次实测生成，论文可直接用 |
| `E2_fig1_feedback_process.pdf` | 矢量 | 24 KB | 同上，300 dpi 矢量版 | ✅ 推荐排版用 PDF |
| `E3_fig1_alm_internal_budget.png` | 2130×870 | 175 KB | **实验 3 核心图（当前版）**：单次 `bspline_alm_correct()` 在固定缓存输入上的 (a) Bézier 约束可行率、(b) 残余最大违反量 vs 迭代预算 I1–I10 | ✅ 本次实测生成 |
| `E3_fig1_alm_internal_budget.pdf` | 矢量 | 23 KB | 同上，矢量版 | ✅ 推荐排版用 PDF |
| `E3_superseded_fullchain_budget.png` | 2130×870 | 151 KB | 旧版实验 3（整链路预算扫描）：连续约束满足率 + 端到端延迟 | ⚠️ **已被取代**（不评价最终轨迹/耗时的新版取代） |
| `E3_superseded_fullchain_budget.pdf` | 矢量 | 25 KB | 同上，矢量版 | ⚠️ 同上 |
| `E3_superseded_ellipse_prior.png` | 2202×839 | 126 KB | 原实验 3（椭圆几何先验）：Fixed Circle / Fixed Ellipse / Learned Ellipse 三列并排的凸走廊与先验椭圆 | ⚠️ **已被取代**，仅可作补充材料 |
| `E3_superseded_ellipse_prior.pdf` | 矢量 | 60 KB | 同上，矢量版 | ⚠️ 同上 |
| `E1_case293_denoising.png` | 2570×1689 | 571 KB | 单任务（case 293）逆扩散过程中的轨迹预测与 ALM 修正；6 面板 | ✅ 历史实测，可作实验 1/2 的可视化佐证 |
| `E1_case293_paper_onerow.png` | 3684×992 | 477 KB | 同一案例的**单行 6 面板排版版**，论文正文已嵌入此版 | ✅ 论文现用版 |
| `E1_random_endpoints_01.png` | 3315×2239 | 820 KB | 随机起终点案例 2 组（索引 0–1），含走廊与轨迹 | ✅ 历史实测 |
| `E1_random_endpoints_23.png` | 3315×2239 | 765 KB | 随机起终点案例 2 组（索引 2–3） | ✅ 历史实测 |
| `E1_random_endpoints_45.png` | 3315×2239 | 835 KB | 随机起终点案例 2 组（索引 4–5） | ✅ 历史实测 |

> 缺口提示：**实验 1（五方法 × 6 指标）与实验 2 第一部分（A0–A3 消融）目前只有表格，没有图**。
> 若需要，可从 `论文整合/实验/results/exp1/per_task_metrics.csv`（6300 行逐任务）直接出图，
> 无需重跑模型。

---

## 二、方法图（M）

| 文件 | 尺寸 | 体积 | 内容 |
|---|---|---|---|
| `M01_framework.png` | 6292×2780 | 14.6 MB | 图 3-1 总体框架 |
| `M02_feature_fusion.png` | 692×1611 | 213 KB | 图 3-2 条件注意力与历史反馈门控融合 |
| `M03a_route_selection.png` | 2052×4644 | 591 KB | 图 3-3a 通路匹配与选择网络 |
| `M03b_safety_geometry.png` | 6584×2312 | 1.05 MB | 图 3-3b 安全几何预测与距离偏置注意力 |
| `M04_denoising.png` | 5456×3408 | 1.19 MB | 图 3-4 安全特征融合与干净控制点预测 |
| `M05_alm_schematic.jpg` | 栅格 | 143 KB | 图 3-5 连续曲线安全约束与 ALM 控制点修正（论文 md 现用） |
| `M05_alm_schematic.svg` | 矢量 | 72 KB | 同上，SVG 矢量版（`TrajSafe_methods.md` 现用） |
| `M06_alm_continuous_safety.png` | 4704×2784 | 648 KB | ALM 连续安全约束示意图（仓库根 `figures/alm_schematic/`） |

---

## 三、原始路径对照

| 汇总文件名 | 原始路径 |
|---|---|
| `E1_case293_denoising.png`、`E1_random_endpoints_*.png` | `outputs/paper_ral/run_20261009_no_projection/custom_endpoints/` |
| `E1_case293_paper_onerow.png`、`M01`–`M05.jpg` | `论文整合/论文/TrajSafe_methods_and_figures/figures/png/` |
| `M05_alm_schematic.svg` | `论文整合/论文/TrajSafe_methods_and_figures/figures/fig3-5_alm.svg` |
| `M06_alm_continuous_safety.png` | `figures/alm_schematic/alm_continuous_safety.png` |
| `E2_fig1_*` | `论文整合/实验/results/exp2/fig_feedback_process.{png,pdf}` |
| `E3_fig1_alm_internal_budget.*` | `outputs/alm_budget_analysis/figures/alm_budget_analysis.{png,pdf}` |
| `E3_superseded_fullchain_budget.*` | `论文整合/实验/results/exp3/alm_budget_safety_runtime.{png,pdf}` |
| `E3_superseded_ellipse_prior.*` | `论文整合/实验/results/exp3_superseded_ellipse_prior/fig3_geometry_priors.{png,pdf}` |

---

## 四、复现命令（重新生成 E2 / E3）

```powershell
$py = "E:\CondaEnvData\envs\GGMPC\python.exe"
$ds = "outputs\paper_ral\run_20261009_no_projection\data\carla_processed_160k4p_c48_square_safe"
$ck = "outputs\paper_ral\run_20261009_no_projection\A\ckpt\best_task.pt"

# 实验 2 图（历史反馈过程双子图）
& $py -u scripts\experiments\exp2_feedback_process.py --stage summarise

# 实验 3 图（当前版：单次 ALM 调用的预算扫描；复用缓存即可，无需重新推理）
& $py -u scripts\alm_budget_analysis.py --config configs\paper_modified.yaml `
    --ckpt $ck --processed $ds --reuse-cache --allow-commit-drift
```

> 实验 3 的**缓存阶段**必须用能加载论文 checkpoint 的代码版本
> （`v3-skeleton-dynamic`；当前 v4 分支的模型架构已变、无法加载该 ckpt）。
> 缓存生成后，**回放/统计阶段**在任何分支都能跑（不做网络前向，只调用 ALM）。

---

## 五、排版注意

* `E2_*` / `E3_*` 的图内文字为英文（Times/DejaVu Serif），若与中文正文混排，建议统一改用 PDF 版。
* `E3_fig1_alm_internal_budget` 的横轴为**单次 ALM 调用**的最大内迭代次数（1/2/4/6/8/10），
  纵轴为约束可行率与残余最大违反量；单位是**场景单位**（1 scene = 80 m，容差 1e-3 scene = 0.08 m）。
  本轮采用 **v4 模型**（4_map16_c48/best_task.pt）；论文 A 权重的同协议结果存档在
  outputs/alm_budget_analysis_paperA_v3/，结论方向一致。
  该实验只评价单次 ALM 修正，不评价最终碰撞率或推理耗时。
* 曲率相关数值（实验 1 表 1、实验 3 表 4）按 **0.5 m 弧长分辨率**评估，需在表注说明。
* `M01_framework.png` 体积 14.6 MB（6292×2780，300 dpi），投稿前建议按期刊要求压缩或改投矢量版。
