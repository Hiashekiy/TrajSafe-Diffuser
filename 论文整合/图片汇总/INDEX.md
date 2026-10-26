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
| `E3_fig1_alm_budget.png` | 2130×870 | 151 KB | **实验 3 核心图**：(a) 连续约束满足率 + 碰撞率 vs ALM 迭代预算、(b) 端到端延迟（batch=1，含 95% CI 带） | ✅ 本次实测生成 |
| `E3_fig1_alm_budget.pdf` | 矢量 | 25 KB | 同上，矢量版 | ✅ 推荐排版用 PDF |
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
| `E3_fig1_*` | `论文整合/实验/results/exp3/alm_budget_safety_runtime.{png,pdf}` |
| `E3_superseded_*` | `论文整合/实验/results/exp3_superseded_ellipse_prior/fig3_geometry_priors.{png,pdf}` |

---

## 四、复现命令（重新生成 E2 / E3）

```powershell
$py = "E:\CondaEnvData\envs\GGMPC\python.exe"

# 实验 2 图（历史反馈过程双子图）
& $py -u scripts\experiments\exp2_feedback_process.py --stage summarise

# 实验 3 图（ALM 迭代预算双子图）
& $py -u scripts\experiments\exp3_alm_budget.py --stage figure
```

两个脚本的输入都已落盘（`exp2/raw_feedback/*_process.csv`、`exp3/timing_interleaved.json`），
因此重画图**不需要重新推理**。

---

## 五、排版注意

* `E2_*` / `E3_*` 的图内文字为英文（Times/DejaVu Serif），若与中文正文混排，建议统一改用 PDF 版。
* `E3_fig1_alm_budget` 的横轴为最大 ALM 内迭代次数，`Default (8/10)` 用星标单独标出，
  引用时须说明 Default ≠ I8/I10（首次 8、后续 10）。
* 曲率相关数值（实验 1 表 1、实验 3 表 4）按 **0.5 m 弧长分辨率**评估，需在表注说明。
* `M01_framework.png` 体积 14.6 MB（6292×2780，300 dpi），投稿前建议按期刊要求压缩或改投矢量版。
