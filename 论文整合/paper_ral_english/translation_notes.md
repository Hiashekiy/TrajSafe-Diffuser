# Translation notes — TrajSafe-Diffuser (CN → EN, IEEE RA-L)

Source: `论文/TrajSafe_methods_and_figures/TrajSafe_paper_optimized.md`
(369 lines, 22 references). Section III of the current revision of that file is
itself the condensed text from `方法论述_精简优化版.md`, so the two agree
sentence-for-sentence.

Target: `paper_ral_english/main.tex` (RA-L two-column, **10 pages**)

The pre-replacement manuscript is archived as `backup/main_v1_original_method.tex.bak`
(14 pages) so the early Method version can still be compared. See §2.1 and §2.2
for the two rounds of source changes.

**No source content was deleted.** All 5 top-level sections, every subsection,
all 27 display equations, all 9 figure images, all 3 tables and all 22
references are present in the English manuscript. No experiment, number,
conclusion, algorithm mechanism or theoretical claim was added, removed or
altered.

---

## 1. Terminology actually used

| 中文 | English used in `main.tex` |
|---|---|
| 扩散运动规划 | diffusion-based motion planning |
| 生成式安全约束 | generative safety constraints |
| 安全几何生成 | safety geometry generation |
| 凸安全走廊 | convex safe corridor |
| 安全条件去噪 | safety-conditioned denoising |
| 历史安全反馈 | historical safety feedback |
| 控制点 | control points |
| 候选通路 | candidate routes |
| 通路匹配 | route matching |
| 局部安全椭圆 | local safety ellipse |
| 增广拉格朗日法 | augmented Lagrangian method (ALM) |
| 起终点补偿 | endpoint compensation |
| 连续轨迹约束 | continuous-trajectory constraints |
| 自适应层归一化 | adaptive layer normalization (AdaLN) |
| 空间距离偏置 | spatial distance bias |
| 夹持三次 B 样条 | clamped cubic B-spline |
| 自由空间 | free space |
| 占据栅格 | occupancy grid |
| 安全中心 | safety center |
| 责任区间 | responsibility interval |
| 半空间 | half-space |
| 接受检查 | acceptance check |
| 逆扩散过程 | reverse diffusion process |
| 预热 | warm-up |
| 内迭代 | inner iteration |
| 门控 | gating |
| 掩码 Softmax | masked softmax |
| 原始—对偶引导 | primal-dual guidance |
| 固定惩罚引导 | fixed-penalty guidance |
| 有符号距离场 | signed distance field (SDF) |

Abbreviations are spelled out at first use and abbreviated afterwards:
DDPM-level terms are not used directly; **DDIM**, **ALM**, **AdaLN**, **SDF**,
**RMSE**, **MLP**, **FFN**, **CA** (cross-attention), **LN** (layer
normalization), **B-spline**, **Bézier**, **CARLA**, **GCS**, **IRIS**,
**FIRI**, **DGD**, **MPD** are all introduced once.

`Bézier` is written as `B\'ezier` throughout so the accent compiles without
`inputenc` issues.

## 2. Structural mapping

| Source | English manuscript |
|---|---|
| 摘要 / 关键词 | Abstract / Index Terms |
| 1 引言 | I. Introduction |
| 2 相关工作 | II. Related Work |
| 3 方法 | III. Method |
| 3.1 总体框架 | III-A Overall Framework |
| 3.2 特征编码与融合 | III-B Feature Encoding and Fusion |
| 3.2(1) 地图与控制点特征编码 | III-B1 Map and Control Point Feature Encoding |
| 3.2(2) 历史安全反馈融合 | III-B2 Historical Safety Feedback Fusion |
| 3.3 安全特征与约束生成 | III-C Safety Feature and Constraint Generation |
| 3.3(1) 通路匹配与选择 | III-C1 Route Matching and Selection |
| 3.3(2) 局部安全几何预测 | III-C2 Local Safety Geometry Prediction |
| 3.4 轨迹去噪与安全修正 | III-D Trajectory Denoising and Safety Correction |
| 3.4(1) 安全条件去噪与端点补偿 | III-D1 Safety-Conditioned Denoising and Endpoint Compensation |
| 3.4(2) 连续轨迹安全约束与增广拉格朗日修正 | III-D2 Continuous-Trajectory Safety Constraints and Augmented Lagrangian Correction |
| 3.4(3) 安全状态更新与历史反馈 | III-D3 Safety State Update and Historical Feedback |
| 3.5 训练策略与损失函数 | III-E Training Strategy and Loss Function |
| 3.5(1) 共享参数的两次前向训练 | III-E1 Two-Forward Training with Shared Parameters |
| 3.5(2) 损失函数 | III-E2 Loss Function |
| 4 实验与分析 | IV. Experiments and Analysis |
| 4.1 实验设置 (4.1.1 / 4.1.2) | IV-A Experimental Setup (IV-A1 / IV-A2) |
| 4.2 不同安全约束方法比较 (4.2.1 / 4.2.2) | IV-B Comparison of Different Safety Constraint Methods (IV-B1 / IV-B2) |
| 4.3 关键组成模块消融 | IV-C Ablation of Key Components |
| 4.4 轨迹生成可视化 (4.4.1 / 4.4.2) | IV-D Visualization of Trajectory Generation (IV-D1 / IV-D2) |
| 5 结论 | V. Conclusion |
| 参考文献 | References |

### Figure / table renumbering

| Source | Manuscript | File in `figures/` |
|---|---|---|
| 图 3-1 总体框架 | **Fig. 1** | `fig1_framework.png` (re-drawn) |
| 图 3-2 条件注意力与历史反馈门控融合 | **Fig. 2** | `fig2_feature_fusion.png` (re-drawn, portrait → single-column float) |
| 图 3-3(a) 候选通路匹配与选择 | **Fig. 3(a)** | `fig3a_route_selection.png` (re-drawn) |
| 图 3-3(b) 局部安全特征与椭圆几何预测 | **Fig. 3(b)** | `fig3b_safety_geometry.png` (re-drawn earlier) |
| 图 3-4(a) 安全条件融合与控制点预测 | **Fig. 4 (upper)** | `fig4_denoising.png` |
| 图 3-4(b) 单层条件去噪模块 | **Fig. 4 (lower)** | (same file — source merged the two) |
| 图 3-5 连续曲线安全约束与 ALM 修正 | **Fig. 5** | `fig5_alm_constraint.jpg` |
| 图 4-1 逆扩散过程中的轨迹预测与安全修正 | **Fig. 6** | `fig6_diffusion_step_correction.png` (re-plotted) |
| ~~图 4-2 不同地图与起终点距离下的轨迹生成结果~~ | **removed** | source image deleted |
| 表 4-1 | **Table I** | — |
| 表 4-2 | **Table II** | — |
| 表 4-3 | **Table III** | — |

The sub-figure pairing required by the task is preserved: 图 3-3(a)/(b) form
Fig. 3 with parts (a) and (b); 图 3-4(a)/(b) form Fig. 4 with parts (a) and (b).

### 2.1 Method section replacement (second revision)

Section III was re-translated from `方法论述_精简优化版.md` and spliced in,
replacing the previous Method text. Nothing outside Section III was touched
apart from the cross-references it already carried.

Effect on the manuscript:

| | Revision 1 (original Method) | Revision 2–4 (condensed Method) |
|---|---:|---:|
| Total pages | 14 | **10** |
| Numbered equations | 27 | **15** |
| Method column-height | 6 927 pt | **4 435 pt** (−36 %) |
| Method share of body text | 54.4 % | **45.4 %** |
| Total body-text column-height | 12 738 pt | **9 778 pt** (−23 %) |
| Figure raster area | 3.56 page-eq | **1.76 page-eq** (−51 %) |

Content changes deliberately made by the new source text (all of them are the
source's own edits, not the translator's):

* **No `L_coarse` any more.** Revision 1 listed an auxiliary coarse control point
  prediction loss; the new text drops it entirely. **The code still computes
  `Lcoarse` (`lambda_coarse = 0.5`, the largest weight in the whole objective)** —
  see §5.5.5. This is now the largest single paper-vs-code discrepancy.
* The loss discussion no longer gives formulas; `L_ctrl`, `L_base`, `L_fb,safe`,
  `L_fb` and the total-objective `L = L_base + ω_fb L_fb` equations are gone.
  Losses are now described qualitatively in three groups. The `ω_fb = 0.5`
  value is no longer stated either.
* The loss groups are regrouped to match the code's names: `L_topo`, `L_shape`,
  `L_IoU`, `L_safe`, `L_alm`, `L_fb,safe`, `L_curve`. The old paper names
  `L_route` / `L_occ` / `L_corr` are gone, which **removes** the naming mismatch
  reported in §5.5.5 of the previous revision.
* The Lagrange multipliers in the augmented Lagrangian are now `μ` instead of
  `λ`, and the objective is written as `J(Q − Q^ref)`. This **resolves** the
  `λ` symbol collision and the undefined `J(·)` reported in revision 1.
* The endpoint-compensation weights are now explicitly constrained to
  `w_0^S = w_{C-1}^G = 1`, `w_0^G = w_{C-1}^S = 0`. This **resolves** the
  under-specified endpoint condition reported in revision 1.
* The training section now states the 25 % feedback-masking probability, and
  explicitly says the feedback correction is set to zero when the raw prediction
  already satisfies the constraints and invalidated when the correction fails
  the check — both match the code (`drop_prob = 0.25`, `already_ok`, `fb_valid`).
* Three sub-figures are now referenced from a single sentence, and the denoising
  block is explicitly declared structurally identical to the conditional
  attention encoding instead of being re-described.

### 2.2 Second source revision (Rev 4)

`TrajSafe_paper_optimized.md` was rewritten again (52 089 → 42 134 bytes;
21:22). Compared against the version Rev 1–3 were built from:

**Method (§III).** The md now contains the condensed Method itself, so §III is
unchanged in the manuscript (it already matched).

**Experiments (§IV).** The only content changes in the whole paper:

* §4.4 renamed 轨迹生成可视化 → 轨迹生成与安全修正可视化, and its two
  sub-subsections collapsed into **one paragraph**. The manuscript's §IV-D
  was rewritten accordingly and ``\subsubsection`` level removed.
* **§4.4.2 (72 tasks across 4 maps, near/medium/far distances) was deleted**,
  together with its figure 图 4-2. Both the section and **Fig. 7** were removed
  from the manuscript and `figures/fig7_multitask_generation.png` deleted.
* §4.4.1 was shortened: the quoted violation values
  (13.16/7.44/8.11/8.90 → 0.79/0.19/0.20/0.08 m), the “no RMSE is computed”
  note and one “the discrete check is not a certificate” caveat are gone.
* Fig. 6's caption was re-worded (the corridor is no longer described as
  “subsequently activated and frozen”).
* Everything else — Abstract, Introduction, Related Work, §IV-A, §IV-B, §IV-C,
  Conclusion, all 22 references — was checked paragraph by paragraph and is
  identical.

**Figures.** 图 3-1 and 图 3-3(b) were re-drawn (both now English-only, and the
stale “See page 2 …” annotation is gone from 图 3-3(b)); 图 4-1 was replaced by
the single-row re-plot; 图 4-2 was deleted. 图 3-2, 图 3-4(a)(b) and 图 3-5 are
byte-identical to before.

**Two content-neutral typesetting changes** were needed to land on 10 pages
(see `layout_report.md` §2.3): Fig. 2 was narrowed from 0.78 to 0.74`\textwidth`,
and the missing in-text reference to Fig. 5 was restored — the condensed source
had dropped its “如图 3-5 所示”, leaving the figure uncited, which is an IEEE
style defect. The restored text is only “As shown in Fig.~\ref{fig:alm}, …”.

### 2.3 Third round: images re-drawn (Rev 5)

The markdown did not change again; only the images under
`论文/.../figures/png/` were re-exported:

| Source file | Change | Aspect |
|---|---|---|
| `fig3-1_framework.png` | re-drawn (1672×941 → 6292×2780) | 1.78 → **2.26** |
| `fig3-2_feature_fusion.png` | re-drawn (2048×1573 → 692×1611) | 1.30 → **0.43 — now portrait**, (a)/(b) stacked vertically |
| `fig3-3a_route_selection.png` | re-drawn (2048×721 → 4416×2580) | 2.84 → **1.71** |
| `fig3-4a_denoising.png` + `fig3-4b_denoising_block.png` | **merged into one file** `fig3-4_denoising.png` (5652×3200) | — → **1.77** |
| `fig3-3b_safety_geometry.png`, `fig3-5_alm.jpg`, `case_293_*.png` | unchanged | — |

Note: the markdown still references `fig3-4a_denoising.png` and
`fig3-4b_denoising_block.png` at lines 147 and 153, but those files no longer
exist. Because the two panels are now one image, the manuscript treats them as a
**single Fig. 4**, and the two source captions were merged into one
(`Upper: … Lower: …`). The in-text “Fig. 4(a)/(b)” references became
“the upper/lower part of Fig. 4”.

Layout consequences (all content-neutral):

* **Fig. 2 became a single-column float** (`0.95\columnwidth`, 239 × 555 pt).
  A portrait figure costs `2 × height` of column space when anchored across both
  columns but only `1 × height` in a single column, so this saved roughly half
  the space and freed a double-column float slot.
* Fig. 3(a)(b) narrowed from 0.85 to 0.75`\textwidth`, absorbing the extra
  height that Fig. 3(a)'s new, squarer aspect introduced.
* **Fig. 3(b) had lost its in-text reference** in the condensed source (the
  older revision read “其中通路匹配与局部几何预测分别如图 3-3(a) 和图 3-3(b)
  所示”). One purely cross-referential sentence was restored — the same
  treatment Fig. 5 received in Rev 4 — and Fig. 2 was tightened to
  `0.95\columnwidth` to absorb the two extra lines. Net: still 10 pages.


`IEEEtran.bst` numbers entries in order of first citation. To keep the
manuscript's citation numbers identical to the source manuscript (so the two
can be cross-checked 1:1), the 22 keys are `\nocite`d in exactly the source
order immediately after `\maketitle`. The resulting mapping is **identical to
the Chinese original**:

`[1]` LaValle · `[2]` Karaman · `[3]` Zucker · `[4]` Ho · `[5]` Janner ·
`[6]` Huang · `[7]` Carvalho · `[8]` Ames · `[9]` Xiao · `[10]` Christopher ·
`[11]` Liang (SMD) · `[12]` Deits · `[13]` Wang · `[14]` Marcucci ·
`[15]` Liang (DGD) · `[16]` Song · `[17]` Chen · `[18]` Piegl · `[19]` Peebles ·
`[20]` Vaswani · `[21]` Nocedal · `[22]` Dosovitskiy.

In-text citations resolve to the same numbers as in the source: Diffuser `[5]`,
SceneDiffuser `[6]`, MPD `[7]`, SafeDiffuser `[9]`, Projected Diffusion Models
`[10]`, SMD `[11]`, IRIS `[12]`, FIRI `[13]`, GCS `[14]`, DGD `[15]`,
CARLA `[22]`.

### Bibliography hygiene

* The Markdown annotations such as `[作者公开版本]`, `[会议正式记录]`,
  `[作者提供的书籍页面]` were **not** reproduced as reference data, as required.
* Every DOI that appears in the source is retained; **no DOI was invented**.
* No URL-only source was promoted into the bibliography as a DOI.
* Author lists were completed only from the information already in the source
  (the source used `et al.`; the full author lists of the cited works were not
  re-derived from the web, so `et al.` was expanded only where the source
  already listed the names). Where a work is normally cited with more authors
  than the source lists, the entry keeps the source's author set.

**Open item (not fixed):** 11 of the 22 entries are not cited anywhere in the
body text (`[1]`–`[4]`, `[8]`, `[16]`–`[21]`). They are retained in the
bibliography and reported, not silently removed or force-cited. See §5 below.

## 4. One deliberate, minimal editorial insertion

The source text of §4.2.2 never cross-references 表 4-2 (Table II). A single
mechanical cross-reference was added:

> “The results are reported in Table II.”

This adds **no** technical content, data, claim or conclusion; it only removes
an orphaned float. All other in-text `\ref`/`\cite` commands reproduce
cross-references that already exist in the source.

## 5. Issues found in the source — reported, NOT modified

Per the task rules, no algorithm mechanism, symbol, number or conclusion was
altered. The following problems were found and are recorded here for the
authors to decide on.

### 5.1 Uncited references (11 of 22)

`[1]` LaValle, `[2]` Karaman & Frazzoli, `[3]` Zucker *et al.* (CHOMP),
`[4]` Ho *et al.* (DDPM), `[8]` Ames *et al.* (CBF), `[16]` Song *et al.*
(DDIM), `[17]` Chen *et al.* (Analog Bits), `[18]` Piegl & Tiller (NURBS),
`[19]` Peebles & Xie (DiT), `[20]` Vaswani *et al.* (Attention),
`[21]` Nocedal & Wright (Numerical Optimization).

An earlier revision of the same manuscript
(`历史版本/TrajSafe_paper_optimized_20261009_before_protocol_change.md`) cited
`[1]`,`[2]`,`[3]` in the introduction, `[4]` for DDPM, `[8]` for CBF, `[16]`
for the DDIM update, `[18]` for the B-spline/Bézier convex hull and `[21]` for
the augmented Lagrangian function. The citations were lost during later
condensation. `[17]`, `[19]`, `[20]` were uncited even then.
*Recommendation:* restore `[16]` at eq. (21), `[18]` at eqs. (1) and (17),
`[21]` at eq. (19), `[4]` at eq. (2), and `[1]`–`[3]`, `[8]` in the
introduction / related work; drop or cite `[17]`, `[19]`, `[20]`.
This is a decision for the authors and was not taken here.

### 5.2 In-image Chinese text (blocks an English submission) — RESOLVED

* **Fig. 6** — **fixed.** Re-plotted by `scripts/plot_dashboard_case293.py`
  with English labels and the violation/correction values moved into the title
  band above the panels (`Step 1 (warm-up)`, `violation 17.35 → 0.00 m`,
  `Pre-ALM prediction`, `Safe trajectory at this step`, `Continuous safe
  corridor`, `Re-specified start`, `Re-specified goal`).
* **Fig. 3(b)** — **fixed by the source re-draw**; it is now English-only and
  the stale cross-reference is gone.
* **Fig. 1** — **fixed by the source re-draw**; English-only.
* **Former Fig. 7** (`random_endpoints_seed20260927_c.png`, all-Chinese labels)
  — **no longer in the paper**: the source deleted both the figure and §4.4.2.

No figure in the current manuscript contains Chinese text.

### 5.3 Numeric inconsistency between §4.4.1 and Fig. 6 — MOOT

In Rev 1–3 the body text quoted violation values (13.16, 7.44, 8.11, 8.90 m →
0.79, 0.19, 0.20, 0.08 m) that could not both describe the same run as Fig. 6's
own annotations (17.35 → 0.00, 17.88 → 0.00, 19.08 → 0.29, 19.59 → 0.00 m).

**The revised source removed the quoted numbers from §4.4.1 entirely**, so the
contradiction no longer appears in the manuscript. Fig. 6's annotations remain
the only numeric statement for that case. Worth keeping in mind if the numbers
are ever re-introduced into the text.

### 5.4 Stale cross-reference inside Fig. 3(b) — FIXED

The old `fig3b_safety_geometry.png` contained “*See page 2 for position encoding
and the complete formulation.*” The source re-drew this figure; the annotation
is gone.

### 5.5 Symbol collisions and under-specified notation

Recorded for the authors; nothing was changed. Items 2, 3 and 4 of revision 1
were **fixed by the new source text** and are kept here only as a record; the
equation numbers below refer to the current (10-page) manuscript.

1. **`b_i` STILL has two meanings (open).** Eq. (8) uses
   `e_i = (log a_i, log b_i, …)` for the second semi-axis of the safety ellipse,
   while eq. (9) uses `R_i = {x | A_i x ≤ b_i}` for the boundary offset of the
   convex region.
2. ~~`λ` has two meanings.~~ **Fixed.** The multipliers in the augmented
   Lagrangian are now `μ`, so `λ_p`, `λ_s` (objective weights) no longer collide
   with the multiplier vector.
3. ~~`J(·)` is never defined.~~ **Fixed.** The ALM problem now writes
   `J(Q − Q^ref) = (λ_p/2)‖B(Q−Q^ref)‖²_F + (λ_s/2)‖D₂B(Q−Q^ref)‖²_F`, so `J`
   is defined by the equation itself.
4. ~~Endpoint-compensation weights are under-specified.~~ **Fixed.** The text
   after eq. (10) now states `w_0^S = w_{C-1}^G = 1` and
   `w_0^G = w_{C-1}^S = 0`.
5. **Loss terms are named but not defined.** In the current text `L_ctrl`,
   `L_boundary`, `L_smooth`, `L_topo`, `L_shape`, `L_IoU`, `L_safe`, `L_alm`,
   `L_fb,safe` and `L_curve` all appear only as names with prose descriptions;
   no formula and no weight value is given for any of them. This is weaker than
   revision 1, which at least gave `L_ctrl`, `L_fb,safe` and a coefficient list
   `w_c, w_a, …` (though never their values). **`L_coarse` has disappeared from
   the paper entirely while remaining the largest term in the code's objective
   (`lambda_coarse = 0.5`).**
6. **`ᾱ_t` inside the spatial distance bias.** Eq. (7) multiplies a squared
   Euclidean distance in metres by the dimensionless diffusion coefficient
   `ᾱ_t`. The text explains the intended effect (the bias strengthens as the
   reverse process proceeds), but the dimensional role of `ᾱ_t` and the
   meaning/units of `σ_geo` and `b_clip` are not given.
7. **Index range in eq. (2).** The equation is written for general `j` while
   the following text restricts it to `j = 1, …, C−2`; the internal/external
   split is only stated in prose.
8. **The two-forward weighting is not stated.** The code applies
   `step2_weight = 0.5` to the three control-space terms of the second forward
   pass only (not to `L_fb,safe` / `L_curve`), and the whole stage-1 + stage-2
   combination is not expressed as an equation any more.

### 5.6 Consistency checks that passed

* 0.625 m/grid × 256 grid = 160 m, matching the stated 160 m × 160 m scenes.
* 16 × 16 = 256 = `C_G` tokens; 32 × 32 = 1024 = `C_E` tokens.
* `N_q = 128` safety centers and `c_i = Γ*(i/(N_q−1))` = `Γ*(i/127)`, matching
  the annotation inside Fig. 3(b).
* 3.33 % of 1260 ≈ 42 collisions, matching “the number of collisions drops from
  42 to zero” in §IV-C.
* Table I, Table II and Table III values were transcribed digit-for-digit from
  the source Markdown.

## 6. What was NOT done

* No figure, table, reference, experiment or conclusion was removed **by this
  translation**. Two things did disappear from the manuscript because the
  **source** deleted them: §4.4.2 (the 72-task, four-map study) and its figure
  图 4-2. The source image file itself is gone from
  `论文/.../figures/png/`, so nothing was lost locally either.
* Section III of revision 1 was **replaced on request** by the translation of
  `方法论述_精简优化版.md`. That replacement removed 12 numbered equations, the
  `L_coarse` term and several explanatory sentences — **all of them because the
  new source text does not contain them**, not because the translator trimmed
  the paper. The superseded Section III is preserved in
  `backup/main_v1_original_method.tex.bak`.
* No experimental value was rounded, recomputed or supplemented.
* No conclusion, caveat or limitation was softened. (One limitation sentence in
  §IV-D and the “no RMSE for this case” note were dropped because the source
  dropped them.)
* No figure was re-drawn, re-coloured, cropped, simplified or re-encoded **by
  this translation**. Fig. 1, Fig. 3(b) and Fig. 4-1 were re-drawn by the source
  / by the project's own plotting script, and the new files were copied in
  byte-for-byte.
* No algorithm mechanism, symbol or notation was “fixed” by the translator (all
  issues above are only reported; the notation fixes in §5.5 items 2–4 came from
  the new source text).
* No new citation, claim or theoretical guarantee was introduced.
* No template parameter (margins, body font, leading, paragraph spacing,
  reference font, float separation) was modified to reduce page count. Two
  things were adjusted to reach 10 pages, both content-neutral and documented in
  `layout_report.md` §2.3: Fig. 2's width (0.78 → 0.74`\textwidth`) and one
  restored in-text reference to Fig. 5. Two display equations were re-broken
  across two lines (`aligned`) to remove overfull boxes; no term was deleted or
  altered.
