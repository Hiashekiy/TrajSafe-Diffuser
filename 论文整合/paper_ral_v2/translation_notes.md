# `paper_ral_v2` — 英文翻译与排版说明

**源稿**：`论文/TrajSafe_methods_and_figures/基于生成式安全约束与闭环修正的扩散模型轨迹规划方法.md`
（29 867 字节 / 300 行，2026-10-23 19:38 版 + 集合记号修正；只读，未改动）

**产物**：`paper_ral_v2/output/TrajSafe_Diffuser_v2_EN.pdf` —— **6 页，无图，3 表**

---

## 0. 当前状态（两轮工作）

| 轮次 | 内容 | 结果 |
|---|---|---|
| 第 1 轮 | §摘要–§3.4 逐段重译（源稿段 1–45） | 5 页 |
| 第 2 轮 | **新增 §IV 实验与结果分析**（源稿第 218–300 行，§4.1–§4.5） | **6 页** |

两轮均**按源稿段落顺序逐段翻译**，每段前用注释标注对应源稿位置
（`% ---------- 源稿 [4.3.1] 段 2 ----------`），便于逐条追溯。

### 逐段核对结果

第 1 轮（§摘要–§3.4）：按段逐条对照，段落数与原稿一致（记录见本节历史版本）。

第 2 轮（§4）——本轮脚本实测：

```
§4 源稿正文段落 = 14   译文正文段落 = 14   => ✅ 一一对应（4.1:3 / 4.2:3 / 4.3.1:2 / 4.3.2:2 / 4.4:3 / 4.5:1）
§4 源稿表格     =  3   译文表格     =  3   => ✅ Table I / II / III
§4 源稿图题     =  2   译文图题     =  2   => 以 LaTeX 注释保留（本版不带图）
数值核对：源稿全部数字片段（105 个不同数值）在译文 §IV 中全部出现，无遗漏、无改写
```

核对脚本先把源稿中的 `$$` 公式替换成段分隔符（源稿存在公式与正文**贴行**的情况），
因此不会漏算公式前后的文字段。

---

## 1. 源稿结构与本版对应关系

| 源稿 | 段落数 | 译文位置 |
|---|---:|---|
| 摘要 | 1 | Abstract |
| 关键词 | 1 | IEEEkeywords |
| 1、引言 | 5 + 1（贡献引导）+ 3（贡献条目） | I. Introduction |
| 2、问题定义 | 3（公式前/后各成段） | II. Problem Definition |
| 3、方法（引言） | **1** | III. Method |
| 3.1(1) 地图与控制点特征编码 | 3 | III-A1 |
| 3.1(2) 历史安全反馈融合 | 3 | III-A2 |
| 3.2(1) 候选骨架通路匹配与选择 | 2 | III-B1 |
| 3.2(2) 局部安全特征提取与凸安全走廊生成 | 5 | III-B2 |
| 3.3(1) 安全条件去噪 | 2 | III-C1 |
| 3.3(2) 连续轨迹安全约束与 ALM 修正 | 9 | III-C2 |
| 3.3(3) 逆扩散更新与历史安全反馈 | 3 | III-C3 |
| 3.4 训练策略与损失函数 | 3 + 表 1 | III-D + Table I |
| 4.1 实验设置 | 3 | IV-A |
| 4.2 不同安全规划方法的性能比较 | 3 + 表 1 | IV-B + Table I |
| 4.3.1 关键模块消融 | 2 + 表 3 | IV-C1 + Table II |
| 4.3.2 历史安全反馈对去噪过程的影响 | 2 + 图 1 题 | IV-C2 |
| 4.4 ALM 迭代预算分析 | 3 + 表 4 | IV-D + Table III |
| 4.5 CARLA 道路场景轨迹生成 | 1 + 图 4-6 题 | IV-E |

`（1）（2）（3）` 三级条目按 IEEE 习惯排成 `\subsubsection`。

### 表号重编（源稿表号不连续）

中文源稿的 `表 1 / 表 3 / 表 4` 是全文编号，`表 2`（效率对比）不在本版译入范围内
（作者决定：严格按指定 md，不补 §4.2.2）。译入后按 IEEE 顺序编号为
**Table I / II / III**，与源稿的对应关系见表内 `\label`：

| 源稿 | 本版 | `\label` |
|---|---|---|
| 表 1 不同安全规划方法的性能比较 | Table I | `tab:methods` |
| 表 3 关键模块消融结果 | Table II | `tab:ablation` |
| 表 4 不同 ALM 迭代预算下的安全性与轨迹质量 | Table III | `tab:budget` |

---

## 2. 排版结构

| 项目 | 第 1 轮 | 当前 |
|---|---:|---:|
| `\section` | 3 | **4** |
| `\subsection` | 4 | **9** |
| `\subsubsection` | 7 | **9** |
| 编号公式 | 13 | 13 |
| 表 | 1 | **3** |
| 图 | 0 | 0 |
| 英文词数（PDF 抽取） | ≈ 3 480 | **≈ 4 820** |

* Table I、Table III 为 `table*`（通栏）；Table II、Table III 落地后经调整，
  Table III 改为单栏 `\scriptsize` 两行表头，避免整页只剩一张表。
* 编译日志：**0 error、0 overfull、1 underfull**（§IV-D 第 2 段，badness 2096，无害）。

---

## 3. 篇幅实测

| 章节 | 页码 | 估计占用 |
|---|---|---:|
| 标题 / 摘要 / I–II | 1–2 | 1.2 页 |
| III. Method（A–D） | 2–5 | 2.8 页 |
| **IV. Experiments and Results（A–E）** | **5–6** | **1.6 页** |
| 合计 | 6 页 | 约 5.9 页（末页接近排满） |

---

## 4. 与源稿的版式偏差（均因"本版不带图"）

| 项 | 处理 |
|---|---|
| 图 1（方法总框架） | 源稿「整体结构如图 1 所示」→ 直接以 "The overall structure … comprises three modules" 起句 |
| 图 3（ALM 修正） | 源稿「如图 3 所示…」→ 直接以 "The augmented Lagrangian method (ALM) is used to…" 起句 |
| **图 1（§4.3.2 反馈过程）** | 源稿「如图 1 所示」→ 改为直接叙述被比较的两个量；图题以注释保存在 `main.tex` 末尾附近 |
| **图 4-6（CARLA 场景）** | 源稿「图 4-6 展示了…」→ 改为 "In the planning results, …"；图题同样以注释保留 |

4 条图题仍保留在中文源稿与 `main.tex` 注释中，将来插图时无需找回。

---

## 5. 仍未解决（需作者决定）

1. **没有任何参考文献**。引言点名 Diffuser、MPD、SafeDiffuser、Projected Diffusion Models，
   并涉及 CBF、DDIM、self-conditioning，全文 `\cite` 为 0。IEEE 稿件必须补。
   `paper_ral_english/references.bib` 已有现成条目：`janner2022diffuser`、`carvalho2025mpd`、
   `xiao2025safediffuser`、`christopher2024projected`、`ames2017cbf`、`song2021ddim`、
   `ho2020ddpm`、`deits2015iris`、`piegl1997nurbs`、`nocedal2006numerical`、`dosovitskiy2017carla`。
2. **仍是不完整稿**：没有相关工作、结论；`\cite` 为 0 时 §IV 也无法与外部工作对比。
   按同一版式估算，补齐后约 **9～10 页**（无图）/ **10～11 页**（带图），
   超出 RA-L 6 页基准，需压缩或走 overlength 费用。
3. **§4.2.2 计算效率（源稿表 2）未译入**——作者本轮决定严格按指定 md。
   内容已备好：`实验/results/exp1/table2_efficiency.md`、`实验/results/paper/PAPER_TABLES.md`
   （82.26 / 51.41 / 56.44 / 27.14 ms，加速 3.03×）。补入后表号需整体后移一位。
4. **§4.5 的场景清单与图件不一致**：正文写"直道路段 / 交叉路口 / 弯道路段"（3 类），
   而 `论文/TrajSafe_methods_and_figures/仿真图片/` 里是两张 2×2 图
   （图4-6 长距离 1/2 + 直行 1/2；图4-7 转弯 1/2 + 弯道 1/2）。插图前需统一。
5. **§4 数值口径**：译文严格照抄源稿数字，未做重算。与
   `实验/results/paper/PAPER_TABLES.md` 一致（如 61.19 / 25.16 / 3.33 / 0.00，
   1260 = 420×3）。若改用 `新版主方法实验更新清单.md` 里的新权重/新配置，全表需重跑。

---

## 6. 编译

```powershell
cd paper_ral_v2
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex     # 两趟；无参考文献，不需要 bibtex
```

> `main.pdf` 被 PDF 阅读器占用时编译会失败（`I can't write on file`），
> 关掉阅读器或改用 `-jobname=v5`。成品见 `output/TrajSafe_Diffuser_v2_EN.pdf`。
