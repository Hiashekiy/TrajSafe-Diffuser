# 中文 md / Word 版说明

## 文件清单

| 文件 | 大小 | 说明 |
|---|---|---|
| `TrajSafe_methods_and_figures/TrajSafe_paper_optimized.md` | 42 KB | **中文原稿，未改动**（只读，一切以此为源） |
| `TrajSafe_methods_and_figures/TrajSafe_paper_optimized_synced.md` | 43 KB | **同步版**（由脚本从原稿生成，补了 3 处图引用差异） |
| `TrajSafe-Diffuser_中文版.docx` | 13.5 MB | Word 忠实版（图片原分辨率） |
| `TrajSafe-Diffuser_中文版_轻量.docx` | 1.2 MB | Word 轻量版（图片 220 dpi），正文完全相同 |

> ✅ 两份 Word 均已由 **synced 版** md 生成，文件名即最新版，无遗留 `_new` 文件。

## 一、同步版 md 做了什么

脚本：`scripts/paper/sync_zh_md.py`（**只读原稿**，原稿 mtime 保持 2026-10-10 21:22 不变）

```powershell
python scripts\paper\sync_zh_md.py
```

它把 Word / 英文 RA-L 版里已经修好、但没回写到 md 的差异补上：

| # | 差异 | 原稿 | 同步版 |
|---|---|---|---|
| 1 | **图 3-4** | 第 147、153 行分别引用 `fig3-4a_denoising.png`、`fig3-4b_denoising_block.png`——**这两个文件已被删除**（源图合并为 `fig3-4_denoising.png`） | 换成单张 `fig3-4_denoising.png`，两条图题合并为一条；正文 "图 3-4(a)/(b)" 改为 "图 3-4 上半部分/下半部分" |
| 2 | **图 3-3(b)** | 3.3 节无引言句，图 3-3(b) 无正文引用（只有 (a) 在正文里被提到） | 3.3 节标题下补回旧版原句：「本节中，通路匹配与局部几何预测分别如图 3-3(a) 和图 3-3(b) 所示。」 |
| 3 | **图 3-5** | "如图 3-5 所示" 被精简版删掉，图 3-5 成为无引用孤立图 | ALM 求解句前补回「如图 3-5 所示，」 |

**没有改动任何技术内容、公式、数据、图片或结论。** 差异就是这 6 行（4 行删 + 4 行改）。
新文件顶部有一段 `<!-- ... -->` HTML 注释记录这三条差异，Markdown 渲染时不显示。

同步后的引用自检（正文提及次数，不含图题）：

```
图 3-1: 1   图 3-2: 2   图 3-3(a): 2   图 3-3(b): 2
图 3-4: 5   图 3-5: 3   图 4-1: 1
```

全部 6 个图片文件都存在，无失效引用。

## 二、Word 版怎么生成

脚本：`scripts/paper/make_zh_docx.py`

环境里没有 pandoc、LibreOffice、python-docx（pip 也被沙箱拦住），所以脚本
**直接按 OOXML 手写 docx 包**（zip + XML），只依赖标准库和 Pillow。

```powershell
# 忠实版（原分辨率）——默认读 synced 版，没有则回退到原稿
python scripts\paper\make_zh_docx.py

# 轻量版（限 220 dpi）
python scripts\paper\make_zh_docx.py "论文\TrajSafe-Diffuser_中文版_轻量.docx" --dpi 220

# 指定源文件
python scripts\paper\make_zh_docx.py --src <某.md>
```

当前两份 Word 均由 **synced 版**生成：20 个标题、296 段、**7 张图**、3 张表、22 条参考文献；
行内公式 107 处 + 独立公式 15 组全部转换，无残留 LaTeX / `$` / `**`。

### 图片在 Word 中的尺寸

| 图 | 原图 | Word 中 |
|---|---|---|
| 图 3-1 总体框架 | 6292×2780 | 15.8 × 7.0 cm |
| 图 3-2 条件注意力与反馈门控 | 692×1611（竖版） | 9.0 × 21.0 cm |
| 图 3-3(a) 通路匹配 | 4416×2580 | 15.8 × 9.2 cm |
| 图 3-3(b) 安全几何 | 6584×2312 | 15.8 × 5.6 cm |
| 图 3-4 去噪与预测（合并图） | 5652×3200 | 15.8 × 9.0 cm |
| 图 3-5 ALM 修正示意 | 2048×1048 | 15.8 × 8.1 cm |
| 图 4-1 逐步修正 | 3684×992 | 15.8 × 4.3 cm |

版面：A4，页边距 2.5 cm，正文中文宋体 / 西文 Times New Roman 五号。

## 三、重要提醒：公式是**近似**转换

Word 不执行 LaTeX，脚本把公式转成 Unicode 纯文本：

```
$\bar\alpha_t$            ->  ᾱₜ
$\mathcal N(0,I)$         ->  𝒩(0,I)
$q_i^\top k_j/\sqrt{d_h}$ ->  qᵢᵀ kⱼ/√(dₕ)
$\frac{a}{b}$             ->  (a)/(b)
$H_\mathrm{ctrl}$         ->  H_(ctrl)      <- 下标映射不到 Unicode 时退化为括号
```

能映射到 Unicode 的上/下标用 Unicode（₀₁₂… ᵀ ⁽ᵗ⁾），映射不到的退化为 `_(…)` / `^(…)`。
分式写成 `(分子)/(分母)`，`aligned` 多行公式按行拆成多段居中公式。

**Word 版适合阅读、评审、批注；公式的权威版本仍是 LaTeX/PDF。**
若要 Word 里是**可编辑的 Word 公式对象**，需要 LaTeX -> OMML，是另一套工作。

## 四、已知限制

* 本机 Word 的 COM 自动化被沙箱拒绝（`CO_E_SERVER_EXEC_FAILURE`），
  **没能用 Word 实际打开验证**。已做的校验：zip 结构完整、5 个 XML/rels 部件
  全部 well-formed、图片关系全部解析、7 个 `<w:drawing>`、3 个 `<w:tbl>`、
  无残留标记。若打开时 Word 提示修复，请告知。
* 未插入页码、页眉页脚。
* `make_zh_docx.py` 会跳过 md 顶部的 `<!-- ... -->` 注释，不会渲染进文档。
