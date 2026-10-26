# TrajSafe-Diffuser — English RA-L manuscript project

English translation and IEEE RA-L two-column typesetting of

* `论文/TrajSafe_methods_and_figures/TrajSafe_paper_optimized.md` — Abstract,
  Introduction, Related Work, Experiments, Conclusion, References
* `论文/TrajSafe_methods_and_figures/方法论述_精简优化版.md` — **Section III
  (Method)**, which replaced the earlier Method translation

The Chinese source manuscripts were **not** modified.

---

## 1. Directory layout

```text
paper_ral_english/
├── main.tex                                   full English manuscript (LaTeX)
├── references.bib                             22 BibTeX entries
├── figures/                                   9 figure files copied from the source
│   ├── fig1_framework.png                     ← fig3-1_framework.png (re-drawn)
│   ├── fig2_feature_fusion.png                ← fig3-2_feature_fusion.png (re-drawn,
│   │                                            now PORTRAIT → single-column float)
│   ├── fig3a_route_selection.png              ← fig3-3a_route_selection.png (re-drawn)
│   ├── fig3b_safety_geometry.png              ← fig3-3b_safety_geometry.png
│   ├── fig4_denoising.png                     ← fig3-4_denoising.png
│   │                                            (source merged 3-4(a)+(b) into one file)
│   ├── fig5_alm_constraint.jpg                ← fig3-5_alm.jpg
│   └── fig6_diffusion_step_correction.png     ← case_293_custom_endpoints_alm.png
│                                                (re-plotted single-row)
├── output/
│   └── TrajSafe_Diffuser_RAL.pdf              compiled PDF (10 pages)
├── backup/
│   └── main_v1_original_method.tex.bak        previous 14-page manuscript
│                                              (Method from TrajSafe_paper_optimized.md)
├── translation_notes.md                       translation + terminology + issue log
├── layout_report.md                           real page count, section budgets, compression plan
├── main.aux / main.bbl / main.blg / main.log  build artefacts
└── README.md                                  this file
```

All figure files are **byte-identical copies** of the originals
(SHA-256 verified at copy time). No figure was re-drawn, re-coloured,
cropped, re-encoded or otherwise modified.

---

## 2. Template

| Item | Value |
|---|---|
| Document class | `IEEEtran` V1.8b (2015/08/26), `\documentclass[journal]{IEEEtran}` |
| Class file | `texmf-dist/tex/latex/ieeetran/IEEEtran.cls` |
| Bibliography style | `IEEEtran.bst` V1.14 (2015/08/26) |
| Layout | IEEE journal two-column, US Letter, 612 × 792 pt |

IEEE RA-L does not ship a separate `.cls`. RA-L manuscripts are prepared with the
standard IEEE journal template (`IEEEtran`, journal mode) distributed by the IEEE
Author Center; `IEEEtran.cls` in *journal* mode therefore **is** the official RA-L
page layout. Page geometry, body font (10 pt Times), leading, column width and
margins are the class defaults and were not overridden anywhere in `main.tex`.

The only class-level parameters touched are float pacing inside individual
`figure*` / `table*` environments, plus `\footnotesize` + `\tabcolsep` inside
`tabular` bodies so that the three wide experiment tables fit inside the column
width. No page-geometry, body-font, line-spacing or paragraph-spacing parameter
was changed.

### Packages used

`amsmath`, `amssymb`, `amsfonts`, `graphicx`, `booktabs`, `array`, `multirow`,
`url`, `hyperref` (with `hidelinks`).

### Package availability note

The local TeX Live 2026 installation (TinyTeX) does **not** contain
`subfig`, `subcaption`, `caption`, `cite`, `threeparttable`, `lineno`,
`cleveref`, `soul` or `mathtools`, and `tlmgr` cannot run in the current
sandbox (registry write denied), so no package could be installed.

Consequences and work-arounds:

* `\usepackage{cite}` is omitted — with `IEEEtran.bst` this only means that
  adjacent citations are not compressed into `[5]–[7]` ranges. Everything else
  is identical.
* `subfig`/`subcaption` are unavailable, so the two composite figures
  (**Fig. 3** = source 图 3-3(a)+(b) and **Fig. 4** = source 图 3-4(a)+(b)) are
  built by stacking the two original images inside one `figure*` and labelling
  the parts `(a)` / `(b)` in the caption. This is a standard IEEE-style
  composite figure; the sub-figure relationship required by the task is
  preserved and no image is lost. Switching to `\subfloat` later only requires
  adding `\usepackage{subfig}` and wrapping the two `\includegraphics` calls.

---

## 3. Build

```powershell
cd paper_ral_english
pdflatex -interaction=nonstopmode main.tex
bibtex   main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
copy main.pdf output\TrajSafe_Diffuser_RAL.pdf
```

Verified with pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026) on Windows.
`latexmk -pdf main.tex` works equally well. All figure paths are
project-relative (`\graphicspath{{figures/}}`), so the directory is
self-contained.

Build result: **`output/TrajSafe_Diffuser_RAL.pdf`, 10 pages, 15 numbered
equations, 7 figures (7 raster images: 4 single-column, 3 double-column),
3 tables, 22 references; 0 undefined references, 0 undefined citations,
0 overfull boxes.**

> **Fig. 4 note.** At single-column width the safety-geometry diagram became
> 251 × 88 pt with ~2 pt labels, i.e. unreadable. It has been restored to a
> double-column `figure*` (437 × 153 pt) and is legible again. See
> `layout_report.md` §7 for the per-figure legibility check.

Revision history:

| | Source | Method source | Figures | Pages |
|---|---|---|---|---:|
| Rev 1 | `TrajSafe_paper_optimized.md` (old) | old md | `\textwidth` / `0.95\textwidth` | 14 |
| Rev 2 | + `方法论述_精简优化版.md` | 精简优化版 | unchanged | 12 |
| Rev 3 | same | 精简优化版 | shrunk, Fig. 6 re-plotted | 11 |
| **Rev 4** | **`TrajSafe_paper_optimized.md` (updated 21:22)** | same (now built into the md) | **new Fig. 1 / Fig. 3(b), Fig. 4-2 dropped** | **10** |
| **Rev 5** | same md, **figures re-drawn again** | **Fig. 1 / 3-2 / 3-3(a) re-drawn; Fig. 3-4(a)+(b) merged into one file** | **10** |
| **Rev 6** | same md, figures unchanged | **Fig. 3(a)(b) split into two separate figures (now Fig. 3 and Fig. 4); Fig. 3 and Fig. 5 → single-column** | **10** |
| **Rev 7** | same md, figures unchanged | **Fig. 2 shrunk to 0.70`\columnwidth`; Fig. 4 restored to double-column** | **10** |

Rev 1 is archived as `backup/main_v1_original_method.tex.bak`. See
`layout_report.md` §2 for the Rev 5 image diff and §1/§4 for the measurements.

### Figure sizes

Every figure width is a single explicit `\includegraphics[width=...]` argument
so it can be tuned in one place. Current widths: Fig. 1 `0.85\textwidth`,
Fig. 2 `0.70\columnwidth`, **Fig. 3 `0.90\columnwidth`**, Fig. 4 `0.85\textwidth`,
Fig. 5 `0.85\textwidth`, Fig. 6 `\columnwidth`, Fig. 7 `\textwidth`. The figures
were consuming 3.56 page-equivalents at Rev 2; they now consume 1.67. All seven
figures were visually verified as legible at 150 dpi.

Fig. 3 and Fig. 5 come from source images that were re-drawn as **portrait /
near-square** layouts (aspect 1.71 -> 0.44 and 1.77 -> 1.60), so both are
rendered noticeably larger per column than before. `build_paper.bat` copies the
current source images in on every run — see `scripts/paper/sync_figures.py`.

### Fig. 6 (case 293) is generated by a script

`scripts/plot_dashboard_case293.py` re-plots Fig. 6 from the Diffusion Lens
cache. Its layout is **one row of five panels, with the step number and the
violation/correction values in a title band above the panels, and one legend row
underneath** — no text is drawn over the maps. Its labels are English
(`LANG = "en"`; set `LANG = "zh"` for the Chinese strings, which are kept in the
same `LABELS` dict).

```powershell
python scripts/plot_dashboard_case293.py                      # canonical path
python scripts/plot_dashboard_case293.py <some\other\path.png> # explicit output
```

**The canonical output path could not be overwritten from this session.** Writing
to `outputs/chapter4/figures/cases/case_293_custom_endpoints_alm.png` is refused with
`access denied` even though every sibling file, the parent directory, and file
creation in that directory all work, and read access succeeds. The bundled
ACL diagnostic classified it as `PRECONDITION` (no effective `WRITE_DAC` for the
caller) and explicitly does not authorise a repair, so no ACL was touched. The
one-row figure is therefore written to
`outputs/chapter4/figures/cases/case_293_custom_endpoints_alm.onerow.png` and copied
into `figures/fig6_diffusion_step_correction.png`. The original two-row image is
preserved as `case_293_custom_endpoints_alm.tworow.bak.png`. Once the object's
permissions are reviewed (or the file is deleted by the user), the script writes
the canonical path unchanged.

---

## 4. Placeholder that must be filled in before submission

The Chinese source contains **no author, affiliation, funding or contact
information**. The `\author{}` block therefore carries explicit placeholders
(`First A. Author`, `Second B. Author`, `Department of XXXX`) plus a visible
`\thanks` note. These must be replaced with the real data.
