#!/usr/bin/env python
"""把中文论文 Markdown 生成为**带图片**的 Word (.docx)。

环境里没有 pandoc / LibreOffice / python-docx，所以在标准库范围内直接按
OOXML 写出 docx 包（zip + XML）。公式按 LaTeX -> Unicode 尽力转换；转换是
近似，权威版本仍是 LaTeX/PDF。

用法：
    python scripts/paper/make_zh_docx.py [输出路径]
"""

from __future__ import annotations

import io
import os
import re
import sys
import zipfile
from pathlib import Path

from PIL import Image
from xml.sax.saxutils import escape as xesc


ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "论文" / "TrajSafe_methods_and_figures" / "figures" / "png"
SRC_ORIG = ROOT / "论文" / "TrajSafe_methods_and_figures" / "TrajSafe_paper_optimized.md"
SRC_SYNCED = (ROOT / "论文" / "TrajSafe_methods_and_figures"
              / "TrajSafe_paper_optimized_synced.md")
# 优先用同步版（图形引用已修好）；没有就用原稿（届时走下面的合并逻辑兜底）
SRC = SRC_SYNCED if SRC_SYNCED.exists() else SRC_ORIG
OUT = ROOT / "论文" / "TrajSafe-Diffuser_中文版.docx"
MAX_DPI: int | None = None   # None = 按原分辨率嵌入

# A4 21.0 x 29.7 cm，页边距 2.5 cm -> 版心宽 16.0 cm
PAGE_W_TW = 11906
MARGIN_TW = 1418
TEXT_W_CM = 16.0
EMU_PER_CM = 360000
MAX_IMG_W_CM = 15.8
MAX_IMG_H_CM = 21.0

EA = "宋体"          # 正文中文
EA_H = "黑体"        # 标题中文
LATIN = "Times New Roman"

# --------------------------------------------------------------------------
# LaTeX -> Unicode
# --------------------------------------------------------------------------
GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ",
    "Omega": "Ω",
}
SYM = {
    "in": "∈", "notin": "∉", "times": "×", "cdot": "·", "odot": "⊙",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠",
    "approx": "≈", "sim": "∼", "simeq": "≃", "rightarrow": "→", "to": "→",
    "leftarrow": "←", "Rightarrow": "⇒", "leftrightarrow": "↔",
    "infty": "∞", "partial": "∂", "nabla": "∇", "pm": "±", "mp": "∓",
    "ell": "ℓ", "ldots": "…", "cdots": "⋯", "dots": "…", "mid": "|",
    "sum": "Σ", "prod": "Π", "int": "∫", "prime": "′", "circ": "∘",
    "ast": "∗", "star": "⋆", "forall": "∀", "exists": "∃", "subset": "⊂",
    "subseteq": "⊆", "cup": "∪", "cap": "∩", "emptyset": "∅", "angle": "∠",
    "perp": "⊥", "parallel": "∥", "log": "log", "exp": "exp", "sin": "sin",
    "cos": "cos", "tan": "tan", "max": "max", "min": "min", "det": "det",
    "arg": "arg", "dim": "dim", "diag": "diag", "clip": "clip",
    "softmax": "softmax", "quad": " ", "qquad": "  ", " ": " ",
    "lVert": "‖", "rVert": "‖", "Vert": "‖", "vert": "|", "lvert": "|",
    "rvert": "|", "top": "ᵀ", "bot": "⊥", "langle": "⟨", "rangle": "⟩",
    "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉",
    "equiv": "≡", "propto": "∝", "triangleq": "≜", "ll": "≪", "gg": "≫",
    "gets": "←", "mapsto": "↦", "wedge": "∧", "vee": "∨", "neg": "¬",
    "degree": "°", "rank": "rank", "tr": "tr", "trace": "tr",
    "argmax": "arg max", "argmin": "arg min", "st": "s.t.",
}
CAL = {"L": "ℒ", "R": "ℛ", "C": "𝒞", "V": "𝒱", "E": "ℰ", "B": "ℬ",
       "N": "𝒩", "I": "ℐ", "D": "𝒟", "P": "𝒫", "S": "𝒮", "G": "𝒢",
       "F": "ℱ", "M": "ℳ", "A": "𝒜", "T": "𝒯", "Q": "𝒬", "O": "𝒪",
       "H": "ℋ", "X": "𝒳", "W": "𝒲", "Z": "𝒵", "U": "𝒰", "J": "𝒥",
       "K": "𝒦", "Y": "𝒴"}
BB = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "C": "ℂ", "Q": "ℚ", "E": "𝔼",
      "P": "ℙ", "1": "𝟙"}
ACCENT = {"widehat": "\u0302", "widetilde": "\u0303", "hat": "\u0302",
          "tilde": "\u0303", "bar": "\u0304", "check": "\u030c",
          "dot": "\u0307", "ddot": "\u0308"}
WRAP = {"mathrm", "operatorname", "mathsf", "text", "mathbf", "mathit",
        "boldsymbol", "bm", "textrm", "mbox"}
DROP = {"left", "right", "bigl", "bigr", "Bigl", "Bigr", "big", "Big",
        "bigg", "Bigg", "displaystyle", "limits", "nolimits", "!", ";",
        ",", ":", ">", "medskip", "smallskip", "centering", "hspace",
        "vspace", "label", "nonumber", "qquad"}
SUPMAP = dict(zip("0123456789+-=()nijkaTbcdefghlmoprstuvwxyz",
                  "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱʲᵏᵃᵀᵇᶜᵈᵉᶠᵍʰˡᵐᵒᵖʳˢᵗᵘᵛʷˣʸᶻ"))
SUBMAP = dict(zip("0123456789+-=()aehijklmnoprstuvx",
                  "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"))
# 已经是上标的字符（如 \top 的 ᵀ）在 ^ 里要保持原样
for _c in "ᵀ":
    SUPMAP[_c] = _c


def _group(s: str, i: int):
    """s[i] must be '{'; return (inner, index_after_closing_brace)."""
    assert s[i] == "{", s[i:i + 10]
    depth, j = 1, i + 1
    while j < len(s) and depth:
        if s[j] == "\\":
            j += 2
            continue
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def _shift(text: str, table: dict, mark: str) -> str:
    if text and all(ch in table for ch in text):
        return "".join(table[ch] for ch in text)
    return f"{mark}({text})"


def _atom(s: str, i: int):
    """取一个数学原子并转成 Unicode：{组} / \\命令 / 单个字符。

    支持无花括号的写法，例如 `\\mathcal V`、`\\bar\\alpha_t`、`^\\top`。
    """
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] == "{":
        inner, j = _group(s, i)
        return conv(inner), j
    m = re.match(r"\\([A-Za-z]+)", s[i:])
    if m:
        name, j = m.group(1), i + m.end()
        if name in DROP:
            return "", j
        if name in ACCENT:
            body, k = _atom(s, j)
            return body + ACCENT[name], k
        if name == "frac":
            a, k = _atom(s, j)
            b, k2 = _atom(s, k)
            return f"({a})/({b})", k2
        if name == "sqrt":
            a, k = _atom(s, j)
            return f"√({a})", k
        if name == "mathcal":
            a, k = _atom(s, j)
            return "".join(CAL.get(c, c) for c in a), k
        if name == "mathbb":
            a, k = _atom(s, j)
            return "".join(BB.get(c, c) for c in a), k
        if name in WRAP:
            return _atom(s, j)
        if name in GREEK:
            return GREEK[name], j
        if name in SYM:
            return SYM[name], j
        return name, j
    return s[i], i + 1


def conv(s: str) -> str:
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            if m:
                name = m.group(1)
                if (name in ACCENT or name in WRAP or name in GREEK
                        or name in SYM or name in DROP
                        or name in ("frac", "sqrt", "mathcal", "mathbb")):
                    text, i = _atom(s, i)
                    out.append(text)
                    continue
                i += m.end()
                continue
            nxt = s[i + 1] if i + 1 < n else ""
            if nxt == "\\":
                out.append("\n")
                i += 2
                continue
            if nxt == "|":
                out.append("‖")
                i += 2
                continue
            if nxt in "{}%&_#":
                out.append(nxt)
                i += 2
                continue
            if nxt == " ":
                out.append(" ")
                i += 2
                continue
            i += 2
            continue
        if c in "_^":
            text, k = _atom(s, i + 1)
            out.append(_shift(text, SUBMAP if c == "_" else SUPMAP,
                              "_" if c == "_" else "^"))
            i = k
            continue
        if c == "{":
            inner, i = _group(s, i)
            out.append(conv(inner))
            continue
        if c == "}":
            i += 1
            continue
        if c == "~":
            out.append(" ")
            i += 1
            continue
        if c == "&":
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def math_unicode(s: str) -> str:
    s = re.sub(r"\\begin\{[a-z*]+\}", "", s)
    s = re.sub(r"\\end\{[a-z*]+\}", "", s)
    s = conv(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*", "\n", s)
    return s.strip()


# --------------------------------------------------------------------------
# OOXML 输出
# --------------------------------------------------------------------------
def run(text, bold=False, size=None, font=None, ea=None, italic=False,
        mono=False):
    rpr = ""
    if font or ea or mono:
        f = "Consolas" if mono else (font or LATIN)
        e = ea or (EA if not mono else EA)
        rpr += (f'<w:rFonts w:ascii="{f}" w:hAnsi="{f}" '
                f'w:eastAsia="{e}"/>')
    if bold:
        rpr += "<w:b/>"
    if italic:
        rpr += "<w:i/>"
    if size:
        rpr += f'<w:sz w:val="{int(round(size * 2))}"/>'
    rpr += f'<w:sz w:val="{int(round((size or 10.5) * 2))}"/>' if not size else ""
    return (f"<w:r>{'<w:rPr>' + rpr + '</w:rPr>' if rpr else ''}"
            f'<w:t xml:space="preserve">{xesc(text)}</w:t></w:r>')


def para(runs="", style=None, align=None, before=0, after=0, line=None,
         ind_first=None, keep_next=False):
    ppr = ""
    if style:
        ppr += f'<w:pStyle w:val="{style}"/>'
    if keep_next:
        ppr += "<w:keepNext/>"
    sp = f'<w:spacing w:before="{before}" w:after="{after}"'
    sp += f' w:line="{line}" w:lineRule="auto"/>' if line else "/>"
    ppr += sp
    if ind_first:
        ppr += f'<w:ind w:firstLineChars="{ind_first}"/>'
    if align:
        ppr += f'<w:jc w:val="{align}"/>'
    return f"<w:p><w:pPr>{ppr}</w:pPr>{runs}</w:p>"


def image_para(rel_id, name, pic_id, w_cm, h_cm):
    cx, cy = int(w_cm * EMU_PER_CM), int(h_cm * EMU_PER_CM)
    return (
        '<w:p><w:pPr><w:jc w:val="center"/>'
        '<w:spacing w:before="120" w:after="60"/></w:pPr><w:r><w:drawing>'
        f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/>'
        '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{pic_id}" name="Picture {pic_id}"/>'
        '<wp:cNvGraphicFramePr><a:graphicFrameLocks '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:nvPicPr><pic:cNvPr id="{pic_id}" name="{xesc(name)}"/>'
        '<pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rel_id}"/>'
        '<a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        '<pic:spPr><a:xfrm><a:off x="0" y="0"/>'
        f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        '</pic:pic></a:graphicData></a:graphic></wp:inline>'
        '</w:drawing></w:r></w:p>'
    )


def table_xml(rows, widths_cm, header=True):
    total = sum(widths_cm)
    tw = int(total * 567)
    grid = "".join(f'<w:gridCol w:w="{int(w * 567)}"/>' for w in widths_cm)
    borders = ("<w:tblBorders>"
               + "".join(f'<w:{s} w:val="single" w:sz="6" w:space="0" '
                          f'w:color="808080"/>'
                          for s in ("top", "left", "bottom", "right",
                                    "insideH", "insideV"))
               + "</w:tblBorders>")
    xml = (f'<w:tbl><w:tblPr><w:tblW w:w="{tw}" w:type="dxa"/>'
           f'<w:jc w:val="center"/>{borders}'
           '<w:tblLayout w:type="fixed"/></w:tblPr>'
           f"<w:tblGrid>{grid}</w:tblGrid>")
    for ri, row in enumerate(rows):
        trpr = "<w:trPr><w:cantSplit/></w:trPr>"
        xml += f"<w:tr>{trpr}"
        for ci, cell in enumerate(row):
            w = int(widths_cm[min(ci, len(widths_cm) - 1)] * 567)
            bold = header and ri == 0
            runs = cell_runs(cell, size=8.5, force_bold=bold)
            xml += (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>'
                    f'<w:vAlign w:val="center"/></w:tcPr>'
                    f'{para(runs, align="center", before=20, after=20)}</w:tc>')
        xml += "</w:tr>"
    return xml + "</w:tbl>"


STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr>
<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/>
<w:sz w:val="21"/><w:szCs w:val="21"/>
</w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:spacing w:line="360" w:lineRule="auto"/></w:pPr></w:pPrDefault>
</w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/>
<w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/>
<w:spacing w:before="240" w:after="240"/></w:pPr>
<w:rPr><w:rFonts w:eastAsia="黑体"/><w:b/><w:sz w:val="36"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>
<w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="0"/>
<w:spacing w:before="280" w:after="140"/><w:keepNext/></w:pPr>
<w:rPr><w:rFonts w:eastAsia="黑体"/><w:b/><w:sz w:val="28"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/>
<w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="1"/>
<w:spacing w:before="220" w:after="110"/><w:keepNext/></w:pPr>
<w:rPr><w:rFonts w:eastAsia="黑体"/><w:b/><w:sz w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/>
<w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="2"/>
<w:spacing w:before="180" w:after="90"/><w:keepNext/></w:pPr>
<w:rPr><w:rFonts w:eastAsia="黑体"/><w:b/><w:sz w:val="22"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="Caption"/>
<w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/>
<w:spacing w:before="60" w:after="180"/></w:pPr>
<w:rPr><w:sz w:val="18"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Math"><w:name w:val="Math"/>
<w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="center"/>
<w:spacing w:before="120" w:after="120"/></w:pPr>
<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math" w:eastAsia="宋体"/>
<w:sz w:val="21"/></w:rPr></w:style>
</w:styles>
"""


# --------------------------------------------------------------------------
# Markdown 解析
# --------------------------------------------------------------------------
RE_IMG = re.compile(r"^!\[([^\]]*)\]\((.+?)\)\s*$")
RE_HEAD = re.compile(r"^(#{1,4})\s+(.*)$")
RE_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")

# 原稿里 fig3-4(a)/(b) 已合并为单张 fig3-4_denoising.png
MERGED_34 = {
    "img": "fig3-4_denoising.png",
    "caption": ("**图 3-4 安全条件融合与控制点预测（上）与单层条件去噪模块（下）。** "
                "上半部分：控制特征通过交叉注意力读取局部安全隐特征，融合特征经三层条件去噪模块、"
                "坐标预测头和端点补偿得到满足起终点位置约束的控制点。"
                "下半部分：单层条件去噪模块通过 AdaLN 注入扩散时间条件，依次进行控制点自注意力、"
                "全局地图交叉注意力和前馈变换，三个子层分别采用残差连接。"),
}
SKIP_FILES = {"fig3-4a_denoising.png", "fig3-4b_denoising_block.png"}

RE_LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")


def clean_ref(line: str) -> str:
    """参考文献行：保留 DOI 说明，去掉'作者公开版本/会议正式记录'之类的注解链接。"""
    def sub(m):
        txt, url = m.group(1), m.group(2)
        if txt.startswith("DOI"):
            return txt
        if re.search(r"[\u4e00-\u9fff]", txt):
            return ""
        return txt
    return RE_LINK.sub(sub, line).rstrip()


def inline_runs(text: str, size=10.5, bold=False):
    """把 `**粗体**` 与 `$行内公式$` 拆成 run 列表。"""
    runs, i = [], 0
    pattern = re.compile(r"\*\*(.+?)\*\*|\$([^$]+)\$")
    for m in pattern.finditer(text):
        if m.start() > i:
            runs.append(run(text[i:m.start()], bold=bold, size=size))
        if m.group(1) is not None:
            runs.append(run(m.group(1), bold=True, size=size))
        else:
            runs.append(run(math_unicode(m.group(2)), bold=bold, size=size,
                            font="Cambria Math"))
        i = m.end()
    if i < len(text):
        runs.append(run(text[i:], bold=bold, size=size))
    return "".join(runs)


def embed_bytes(path: Path, wcm: float) -> bytes:
    """返回要嵌入 docx 的图片字节。给了 --dpi 就按打印尺寸降采样。"""
    raw = path.read_bytes()
    if not MAX_DPI:
        return raw
    with Image.open(path) as im:
        target = max(1, int(round(wcm / 2.54 * MAX_DPI)))
        if im.width <= target:
            return raw
        h = max(1, int(round(im.height * target / im.width)))
        im = im.convert("RGB") if im.mode in ("P", "RGBA", "LA") else im
        small = im.resize((target, h), Image.LANCZOS)
        buf = io.BytesIO()
        fmt = "JPEG" if path.suffix.lower() in (".jpg", ".jpeg") else "PNG"
        small.save(buf, fmt, optimize=True, quality=92,
                   **({"dpi": (MAX_DPI, MAX_DPI)} if fmt == "PNG" else {}))
        return buf.getvalue()


def cell_runs(text: str, size: float, force_bold: bool = False) -> str:
    """表格单元格：处理 `**粗体**` 与行内公式。表头整格加粗。"""
    if force_bold:
        return inline_runs(text.replace("**", ""), size=size, bold=True)
    return inline_runs(text, size=size)


def parse(path: Path):
    """产出 block 序列。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks, i, n = [], 0, len(lines)
    skip_caption = False
    in_comment = False
    while i < n:
        raw = lines[i]
        line = raw.strip()
        # 跳过 HTML 注释（同步版 md 顶部的改动说明）
        if in_comment:
            if "-->" in line:
                in_comment = False
            i += 1
            continue
        if line.startswith("<!--"):
            if "-->" not in line:
                in_comment = True
            i += 1
            continue
        if not line:
            i += 1
            continue
        m = RE_HEAD.match(line)
        if m:
            blocks.append(("head", len(m.group(1)), m.group(2).strip()))
            i += 1
            continue
        m = RE_IMG.match(line)
        if m:
            src = m.group(2).replace("\\", "/")
            fname = os.path.basename(src)
            if fname in SKIP_FILES:
                # 3-4(a)/(b) 已被合并成一张图：第一次出现时插合并图 + 合并图题，
                # 两次都要丢掉紧随其后的原图题。
                if not any(b[0] == "merged34" for b in blocks):
                    blocks.append(("merged34",))
                skip_caption = True
            else:
                blocks.append(("img", fname))
            i += 1
            continue
        if line.startswith("$$"):
            buf = []
            if line.strip() != "$$":
                buf.append(line.strip().strip("$"))
            i += 1
            while i < n and not lines[i].strip().startswith("$$"):
                buf.append(lines[i])
                i += 1
            i += 1
            blocks.append(("math", "\n".join(buf).strip()))
            continue
        if RE_TABLE_ROW.match(line):
            rows = []
            while i < n and RE_TABLE_ROW.match(lines[i].strip()):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows))
            continue
        if line.startswith("**图") or line.startswith("**表"):
            if skip_caption:
                skip_caption = False
                i += 1
                continue
            blocks.append(("caption", line))
            i += 1
            continue
        if re.match(r"^\[\d+\]\s", line):
            blocks.append(("ref", clean_ref(line)))
            i += 1
            continue
        if line.startswith("**关键词"):
            blocks.append(("kw", line))
            i += 1
            continue
        if line.startswith("**（") or line.startswith("**（"):
            blocks.append(("sub", line))
            i += 1
            continue
        blocks.append(("para", line))
        i += 1
    return blocks


def img_size_cm(path: Path):
    with Image.open(path) as im:
        w, h = im.size
    r = w / h
    wcm = MAX_IMG_W_CM
    hcm = wcm / r
    if hcm > MAX_IMG_H_CM:
        hcm = MAX_IMG_H_CM
        wcm = hcm * r
    return wcm, hcm


def build(blocks, out: Path):
    body, media, rels = [], [], []
    pic_id = 1

    def add_image(fname):
        nonlocal pic_id
        p = FIGDIR / fname
        if not p.exists():
            print(f"   [skip missing image] {fname}")
            return ""
        wcm, hcm = img_size_cm(p)
        part = f"image{len(media) + 1}{p.suffix.lower()}"
        media.append((part, embed_bytes(p, wcm)))
        rid = f"rId{100 + len(media)}"
        rels.append((rid, part))
        xml = image_para(rid, fname, pic_id, wcm, hcm)
        pic_id += 1
        return xml

    for b in blocks:
        kind = b[0]
        if kind == "head":
            lvl, text = b[1], b[2]
            if lvl == 1:
                body.append(para(inline_runs(text, size=18, bold=True),
                                 style="Title", align="center"))
            else:
                style = {2: "Heading1", 3: "Heading2", 4: "Heading3"}[lvl]
                body.append(para(inline_runs(text, size={2: 14, 3: 12, 4: 11}[lvl],
                                             bold=True), style=style))
        elif kind == "merged34":
            body.append(add_image(MERGED_34["img"]))
            body.append(para(inline_runs(MERGED_34["caption"], size=9),
                             style="Caption"))
        elif kind == "img":
            body.append(add_image(b[1]))
        elif kind == "math":
            for ln in math_unicode(b[1]).split("\n"):
                if ln.strip():
                    body.append(para(run(ln, size=10.5, font="Cambria Math"),
                                     style="Math"))
        elif kind == "table":
            rows = b[1]
            if not rows:
                continue
            ncol = max(len(r) for r in rows)
            widths = [TEXT_W_CM / ncol] * ncol
            body.append(table_xml(rows, widths))
            body.append(para("", after=120))
        elif kind == "caption":
            body.append(para(inline_runs(b[1], size=9), style="Caption"))
        elif kind == "ref":
            body.append(para(inline_runs(b[1], size=9.5), before=0, after=30))
        elif kind == "kw":
            body.append(para(inline_runs(b[1], size=10.5), before=120, after=120))
        elif kind == "sub":
            body.append(para(inline_runs(b[1], size=11, bold=True),
                             before=140, after=60, keep_next=True))
        else:
            body.append(para(inline_runs(b[1], size=10.5), ind_first=200))

    sect = (f'<w:sectPr><w:pgSz w:w="{PAGE_W_TW}" w:h="16838"/>'
            f'<w:pgMar w:top="{MARGIN_TW}" w:right="{MARGIN_TW}" '
            f'w:bottom="{MARGIN_TW}" w:left="{MARGIN_TW}" '
            'w:header="851" w:footer="992" w:gutter="0"/>'
            '<w:cols w:space="425"/><w:docGrid w:linePitch="312"/></w:sectPr>')
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
           '<w:document '
           'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
           'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
           'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
           'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
           'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
           "<w:body>" + "".join(body) + sect + "</w:body></w:document>")

    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Default Extension="png" ContentType="image/png"/>'
          '<Default Extension="jpg" ContentType="image/jpeg"/>'
          '<Default Extension="jpeg" ContentType="image/jpeg"/>'
          '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
          '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
          "</Types>")
    top = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
           '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
           '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
           "</Relationships>")
    drels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
             '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
             '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
             + "".join(
                 f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{part}"/>'
                 for rid, part in rels)
             + "</Relationships>")

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", top)
        z.writestr("word/document.xml", doc)
        z.writestr("word/_rels/document.xml.rels", drels)
        z.writestr("word/styles.xml", STYLES)
        for part, blob in media:
            z.writestr(f"word/media/{part}", blob)
    return len(media), len(body)


def main():
    global MAX_DPI, SRC
    args = sys.argv[1:]
    if "--src" in args:
        k = args.index("--src")
        SRC = Path(args[k + 1]).resolve()
        del args[k:k + 2]
    if "--dpi" in args:
        k = args.index("--dpi")
        MAX_DPI = int(args[k + 1])
        del args[k:k + 2]
    out = Path(args[0]) if args else OUT
    print("source:", SRC)
    print("max dpi:", MAX_DPI or "original")
    blocks = parse(SRC)
    kinds = {}
    for b in blocks:
        kinds[b[0]] = kinds.get(b[0], 0) + 1
    print("blocks:", kinds)
    try:
        nmedia, npar = build(blocks, out)
    except PermissionError:
        alt = out.with_name(out.stem + "_new" + out.suffix)
        print(f"!! {out.name} 被占用（多半在 Word 里开着），改写到 {alt.name}")
        nmedia, npar = build(blocks, alt)
        out = alt
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB, "
          f"{nmedia} images, {npar} block elements)")

if __name__ == "__main__":
    main()
