from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


def set_run_font(run, latin="Times New Roman", east_asia="宋体", size=Pt(10.5), bold=None):
    run.font.name = latin
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asia)
    run.font.size = size
    run.font.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        run.bold = bold


def shade_cell(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=90, bottom=80, end=90):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, text, end])
    set_run_font(run, size=Pt(9))


def latex_to_readable(text):
    replacements = {
        r"\mathbb{R}": "ℝ", r"\mathcal{N}": "𝒩", r"\mathcal{G}": "𝒢",
        r"\mathcal{V}": "𝒱", r"\mathcal{E}": "ℰ", r"\mathcal{P}": "𝒫",
        r"\mathcal{C}": "𝒞", r"\mathcal{S}": "𝒮", r"\operatorname{area}": "area",
        r"\operatorname{softmax}": "softmax", r"\mathrm{raw}": "raw",
        r"\mathrm{safe}": "safe", r"\mathrm{raw}": "raw", r"\top": "ᵀ",
        r"\rightarrow": "→", r"\in": "∈", r"\le": "≤", r"\ge": "≥",
        r"\mid": "|", r"\sum": "∑", r"\sqrt": "√", r"\odot": "⊙",
        r"\sigma": "σ", r"\lambda": "λ", r"\alpha": "α", r"\bar": "",
        r"\hat": "", r"\widetilde": "", r"\Delta": "Δ", r"\Gamma": "Γ",
        r"\theta": "θ", r"\rho": "ρ", r"\pi": "π", r"\ldots": "…",
        r"\left": "", r"\right": "", r"\begin{bmatrix}": "[",
        r"\end{bmatrix}": "]", r"\quad": "  ", r"\,": " ", r"\\": "; ",
    }
    result = " ".join(text.split())
    for old, new in replacements.items():
        result = result.replace(old, new)
    result = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", result)
    result = re.sub(r"\\text\{([^{}]+)\}", r"\1", result)
    result = re.sub(r"\\(?:min|max|exp|cos|sin)\b", lambda m: m.group(0)[1:], result)
    result = result.replace("{", "(").replace("}", ")")
    result = re.sub(r"\\[A-Za-z]+", "", result)
    result = re.sub(r"\s+", " ", result).strip()
    return result


def add_native_equation(paragraph, latex):
    math_para = OxmlElement("m:oMathPara")
    math = OxmlElement("m:oMath")
    math_run = OxmlElement("m:r")
    math_text = OxmlElement("m:t")
    math_text.text = latex_to_readable(latex)
    math_run.append(math_text)
    math.append(math_run)
    math_para.append(math)
    paragraph._p.append(math_para)


def clean_inline(text):
    text = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("`", "")
    text = text.replace("**", "")
    text = re.sub(r"\$([^$]+)\$", r"\1", text)
    return text


def add_rich_text(paragraph, text, size=Pt(10.5), east_asia="宋体"):
    # Preserve bold spans; render inline formulas in Cambria Math without raw dollar signs.
    parts = re.split(r"(\*\*.*?\*\*|\$.*?\$|`.*?`)", text)
    for part in parts:
        if not part:
            continue
        bold = part.startswith("**") and part.endswith("**")
        math = part.startswith("$") and part.endswith("$")
        code = part.startswith("`") and part.endswith("`")
        content = part[2:-2] if bold else part[1:-1] if (math or code) else part
        if math:
            content = latex_to_readable(content)
        content = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", content)
        run = paragraph.add_run(content)
        set_run_font(
            run,
            latin="Cambria Math" if math else "Consolas" if code else "Times New Roman",
            east_asia="等线" if code else east_asia,
            size=Pt(9) if code else size,
            bold=bold,
        )


def configure_styles(doc):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.first_line_indent = Pt(21)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    normal.paragraph_format.space_after = Pt(0)

    title = styles["Title"]
    title.font.name = "Times New Roman"
    title._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    title.font.size = Pt(22)
    title.font.bold = True
    title.font.color.rgb = RGBColor(0, 0, 0)
    title.paragraph_format.space_after = Pt(24)

    for name, size, east, before, after in (
        ("Heading 1", 16, "黑体", 18, 12),
        ("Heading 2", 14, "黑体", 14, 8),
        ("Heading 3", 12, "黑体", 10, 6),
    ):
        style = styles[name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), east)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True


def add_table(doc, rows):
    if not rows:
        return
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    table = doc.add_table(rows=len(rows), cols=width)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    table.style = "Table Grid"
    set_repeat_table_header(table.rows[0])
    for r_idx, row in enumerate(rows):
        for c_idx, value in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            if r_idx == 0:
                shade_cell(cell, "D9EAF7")
            elif r_idx % 2 == 0:
                shade_cell(cell, "F7FAFC")
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.first_line_indent = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.05
            add_rich_text(paragraph, clean_inline(value), size=Pt(7.5 if width >= 8 else 8.5), east_asia="宋体")
            for run in paragraph.runs:
                if r_idx == 0:
                    run.bold = True
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_body_paragraph(doc, text, bullet=False):
    paragraph = doc.add_paragraph(style="List Bullet" if bullet else "Normal")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if bullet:
        paragraph.paragraph_format.first_line_indent = Pt(0)
        paragraph.paragraph_format.left_indent = Pt(21)
    add_rich_text(paragraph, text)
    return paragraph


def build_docx(source_path: Path, output_path: Path, equation_dir: Path):
    source = source_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.7)
    section.right_margin = Cm(2.4)
    add_page_number(section.footer.paragraphs[0])
    configure_styles(doc)

    equation_index = 0
    paragraph_buffer = []
    i = 0
    first_h1 = True

    def flush_paragraph():
        nonlocal paragraph_buffer
        if paragraph_buffer:
            add_body_paragraph(doc, "".join(part.strip() for part in paragraph_buffer))
            paragraph_buffer = []

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            i += 1
            continue

        if stripped == "$$":
            flush_paragraph()
            eq_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                eq_lines.append(lines[i])
                i += 1
            equation_index += 1
            image_path = equation_dir / f"equation_{equation_index:02d}.png"
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(5)
            p.paragraph_format.space_after = Pt(5)
            if image_path.exists():
                run = p.add_run()
                run.add_picture(str(image_path), width=Inches(4.6))
            else:
                add_native_equation(p, " ".join(eq_lines))
            i += 1
            continue

        heading = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            text = clean_inline(heading.group(2))
            if level == 1:
                p = doc.add_paragraph(style="Title")
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_rich_text(p, text, size=Pt(22), east_asia="黑体")
            else:
                word_level = min(level - 1, 3)
                if word_level == 1 and not first_h1:
                    doc.add_page_break()
                p = doc.add_paragraph(style=f"Heading {word_level}")
                p.paragraph_format.first_line_indent = Pt(0)
                add_rich_text(p, text, size=Pt(16 if word_level == 1 else 14 if word_level == 2 else 12), east_asia="黑体")
                if word_level == 1:
                    first_h1 = False
            i += 1
            continue

        image_match = re.match(r"^!\[([^]]*)\]\(([^)]+)\)$", stripped)
        if image_match:
            flush_paragraph()
            image_path = (source_path.parent / image_match.group(2)).resolve()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            if image_path.exists():
                p.add_run().add_picture(str(image_path), width=Inches(5.7))
            else:
                add_rich_text(p, f"[图像缺失：{image_match.group(1)}]")
            i += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            parsed = []
            for idx, row in enumerate(table_lines):
                cells = [cell.strip() for cell in row.strip("|").split("|")]
                if idx == 1 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    continue
                parsed.append(cells)
            add_table(doc, parsed)
            continue

        if re.match(r"^(图|表)\d+-\d+\s", stripped):
            flush_paragraph()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Pt(0)
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.keep_with_next = False
            add_rich_text(p, stripped, size=Pt(9), east_asia="宋体")
            i += 1
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            add_body_paragraph(doc, stripped[2:], bullet=True)
            i += 1
            continue

        paragraph_buffer.append(stripped)
        i += 1

    flush_paragraph()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.core_properties.title = "TrajSafe-Diffuser 生成式安全约束下的扩散运动规划方法"
    doc.core_properties.subject = "学术论文"
    doc.core_properties.author = ""
    doc.core_properties.keywords = "扩散模型, 运动规划, 安全约束, 凸安全走廊"
    doc.save(output_path)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: generate_thesis_docx.py SOURCE.md OUTPUT.docx EQUATION_DIR")
    build_docx(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())
