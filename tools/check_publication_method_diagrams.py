"""Check editable cells, rendered math, exports and scoped manuscript edits."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as E
import fitz
from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'paper/figures/methods/publication-v2'
PAPER = ROOT/'paper/TrajSafe-Diffuser：生成式安全约束下的扩散运动规划方法.md'


def main():
    paper = PAPER.read_text(encoding='utf-8')
    old = (ROOT/'paper/backup/TrajSafe-Diffuser：生成式安全约束下的扩散运动规划方法-整体修订前-20261001-论文口径.md').read_text(encoding='utf-8')
    start, stop = '## 3 方法', '## 4 实验与结果分析'
    # This edition revises related work, methods and experimental prose.
    # The user-confirmed abstract/introduction and all result tables are fixed.
    assert paper[:paper.index('## 2 相关工作')] == old[:old.index('## 2 相关工作')]
    def result_tables(text):
        return [block for block in re.findall(r'(?:^\|.*\n)+', text, re.M)
                if '碰撞率/%' in block or '时间/ms' in block]
    assert len(result_tables(paper)) == 3
    assert result_tables(paper) == result_tables(old), 'Experimental tables changed'
    assert not any(term in paper for term in ('末端投影','末端硬投影','L1 硬投影'))
    assert paper.count(r'\(') == paper.count(r'\)')
    method = paper[paper.index(start):paper.index(stop)]
    assert '$$' not in method
    figures = re.findall(r'fig3-(\d)\.svg', method)
    assert figures == ['1','2','3','4','5']
    for link in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',paper):
        assert (PAPER.parent/link).is_file(), link
    # Old source documents are unchanged, including the hand-edited overview.
    for record in json.loads((OUT.parent/'figure-map.json').read_text(encoding='utf-8')):
        assert hashlib.sha256(Path(record['source']).read_bytes()).hexdigest() == record['source_sha256']
    checks=[]
    for record in json.loads((OUT/'figure-map.json').read_text(encoding='utf-8')):
        source = Path(record['source'])
        assert source.read_bytes() == Path(record['publication_copy']).read_bytes()
        doc=E.parse(source)
        assert doc.find('.//mxGraphModel').get('math') == '1'
        vertices = doc.findall('.//mxCell[@vertex="1"]')
        edges=doc.findall('.//mxCell[@edge="1"]')
        ids={c.get('id') for c in doc.findall('.//mxCell')}
        labels=''.join(c.get('value','') for c in vertices)
        assert labels.count(r'\(') == labels.count(r'\)') and '$$' not in labels
        assert not any('image=' in c.get('style','') for c in vertices)
        for c in edges:
            for endpoint in ('source','target'):
                if c.get(endpoint): assert c.get(endpoint) in ids
        svg_text=(OUT/record['svg']).read_text(encoding='utf-8')
        svg=E.fromstring(svg_text)
        assert not any(c.tag.endswith('image') for c in svg.iter())
        assert 'data-mml-node' in svg_text or 'MJX' in svg_text, 'MathJax glyphs missing'
        with Image.open(OUT/record['png']) as im:
            assert min(im.size)>1900
            assert sum(ImageStat.Stat(im.convert('RGB')).var)>100
            size=im.size
        pdf=fitz.open(OUT/record['pdf'])
        assert len(pdf)==1 and len(pdf[0].get_drawings())>20
        # Render actual exported vector PDF at intended full-page width 170 mm.
        page=pdf[0]
        scale=(170/25.4*96)/page.rect.width
        preview=page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False)
        preview.save(str(OUT/f"fig{record['number']}-print-preview.png"))
        font_sizes=[]
        for c in vertices:
            if c.get('value'):
                match=re.search(r'fontSize=(\d+)',c.get('style',''))
                font_sizes.append(int(match.group(1)))
        crop_width = size[0]/3
        min_pt = min(font_sizes)/crop_width*(170/25.4*72)
        assert min_pt >= 7, (record['number'],min_pt)
        checks.append({'number':record['number'],'native_vertices':len(vertices),'native_edges':len(edges),
                       'connected_edges':sum(bool(c.get('source') or c.get('target')) for c in edges),
                       'embedded_images':0,'png_size':size,'rendered_math':True,
                       'min_label_pt_at_170mm':round(min_pt,2),'vector_pdf':True})
    result={'scope':'manuscript-authoritative revision and 4 native method figures',
            'abstract_and_introduction_unchanged':True,'experimental_tables_unchanged':True,
            'old_sources_unchanged':True,'image_links_valid':True,'inline_math_only':True,
            'intended_width_mm':170,'figures':checks}
    (OUT/'publication-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
