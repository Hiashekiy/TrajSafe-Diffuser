"""Verify scoped manuscript edits and the actual draw.io export artifacts."""
from pathlib import Path
import re
import json
import hashlib
import xml.etree.ElementTree as ET
from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper' / 'TrajSafe-Diffuser：生成式安全约束下的扩散运动规划方法.md'
OUT = ROOT / 'paper' / 'figures' / 'methods'

def main():
    text = PAPER.read_text(encoding='utf-8')
    backup = sorted((ROOT / 'paper' / 'backup').glob('*方法重写前-*.md'))[-1]
    old = backup.read_text(encoding='utf-8')
    start, stop = '## 3 方法', '## 4 实验与结果分析'
    method = text[text.index(start):text.index(stop)]
    assert text[:text.index(start)] == old[:old.index(start)], 'Front matter changed'
    assert text[text.index(stop):] == old[old.index(stop):], 'Experiments or conclusion changed'
    paths = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text)
    missing = [p for p in paths if not (PAPER.parent / p).is_file()]
    assert not missing, missing
    figures = re.findall(r'fig3-(\d)\.svg', method)
    assert figures == ['1', '2', '3', '4', '5'], figures
    assert method.count(r'\(') == method.count(r'\)')
    assert '$$' not in method
    manifest = json.loads((OUT / 'figure-map.json').read_text(encoding='utf-8'))
    exports = []
    for record in manifest:
        assert hashlib.sha256(Path(record['source']).read_bytes()).hexdigest() == record['source_sha256']
        svg = ET.parse(OUT / record['svg']).getroot()
        for element in svg.iter():
            if element.tag.endswith('image'):
                href = element.get('href') or element.get('{http://www.w3.org/1999/xlink}href')
                assert not href or href.startswith(('data:', '#')), href
        with Image.open(OUT / record['png']) as im:
            assert min(im.size) > 1000
            assert sum(ImageStat.Stat(im.convert('RGB')).var) > 100, 'Blank image'
            size = list(im.size)
        assert (OUT / record['pdf']).read_bytes().startswith(b'%PDF')
        exports.append(dict(number=record['number'], png_size=size, svg_bytes=(OUT / record['svg']).stat().st_size))
    report = dict(paper=str(PAPER), backup=str(backup), scope='method section only',
        other_sections_unchanged=True, image_links=len(paths), missing_links=missing,
        figure_order=figures, inline_formulas=method.count(r'\('),
        method_chinese_characters=len(re.findall(r'[\u4e00-\u9fff]', method)),
        original_diagrams_unchanged=True, exports=exports)
    (OUT / 'integration-check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
