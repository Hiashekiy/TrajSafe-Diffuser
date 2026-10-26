"""Prepare publication copies without modifying the user's draw.io sources."""
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'paper' / 'drawio'
OUT = ROOT / 'paper' / 'figures' / 'methods'
COPIES = OUT / 'drawio'

def prepare():
    COPIES.mkdir(parents=True, exist_ok=True)
    records = []
    for number in range(1, 6):
        source = (SOURCE / 'TrajSafe-Diffuser_arch-main.framed.drawio' if number == 1
                  else next(SOURCE.glob(f'*图3-{number}_*原排版修订版.drawio')))
        tree = ET.parse(source)
        model = tree.find('.//mxGraphModel')
        model.set('math', '1')
        changes = []
        parent = model.find('root')
        for cell in list(parent):
            ident = cell.get('id')
            value = cell.get('value', '')
            if ident in ('title', 'rev_title', 'T'):
                parent.remove(cell)
                changes.append('Removed internal figure title; use manuscript caption')
                continue
            updated = value
            if number == 1:
                updated = updated.replace('[B, 32, 2]', '[B, 48, 2]')
                if ident == 'p1_ffbcap':
                    updated = r'\(h_t\)'
            if number == 3 and ident == 'bias':
                updated = updated.replace(r'\bar\alpha_t^2', r'\bar\alpha_t')
            if number == 3 and ident == 'curve':
                updated = 'CurveDecoder<br>固定归一化弧长采样'
            if number == 5:
                formulas = {
                    'raw': r'当前干净预测　\(Q_0^{\mathrm{pred}}\quad[B,C,2]\)',
                    'pack': r'冻结连续约束　\(A_j E_\ell Q-b_j\leq0\)',
                    'g': r'\(g(Q)\quad[B,P,4,F_{\max}]\)',
                    'dual': r'\(\omega=[\lambda+\rho g(Q)]_+\)',
                    'lam': r'\(\omega\quad[B,P,4,F_{\max}]\)　解析梯度权重',
                    'clip': r'对偶更新　\(\lambda\leftarrow[\lambda+\rho g(Q_{\mathrm{new}})]_+\)',
                    'tol': r'\(\max[g(Q_{\mathrm{new}})]_+\leq\varepsilon_{\mathrm{alm}}\)，或达到迭代预算',
                    'q0s': r'\(Q_0^{\mathrm{safe}}\quad[B,C,2]\)',
                    'ddim': r'\(Q_{t^-}=\mathrm{DDIM}(Q_t,Q_0^{\mathrm{safe}})\)',
                    'hard': r'固定端点　\(q_{t^-,0}=S,\quad q_{t^-,C-1}=G\)',
                    'qnext': r'\(Q_{t^-}\quad[B,C,2]\)',
                    'fin': r'\([Q_0^{\mathrm{safe}};\Delta Q;v]\quad[B,C,5]\)',
                    'fhf': r'\(H_{\mathrm{fb}}\quad[B,C,D]\)',
                    'fcat': r'\([H_{\mathrm{ctrl}};H_{\mathrm{fb}}]\quad[B,C,2D]\)',
                    'op': r'\(G_{\mathrm{fb}}=v\,\mathrm{sigmoid}(\cdot)\)<br>\(\widetilde H_{\mathrm{ctrl}}=H_{\mathrm{ctrl}}+G_{\mathrm{fb}}\odot H_{\mathrm{fb}}\)',
                    'hcin': r'\(H_{\mathrm{ctrl}}\quad[B,C,D]\)',
                    'hc2': r'\(\widetilde H_{\mathrm{ctrl}}\quad[B,C,D]\)　主干输出后注入',
                    'N': 'ΔQ：安全控制点减原始预测；v：历史反馈有效标志。<br>P：Bézier 段数；Fmax：最大半平面数。最多 10 次内迭代，ρ=5，η=0.05，停止容差 0.001。',
                }
                updated = formulas.get(ident, updated)
                if ident == 'op':
                    cell.set('style', cell.get('style').replace('fontSize=13;', 'fontSize=11;'))
                    cell.find('mxGeometry').set('height', '44')
            if updated != value:
                cell.set('value', updated)
                changes.append(f'{ident}: {value} -> {updated}')
        dest = COPIES / f'fig3-{number}.drawio'
        tree.write(dest, encoding='utf-8', xml_declaration=True)
        ET.parse(dest)
        records.append(dict(number=f'3-{number}', source=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            publication_copy=str(dest), adjustments=changes,
            svg=f'fig3-{number}.svg', png=f'fig3-{number}.png', pdf=f'fig3-{number}.pdf'))
    (OUT / 'figure-map.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Prepared {len(records)} publication copies in {COPIES}')

if __name__ == '__main__':
    prepare()
