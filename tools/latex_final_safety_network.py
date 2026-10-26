from pathlib import Path
import json,xml.etree.ElementTree as E
p=Path('paper/figures/final-safety-network/final-safety-network.drawio');tr=E.parse(p);tr.find('.//mxGraphModel').set('math','1');r=tr.find('.//mxGraphModel/root');cells={z.get('id'):z for z in r}
def m(s):return r'\( '+s+r' \)'
labels={'path':('Control',r'H_{\mathrm{ctrl}}','48 \\times 128'),'ell':('Safety',r'H_{\mathrm{ell}}','128 \\times 128'),'safe':('Safety-fused',r'H_{\mathrm{fuse}}','48 \\times 128'),'f':('Fused','F','48 \\times 128'),'hc':('Clean',r'H_{\mathrm{clean}}','48 \\times 128'),'raw':('Raw',r'Q_{\mathrm{raw}}','48 \\times 2'),'fixed':('Corrected',r'Q_{\mathrm{bdry}}','48 \\times 2'),'map':('Map',r'C_G','256 \\times 128'),'rc':('Raw',r'Q_{\mathrm{raw}}','48 \\times 2'),'bc':('Corrected',r'Q_{\mathrm{bdry}}','48 \\times 2')}
for i,(name,sym,dim) in labels.items():cells[i+'_caption'].set('value',f'<b>{name}</b> '+m(sym)+'<br>'+m(dim))
values={
 'ht':'Time embedding '+m('h_t')+'<br>'+m('128'),
 'sca':'Safety cross-attention<br>4 heads · '+m(r'W_{\mathrm{out}}'),
 'mlp':'Fusion MLP<br>'+m(r'\mathrm{LN}(128)\to256')+'<br>GELU '+m(r'\to128'),
 'den':'FinalDenoiser '+m(r'\times3')+'<br>Independent TrajBlock parameters',
 'head':'Shared '+m(r'\mathrm{head}_p')+'<br>Linear '+m(r'128\to2'),
 'cg':'Global map memory '+m('C_G')+'<br>'+m(r'256\times128'),
 'sg':'Start '+m('S')+' / Goal '+m('G'),
 'sgc':'Start '+m('S')+' / Goal '+m('G'),
 'x':m('X')+'<br>'+m(r'48\times128'),
 'xp':m('X^+')+'<br>'+m(r'48\times128'),
 'op1':'Map cross-attention<br>4 heads · '+m(r'W_{\mathrm{out}}'),
 'op2':'FFN<br>'+m(r'128\to512\to128')+'<br>GELU',
 'tb':m(r'h_t\to\mathrm{Linear}(128,256)\to\mathrm{split}(\gamma,\beta)')+'<br>Separate modulation projection in each AdaLN',
 'ada':m(r'\mathrm{AdaLN}(X,h_t)=(1+\gamma(h_t))\odot\mathrm{LN}(X)+\beta(h_t)'),
 'bias':'Self-attention bias:<br>'+m(r'B_{ij}^{h}=b_h(|i-j|)')+' for '+m('C=48')+'.',
 'err':'Endpoint errors<br>'+m(r'e_s=S-Q_{\mathrm{raw},0}')+'<br>'+m(r'e_g=G-Q_{\mathrm{raw},47}'),
 'weights':'Fixed correction profiles<br>'+m(r'w^s=[1,0.75,0.5,0.25,0,\ldots]')+'<br>'+m(r'w^g=\mathrm{reverse}(w^s)'),
 'delta':'Compensation<br>'+m(r'\Delta Q_i=w_i^s e_s+w_i^g e_g'),
 'eq':m(r'Q_{\mathrm{bdry},0}=S,\quad Q_{\mathrm{bdry},47}=G')+'; only the first / last four controls are shifted.',
 'down':'Curve: '+m(r'128\times2')+'<br>Controls → downstream ALM',
 'fusion_formula':m(r'H_{\mathrm{fuse}}=H_{\mathrm{ctrl}}+\mathrm{CrossAttention}\!\left(\mathrm{AdaLN}(H_{\mathrm{ctrl}},h_t),H_{\mathrm{ell}}\right)'),
 'meta':m(r'C=48\;\cdot\;Q=128\;\cdot\;D=128')+' · 4 heads · '+m(r'd_h=32')+' · batch omitted',
 'kvnote':m('K,V')
}
for i,v in values.items():cells[i].set('value',v)
for z in r:
 if z.get('edge')=='1' and z.get('value') in ['Q','K, V']:z.set('value',m(z.get('value').replace(' ','')))
assert not any('h_t' in z.get('value','') and r'\(' not in z.get('value','') for z in r)
E.indent(tr,space='  ');tr.write(p,encoding='utf-8',xml_declaration=True)
j=p.with_suffix('.json');data=json.loads(j.read_text(encoding='utf-8'));data['vertices']=[{'id':z.get('id'),'value':z.get('value'),'geometry':dict(z.find('mxGeometry').attrib)} for z in r if z.get('vertex')=='1'];data['math_format']='LaTeX, drawio math=1';j.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print('LaTeX labels updated')
