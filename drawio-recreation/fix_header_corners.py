"""Top-only rounded title bands, inset from the outer panel's exact corner radius."""
from pathlib import Path
import base64, urllib.parse, zlib
import xml.etree.ElementTree as E

HERE=Path(__file__).resolve().parent
FILE=HERE/'总体框架图.drawio'
BACKUP=HERE/'总体框架图-标题圆角修正前备份.drawio'
if not BACKUP.exists():BACKUP.write_bytes(FILE.read_bytes())
tree=E.parse(FILE);root=tree.find('.//root')
prefix='tQv-43lgluLC_yPoRtkz-'
cells={c.get('id'):c for c in root}
for header,panel in [(530,529),(5,2),(6,3)]:
    c=cells[prefix+str(header)];p=cells[prefix+str(panel)]
    g=c.find('mxGeometry');pg=p.find('mxGeometry')
    w,h=float(g.get('width')),float(g.get('height'))
    # draw.io relative arcSize is a percentage of the shorter panel dimension.
    ps=dict(s.split('=',1) for s in p.get('style').split(';') if '=' in s)
    outer=min(float(pg.get('width')),float(pg.get('height')))*float(ps.get('arcSize',15))/100
    inset=max(float(g.get('x'))-float(pg.get('x')),float(g.get('y'))-float(pg.get('y')))
    r=max(0,outer-inset);k=.5522847498307936
    path=f'<move x="0" y="{h}"/><line x="0" y="{r}"/><curve x1="0" y1="{r*(1-k)}" x2="{r*(1-k)}" y2="0" x3="{r}" y3="0"/><line x="{w-r}" y="0"/><curve x1="{w-r*(1-k)}" y1="0" x2="{w}" y2="{r*(1-k)}" x3="{w}" y3="{r}"/><line x="{w}" y="{h}"/><close/>'
    shape=f'<shape name="TopRoundedHeader" w="{w}" h="{h}" aspect="variable" strokewidth="inherit"><background><path>{path}</path></background><foreground><fillstroke/></foreground></shape>'
    comp=zlib.compressobj(wbits=-15);raw=urllib.parse.quote(shape,safe="~()*!.'-").encode()
    encoded=base64.b64encode(comp.compress(raw)+comp.flush()).decode()
    parts=c.get('style').strip(';').split(';')
    parts=[s for s in parts if not s.startswith(('shape=','rounded=','arcSize='))]
    c.set('style',f'shape=stencil({encoded});'+ ';'.join(parts)+';')
    print(f'Header {header}: top corner radius={r}; bottom corners remain square')
E.indent(tree,space='  ');tree.write(FILE,encoding='utf-8',xml_declaration=True)
for name in ['总体框架图-美化版.drawio','总体框架图-几何语义修订版.drawio']:(HERE/name).write_bytes(FILE.read_bytes())
