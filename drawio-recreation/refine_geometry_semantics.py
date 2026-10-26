"""Separate ellipse predictions, convex corridor construction and final spline."""
from pathlib import Path
import base64, copy, importlib.util, json, math, re, urllib.parse, zlib
import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import BSpline

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('thumbnails_base',HERE/'redraw_thumbnails.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
FILE=HERE/'总体框架图.drawio'
BACKUP=HERE/'总体框架图-语义修订前备份.drawio'
if not BACKUP.exists():BACKUP.write_bytes(FILE.read_bytes())
tree=ET.parse(BACKUP);model=tree.find('.//mxGraphModel');root=model.find('root')
prefix='tQv-43lgluLC_yPoRtkz-'
cells={c.get('id'):c for c in root}
def cell(n):return cells[prefix+str(n)]
def style(c,**updates):
    parts=c.get('style','').strip(';').split(';');d={};heads=[]
    for s in parts:
        if '=' in s:k,v=s.split('=',1);d[k]=v
        elif s:heads.append(s)
    for k,v in updates.items():
        if v is None:d.pop(k,None)
        else:d[k]=str(v)
    c.set('style',';'.join(heads+[f'{k}={v}' for k,v in d.items()])+';')

def samples_at(path,progress):
    dense=np.asarray(base.resample(path,1001))
    return dense[np.round(np.asarray(progress)*1000).astype(int)]

def ellipse_parameters():
    output=[]
    centers=samples_at(base.ROUTES[1],np.linspace(.055,.945,11))
    dense=np.asarray(base.resample(base.ROUTES[1],1001))
    for t,p in zip(np.linspace(.055,.945,11),centers):
        index=int(t*1000);v=dense[min(index+12,1000)]-dense[max(index-12,0)]
        angle=math.atan2(v[1],v[0]);a,b=17.,8.5
        for _ in range(40):
            poly=[(p[0]+a*math.cos(q)*math.cos(angle)-b*math.sin(q)*math.sin(angle),p[1]+a*math.cos(q)*math.sin(angle)+b*math.sin(q)*math.cos(angle)) for q in np.linspace(0,2*math.pi,81)[:-1]]
            if base.path_safe(poly,True):break
            a*=.94;b*=.94
        output.append((p,a,b,angle))
    return output

def convex_corridor():
    dense=np.asarray(base.resample(base.ROUTES[1],29));output=[]
    template=np.array([[-.72,-1],[.48,-1],[1,-.28],[.8,.7],[-.5,1],[-1,.25]])
    for i,p in enumerate(dense):
        v=dense[min(i+1,len(dense)-1)]-dense[max(i-1,0)];angle=math.atan2(v[1],v[0])
        rot=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        a,b=16.,10.5
        for _ in range(50):
            poly=template*np.array([a,b])@rot.T+p
            if base.path_safe(poly.tolist(),True):break
            a*=.94;b*=.94
        output.append(poly.tolist())
    # Convex bridge pieces join adjacent assigned regions across tight bends.
    for i in range(len(dense)-1):
        if base.inside(dense[i+1],output[i]) or base.inside(dense[i],output[i+1]):continue
        a,b=dense[i],dense[i+1];d=b-a;normal=np.array([-d[1],d[0]])/np.linalg.norm(d)
        width=3.
        for _ in range(40):
            poly=[a-normal*width,b-normal*width,b+normal*width,a+normal*width]
            if base.path_safe([p.tolist() for p in poly],True):break
            width*=.9
        output.append([p.tolist() for p in poly])
    return output

ELLIPSES=ellipse_parameters();REGIONS=convex_corridor()
CONTROLS=np.asarray(base.resample(base.ROUTES[1],15))
KNOTS=np.r_[np.zeros(4),np.linspace(0,1,len(CONTROLS)-2)[1:-1],np.ones(4)]
SPLINE=BSpline(KNOTS,CONTROLS,3)
CURVE=SPLINE(np.linspace(0,1,1201))
assert base.path_safe(CURVE.tolist()),'The final illustrative spline crosses an obstacle'

def ellipse(thumb,p,a,b,angle):
    # Transform a real ellipse into the common map's 200 x 160 coordinates.
    rot=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    u,s,_=np.linalg.svd(np.diag([1,.8])@rot@np.diag([a,b]))
    deg=math.degrees(math.atan2(u[1,0],u[0,0]))
    x,y=float(p[0]),float(p[1]*.8)
    thumb.vertex(f'ellipse;rotation={deg};fillColor=#f3cf91;fillOpacity=24;strokeColor=#bd8138;strokeWidth=1.65;',x-s[0],y-s[1],s[0]*2,s[1]*2)
    thumb.svg.append(f'<ellipse cx="{x}" cy="{y}" rx="{s[0]}" ry="{s[1]}" transform="rotate({deg} {x} {y})" fill="#f3cf91" fill-opacity=".24" stroke="#bd8138" stroke-width="1.65"/>')

def spline_shape(thumb):
    pieces=[]
    breaks=np.unique(KNOTS);derivative=SPLINE.derivative()
    first=SPLINE(0);pieces.append(f'<move x="{first[0]}" y="{first[1]*.8}"/>')
    svgpath=f'M {first[0]} {first[1]*.8}'
    for a,b in zip(breaks,breaks[1:]):
        q=np.vstack([SPLINE(a)+(b-a)/3*derivative(a),SPLINE(b)-(b-a)/3*derivative(b),SPLINE(b)])
        q[:,1]*=.8
        pieces.append('<curve '+ ' '.join(f'x{i+1}="{p[0]}" y{i+1}="{p[1]}"' for i,p in enumerate(q))+'/>')
        svgpath+=' C '+' '.join(f'{p[0]} {p[1]}' for p in q)
    source='<shape name="CubicBSpline" w="200" h="160" aspect="variable" strokewidth="inherit"><foreground><path>'+''.join(pieces)+'</path><stroke/></foreground></shape>'
    encoded=urllib.parse.quote(source,safe="~()*!.'-").encode();c=zlib.compressobj(wbits=-15)
    packed=base64.b64encode(c.compress(encoded)+c.flush()).decode()
    thumb.vertex(f'shape=stencil({packed});fillColor=none;strokeColor=#d74046;strokeWidth=2.5;',0,0,200,160)
    thumb.svg.append(f'<path d="{svgpath}" fill="none" stroke="#d74046" stroke-width="2.5" stroke-linecap="round"/>')

def make(kind):
    thumb=base.Thumbnail('occupancy','revision_'+kind)
    if kind=='predicted_ellipses':
        thumb.line(base.ROUTES[1],'#8895a1',1.1,True)
        for p,a,b,theta in ELLIPSES:ellipse(thumb,p,a,b,theta)
    else:
        for i,p in enumerate(REGIONS):thumb.polygon(p,'#b8ddbe','#48985f',39,.9)
        # Keep each convex boundary legible rather than a blurred band.
        for c in thumb.cells[9:]:
            if 'shape=stencil' in c.get('style',''):style(c,strokeOpacity=72)
        thumb.svg=[s.replace('stroke-opacity=".4"','stroke-opacity=".72"') if '<polygon' in s and '#b8ddbe' in s else s for s in thumb.svg]
        if kind=='convex_corridor':thumb.line(base.ROUTES[1],'#d85a59',1.7)
        else:
            thumb.line(CONTROLS.tolist(),'#8b9baa',.9,True)
            spline_shape(thumb)
            for p in CONTROLS:thumb.dot(p,'#246fa4',2.3)
    return thumb

def replace_group(n,frame_id,thumb):
    group=cell(n);gid=group.get('id');g=group.find('mxGeometry');w,h=float(g.get('width')),float(g.get('height'))
    keep=prefix+str(frame_id)
    descendants=set([gid]);remove=[]
    while True:
        new=[c for c in root if c.get('parent') in descendants and c.get('id') not in descendants]
        if not new:break
        descendants.update(c.get('id') for c in new);remove+=new
    for c in remove:
        if c.get('id')!=keep:root.remove(c)
    at=list(root).index(cells[keep])+1
    scale=min((w-8)/200,(h-8)/160);ox,oy=(w-200*scale)/2,(h-160*scale)/2
    for i,original in enumerate(thumb.cells):
        c=copy.deepcopy(original);c.set('parent',gid);c.set('id',f'geom_revision_{n}_{i}')
        c.set('style',re.sub(r'strokeWidth=([\d.]+);',lambda m:f'strokeWidth={float(m[1])*scale};',c.get('style')))
        for geom in c.findall('mxGeometry'):
            if geom.get('relative')!='1':
                for key in ['x','y','width','height']:
                    if key in geom.attrib:geom.set(key,str(float(geom.get(key))*scale+{'x':ox,'y':oy}.get(key,0)))
            for p in geom.iter('mxPoint'):
                p.set('x',str(float(p.get('x'))*scale+ox));p.set('y',str(float(p.get('y'))*scale+oy))
        root.insert(at+i,c)

assets={kind:make(kind) for kind in ['predicted_ellipses','convex_corridor','smooth_trajectory']}
replace_group(212,213,assets['predicted_ellipses'])
replace_group(461,462,assets['convex_corridor'])
replace_group(321,322,assets['smooth_trajectory'])

# All mathematical symbols use inline MathJax delimiters, never HTML subscripts.
labels={
151:r'Skeleton Candidates  \(\{\Gamma_1,\Gamma_2,\ldots,\Gamma_m\}\)',
190:r'\(\cdots\)',198:r'\(\Gamma^\ast\)',
286:r'\(H_{\mathrm{geo}}\)',294:r'\(F\)',299:r'\(H_{\mathrm{geo}}\)',
301:r'Iterative Denoising \((t=T,\ldots,0)\)',
306:r'Trajectory<br>Denoiser<br>\(\varepsilon_{\theta}(\cdot)\)',
385:r'\(\tau^\ast\)',443:r'Refinement History \(\mathcal{H}_{t+1}\)',
528:r'Noisy Control Points \(Q_{t+1}\)',543:r'Occupancy Map \(\mathcal{O}\)',
573:r'Noisy Control Points \(Q_t\)',626:r'Refinement History \(\mathcal{H}_t\)',
657:r'\(F_{\mathrm{map}}\)',658:r'\(F_{\mathrm{ctrl}}\)',659:r'\(F_{\mathrm{fb},t}\)',671:r'\(F\)',
681:'Predicted<br>Ellipses',211:'Convex Safe Corridor',
}
for n,value in labels.items():
    c=cell(n);c.set('value',value);style(c,math=1,html=1,fontFamily='Times New Roman')
style(cell(681),fontSize=21)
style(cell(211),fontSize=18)
cell(211).find('mxGeometry').attrib.update(x='1338',width='178',y='526',height='28')
# MathJax labels require slightly more width than the old HTML subscript labels.
cell(659).find('mxGeometry').attrib.update(x='457',width='108')
cell(443).find('mxGeometry').attrib.update(x='786',width='282')
model.set('math','1')
tree.getroot().set('modified','2026-10-26')
all_ids={c.get('id') for c in root}
assert len(all_ids)==len(list(root))
for c in root:
    for k in ['parent','source','target']:
        assert c.get(k) is None or c.get(k) in all_ids,(c.get('id'),k,c.get(k))
ET.indent(tree,space='  ');tree.write(FILE,encoding='utf-8',xml_declaration=True)
for name in ['总体框架图-美化版.drawio','总体框架图-几何语义修订版.drawio']:(HERE/name).write_bytes(FILE.read_bytes())
dest=HERE/'geometry-revision';dest.mkdir(exist_ok=True)
for kind,asset in assets.items():(dest/(kind+'.svg')).write_text(asset.document(),encoding='utf-8')
sheet='<svg xmlns="http://www.w3.org/2000/svg" width="1560" height="500" viewBox="0 0 1560 500"><rect width="1560" height="500" fill="white"/>'
titles=['Predicted ellipses','Constructed convex corridor','Smooth B-spline with control points']
for i,((kind,asset),title) in enumerate(zip(assets.items(),titles)):
    x=20+i*520
    sheet+=f'<text x="{x}" y="32" font-family="Times New Roman" font-size="25">{title}</text><g transform="translate({x} 60) scale(2.4)">'+''.join(asset.svg)+'</g>'
(dest/'comparison.svg').write_text(sheet+'</svg>',encoding='utf-8')
validation={'predicted_ellipses':len(ELLIPSES),'convex_regions_with_bridges':len(REGIONS),'final_spline_degree':3,'final_spline_controls':len(CONTROLS),'spline_collision_sampling':1201,'spline_collision_free':True,'corridor_regions_collision_free':all(base.path_safe(p,True) for p in REGIONS),'mathjax_enabled':True,'mathjax_labels':sum('\\(' in c.get('value','') for c in root),'history_input':r'\mathcal{H}_t','history_feedback':r'\mathcal{H}_{t+1}','note':'Synthetic schematic geometry; the visible control points are actual B-spline controls, not curve samples.'}
(dest/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
print(json.dumps(validation))
