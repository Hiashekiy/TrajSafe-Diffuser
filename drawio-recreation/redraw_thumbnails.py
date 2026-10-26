"""Deterministic editable diagram thumbnails; no screenshot or raster embedding."""
from pathlib import Path
import base64, copy, json, math, re, urllib.parse, zlib
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
NAME = '特征融合与安全轨迹去噪流程图'
OBSTACLES = [
 [(0,0),(61,0),(59,14),(38,28),(30,51),(0,57)],
 [(85,43),(111,40),(130,49),(135,67),(126,88),(103,88),(83,82),(79,63)],
 [(79,110),(103,108),(118,120),(118,141),(105,155),(79,152),(70,136),(71,119)],
 [(0,81),(26,75),(33,87),(31,116),(20,134),(0,135)],
 [(183,62),(200,56),(200,146),(184,150),(179,132),(181,105)],
 [(0,194),(53,193),(59,200),(0,200)],
 [(84,194),(122,188),(158,186),(168,200),(84,200)],
 [(156,0),(200,0),(200,10),(181,12),(164,13)],
]
ROUTES = [
 [(20,177),(44,164),(49,132),(49,99),(55,65),(65,32),(108,22),(150,23),(180,23)],
 [(20,177),(42,171),(59,162),(61,137),(60,114),(70,98),(101,98),(139,99),(151,78),(154,52),(165,34),(180,23)],
 [(20,177),(54,180),(100,178),(144,169),(160,146),(161,115),(163,86),(164,58),(180,23)],
]

def inside(p, poly):
    result = False
    for a,b in zip(poly,poly[1:]+poly[:1]):
        if (a[1]>p[1]) != (b[1]>p[1]) and p[0] < (b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0]:
            result = not result
    return result

def safe(p):
    return 0 <= p[0] <= 200 and 0 <= p[1] <= 200 and not any(inside(p,o) for o in OBSTACLES)

def path_safe(path, closed=False):
    pairs = list(zip(path,path[1:])) + ([(path[-1],path[0])] if closed else [])
    for a,b in pairs:
        n = max(1,math.ceil(math.dist(a,b)*4))
        if not all(safe((a[0]+(b[0]-a[0])*j/n,a[1]+(b[1]-a[1])*j/n)) for j in range(n+1)):
            return False
    return True

def resample(path, count):
    lengths = [0]
    for a,b in zip(path,path[1:]): lengths.append(lengths[-1]+math.dist(a,b))
    out = []
    for i in range(count):
        d = lengths[-1]*i/(count-1)
        j = next((j for j in range(1,len(lengths)) if lengths[j]>=d),len(lengths)-1)
        t = (d-lengths[j-1])/(lengths[j]-lengths[j-1])
        out.append(tuple(path[j-1][k]+t*(path[j][k]-path[j-1][k]) for k in (0,1)))
    return out

CENTER = resample(ROUTES[1],17)
NOISY = [(p if i in (0,16) else (p[0]+[0,2,-2.2,1.6,-1.3][i%5],p[1]+[0,-1.8,2.2,-1.7,1.4][i%5])) for i,p in enumerate(CENTER)]
def feedback_before():
    # A visible single ALM correction, with the original points near obstacles.
    for shift in (6.5,6,5.5,5,4):
        points = []
        for i,p in enumerate(CENTER):
            if i in (0,16):
                points.append(p)
                continue
            a,b = CENTER[i-1],CENTER[i+1]
            dx,dy = b[0]-a[0],b[1]-a[1]
            length = math.hypot(dx,dy)
            points.append((p[0]-dy/length*shift,p[1]+dx/length*shift))
        if path_safe(points): return points
    raise ValueError('No valid illustrative feedback polygon')

BEFORE = feedback_before()

def corridor():
    dense = resample(ROUTES[1],51)
    regions = []
    for i,p in enumerate(dense):
        a,b = dense[max(0,i-1)],dense[min(len(dense)-1,i+1)]
        theta = math.atan2(b[1]-a[1],b[0]-a[0])
        ra,rb = 12,10+(math.sin(i*.5)+1)*1.8
        for _ in range(40):
            poly = [(p[0]+ra*math.cos(t)*math.cos(theta)-rb*math.sin(t)*math.sin(theta),p[1]+ra*math.cos(t)*math.sin(theta)+rb*math.sin(t)*math.cos(theta)) for t in (2*math.pi*j/20 for j in range(20))]
            if path_safe(poly,True) and all(safe(v) for v in poly): break
            ra,rb = ra*.9,rb*.9
        regions.append(poly)
    assert all(inside(dense[i+1], regions[i]) or inside(dense[i],regions[i+1]) for i in range(len(dense)-1)), 'corridor gap'
    return regions

REGIONS = corridor()
def f(v): return f'{v:.4f}'.rstrip('0').rstrip('.') if v else '0'
def stencil(poly):
    xs,ys = zip(*poly); x,y = min(xs),min(ys); w,h = max(xs)-x,max(ys)-y
    path = ''.join(f'<{"line" if i else "move"} x="{f(a-x)}" y="{f(b-y)}"/>' for i,(a,b) in enumerate(poly))
    source = f'<shape name="EditablePolygon" w="{f(w)}" h="{f(h)}" aspect="variable" strokewidth="inherit"><background><path>{path}<close/></path></background><foreground><fillstroke/></foreground></shape>'
    compressor = zlib.compressobj(wbits=-15)
    packed = base64.b64encode(compressor.compress(urllib.parse.quote(source,safe="~()*!.'-").encode())+compressor.flush()).decode()
    return packed,(x,y,w,h)

class Thumbnail:
    def __init__(self,kind,tag):
        self.kind,self.tag,self.cells,self.svg = kind,tag,[],[]
        self.rect(0,0,200,160,'#ffffff')
        for o in OBSTACLES: self.polygon(o,'#c6d1dc','#adbdca',100,.65)
        if kind in ('corridor','final'):
            for i,p in enumerate(REGIONS): self.polygon(p,'#78bf8d','#68aa7e' if i%4==0 else 'none',18,.45)
        if kind.startswith('candidate') or kind in ('selected','corridor','final'):
            route = ROUTES[int(kind[-1])-1] if kind.startswith('candidate') else ROUTES[1]
            self.line(route,'#db4d50',1.95)
        if kind=='noisy':
            self.line(NOISY,'#3d7eb2',1.2,True)
            for p in NOISY: self.dot(p,'#377db5',2.2)
        if kind=='history':
            self.line(BEFORE,'#90a2b5',1.1,True)
            for p in BEFORE: self.dot(p,'#8e9fb1',1.65)
            self.line(CENTER,'#69aa82',1.5)
            for i in (3,5,8,10,13): self.line([BEFORE[i],CENTER[i]],'#628973',.9,False,True)
            for p in CENTER: self.dot(p,'#66aa7d',2)

    def cell(self,style,geometry,edge=False):
        c = ET.Element('mxCell',id=f'{self.tag}_{len(self.cells)}',value='',style=style,parent=self.tag,**({'edge':'1'} if edge else {'vertex':'1'}))
        c.append(geometry); self.cells.append(c)

    def vertex(self,style,x,y,w,h):
        self.cell(style,ET.Element('mxGeometry',x=f(x),y=f(y),width=f(w),height=f(h),**{'as':'geometry'}))

    def rect(self,x,y,w,h,color):
        self.vertex(f'rounded=0;fillColor={color};strokeColor=none;',x,y,w,h)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}"/>')

    def polygon(self,poly,fill,stroke='none',opacity=100,sw=.5):
        poly = [(x,y*.8) for x,y in poly]
        encoded,box = stencil(poly)
        self.vertex(f'shape=stencil({encoded});fillColor={fill};strokeColor={stroke};strokeWidth={sw};fillOpacity={opacity};strokeOpacity=40;',*box)
        pts = ' '.join(f'{f(x)},{f(y)}' for x,y in poly)
        self.svg.append(f'<polygon points="{pts}" fill="{fill}" fill-opacity="{opacity/100}" stroke="{stroke}" stroke-opacity=".4" stroke-width="{sw}" stroke-linejoin="round"/>')

    def line(self,points,color,width=1.7,dash=False,arrow=False):
        p = [(x,y*.8) for x,y in points]
        geom = ET.Element('mxGeometry',relative='1',**{'as':'geometry'})
        for pt,role in ((p[0],'sourcePoint'),(p[-1],'targetPoint')):
            ET.SubElement(geom,'mxPoint',x=f(pt[0]),y=f(pt[1]),**{'as':role})
        if len(p)>2:
            arr = ET.SubElement(geom,'Array',**{'as':'points'})
            for x,y in p[1:-1]: ET.SubElement(arr,'mxPoint',x=f(x),y=f(y))
        self.cell(f'edgeStyle=none;html=1;rounded=0;endArrow={"block" if arrow else "none"};endSize=4;endFill=1;strokeColor={color};strokeWidth={width};'+('dashed=1;dashPattern=3 3;' if dash else ''),geom,True)
        pts = ' '.join(f'{f(x)},{f(y)}' for x,y in p)
        self.svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"'+(' stroke-dasharray="3 3"' if dash else '')+(' marker-end="url(#correction)"' if arrow else '')+'/>')

    def dot(self,p,color,r):
        x,y = p[0],p[1]*.8
        self.vertex(f'ellipse;fillColor={color};strokeColor=#ffffff;strokeWidth=.65;',x-r,y-r,2*r,2*r)
        self.svg.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="{r}" fill="{color}" stroke="white" stroke-width=".65"/>')

    def document(self):
        return '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="800" viewBox="0 0 200 160">'+DEFS+''.join(self.svg)+'</svg>'

DEFS = '<defs><marker id="correction" markerWidth="4" markerHeight="4" refX="3.5" refY="2" orient="auto"><path d="M0 0L4 2L0 4Z" fill="#628973"/></marker></defs>'
PLACEMENTS = [
 ('v11',9,'occupancy',33,100,167,95),('v21',6,'noisy',33,243,167,87),('v28',10,'history',33,380,167,87),
 ('v88',12,'candidate1',914,136,110,117),('v100',12,'candidate2',1052,136,102,117),('v112',11,'candidate3',1196,136,100,117),
 ('v133',11,'selected',1339,170,171,161),('v145',13,'corridor',1339,394,171,160),('v204',13,'final',1339,714,171,161),('v224',10,'history',67,916,129,78),
]

def main():
    file = HERE/(NAME+'.drawio')
    backup = HERE/(NAME+'-替换前备份.drawio')
    if not backup.exists(): backup.write_bytes(file.read_bytes())
    svgfile = HERE/(NAME+'.svg')
    svgbackup = HERE/(NAME+'-替换前备份.svg')
    if not svgbackup.exists(): svgbackup.write_bytes(svgfile.read_bytes())
    tree = ET.parse(backup); root = tree.find('.//root')
    preview = svgbackup.read_text(encoding='utf-8')
    kinds = ['occupancy','noisy','history','candidate1','candidate2','candidate3','selected','corridor','final']
    assets = {k:Thumbnail(k,'asset_'+k) for k in kinds}
    dest = HERE/'thumbnails'; dest.mkdir(exist_ok=True)
    for i,k in enumerate(kinds): (dest/f'{i+1:02d}_{k}.svg').write_text(assets[k].document(),encoding='utf-8')
    for index,(start,count,kind,x,y,w,h) in enumerate(PLACEMENTS):
        a = assets[kind]; group = f'thumbnail_{index}_{kind}'
        children = list(root); at = next(i for i,c in enumerate(children) if c.get('id')==start)
        for c in children[at:at+count]: root.remove(c)
        g = ET.Element('mxCell',id=group,value='',style='group;',vertex='1',connectable='0',parent='1')
        ET.SubElement(g,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
        frame = ET.Element('mxCell',id=group+'_frame',value='',style='rounded=1;arcSize=12;fillColor=#ffffff;strokeColor=#9baec1;strokeWidth=1.4;',vertex='1',parent=group)
        ET.SubElement(frame,'mxGeometry',x='0',y='0',width=str(w),height=str(h),**{'as':'geometry'})
        k = min((w-8)/200,(h-8)/160); ox,oy = (w-200*k)/2,(h-160*k)/2
        updated = [g,frame]
        for original in a.cells:
            c = copy.deepcopy(original); c.set('id',f'placed{index}_'+c.get('id')); c.set('parent',group)
            c.set('style',re.sub(r'strokeWidth=([\d.]+);',lambda m:f'strokeWidth={f(float(m[1])*k)};',c.get('style')))
            for geom in c.iter('mxGeometry'):
                if geom.get('relative')=='1': continue
                for key in ('x','y','width','height'):
                    if key in geom.attrib: geom.set(key,f(float(geom.get(key))*k+({'x':ox,'y':oy}.get(key,0))))
            for p in c.iter('mxPoint'):
                p.set('x',f(float(p.get('x'))*k+ox)); p.set('y',f(float(p.get('y'))*k+oy))
            updated.append(c)
        for offset,c in enumerate(updated): root.insert(at+offset,c)
        needle = f'<rect x="{x}" y="{y}" width="{w}" height="{h}"'
        pos = preview.index(needle); end = pos
        for _ in range(count):
            match = re.match(r'\s*<(?:rect|path|circle)\b[^>]*?/>',preview[end:])
            assert match, start
            end += match.end()
        replacement = f'<g><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="white" stroke="#9baec1" stroke-width="1.4"/><g transform="translate({f(x+ox)} {f(y+oy)}) scale({f(k)})">'+''.join(a.svg)+'</g></g>'
        preview = preview[:pos]+replacement+preview[end:]
    for c in list(root):
        if c.get('id')=='v39': root.remove(c)
    preview = re.sub(r'<text x="178\.5"[\s\S]*?</text>','',preview)
    preview = preview.replace('<defs>','<defs>'+DEFS[6:-7],1)
    ET.indent(tree,space='  '); tree.write(file,encoding='utf-8',xml_declaration=True)
    svgfile.write_text(preview,encoding='utf-8')
    (HERE/(NAME+'-小图重绘版.drawio')).write_bytes(file.read_bytes())
    all_cells = list(tree.iter('mxCell')); ids = [c.get('id') for c in all_cells]
    assert len(ids)==len(set(ids))
    assert all(c.get('parent') in ids for c in all_cells if c.get('parent'))
    assert all(path_safe(p) for p in ROUTES+[NOISY,BEFORE,CENTER])
    assert all(path_safe(p,True) for p in REGIONS)
    report = {'updated_positions':len(PLACEMENTS),'native_cells':len(all_cells),'shared_map':True,'selected_equals_candidate_2':True,'representative_control_points':len(CENTER),'convex_corridor_regions':len(REGIONS),'collision_sampling_step_max':.25,'diagram_raster_images':sum('shape=image' in c.get('style','') for c in all_cells),'note':'Illustrative geometry, not planner output or a safety proof. SVG full preview inherits prior rendering; thumbnails share exact geometry with drawio stencils.'}
    (dest/'geometry.json').write_text(json.dumps({'obstacles':OBSTACLES,'routes':ROUTES,'control_points':NOISY,'feedback_before':BEFORE,'feedback_after':CENTER,'corridor_regions':REGIONS},indent=2),encoding='utf-8')
    (dest/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    sheet = '<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="640" viewBox="0 0 1440 640">'+DEFS+'<rect width="1440" height="640" fill="#f4f7fa"/>'
    titles = ['Occupancy Map','Noisy Control Points','Refinement History','Skeleton Candidate 1','Skeleton Candidate 2','Skeleton Candidate 3','Selected Route','Safety Corridor']
    for i,(kind,title) in enumerate(zip(kinds,titles)):
        col,row = i%4,i//4; tx,ty = 25+col*360,40+row*320
        sheet += f'<text x="{tx}" y="{ty-14}" font-family="Times New Roman" font-size="21" fill="#273344">{i+1}. {title}</text><g transform="translate({tx} {ty}) scale(1.5)">'+''.join(assets[kind].svg)+'</g>'
    (dest/'contact-sheet.svg').write_text(sheet+'</svg>',encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__': main()
