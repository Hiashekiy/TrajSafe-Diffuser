"""Polish the user's edited overall framework while keeping its principal flow."""
from pathlib import Path
import copy, re, json
import xml.etree.ElementTree as E

HERE=Path(__file__).resolve().parent
FILE=HERE/'总体框架图.drawio'
BACKUP=HERE/'总体框架图-美化前备份.drawio'
if not BACKUP.exists(): BACKUP.write_bytes(FILE.read_bytes())
tree=E.parse(BACKUP); model=tree.find('.//mxGraphModel'); root=model.find('root')
cells={c.get('id'):c for c in root}
prefix='tQv-43lgluLC_yPoRtkz-'
def cell(n):return cells[prefix+str(n)]
def style(c,**updates):
    tokens=c.get('style','').split(';'); parts={}
    head=[]
    for t in tokens:
        if '=' in t:
            k,v=t.split('=',1);parts[k]=v
        elif t:head.append(t)
    for k,v in updates.items():
        if v is None:parts.pop(k,None)
        else:parts[k]=str(v)
    c.set('style',';'.join(head+[f'{k}={v}' for k,v in parts.items()])+';')
def geom(n,x=None,y=None,w=None,h=None):
    c=cell(n);g=c.find('mxGeometry')
    for key,value in [('x',x),('y',y),('width',w),('height',h)]:
        if value is not None:g.set(key,str(value))
    return c
def resize_group(n,x,y,w,h):
    c=cell(n);g=c.find('mxGeometry');sx=w/float(g.get('width'));sy=h/float(g.get('height'))
    descendants=[]; parents={c.get('id')}
    while True:
        added=[z for z in root if z.get('parent') in parents and z not in descendants]
        if not added:break
        descendants+=added;parents.update(z.get('id') for z in added)
    for z in descendants:
        for g0 in z.findall('mxGeometry'):
            if g0.get('relative')!='1':
                for key,s in [('x',sx),('width',sx),('y',sy),('height',sy)]:
                    if key in g0.attrib:g0.set(key,str(float(g0.get(key))*s))
            for p in g0.iter('mxPoint'):
                if p.get('x') is not None:p.set('x',str(float(p.get('x'))*sx))
                if p.get('y') is not None:p.set('y',str(float(p.get('y'))*sy))
    geom(n,x,y,w,h)
def edge(n,points,source=None,target=None,color=None,dashed=False):
    c=cell(n)
    for k,v in [('source',source),('target',target)]:
        if v is None:c.attrib.pop(k,None)
        else:c.set(k,prefix+str(v))
    g=c.find('mxGeometry');g.clear();g.attrib.update(relative='1',**{'as':'geometry'})
    for p,role in [(points[0],'sourcePoint'),(points[-1],'targetPoint')]:E.SubElement(g,'mxPoint',x=str(p[0]),y=str(p[1]),**{'as':role})
    if len(points)>2:
        arr=E.SubElement(g,'Array',**{'as':'points'})
        for p in points[1:-1]:E.SubElement(arr,'mxPoint',x=str(p[0]),y=str(p[1]))
    style(c,edgeStyle='none',rounded=1,arcSize=12,strokeWidth=1.7,endArrow='block',endFill=1,endSize=7,strokeColor=color or '#526878',dashed=1 if dashed else 0,exitX=None,exitY=None,entryX=None,entryY=None,opacity=100)

# Flat, restrained fills and consistent typography. The user-defined palette stays.
for c in root:
    if c.get('vertex')=='1':
        style(c,gradientColor=None,gradientDirection=None,shadow=0)
        if c.get('value'):style(c,fontFamily='Times New Roman')
    if c.get('parent')=='1' and c.get('edge'):style(c,strokeWidth=1.7,endSize=7,rounded=1,arcSize=12)

# Equal panel padding and header hierarchy.
geom(529,502,370,845,550);geom(2,1369,370,653,550)
geom(530,504,372,841,58);geom(5,1371,372,649,58)
style(cell(529),fillColor='#fbfdff',strokeColor='#a5c7ed',strokeWidth=1.4,arcSize=5)
style(cell(2),fillColor='#fffdfa',strokeColor='#e8b67d',strokeWidth=1.4,arcSize=5)
for n in [531,8,9]:style(cell(n),fontSize=32,fontStyle=1)
geom(531,522,379,806,43);geom(8,1389,379,614,43)
style(cell(530),fillColor='#e9f2fc');style(cell(5),fillColor='#fff1e1');style(cell(6),fillColor='#eaf4eb')

# Equal image sizes, encoder baselines and token bars.
for n,y in [(532,457),(544,594),(574,731)]:resize_group(n,530,y,167,96)
for n,y in [(543,559),(573,696),(626,833)]:geom(n,y=y,h=30);style(cell(n),fontSize=21)
for n,y in [(630,453),(631,590),(632,727)]:geom(n,y=y,h=104);style(cell(n),fontSize=22,strokeWidth=1.4)
for ids,y in [([642,643,644,645],473),([646,647,648,649],610),([650,651,652,653],747)]:
    for n in ids:geom(n,y=y,h=64)
for n,y in [(657,541),(658,678),(659,815)]:geom(n,y=y,h=30)
geom(661,1089,450,109,420);style(cell(661),arcSize=18,strokeWidth=1.5)
geom(662,1094,627,99,66);style(cell(662),fontSize=23)
for n in [667,668,669,670]:geom(n,y=611,h=98)
geom(671,y=715)
edge(666,[(1200,660),(1241,660)],color='#2663a9')
for a,b,c,y in [(634,635,663,505),(636,637,664,642),(638,639,665,779)]:
    color={505:'#709dce',642:'#d79936',779:'#4ba174'}[y]
    edge(a,[(701,y),(749,y)],color=color)
    edge(b,[(899,y),(949,y)],color=color)
    edge(c,[(1034,y),(1085,y)],color=color,dashed=True)

# Candidates: the caption has a real row, separate from the thumbnails.
geom(150,1391,450,412,160);style(cell(150),strokeWidth=1.2,arcSize=10,dashPattern='6 4')
geom(151,1402,456,390,30);style(cell(151),fontSize=21)
for n,x in [(153,1406),(165,1540),(177,1674)]:resize_group(n,x,493,110,104)
edge(189,[(1520,545),(1536,545)],color='#71879b')
geom(190,1650,527,22,30);style(cell(190),fontSize=23)
geom(192,1450,628,280,64);style(cell(192),arcSize=20,strokeWidth=1.4)
geom(193,1460,638,260,44);style(cell(193),fontSize=23)
edge(191,[(1590,611),(1590,626)],source=150,target=192,color='#c68b46')
edge(197,[(1323,660),(1448,660)],target=192,color='#2366ac')
resize_group(451,1450,732,280,70)
geom(196,7,12,266,46);style(cell(196),fontSize=23)
style(cell(195),arcSize=20,strokeWidth=1.4)
edge(457,[(1590,694),(1590,730)],source=192,target=195,color='#637281')
style(cell(457),exitX=.5,exitY=1,entryX=.5,entryY=0)
geom(198,1602,696,63,30);style(cell(198),fontSize=21)
cell(198).set('value','<i>Γ*</i>')

# Geometry feature caption, bars and output variable have separate space.
geom(674,1435,812,310,30);cell(674).set('value','Safety Geometry Features');style(cell(674),fontSize=21)
for i,n in enumerate(range(279,286)):geom(n,1492+i*25,851,16,47)
geom(286,1670,857,72,32);style(cell(286),fontSize=22)
edge(675,[(1590,803),(1590,849)],color='#7a8c9d')
# Treat the caption as a label on this short output line, with a clean paper fill.
style(cell(674),fillColor='#fffdfa')
root.remove(cell(674));root.append(cell(674))

# Two right-hand views stay in the user's order and retain their map contents.
geom(679,1818,450,182,454);style(cell(679),strokeWidth=1.2,arcSize=10,dashPattern='6 4')
geom(681,1826,462,166,54);cell(681).set('value','Safety Corridor<br>Construction');style(cell(681),fontSize=20,fontStyle=1)
resize_group(212,1834,528,150,146.5)
resize_group(461,1834,726,150,146.5)
geom(211,1824,876,170,28);style(cell(211),fontSize=21)
edge(524,[(1909,676),(1909,724)],source=213,target=462,color='#637281')
style(cell(524),exitX=.5,exitY=1,entryX=.5,entryY=0)
edge(525,[(1650,898),(1650,908),(1777,908),(1777,601),(1832,601)],source=285,target=213,color='#7f929f')
style(cell(525),exitX=.5,exitY=1,entryX=0,entryY=.5)

# Give the iterative chain an even rhythm and a separate lane for each feedback.
bottom_ids=[]
for c in root:
    g=c.find('mxGeometry')
    if c.get('parent')!='1' or g is None:continue
    if c.get('edge'):
        pts=list(g.iter('mxPoint'))
        if pts and min(float(p.get('y','0')) for p in pts)>=900:
            for p in pts:p.set('y',str(float(p.get('y'))+52))
    elif float(g.get('y','0'))>=900:
        g.set('y',str(float(g.get('y'))+52));bottom_ids.append(c.get('id'))
geom(3,503,954,1519,360);style(cell(3),fillColor='#fbfdfb',strokeWidth=1.4,arcSize=5)
geom(6,505,956,1515,57);geom(9,527,964,1450,43)
geom(300,738,1023,1070,180);style(cell(300),strokeWidth=1.3,arcSize=10,dashPattern='6 4',strokeColor='#81918d')
geom(301,1000,1030,546,32);style(cell(301),fontSize=22)
for n in [303,305,307,309,311,313]:
    geom(n,y=1083,h=99);style(cell(n),arcSize=18,strokeWidth=1.4)
for n in [304,306,308,310,312,314]:style(cell(n),fontSize=22)
geom(304,769,1101,89,62);geom(314,1677,1102,100,61)
geom(312,1460,1098,177,69)
resize_group(321,1840,1056.25,163,153.5)
geom(384,1836,1217,171,31);style(cell(384),fontSize=22)
geom(385,1890,1248,63,29);style(cell(385),fontSize=22)
edge(302,[(698,1133),(758,1133)],target=303,color='#43966b')
for n,a,b in [(315,303,305),(316,305,307),(317,307,309),(318,309,311),(319,311,313)]:
    ga=cell(a).find('mxGeometry');gb=cell(b).find('mxGeometry')
    edge(n,[(float(ga.get('x'))+float(ga.get('width'))+2,1133),(float(gb.get('x'))-2,1133)],source=a,target=b,color='#637281')
edge(320,[(1786,1133),(1838,1133)],source=313,target=321,color='#71ad8c')
style(cell(320),exitX=1,exitY=.5,entryX=0,entryY=.5)
edge(444,[(1548,1184),(1548,1233),(1270,1233)],source=311,color='#60a883',dashed=True)
style(cell(444),exitX=.5,exitY=1)
geom(443,1284,1204,250,28);style(cell(443),fontSize=21,align='center')
edge(673,[(1727,1184),(1727,1277),(1455,1277)],source=313,color='#60a883',dashed=True)
style(cell(673),exitX=.5,exitY=1)
geom(528,1468,1246,248,28);style(cell(528),fontSize=21,align='center')

# Remove only the two pale, unbound arrow fragments left above the chain.
for n in [388,389]:root.remove(cell(n))

# A single landscape canvas, without the editor's A4 seams or empty coordinates.
for c in root:
    if c.get('parent')!='1':continue
    g=c.find('mxGeometry')
    if g is None:continue
    if g.get('relative')!='1':
        if 'x' in g.attrib:g.set('x',str(float(g.get('x'))-482))
        if 'y' in g.attrib:g.set('y',str(float(g.get('y'))-350))
    for p in g.iter('mxPoint'):
        if 'x' in p.attrib:p.set('x',str(float(p.get('x'))-482))
        if 'y' in p.attrib:p.set('y',str(float(p.get('y'))-350))
model.attrib.update(pageWidth='1560',pageHeight='985',pageScale='1',grid='0',dx='1560',dy='985')
E.indent(tree,space='  ');tree.write(FILE,encoding='utf-8',xml_declaration=True)
(HERE/'总体框架图-美化版.drawio').write_bytes(FILE.read_bytes())
ids={c.get('id') for c in root}
assert len(ids)==len(list(root))
for c in root:
    for key in ['source','target','parent']:
        assert c.get(key) is None or c.get(key) in ids,(c.get('id'),key,c.get(key))
print(json.dumps({'cells':len(ids),'backup':str(BACKUP),'output':str(FILE),'removed_decorative_fragments':2},ensure_ascii=False))
