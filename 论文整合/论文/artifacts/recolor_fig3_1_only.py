import base64, io, re, sys, shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
p = Path('论文/TrajSafe_methods_and_figures/figures/fig3-1_framework.drawio')
backup = p.with_name('fig3-1_framework.before-flat-palette.drawio')
if not backup.exists():
    shutil.copy2(p,backup)
original = ET.parse(backup)
tree = ET.parse(backup)
blue, purple, green, orange, ink = '#536D85', '#9F803E', '#60768A', '#9F803E', '#1F2933'
def paint(c, **colors):
    s=c.get('style','')
    for k,v in colors.items():
        if re.search(r'(?:^|;)'+k+r'=',s):
            s=re.sub(r'(?:(?<=;)|^)'+k+r'=[^;]*',k+'='+v,s)
        else:
            s=s.rstrip(';')+';'+k+'='+v+';'
    c.set('style',s)

for c in tree.findall('.//mxCell'):
    if c.get('id') in ('0','1'): continue
    n=c.get('id').split('-')[-1]
    paint(c,fontColor=ink)
    if c.get('vertex')=='1' and 'image=' not in c.get('style',''):
        paint(c,gradientColor='none')
    if n in ('92','94','95'):
        paint(c,darkOpacity='0',darkOpacity2='0')
    # Inline HTML colors must agree with the cell's font color.
    v=c.get('value')
    if v:
        c.set('value',re.sub(r'(?<![\w-])color:\s*(?:rgb\([^)]*\)|#[0-9a-fA-F]{3,8}|[a-zA-Z]+)', 'color: '+ink,v))
    if c.get('edge')=='1':
        color = '#536A72' if n=='81' else '#957534' if n in ('78','84','82') else '#485B6C'
        if n in ('39','83','173','176'): color='#8193A2'
        paint(c,strokeColor=color)
    if n=='8': paint(c,strokeColor='#8698A7',fillColor='#FFFFFF')
    if n=='105': paint(c,strokeColor='#AC9975',fillColor='#FFFFFF')
    if n=='1': paint(c,strokeColor='#8698A7',fillColor='#FFFFFF')
    if n in ('92','94','95','30'): paint(c,strokeColor=blue,fillColor='#E2EBF3')
    if n in ('111','136'): paint(c,strokeColor=purple,fillColor='#F0E5CD')
    if n in ('2','3','4','5','6'): paint(c,strokeColor=green,fillColor='#E5ECF2')
    if n=='7': paint(c,strokeColor=orange,fillColor='#F0E5CD')
    if n=='24': paint(c,strokeColor='#718697',fillColor='#EBF0F4')
    if n in ('40','175'): paint(c,strokeColor='#94A5B3',fillColor='#F1F4F7')
    if n=='180': paint(c,strokeColor='#9F803E')  # preserve the original enclosing box
    if n in ('43','114','45'): paint(c,fontColor='#293E52')
    # Recolor the five existing feature icons; retain their pixels, shape and alpha.
    if n in ('15','17','31','32','133'):
        s=c.get('style','')
        m=re.search(r'image=data:image/png,([^;]+)',s)
        if m:
            im=Image.open(io.BytesIO(base64.b64decode(m[1]))).convert('RGBA')
            a=np.array(im); rgb=a[:,:,:3].astype(float)
            saturation=(rgb.max(axis=2)-rgb.min(axis=2))/255
            mask=(saturation>.10)&(a[:,:,3]>0)
            color='#A78945' if n=='133' else '#4D6D89'
            target=np.array([int(color[k:k+2],16) for k in (1,3,5)])
            # Preserve antialiasing at white edges while replacing only colored pixels.
            strength=np.clip((255-rgb.min(axis=2))/(255-min(target)),0,1)
            strength[strength>=.65]=1  # flat color within each glyph; retain antialiasing at edges
            recolored=255-(255-target)*strength[:,:,None]
            a[:,:,:3][mask]=np.rint(recolored[mask]).astype('uint8')
            out=io.BytesIO(); Image.fromarray(a).save(out,format='PNG')
            c.set('style',s[:m.start(1)]+base64.b64encode(out.getvalue()).decode()+s[m.end(1):])

# Verify every structural attribute, geometry, endpoint and non-color style is unchanged.
allowed={'fontColor','strokeColor','fillColor','image','gradientColor','darkOpacity','darkOpacity2'}
def non_color_style(s):
    return [part for part in s.split(';') if part and part.split('=',1)[0] not in allowed]
before=original.findall('.//mxCell'); after=tree.findall('.//mxCell')
assert len(before)==len(after)
for a,b in zip(before,after):
    assert a.get('id')==b.get('id')
    assert {k:v for k,v in a.attrib.items() if k not in ('style','value')} == {k:v for k,v in b.attrib.items() if k not in ('style','value')}
    assert ET.tostring(a.find('mxGeometry'))==ET.tostring(b.find('mxGeometry')) if a.find('mxGeometry') is not None else b.find('mxGeometry') is None
    assert non_color_style(a.get('style',''))==non_color_style(b.get('style','')),a.get('id')
    assert re.sub(r'(?<![\w-])color:\s*(?:rgb\([^)]*\)|#[0-9a-fA-F]{3,8}|[a-zA-Z]+)','COLOR',a.get('value','')) == re.sub(r'(?<![\w-])color:\s*(?:rgb\([^)]*\)|#[0-9a-fA-F]{3,8}|[a-zA-Z]+)','COLOR',b.get('value',''))
tree.write(p,encoding='utf-8',xml_declaration=True)
print('Applied flat fills and stronger contrast; removed cube face shading.')
print('Verified all cell IDs, geometry, source/target references, text and non-color styles remain unchanged.')
