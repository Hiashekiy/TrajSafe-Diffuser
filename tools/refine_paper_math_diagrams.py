"""Two portrait publication diagrams, native editable draw.io + inline LaTeX.

The PNG is an independent layout preview. It is never embedded in the draw.io.
"""
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
import shutil
import io
import re
import math
import json
from PIL import Image, ImageDraw, ImageFont
from matplotlib.mathtext import math_to_image

BASE=Path(__file__).resolve().parents[1]/'paper'/'drawio'
INK='#263449'; BLUE='#426E99'; TEAL='#3B7C79'; GRAY='#718096'
PALETTE={'tensor':('#FFFFFF',GRAY),'attn':('#EAF1F8',BLUE),
         'norm':('#F0EBF7','#8672A2'),'mlp':('#FAF2E2','#AA8853'),
         'fixed':('#F3F5F7',GRAY),'op':('#FFFFFF',INK),'text':('#FFFFFF','#FFFFFF')}

def F(tex): return r'\('+tex+r'\)'

class Figure:
    def __init__(self,title,w,h):
        self.title=title;self.w=w;self.h=h;self.nodes={};self.edges=[];self.frames=[]
    def node(self,id,x,y,w,h,lines,kind='tensor',size=16):
        self.nodes[id]=dict(id=id,x=x,y=y,w=w,h=h,lines=lines,kind=kind,size=size)
        return id
    def frame(self,title,x,y,w,h):self.frames.append(dict(title=title,x=x,y=y,w=w,h=h))
    def edge(self,s,t,sp='b',tp='t',points=(),cond=False,label=''):
        self.edges.append(dict(s=s,t=t,sp=sp,tp=tp,points=list(points),cond=cond,label=label))
    def endpoint(self,id,port):
        n=self.nodes[id];x,y,w,h=n['x'],n['y'],n['w'],n['h']
        return {'t':(x+w/2,y),'b':(x+w/2,y+h),'l':(x,y+h/2),'r':(x+w,y+h/2)}[port]
    def save(self,name):
        root=ET.Element('mxfile',host='app.diagrams.net',agent='Codex',type='device')
        page=ET.SubElement(root,'diagram',id='paper-'+name[20:23],name=self.title)
        model=ET.SubElement(page,'mxGraphModel',dx=str(self.w),dy=str(self.h),grid='0',guides='1',
            page='1',pageScale='1',pageWidth=str(self.w),pageHeight=str(self.h),math='1',shadow='0',background='#FFFFFF')
        cells=ET.SubElement(model,'root');ET.SubElement(cells,'mxCell',id='0');ET.SubElement(cells,'mxCell',id='1',parent='0')
        for i,n in enumerate(self.frames):
            c=ET.SubElement(cells,'mxCell',id='section'+str(i),value=n['title'],parent='1',vertex='1',
                style='rounded=0;fillColor=none;strokeColor=#CBD5E1;strokeWidth=1;html=1;align=left;verticalAlign=top;spacingLeft=14;spacingTop=10;fontFamily=Microsoft YaHei;fontSize=17;fontStyle=1;fontColor=#263449;')
            ET.SubElement(c,'mxGeometry',x=str(n['x']),y=str(n['y']),width=str(n['w']),height=str(n['h']),attrib={'as':'geometry'})
        for n in self.nodes.values():
            fill,stroke=PALETTE[n['kind']]
            shape='ellipse;aspect=fixed;' if n['kind']=='op' else ('text;' if n['kind']=='text' else 'rounded=1;arcSize=9;')
            dash='dashed=1;dashPattern=5 3;' if n['kind']=='fixed' else ''
            style=shape+dash+f'html=1;whiteSpace=wrap;fillColor={fill};strokeColor={stroke};strokeWidth=1.3;fontFamily=Microsoft YaHei;fontSize={n["size"]};fontColor={INK};align=center;verticalAlign=middle;spacing=5;'
            c=ET.SubElement(cells,'mxCell',id=n['id'],value='<br>'.join(n['lines']),style=style,parent='1',vertex='1')
            ET.SubElement(c,'mxGeometry',x=str(n['x']),y=str(n['y']),width=str(n['w']),height=str(n['h']),attrib={'as':'geometry'})
        ports={'t':('0.5','0'),'b':('0.5','1'),'l':('0','0.5'),'r':('1','0.5')}
        for i,e in enumerate(self.edges):
            a,b=ports[e['sp']],ports[e['tp']]
            color=TEAL if e['cond'] else INK
            style=f'edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor={color};strokeWidth=1.5;endArrow=block;endSize=7;endFill=1;exitX={a[0]};exitY={a[1]};entryX={b[0]};entryY={b[1]};exitDx=0;exitDy=0;entryDx=0;entryDy=0;fontSize=13;labelBackgroundColor=#FFFFFF;'
            if e['cond']:style+='dashed=1;dashPattern=5 3;'
            c=ET.SubElement(cells,'mxCell',id='edge'+str(i),source=e['s'],target=e['t'],value=e['label'],edge='1',parent='1',style=style)
            g=ET.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
            if e['points']:
                arr=ET.SubElement(g,'Array',attrib={'as':'points'})
                for x,y in e['points']:ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))
        ET.indent(root,space='  ')
        data=ET.tostring(root,encoding='utf-8',xml_declaration=True)
        ET.fromstring(data)
        assert b'$$' not in data and b'data:image' not in data
        p=BASE/name
        if p.exists():
            dest=BASE/'backup'/('inline-math-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
            dest.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest/p.name)
        p.write_bytes(data)
        self.render(p.with_suffix('.preview.png'))
        return str(p)
    def render(self,path):
        scale=2
        im=Image.new('RGB',(self.w*scale,self.h*scale),'white');d=ImageDraw.Draw(im)
        fontpath='C:/Windows/Fonts/msyh.ttc'
        fonts={}
        def font(sz):
            if sz not in fonts:fonts[sz]=ImageFont.truetype(fontpath,sz*scale)
            return fonts[sz]
        for n in self.frames:
            x,y,w,h=n['x'],n['y'],n['w'],n['h']
            d.rectangle([x*scale,y*scale,(x+w)*scale,(y+h)*scale],outline='#CBD5E1',width=2)
            d.text(((x+14)*scale,(y+10)*scale),n['title'],font=font(17),fill=INK)
        # Edges first, then nodes, ensuring connectors never obscure the labels.
        for e in self.edges:
            points=[self.endpoint(e['s'],e['sp'])]+e['points']+[self.endpoint(e['t'],e['tp'])]
            color=TEAL if e['cond'] else INK
            for a,b in zip(points,points[1:]):
                if e['cond']:
                    dist=math.dist(a,b)
                    for offset in range(0,int(dist),10):
                        stop=min(offset+6,dist)
                        p=tuple((a[k]+(b[k]-a[k])*offset/max(dist,1))*scale for k in (0,1))
                        q=tuple((a[k]+(b[k]-a[k])*stop/max(dist,1))*scale for k in (0,1))
                        d.line([p,q],fill=color,width=3)
                else:d.line([(a[0]*scale,a[1]*scale),(b[0]*scale,b[1]*scale)],fill=color,width=3)
            a,b=points[-2:];angle=math.atan2(b[1]-a[1],b[0]-a[0]);length=9
            arrow=[(b[0]*scale,b[1]*scale)]+[((b[0]-length*math.cos(angle+t))*scale,(b[1]-length*math.sin(angle+t))*scale) for t in (-.45,.45)]
            d.polygon(arrow,fill=color)
        for n in self.nodes.values():
            x,y,w,h=n['x'],n['y'],n['w'],n['h'];box=[x*scale,y*scale,(x+w)*scale,(y+h)*scale]
            fill,stroke=PALETTE[n['kind']]
            if n['kind']=='op':d.ellipse(box,fill=fill,outline=stroke,width=3)
            elif n['kind']!='text':d.rounded_rectangle(box,radius=6*scale,fill=fill,outline=stroke,width=3)
            pieces=[]
            for line in n['lines']:
                if line.startswith(r'\(') and line.endswith(r'\)'):
                    buf=io.BytesIO();math_to_image('$'+line[2:-2]+'$',buf,format='png',dpi=180,color=INK)
                    piece=Image.open(buf).convert('RGBA')
                    # Render math at a consistent point size, then scale to fit a node only if necessary.
                    desired=n['size']*scale*1.10
                    factor=desired/max(piece.height,1)
                    factor=min(factor,(w-16)*scale/max(piece.width,1))
                    piece=piece.resize((max(1,int(piece.width*factor)),max(1,int(piece.height*factor))),Image.Resampling.LANCZOS)
                else:
                    f=font(n['size']);bounds=d.textbbox((0,0),line,font=f)
                    piece=Image.new('RGBA',(int(d.textlength(line,font=f))+4,bounds[3]-bounds[1]+6),(255,255,255,0))
                    ImageDraw.Draw(piece).text((2,3-bounds[1]),line,font=f,fill=INK)
                    if piece.width>(w-12)*scale:
                        factor=(w-12)*scale/piece.width
                        piece=piece.resize((int(piece.width*factor),int(piece.height*factor)),Image.Resampling.LANCZOS)
                pieces.append(piece)
            height=sum(p.height for p in pieces)+8*(len(pieces)-1)
            assert height<=(h-4)*scale,(n['id'],height,h)
            yy=int((y+h/2)*scale-height/2)
            for piece in pieces:
                im.paste(piece,(int((x+w/2)*scale-piece.width/2),yy),piece);yy+=piece.height+8
        im.save(path)

def fig33():
    f=Figure('图3-3 骨架匹配与安全几何',1050,1540)
    f.node('title',40,10,970,45,['图 3-3　骨架匹配与安全几何生成'],'text',21)
    f.frame('(a) 骨架编码与拓扑匹配',40,75,970,350)
    f.node('gamma',105,125,300,55,[F(r'\Gamma\in\mathbb{R}^{B\times M\times L\times 2}')])
    f.node('enc',105,215,300,70,['SkeletonEncoder ×2','坐标编码 + 自注意力 + FFN'],'attn')
    f.node('hs',105,325,300,55,[F(r'H_S\in\mathbb{R}^{B\times M\times L\times D}')])
    f.node('hc',615,125,290,55,[F(r'\widetilde H_{\mathrm{ctrl}}\in\mathbb{R}^{B\times C\times D}')])
    f.node('match',615,215,290,70,['MatchBlock',F(r'Q=\widetilde H_{\mathrm{ctrl}},\quad K,V=H_S')],'attn')
    f.node('topo',615,325,290,65,['TopologyHead',F(r'R\ \longrightarrow\ \pi\ \longrightarrow\ m^*')],'mlp')
    f.edge('gamma','enc');f.edge('enc','hs');f.edge('hc','match');f.edge('hs','match','r','l',[(500,352.5),(500,250)])
    f.edge('match','topo')
    f.frame('(b) 安全查询构建',40,455,970,355)
    f.node('sel',115,510,310,55,[F(r'H_S[m^*]\in\mathbb{R}^{B\times Q\times D}')])
    f.node('selg',635,510,280,55,[F(r'\Gamma_{m^*},\quad s_i=i/(Q-1)')])
    f.node('sq',115,595,310,60,['SafetyQueryHead',F(r'D\rightarrow256\rightarrow D')],'mlp')
    f.node('curve',635,595,280,60,['CurveDecoder','固定参数位置采样'],'fixed')
    f.node('hg',115,685,310,55,[F(r'H_\Gamma\in\mathbb{R}^{B\times Q\times D}')])
    f.node('center',635,685,280,55,[F(r'c\in\mathbb{R}^{B\times Q\times2}')])
    f.node('qe',280,765,490,38,[F(r'q_e=H_\Gamma+\mathrm{MLP}_c(\Phi_{xy}(c))')],'mlp')
    f.edge('topo','sel','l','t',[(555,357.5),(555,440),(270,440)])
    f.edge('topo','selg','b','t',[(760,435),(775,435)])
    f.edge('sel','sq');f.edge('sq','hg');f.edge('selg','curve');f.edge('curve','center')
    f.edge('hg','qe','b','l',[(270,784)])
    f.edge('center','qe','b','r',[(775,784)])
    f.frame('(c) 椭圆几何预测',40,840,970,455)
    f.node('ad',360,890,330,52,['AdaLN',F(r'q_e,\ h_t')],'norm')
    f.node('ce',75,980,220,60,[F(r'C_E\in\mathbb{R}^{B\times N_E\times D}')])
    f.node('att',360,985,330,65,['中心偏置交叉注意力',F(r'\mathrm{MHA}(W_Qq_e^{\mathrm{norm}},K,V;B_{\mathrm{geo}})')],'attn')
    f.node('bias',735,960,240,100,[F(r'd_{ij}=\Vert g_j-c_i\Vert'),F(r'B_{ij}^{\mathrm{geo}}=-\bar\alpha_t^2 d_{ij}^2/(2\sigma^2)'),F(r'\mathrm{clip}(B_{ij}^{\mathrm{geo}},-8,0)')],'text',15)
    f.node('hell',360,1090,330,55,[F(r'H_{\mathrm{ell}}=q_e+W_O A_E')])
    f.node('shapehead',360,1185,330,80,['EllipseShapeHead','AdaLN + 逐查询点 MLP',F(r'D\rightarrow256\rightarrow256\rightarrow4')],'mlp')
    f.edge('qe','ad');f.edge('ad','att');f.edge('ce','att','r','l',cond=True)
    f.edge('bias','att','l','r',[(715,1010),(715,1017.5)],True)
    f.edge('att','hell');f.edge('hell','shapehead')
    f.edge('qe','hell','l','l',[(325,784),(325,1117.5)])
    f.node('shape',80,1330,600,60,[F(r'\mathrm{shape}_4=(\log a,\log b,\cos2\theta,\sin2\theta)')])
    f.node('geo',730,1320,280,80,['固定几何构造','半平面 → 凸区域 → 走廊'],'fixed')
    f.edge('shapehead','shape','b','t',[(525,1305),(380,1305)])
    f.edge('shape','geo','r','l',[(705,1360)])
    f.node('foot',40,1430,970,85,[F(r'M=4,\ C=48,\ Q=L=128,\ D=128,\ N_E=1024'),
        '蓝色：注意力；紫色：AdaLN；黄色：MLP；灰色虚线框：固定算子。'], 'text',14)
    return f

def fig34():
    f=Figure('图3-4 特征融合与最终去噪',1050,1450)
    f.node('title',40,10,970,45,['图 3-4　安全特征融合与最终控制点去噪'],'text',21)
    f.frame('(a) 安全特征映射与三流融合',40,75,970,400)
    f.node('hc',80,125,250,55,[F(r'\widetilde H_{\mathrm{ctrl}}\ [B,C,D]')])
    f.node('hp',400,125,220,55,[F(r'H_{\mathrm{path}}\ [B,C,D]')])
    f.node('hell',720,125,230,55,[F(r'H_{\mathrm{ell}}\ [B,Q,D]')])
    f.node('safe',655,220,310,85,['SafetyControlFusion',F(r'Q=\mathrm{AdaLN}(H_{\mathrm{path}},h_t)'),F(r'K,V=H_{\mathrm{ell}}')],'attn',15)
    f.node('cat',290,345,490,42,[F(r'[\widetilde H_{\mathrm{ctrl}};H_{\mathrm{path}};H_{\mathrm{safe}}^{\mathrm{ctrl}}]')],'tensor')
    f.node('fuse',290,410,490,50,['FusionMLP：LN → Linear → GELU → Linear',F(r'3D\rightarrow2D\rightarrow D')],'mlp',14)
    f.edge('hell','safe','b','t',[(835,200),(810,200)])
    f.edge('hp','safe','r','l',[(640,152.5),(640,262.5)])
    f.edge('hc','cat','b','l',[(205,366)])
    f.edge('hp','cat','b','t',[(510,325),(535,325)])
    f.edge('safe','cat','b','r',[(810,366)])
    f.edge('cat','fuse')
    f.frame('(b) TrajBlock ×3',40,510,970,675)
    f.node('ad1',370,565,330,45,['AdaLN'],'norm')
    f.node('sa',370,635,330,65,['多头自注意力','4 heads + 可学习相对索引偏置'],'attn')
    f.node('p1',517,725,36,36,[F(r'\oplus')],'op',19)
    f.node('ad2',370,785,330,45,['AdaLN'],'norm')
    f.node('ca',370,855,330,60,['场景交叉注意力',F(r'Q=X^{\mathrm{norm}},\quad K,V=C_G')],'attn')
    f.node('p2',517,940,36,36,[F(r'\oplus')],'op',19)
    f.node('ad3',370,1000,330,45,['AdaLN'],'norm')
    f.node('ffn',370,1070,330,65,['FFN（GELU）',F(r'D\rightarrow512\rightarrow D')],'mlp')
    f.node('p3',517,1150,36,36,[F(r'\oplus')],'op',19)
    f.node('ht',75,575,185,60,[F(r'h_t\ [B,D]')])
    f.node('cg',775,855,190,60,[F(r'C_G\ [B,N_G,D]')])
    f.edge('fuse','ad1')
    for a,b in [('ad1','sa'),('sa','p1'),('p1','ad2'),('ad2','ca'),('ca','p2'),('p2','ad3'),('ad3','ffn'),('ffn','p3')]:f.edge(a,b)
    f.edge('fuse','p1','l','l',[(335,435),(335,743)])
    f.edge('p1','p2','l','l',[(315,743),(315,958)])
    f.edge('p2','p3','l','l',[(295,958),(295,1168)])
    for a,y in [('ad1',587.5),('ad2',807.5),('ad3',1022.5)]:
        f.edge('ht',a,'r','l',[(275,605),(275,y)],True)
    f.edge('cg','ca','l','r',cond=True)
    f.node('repeat',770,1030,195,95,['同一结构串联 3 层','与控制主干参数独立'],'text',14)
    f.node('head',370,1220,330,52,['共享输出投影',F(r'\mathrm{Linear}(D,2)')],'mlp',15)
    f.node('bd',370,1295,330,52,['BoundaryDecoder','固定权重端点补偿'],'fixed',15)
    f.node('sg',80,1295,190,52,[F(r'[S;G]\ [B,2,2]')])
    f.node('out',740,1295,240,52,[F(r'\widehat Q_0\ [B,C,2]')])
    f.edge('p3','head');f.edge('head','bd');f.edge('sg','bd','r','l',cond=True);f.edge('bd','out','r','l')
    f.node('foot',40,1380,970,55,[F(r'C=48,\ D=128,\ N_G=256;\quad \oplus:\ \mathrm{residual\ addition}'),
        '输出送入图 3-5 的安全修正；虚线箭头表示条件输入。'],'text',14)
    return f

if __name__=='__main__':
    result=[]
    for f,name in [(fig33(),'TrajSafe-Diffuser_图3-3_§3.2骨架匹配与安全几何-原排版修订版.drawio'),
                   (fig34(),'TrajSafe-Diffuser_图3-4_§3.3融合与最终去噪-原排版修订版.drawio')]:
        p=f.save(name)
        result.append(dict(file=p,nodes=len(f.nodes),edges=len(f.edges),math=1,images=0,display_math=0))
    (BASE/'行内公式论文排版-校验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
