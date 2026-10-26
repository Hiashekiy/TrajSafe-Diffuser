"""Native, editable method figures with a two-level publication layout.

No bitmap/SVG embedding. Labels use draw.io MathJax inline delimiters.
Outputs are a new edition; existing hand-edited diagrams are never overwritten.
"""
from pathlib import Path
import xml.etree.ElementTree as E
import json
import shutil
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/figures/methods/publication-v2'
SRC = ROOT / 'paper/drawio/论文最终插图'
INK = '#263445'
COLORS = {'blue': ('#EAF2FA', '#477BA6'), 'green': ('#EAF5F0', '#37816D'),
          'amber': ('#FAF2E5', '#AC833D'), 'purple': ('#F0EBF6', '#8670A4'),
          'gray': ('#F4F5F7', '#8893A0'), 'white': ('#FFFFFF', '#748394')}


class Figure:
    def __init__(self, number, width, height):
        self.number, self.width, self.height = number, width, height
        self.doc = E.Element('mxfile', host='app.diagrams.net', version='31.5.3')
        diagram = E.SubElement(self.doc, 'diagram', id=f'publication-3-{number}', name=f'图3-{number}')
        model = E.SubElement(diagram, 'mxGraphModel', dx=str(width), dy=str(height),
                             grid='0', guides='1', tooltips='1', connect='1', arrows='1',
                             fold='1', page='1', pageScale='1', pageWidth=str(width),
                             pageHeight=str(height), math='1', shadow='0', background='#FFFFFF')
        self.root = E.SubElement(model, 'root')
        E.SubElement(self.root, 'mxCell', id='0')
        E.SubElement(self.root, 'mxCell', id='1', parent='0')
        self.count = 0

    def box(self, ident, label, x, y, w, h, color='white', extra='', font=19):
        fill, stroke = COLORS[color]
        style = (f'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor={fill};'
                 f'strokeColor={stroke};strokeWidth=1.4;fontColor={INK};fontSize={font};'
                 'fontFamily=Microsoft YaHei;spacing=7;shadow=0;' + extra)
        cell = E.SubElement(self.root, 'mxCell', id=ident, value=label, style=style, vertex='1', parent='1')
        E.SubElement(cell, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), **{'as': 'geometry'})
        return ident

    def text(self, ident, label, x, y, w, h=28, font=19, bold=False, align='center'):
        return self.box(ident, label, x, y, w, h, extra=f'fillColor=none;strokeColor=none;align={align};fontStyle={int(bold)};spacing=0;', font=font)

    def panel(self, ident, label, x, y, w, h):
        self.box(ident, '', x, y, w, h, extra='rounded=0;strokeColor=#D5DCE3;fillColor=#FFFFFF;')
        self.text(ident+'-label', label, x+16, y+10, w-32, 30, font=21, bold=True, align='left')

    def line(self, points, dashed=False, arrow=True, color=INK, label='', width=1.7):
        self.count += 1
        style = (f'edgeStyle=none;rounded=0;html=1;strokeWidth={width};strokeColor={color};'
                 f'endArrow={"block" if arrow else "none"};endFill=1;endSize=7;'
                 f'dashed={int(dashed)};dashPattern=6 4;fontSize=17;fontFamily=Microsoft YaHei;'
                 'fontColor=#263445;labelBackgroundColor=#FFFFFF;')
        cell = E.SubElement(self.root, 'mxCell', id=f'e{self.count}', value=label, style=style, edge='1', parent='1')
        geom = E.SubElement(cell, 'mxGeometry', relative='1', **{'as': 'geometry'})
        for key, p in [('sourcePoint', points[0]), ('targetPoint', points[-1])]:
            E.SubElement(geom, 'mxPoint', x=str(p[0]), y=str(p[1]), **{'as': key})
        if len(points) > 2:
            arr = E.SubElement(geom, 'Array', **{'as': 'points'})
            for x, y in points[1:-1]:
                E.SubElement(arr, 'mxPoint', x=str(x), y=str(y))

    def stack(self, ident, label, x, y, w, h, color='blue'):
        self.box(ident+'-back2', '', x+10, y-10, w, h, color)
        self.box(ident+'-back1', '', x+5, y-5, w, h, color)
        return self.box(ident, label, x, y, w, h, color)

    def circle(self, ident, label, x, y, size=30):
        return self.box(ident, label, x, y, size, size, extra='shape=ellipse;rounded=0;spacing=0;', font=21)

    def save(self, name):
        # Bind native edge terminals to module cells, so moving a module in
        # draw.io also moves its incident arrows. Keep explicit bend points.
        vertices = []
        for c in self.root.findall('mxCell'):
            g = c.find('mxGeometry')
            if c.get('vertex') == '1' and g is not None and c.get('value'):
                vertices.append((c, *(float(g.get(k)) for k in ('x','y','width','height'))))
        for c in self.root.findall('mxCell'):
            if c.get('edge') != '1':
                continue
            g = c.find('mxGeometry')
            for terminal, key, prefix in [('source','sourcePoint','exit'), ('target','targetPoint','entry')]:
                p = g.find(f"mxPoint[@as='{key}']")
                px, py = float(p.get('x')), float(p.get('y'))
                for v,x,y,w,h in reversed(vertices):
                    on_edge = ((abs(px-x)<.1 or abs(px-x-w)<.1) and y<=py<=y+h or
                               (abs(py-y)<.1 or abs(py-y-h)<.1) and x<=px<=x+w)
                    if on_edge:
                        c.set(terminal,v.get('id'))
                        c.set('style',c.get('style')+f'{prefix}X={(px-x)/w};{prefix}Y={(py-y)/h};{prefix}Perimeter=0;')
                        break
        SRC.mkdir(parents=True, exist_ok=True)
        dest = SRC / name
        if dest.exists():
            backup = SRC.parent / 'backup' / ('publication-v2-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, backup / dest.name)
        E.indent(self.doc)
        E.ElementTree(self.doc).write(dest, encoding='utf-8', xml_declaration=True)
        E.parse(dest)
        copy = OUT / 'drawio' / f'fig3-{self.number}.drawio'
        shutil.copy2(dest, copy)
        return {'number': f'3-{self.number}', 'source': str(dest), 'publication_copy': str(copy),
                'svg': f'fig3-{self.number}.svg', 'png': f'fig3-{self.number}.png',
                'pdf': f'fig3-{self.number}.pdf', 'page': [self.width, self.height]}


def fig32():
    f = Figure(2, 1040, 760)
    f.panel('a', '(a) 控制主干与反馈注入', 12, 12, 470, 696)
    f.panel('b', '(b) 单个 TrajBlock', 502, 12, 526, 696)
    f.box('qt', r'带噪控制点 \(Q_t\)', 120, 66, 245, 52)
    f.box('enc', '控制点编码器', 120, 150, 245, 56, 'amber')
    f.stack('backbone', 'TrajBlock ×8', 120, 253, 245, 78)
    f.text('ctrl', r'\(H_{\mathrm{ctrl}}\)', 158, 362, 170, 32)
    f.box('fb', '门控反馈注入', 120, 429, 245, 60, 'green')
    f.box('out', r'\(\widetilde H_{\mathrm{ctrl}}\)', 120, 552, 245, 55)
    for a,b in [(118,150),(206,253),(331,362),(394,429),(489,552)]:
        f.line([(242,a),(242,b)])
    f.text('cond', r'\(h_t\)<br>\(C_G\)', 27, 223, 86, 48, font=18)
    f.line([(70,271),(70,287),(120,287)], True, color='#37816D')
    f.box('history', r'历史反馈<br>\(H_{\mathrm{fb}},v\)', 30, 497, 155, 52, 'green', font=17)
    f.line([(106,497),(106,460),(120,460)], True, color='#37816D')
    f.text('detail', '反馈机制见图 3-5(b)', 217, 503, 235, 26, font=17)
    f.text('branches', '输出送入骨架匹配与粗预测分支', 49, 628, 396, 28, font=18)
    f.line([(242,607),(242,625)])
    f.text('xin', r'输入 \(X\)', 716, 59, 130)
    ys = [111, 285, 459]
    operations = [('多头自注意力', '4 heads · 相对索引偏置', 'blue'),
                  ('场景交叉注意力', r'\(K,V=C_G\)', 'blue'),
                  ('前馈网络', r'\(D\rightarrow512\rightarrow D\)', 'amber')]
    for i,(y,(op,sub,color)) in enumerate(zip(ys,operations)):
        f.box(f'n{i}', r'AdaLN · \(h_t\)', 674, y, 245, 42, 'purple', font=18)
        f.box(f'op{i}', op+'<br>'+sub, 674, y+59, 245, 58, color, font=18)
        f.circle(f'add{i}', '⊕', 781, y+129, 30)
        f.line([(796,y+42),(796,y+59)])
        f.line([(796,y+117),(796,y+129)])
        tap = y-20 if i == 0 else y-10
        f.line([(796,tap),(641,tap),(641,y+144),(781,y+144)])
        if i < 2:
            f.line([(796,y+159),(796,ys[i+1])])
    f.line([(796,87),(796,111)])
    f.line([(796,618),(796,653)])
    f.text('xout', r'输出 \(X^{\mathrm{out}}\)', 699, 657, 200)
    f.box('cg', r'\(C_G\)', 941, 353, 63, 40, font=17)
    f.line([(941,373),(919,373)], True, color='#37816D')
    f.text('legend', r'实线：特征传递　虚线：条件输入　⊕：残差相加　\(C=48,\ D=128\)', 25, 720, 990, 28, font=17)
    return f.save('TrajSafe-Diffuser_图3-2_控制主干与TrajBlock-论文插图版.drawio')


def scene(f, tag, x, y, mode):
    # Conceptual native vector scene, not an experimental result.
    f.box(tag+'-bg', '', x, y, 280, 164, extra='rounded=0;strokeColor=#D5DCE3;')
    for j,(ox,oy,ow,oh) in enumerate([(98,0,46,51),(139,117,56,47),(232,0,48,22)]):
        f.box(tag+f'-obs{j}', '', x+ox, y+oy, ow, oh, 'gray', extra='rounded=0;fillColor=#D8DDE3;strokeColor=#A1ABB6;')
    pts = [(x+22,y+130),(x+68,y+91),(x+123,y+78),(x+177,y+87),(x+224,y+71),(x+258,y+47)]
    if mode == 'corridor':
        for j,(cx,cy,w,h,rot) in enumerate([(68,95,109,58,-26),(126,81,105,58,0),(186,83,104,54,-12),(230,64,68,47,-30)]):
            f.box(tag+f'-region{j}', '', x+cx-w/2,y+cy-h/2,w,h,'green',
                  extra=f'shape=hexagon;rounded=0;size=0.16;rotation={rot};fillOpacity=35;strokeWidth=1.1;')
    if mode == 'ellipse':
        for j,(cx,cy,rot) in enumerate([(68,91,-30),(123,78,0),(177,87,4),(224,71,-28)]):
            f.box(tag+f'-ell{j}', '',x+cx-39,y+cy-23,78,46,'amber',
                  extra=f'shape=ellipse;rounded=0;rotation={rot};fillOpacity=35;strokeWidth=1.2;')
            f.circle(tag+f'-center{j}', '',x+cx-3,y+cy-3,6)
    if mode == 'skeleton':
        f.line([(x+22,y+130),(x+51,y+65),(x+164,y+60),(x+258,y+47)], True, False, '#9AA4AF', width=1.2)
    f.line(pts, arrow=False, color='#3678A4', width=2.5)
    f.circle(tag+'-start','',x+17,y+125,10)
    f.circle(tag+'-goal','',x+253,y+42,10)
    f.text(tag+'-S','S',x+8,y+139,28,22,font=17)
    f.text(tag+'-G','G',x+252,y+14,28,22,font=17)


def fig33():
    f = Figure(3, 1040, 870)
    f.panel('a', '(a) 骨架匹配与安全几何网络', 12, 12, 1016, 539)
    f.box('gamma', r'候选骨架 \(\{\Gamma_m\}\)', 36, 72, 208, 50)
    f.stack('skel', 'SkeletonEncoder ×2', 277, 72, 230, 50)
    f.box('hs', r'\(H_S\)', 549, 72, 112, 50)
    f.box('ctrl', r'控制特征<br>\(\widetilde H_{\mathrm{ctrl}}\)', 733, 65, 235, 64)
    f.line([(244,97),(277,97)])
    f.line([(507,97),(549,97)])
    f.box('match', r'MatchBlock · \(h_t\)', 733, 164, 235, 55, 'blue')
    f.line([(850,129),(850,164)])
    f.line([(661,97),(694,97),(694,190),(733,190)])
    f.box('topo', r'TopologyHead<br>选择 \(m^*\)', 733, 248, 235, 62, 'amber')
    f.line([(850,219),(850,248)])
    f.box('path', r'PathFeatureHead<br>\(R_{m^*}\rightarrow H_{\mathrm{path}}\)', 733, 348, 235, 62, 'amber',font=18)
    f.line([(850,310),(850,348)],True,color='#748394')
    f.line([(968,190),(990,190),(990,379),(968,379)])
    f.text('r', r'\(R\)',996,284,25,28,font=17)
    f.text('pathout', '通路特征 → 图 3-4', 730, 441, 244, 28, font=17)
    f.line([(850,410),(850,440)])
    f.box('select', r'所选骨架 \(\Gamma_{m^*}\)<br>及其 token', 36, 164, 208, 62)
    f.line([(140,122),(140,164)])
    f.line([(605,122),(605,144),(24,144),(24,178),(36,178)])
    f.line([(850,310),(850,327),(24,327),(24,195),(36,195)],True,color='#748394')
    f.box('sq', 'SafetyQueryHead', 277, 164, 230, 56, 'amber')
    f.line([(244,190),(277,190)])
    f.box('center', r'固定弧长采样<br>中心 \(c_i\)', 36, 257, 208, 58, 'gray', extra='dashed=1;')
    f.line([(140,226),(140,257)])
    f.box('query', r'安全查询 \(q_e\)<br>内容特征 + 中心编码', 277, 257, 230, 58, 'amber',font=18)
    f.line([(392,220),(392,257)])
    f.line([(244,286),(277,286)])
    f.box('geom', '中心偏置交叉注意力', 277, 355, 295, 64, 'blue')
    f.line([(392,315),(392,355)])
    f.box('mem', r'局部场景 \(C_E\)<br>时间 \(h_t\)', 36, 357, 208, 60, font=18)
    f.line([(244,387),(277,387)],True,color='#37816D')
    f.box('shape', r'EllipseShapeHead · \(h_t\)', 277, 460, 270, 54, 'amber',font=18)
    f.line([(392,419),(392,460)])
    f.text('hell', r'\(H_{\mathrm{ell}}\) → 图 3-4', 518, 440, 203, 27,font=17)
    f.line([(572,387),(600,387),(600,438)])
    f.text('shapecap', r'椭圆形状 \((a_i,b_i,\theta_i)\)', 54, 559, 345, 29,font=18)
    f.line([(392,514),(392,557)])
    f.panel('b', '(b) 从所选通路到连续凸走廊（几何示意）', 12, 602, 1016, 250)
    for tag,x,mode,title in [('s',35,'skeleton','所选骨架通路'),('e',381,'ellipse','固定中心与预测椭圆'),('c',725,'corridor','确定性几何构造：凸走廊')]:
        scene(f, tag, x, 650, mode)
        f.text(tag+'-title',title,x,821,280,24,font=17)
    f.line([(323,734),(370,734)])
    f.line([(669,734),(715,734)])
    return f.save('TrajSafe-Diffuser_图3-3_骨架匹配与安全几何-论文插图版.drawio')


def fig34():
    f = Figure(4, 1040, 678)
    f.panel('a', '(a) 安全信息对齐与三流融合', 12, 12, 1016, 364)
    f.box('ctrl', r'控制特征<br>\(\widetilde H_{\mathrm{ctrl}}\)', 45, 75, 235, 65)
    f.box('path', r'通路特征<br>\(H_{\mathrm{path}}\)', 356, 75, 235, 65)
    f.box('ell', r'椭圆特征<br>\(H_{\mathrm{ell}}\)', 740, 75, 235, 65, 'green')
    f.box('attn', r'安全交叉注意力<br>\(N_q\) 个椭圆 → \(C\) 个控制点', 664, 175, 335, 65,'blue',font=18)
    f.line([(857,140),(857,175)])
    f.line([(591,107),(622,107),(622,207),(664,207)])
    f.text('q', 'Query', 535, 174, 75,22,font=16)
    f.text('kv', 'Key / Value', 866, 144, 120,22,font=16)
    f.box('cat', r'通道拼接 \([\widetilde H_{\mathrm{ctrl}};H_{\mathrm{path}};H_{\mathrm{safe}}^{\mathrm{ctrl}}]\)', 260, 282, 522, 55,font=18)
    f.line([(162,140),(162,309),(260,309)])
    f.line([(474,140),(474,282)])
    f.line([(831,240),(831,309),(782,309)])
    f.panel('b', '(b) 特征精炼与控制点输出',12,407,1016,248)
    f.box('mlp', r'FusionMLP<br>\(3D\rightarrow2D\rightarrow D\)',45,478,217,66,'amber',font=18)
    f.line([(521,337),(521,391),(153,391),(153,478)])
    f.stack('denoise','FinalDenoiser<br>TrajBlock ×3',307,478,227,66,'blue')
    f.box('head',r'共享输出头<br>\(D\rightarrow2\)',581,478,161,66,'amber',font=18)
    f.box('boundary','BoundaryDecoder<br>固定端点补偿',788,478,212,66,'gray',extra='dashed=1;',font=18)
    for a,b in [(262,307),(534,581),(742,788)]: f.line([(a,511),(b,511)])
    f.text('cond',r'\(h_t,\ C_G\)',327,560,190,28,font=18)
    f.line([(420,560),(420,544)],True,color='#37816D')
    f.text('ends',r'\(S,G\)',817,561,154,28,font=18)
    f.line([(894,561),(894,544)],True,color='#37816D')
    f.text('out',r'\(\widehat Q_0\in\mathbb R^{C\times2}\)',770,610,243,30,font=20)
    f.line([(1000,511),(1014,511),(1014,604),(894,604),(894,610)])
    f.text('note','TrajBlock 结构见图 3-2(b)；两套堆叠参数独立。',38,610,706,28,font=17)
    return f.save('TrajSafe-Diffuser_图3-4_融合与最终去噪-论文插图版.drawio')


def fig35():
    f = Figure(5,1040,762)
    f.panel('a','(a) 推理期安全修正与反向采样',12,12,588,690)
    f.panel('b','(b) 历史安全反馈的门控注入',620,12,408,690)
    f.box('pred',r'网络干净预测 \(\widehat Q_0\)',145,70,329,54)
    f.box('constraints','冻结的连续凸走廊约束',38,166,230,66,'gray',extra='dashed=1;',font=18)
    f.box('alm','ALM 控制点修正<br>近端更新 · 位移限幅 · 对偶更新',300,166,266,92,'green',font=17)
    f.line([(310,124),(310,143),(433,143),(433,166)])
    f.line([(268,199),(300,199)])
    f.line([(566,210),(581,210),(581,274),(433,274),(433,258)],True,color='#748394')
    f.box('safe',r'当前修正结果 \(Q_0^{\mathrm{safe}}\)',145,310,329,54)
    f.line([(433,258),(433,288),(310,288),(310,310)])
    f.box('ddim','DDIM 状态更新<br>重新固定起终点',145,403,329,62,'blue')
    f.line([(310,364),(310,403)])
    f.text('qt',r'当前 \(Q_t\)',32,417,100,32,font=18)
    f.line([(132,434),(145,434)])
    f.box('next',r'下一反向步 \(Q_{t^-}\)',145,505,329,54)
    f.line([(310,465),(310,505)])
    f.line([(145,532),(26,532),(26,97),(145,97)],True,color='#748394')
    f.text('return','重新进入网络',38,565,203,28,font=17)
    f.box('final','采样结束：B 样条曲线解码<br>连续约束检查 · 密集碰撞评价',145,615,420,62,'gray',extra='dashed=1;',font=17)
    f.line([(310,559),(310,615)])
    f.box('check','历史缓存更新<br>约束检查通过时写入',664,70,320,60,'green',font=18)
    f.line([(474,337),(610,337),(610,99),(664,99)],True,color='#37816D')
    f.box('features',r'历史控制点 · 修正量 · 有效标志<br>\([Q_{\mathrm{hist}}^{\mathrm{safe}};\Delta Q_{\mathrm{hist}};v]\)',648,170,352,68,font=18)
    f.line([(824,130),(824,170)])
    f.box('enc',r'FeedbackEncoder<br>\(5\rightarrow64\rightarrow D\)',689,277,270,65,'amber',font=19)
    f.line([(824,238),(824,277)])
    f.box('hfb',r'反馈特征 \(H_{\mathrm{fb}}\)',689,378,270,52,'green')
    f.line([(824,342),(824,378)])
    f.box('gate',r'有效性门控残差<br>\(\widetilde H_{\mathrm{ctrl}}=H_{\mathrm{ctrl}}+G_{\mathrm{fb}}\odot H_{\mathrm{fb}}\)',646,493,356,76,'green',font=17)
    f.line([(824,430),(824,493)])
    f.box('hctrl',r'下一次主干<br>\(H_{\mathrm{ctrl}}\)',646,443,161,43,font=16)
    f.line([(672,486),(672,493)])
    f.text('gateformula',r'\(G_{\mathrm{fb}}=v\,\sigma(W_g[H_{\mathrm{ctrl}};H_{\mathrm{fb}}]+b_g)\)',643,593,365,32,font=16)
    f.text('fbout','注入一次 → 骨架匹配与最终去噪',638,648,372,27,font=17)
    f.line([(824,569),(824,584)],arrow=False)
    f.line([(824,626),(824,646)])
    f.text('legend','实线：当前计算　灰色虚线：采样回环　绿色虚线：跨步历史反馈',29,716,980,30,font=17)
    return f.save('TrajSafe-Diffuser_图3-5_安全修正与历史反馈-论文插图版.drawio')


def main():
    (OUT/'drawio').mkdir(parents=True,exist_ok=True)
    records=[fig32(),fig33(),fig34(),fig35()]
    (OUT/'figure-map.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(records,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
