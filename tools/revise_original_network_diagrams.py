"""Revise labels and level of detail while preserving the user's original layout."""
from pathlib import Path
import copy
import re
import xml.etree.ElementTree as ET
import json

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / 'paper' / 'drawio'
FILES = [
    'TrajSafe-Diffuser_图3-2_§3.3控制主干8×TrajBlock.drawio',
    'TrajSafe-Diffuser_图3-3_§3.2骨架匹配与安全几何.drawio',
    'TrajSafe-Diffuser_图3-4_§3.3融合与最终去噪.drawio',
    'TrajSafe-Diffuser_图3-5_§3.4闭环安全修正与反馈.drawio',
]

COLORS = {
    'norm': ('#EDE9FE', '#7C3AED'),
    'attn': ('#DBEAFE', '#2563EB'),
    'mlp': ('#FEF3C7', '#B7791F'),
    'fixed': ('#F1F5F9', '#64748B'),
    'tensor': ('#FFFFFF', '#475569'),
}

def style_dict(cell):
    result = {}
    for part in cell.get('style', '').split(';'):
        if '=' in part:
            k, v = part.split('=', 1)
            result[k] = v
        elif part:
            result[part] = None
    return result

def restyle(cell, **kwargs):
    s = style_dict(cell)
    s.update({k: str(v) for k, v in kwargs.items()})
    cell.set('style', ';'.join(k if v is None else k+'='+v for k,v in s.items())+';')

def set_kind(cell, kind):
    fill, stroke = COLORS[kind]
    restyle(cell, fillColor=fill, strokeColor=stroke, fontSize=13,
            fontFamily='Microsoft YaHei', fontColor='#1F2937', strokeWidth=1.4)
    if kind == 'fixed':
        restyle(cell, dashed=1, dashPattern='5 3')
    elif kind != 'tensor':
        restyle(cell, dashed=0, rounded=1, arcSize=8)

def add_vertex(root, id, value, x,y,w,h, kind='tensor'):
    c = ET.SubElement(root, 'mxCell', id=id, value=value, vertex='1', parent='1',
                      style='rounded=1;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;')
    ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
    set_kind(c,kind)
    return c

def add_edge(root,id,source,target,value='',cond=False):
    c=ET.SubElement(root,'mxCell',id=id,source=source,target=target,value=value,edge='1',parent='1',
                    style='edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeWidth=1.5;')
    ET.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
    if cond:
        restyle(c,strokeColor='#0F766E',dashed=1,dashPattern='5 3')
    return c

def route(c,points,**ports):
    g=c.find('mxGeometry')
    for child in list(g):
        if child.tag=='Array':g.remove(child)
    arr=ET.SubElement(g,'Array',attrib={'as':'points'})
    for x,y in points:ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))
    restyle(c,**ports)

def math_html(v):
    # Native HTML subscripts/superscripts render without MathJax or rare glyph fonts.
    v=v.replace('H̃_ctrl','H<sub>ctrl</sub><sup>fb</sup>').replace('Q̃_raw','Q<sub>raw</sub>')
    v=v.replace('Q̂_0','Q<sub>0</sub><sup>pred</sup>').replace('X⁰','X<sup>0</sup>')
    v=v.replace('shape₄','shape<sub>4</sub>').replace('𝒞','C_safe').replace('𝒫_j','P_j')
    v=v.replace('ᾱ_t²','ᾱ<sub>t</sub><sup>2</sup>').replace('10⁻³','10<sup>−3</sup>')
    for token in ('ctrl','safe','path','ell','clean','attn','norm','geo','max','fb','raw','next','new'):
        v=re.sub(r'\b([A-Za-zΔω]+)_'+token+r'\b',r'\1<sub>'+token+'</sub>',v)
    for token in ('G','E','S','T','F','C','Γ','xy','1D','h','t','i','j','e'):
        v=re.sub(r'\b([A-Za-zγβαΦσ]+)_'+token+r'\b',r'\1<sub>'+token+'</sub>',v)
    return v

def latex_to_label(v):
    # Existing edge-label cells also need to stop showing raw TeX when math is off.
    replacements={'\\oplus':'⊕','\\odot':'⊙','\\sigma':'σ','\\lambda':'λ',
                  '\\rho':'ρ','\\eta':'η','\\Delta':'Δ','\\to':'→',
                  '\\Phi':'Φ','\\gamma':'Γ','\\cdot':'·'}
    v=re.sub(r'<span[^>]*>|</span>','',v)
    v=v.replace('\\(','').replace('\\)','')
    for a,b in replacements.items(): v=v.replace(a,b)
    v=re.sub(r'\\(?:mathrm|text)\{([^}]+)\}',r'\1',v)
    v=v.replace('{','').replace('}','')
    return v

LABELS = [
{
 'bb_t0':'<b>X⁰</b><br>[B,C,D]', 'bb_cg':'<b>C_G</b><br>[B,N_G,D]', 'bb_ht':'<b>h_t</b><br>[B,D]',
 'bb_fr':'<b>TrajBlock × N_T</b>　N_T=8',
 'bb_ad1':'AdaLN(X, h_t)', 'bb_qkv':'Q,K,V = Linear(X_norm)　D→3D',
 'bb_h0':'Head 1<br>d_h=32','bb_h1':'Head 2<br>d_h=32','bb_h2':'Head 3<br>d_h=32','bb_h3':'Head 4<br>d_h=32',
 'bb_cat':'Concat heads → Linear(D,D)', 'bb_p1':'⊕','bb_p2':'⊕','bb_p3':'⊕',
 'bb_ad2':'AdaLN(X, h_t)', 'bb_ca':'场景交叉注意力　Q=X_norm，K/V=C_G',
 'bb_ad3':'AdaLN(X, h_t)', 'bb_ffn':'FFN　D→512→D（GELU）',
 'bb_fb':'⊕','bb_out':'<b>H̃_ctrl</b><br>[B,C,D]',
 'fb_f1':'<b>FeedbackEncoder</b>　逐控制点共享',
 'fb_in1':'Q_safe<br>[B,C,2]', 'fb_in2':'ΔQ<br>[B,C,2]', 'fb_in3':'v<br>[B]',
 'fb_cat5':'Concat(Q_safe, ΔQ, v)　[B,C,5]',
 'fb_l1':'Linear(5,64)','fb_silu':'SiLU','fb_l2':'Linear(64,D)　零初始化',
 'fb_hfb':'h_fb　[B,C,D]', 'fb_f2':'<b>FeedbackFusion</b>　主干输出后注入',
 'fb_cat2':'Concat(H_ctrl, h_fb)　[B,C,2D]', 'fb_lg':'Linear(2D,D)',
 'fb_sig':'Sigmoid','fb_val':'⊙ v（按批次广播）','fb_mul':'⊙ h_fb','fb_add':'⊕ H_ctrl',
 'fb_out':'H̃_ctrl　[B,C,D]',
},
{
 'T':'图 3-3　骨架匹配与安全几何生成',
 'sm':'候选骨架 Γ　[B,M,L,2]', 'pe':'坐标嵌入 Φ_xy + 序列位置编码 PE_1D',
 'fr1':'<b>SkeletonEncoder × N_S</b>　N_S=2',
 'sk':'相对位置自注意力 + FFN<br>4 heads，D=128',
 'hs':'H_S　[B,M,L,D]', 'fr2':'<b>MatchBlock</b>',
 'mm':'交叉注意力（Q=H̃_ctrl，K/V=H_S）<br>残差 → LayerNorm → FFN → 残差',
 'r':'R　[B,M,C,D]', 'th':'TopologyHead：MeanPool_C → MLP → masked Softmax',
 'hctrl':'H̃_ctrl<br>[B,C,D]',
 'fr3':'<b>EllipseGeometry</b>　中心偏置的局部地图交互',
 'hsq':'H_S[m*]　[B,Q,D]', 'sqh':'SafetyQueryHead：LN → D→256→D（SiLU）',
 'sprog':'固定弧长进度　s_i=i/(Q−1)', 'cdec':'CurveDecoder(Γ_m*, s) → c　[B,Q,2]',
 'qe':'q_e = H_Γ + MLP_C(Φ_xy(c))',
 'qq':'Q_attn = W_Q · AdaLN(q_e, h_t)', 'ce':'C_E<br>[B,N_E,D]',
 'kv':'K=W_K C_E，V=W_V C_E',
 'bias':'B_geo = clip(−ᾱ_t² ‖g_j−c_i‖²/(2σ²), −8, 0)',
 'att':'A_E = W_O · MHA(Q_attn,K,V; B_geo)',
 'hell':'H_ell = q_e + A_E　[B,Q,D]',
 'sch':'σ=0.25<br>g_j：地图 token 中心',
 'fr4':'<b>EllipseShapeHead</b>　逐查询点共享',
 'ead':'AdaLN(H_ell, h_t)', 'el1':'Linear(D,256) → SiLU',
 'el2':'Linear(256,256) → SiLU','el3':'Linear(256,4)',
 'sh':'shape₄ = (log a, log b, cos 2θ, sin 2θ)　[B,Q,4]',
 'det':'固定几何：椭圆 → 分离半平面 → 凸区域 → 连续性检查',
 'corr':'安全走廊 𝒞 = {𝒫_j}', 'glab':'中心偏置示意<br>深色：较高权重',
 'N':'B：批量；M=4：候选数；C=48：控制点数；Q=L=128：安全查询/骨架 token 数<br>D=128；N_E=1024。m* 为选中候选索引；灰色虚线框：无参数算子。',
},
{
 'T':'图 3-4　三流融合与最终控制点去噪',
 'hc':'H̃_ctrl　[B,C,D]　控制特征', 'hp':'H_path　[B,C,D]　路径匹配特征',
 'hs':'H_safe^ctrl　[B,C,D]　安全→控制交叉注意力输出',
 'cat':'Concat(H̃_ctrl, H_path, H_safe^ctrl)　[B,C,3D]',
 'ffr':'<b>FusionMLP</b>', 'ln':'LayerNorm(3D)', 'l1':'Linear(3D,2D)',
 'gel':'GELU','l2':'Linear(2D,D)','f':'F　[B,C,D]',
 'dfr':'<b>FinalDenoiser：TrajBlock × N_F</b>　N_F=3',
 'ad1':'AdaLN(X, h_t)','qkv':'Q,K,V = Linear(X_norm)　D→3D',
 'mh1':'4-head self-attention + 相对索引偏置 → W_O',
 'r1':'⊕','ad2':'AdaLN(X, h_t)',
 'kv2':'场景交叉注意力　Q=X_norm，K/V=C_G',
 'r2':'⊕','ad3':'AdaLN(X, h_t)','ffn':'FFN　D→512→D（GELU）','r3':'⊕',
 'blk':'每层含 3 个预归一化子层与残差；3 层参数独立。',
 'cout':'H_clean　[B,C,D]','cg':'C_G<br>[B,N_G,D]','ht':'h_t<br>[B,D]',
 'hfr':'<b>输出投影</b>　与粗预测共享 head_p',
 'hp2':'Linear(D,2)　逐控制点共享', 'raw':'Q̃_raw　[B,C,2]',
 'bd':'BoundaryDecoder：端点误差的固定权重补偿',
 'out':'Q̂_0　[B,C,2]　送入安全修正（图 3-5）',
 'N':'C=48，D=128，N_G=256；⊕ 表示残差相加。<br>FinalDenoiser 与 8 层控制主干结构相同、参数独立。BoundaryDecoder 无可学习参数。',
 'FT':'AdaLN(X,h_t) = (1+γ(h_t)) ⊙ LN(X) + β(h_t)',
},
{
 'T':'图 3-5　控制点安全修正与跨步反馈',
 'L1':'(a) 固定算子：控制点空间 ALM', 'R1':'(b) 可学习：反馈编码与门控',
 'raw':'Q̂_0　[B,C,2]　当前干净预测',
 'pack':'冻结的连续走廊约束　A_j E_lr Q − b_j ≤ 0',
 'g':'违规 g　[B,P,4,F_max]',
 'dual':'约束权重 ω = ReLU(λ + ρg)',
 'lam':'ω　[B,P,4,F_max]（供解析梯度使用）',
 'step':'近端更新：解析约束梯度 + Cholesky 求解',
 'sm':'曲线位移限幅 → 控制点轴平滑 → 固定端点',
 'clip':'对偶更新　λ ← ReLU(λ + ρg_new)',
 'tol':'max ReLU(g_new) ≤ ε，或达到 K 次内迭代',
 'q0s':'Q_0^safe　[B,C,2]',
 'ddim':'DDIM(q_t, Q_0^safe) → q_{t−1}',
 'hard':'固定端点　q_{t−1,0}=S，q_{t−1,C−1}=G',
 'qnext':'q_{t−1}　[B,C,2]',
 'out':'最终 L1 硬投影 → B-spline 解码 → 512 点验证',
 'warm':'3 步预热后激活；冻结走廊与约束包。<br>仅约束验证通过的结果更新历史反馈。',
 'fr':'<b>FeedbackEncoder + FeedbackFusion</b>',
 'fin':'Concat(Q_0^safe, ΔQ, v)　[B,C,5]',
 'fl1':'Linear(5,64)','fsi':'SiLU','fl2':'Linear(64,D)　零初始化',
 'fhf':'h_fb　[B,C,D]','fcat':'Concat(H_ctrl, h_fb)　[B,C,2D]',
 'flg':'Linear(2D,D)',
 'op':'G = Sigmoid(·) ⊙ v<br>H̃_ctrl = H_ctrl + G ⊙ h_fb',
 'hcin':'H_ctrl<br>[B,C,D]', 'hc2':'H̃_ctrl　[B,C,D]<br>主干输出后注入（图 3-2）',
 'N':'ΔQ = Q_0^safe − Q̂_0；v∈{0,1} 为历史反馈有效标志。<br>P：Bézier 段数；4：每段控制点数；F_max：最大半平面数。K≤10，ρ=5，η=0.05，ε=10⁻³。',
 'F':'当前 C48 配置启用最终 L1 硬投影（final_project=true）；图中 ALM 与几何算子均无可学习参数。',
}]

report=[]
for number,(filename,labels) in enumerate(zip(FILES,LABELS),2):
    source=DIR/filename
    tree=ET.parse(source)
    model=tree.getroot().find('.//mxGraphModel')
    root=model.find('root')
    original={c.get('id'):copy.deepcopy(c) for c in root}
    removed=set()
    if number==2:
        # Discard an unfinished copy outside the canvas; retain central block and right feedback branch.
        for c in list(root):
            if (c.get('id') or '').startswith('YoYnVApzswjtIt1ukSXO-'):
                removed.add(c.get('id')); root.remove(c)
        for c in list(root):
            if c.get('source') in removed or c.get('target') in removed or c.get('parent') in removed:
                removed.add(c.get('id')); root.remove(c)
        model.set('pageWidth','1800'); model.set('pageHeight','1040')
    elif number==3:
        model.set('pageHeight','1800')
    elif number==4:
        model.set('pageHeight','1660')
    else:
        model.set('pageWidth','1470'); model.set('pageHeight','1100')
    model.set('math','0')
    for c in root:
        cid=c.get('id')
        if cid in labels: c.set('value',labels[cid])
        else: c.set('value',latex_to_label(c.get('value','')))
        if c.get('vertex')=='1':
            restyle(c,fontFamily='Microsoft YaHei',fontColor='#1F2937')
            if c.find('mxGeometry').get('relative')=='1':
                restyle(c,fontSize=11)
                continue
            v=c.get('value',''); st=c.get('style','')
            if 'text;' in st or cid in ('T','L1','R1','N','F','FT','blk','sch','glab'):
                restyle(c,fontSize=12 if cid not in ('T','L1','R1') else 15)
            elif 'AdaLN' in v: set_kind(c,'norm')
            elif any(t in v for t in ('注意力','attention','Head ','Q,K,V','Q_attn','K=W_K','MHA')): set_kind(c,'attn')
            elif any(t in v for t in ('Linear','SiLU','GELU','LayerNorm','FFN','Concat','Sigmoid','TopologyHead','QueryHead')): set_kind(c,'mlp')
            elif cid in ('det','cdec','sprog','pack','dual','step','sm','clip','tol','ddim','hard','out') and number in (3,5): set_kind(c,'fixed')
            elif cid=='bd': set_kind(c,'fixed')
            elif 'shape=cube' in st: set_kind(c,'tensor')
        if c.get('edge')=='1':
            restyle(c,fontFamily='Microsoft YaHei',fontSize=11,strokeWidth=1.5,fontColor='#475569')
    lookup={c.get('id'):c for c in root}
    if number==2:
        lookup['bb_er2'].set('source','bb_p1')
        add_vertex(root,'rev_title','图 3-2　控制点主干与历史反馈',380,90,660,30)
        restyle(root[-1],fillColor='none',strokeColor='none',align='left',fontSize=17,fontStyle=1)
        restyle(lookup['bb_fr'],verticalAlign='top',align='left',spacingTop=8,spacingLeft=10)
        # Increase the head labels to fit their two lines, preserving their locations.
        for id in ('bb_h0','bb_h1','bb_h2','bb_h3'):
            lookup[id].find('mxGeometry').set('height','42')
            restyle(lookup[id],fontSize=11)
        add_edge(root,'rev_norm_qkv','bb_ad1','bb_qkv')
        add_edge(root,'rev_final_output','bb_fb','bb_out')
        c=add_edge(root,'rev_residual1','bb_t0','bb_p1')
        route(c,[(405,110),(405,390)],exitX=0,exitY=0.5,entryX=0,entryY=0.5)
        for target in ('bb_ad2','bb_ad3'):
            c=add_edge(root,'rev_time_'+target,'bb_ht',target,cond=True)
            y=float(lookup[target].find('mxGeometry').get('y'))+15
            route(c,[(1080,90),(1080,y)],exitX=1,exitY=0.5,entryX=1,entryY=0.5)
        c=lookup['f7rEOZ8qxl4HRO4dl9AD-1']
        route(c,[(410,90),(410,482)],exitX=0,exitY=0.5,entryX=0,entryY=0.5)
        c=lookup['fb_e18']
        c.set('target','bb_fb');c.set('source','fb_mul')
        c.set('value','门控增量（右侧为展开）')
        c=add_edge(root,'rev_ctrl_gate','bb_p3','fb_cat2','H_ctrl',True)
        route(c,[(1200,660),(1200,442)],exitX=1,exitY=0.5,entryX=0,entryY=0.5)
        add_vertex(root,'rev_input_note','X⁰ = MLP_T(Φ_xy(Q_t)) + PE_1D<br>C=48，D=128，N_G=256；⊕：残差相加',380,930,663,64)
        add_vertex(root,'rev_no_fb','v=0 时：H̃_ctrl=H_ctrl<br>右侧门控是中央 ⊕ 的展开',1287,780,380,60)
    if number==3:
        lookup['th'].set('value','TopologyHead：MeanPool_C → MLP<br>masked Softmax → m*')
        # Queries, keys/values and bias are parallel inputs, not sequential layers.
        lookup['E13'].set('target','att')
        route(lookup['E13'],[(190,947),(190,1085)],exitX=0,exitY=0.5,entryX=0,entryY=0.5)
        lookup['E15'].set('target','att')
        route(lookup['E15'],[(875,993),(875,1085)],exitX=1,exitY=0.5,entryX=1,entryY=0.5)
        add_vertex(root,'rev_ht','h_t　[B,D]',40,923,140,50)
        add_edge(root,'rev_geometry_time','rev_ht','qq','',True)
        add_edge(root,'rev_shape_time','rev_ht','ead','',True)
        c=add_edge(root,'rev_center_bias','cdec','bias','c, ᾱ_t',True)
        route(c,[(945,842),(945,1039)],exitX=1,exitY=0.5,entryX=1,entryY=0.5)
    if number==4:
        for id in ('r1','r2','r3'):
            g=lookup[id].find('mxGeometry'); g.set('x','586'); g.set('width','28');g.set('height','28')
            restyle(lookup[id],ellipse=1,rounded=0,aspect='fixed',fontSize=18)
        add_vertex(root,'rev_hell','H_ell　[B,Q,D]',40,158,220,54)
        add_edge(root,'rev_safe_attn','rev_hell','hs','K/V')
        # Express the actual safety-to-control operator at the original third input position.
        lookup['hs'].set('value','SafetyControlFusion：Q=AdaLN(H_path,h_t)，K/V=H_ell<br>H_safe^ctrl = H_path + CrossAttn　[B,C,D]')
        lookup['hs'].find('mxGeometry').set('height','48')
        set_kind(lookup['hs'],'attn')
        add_edge(root,'rev_path_cond','hp','hs','Q')
        add_edge(root,'rev_time_cond','ht','hs','h_t',True)
        add_vertex(root,'rev_sg','S,G　[B,2,2]',40,1340,220,54)
        add_edge(root,'rev_sg_edge','rev_sg','bd','端点条件',True)
        add_edge(root,'rev_denoiser_time','ht','ad2','',True)
        for a,b,y,lane in [('f','r1',729,255),('r1','r2',861,270),('r2','r3',987,285)]:
            c=add_edge(root,'rev_res_'+b,a,b)
            ay=float(lookup[a].find('mxGeometry').get('y'))+float(lookup[a].find('mxGeometry').get('height'))/2
            route(c,[(lane,ay),(lane,y)],exitX=0,exitY=0.5,entryX=0,entryY=0.5)
        # The explanatory note is not a computational layer.
        lookup['E19'].set('target','cout')
        root.remove(lookup['E20'])
        lookup['blk'].find('mxGeometry').set('x','970')
        lookup['blk'].find('mxGeometry').set('width','230')
    if number==5:
        g=lookup['hcin'].find('mxGeometry');g.set('width','180');g.set('height','60');g.set('x','1260');g.set('y','380')
        g=lookup['fr'].find('mxGeometry');g.set('height','400')
        restyle(lookup['warm'],fontSize=11)
        add_edge(root,'rev_feedback_ctrl','hcin','fcat','H_ctrl')
        add_edge(root,'rev_feedback_res','hcin','op','残差')
        lookup['E23'].set('value','Q_safe, ΔQ, v（验证通过）')
        c=add_edge(root,'rev_alm_loop','tol','g','未收敛：下一内迭代')
        route(c,[(275,487),(275,214)],exitX=0,exitY=0.5,entryX=0,entryY=0.5)
    for c in root:
        c.set('value',math_html(c.get('value','')))
        if c.get('vertex')=='1' and c.get('id') not in ('rev_title',):
            st=style_dict(c)
            if st.get('shape')=='cube' and c.get('id') not in ('bb_t0','bb_cg','bb_ht','bb_out','fb_in1','fb_in2','fb_in3','fb_hfb','fb_out','sm','hs','r','hsq','ce','hell','sh','corr','hc','hp','f','cout','cg','ht','raw','out','g','lam','q0s','qnext','fhf','hc2'):
                restyle(c,shape='rectangle',rounded=1)
    # Keep frames before inner cells in document order.
    frames=[]
    for c in list(root):
        if c.get('id') in ('bb_fr','fb_f1','fb_f2','fr1','fr2','fr3','fr4','ffr','dfr','hfr','fr'):
            frames.append(c);root.remove(c)
            restyle(c,fillColor='none',strokeColor='#94A3B8',fontSize=13,fontStyle=1,
                    verticalAlign='top',align='left',spacingTop=6,spacingLeft=8)
    for i,c in enumerate(frames):root.insert(i+2,c)
    ids={c.get('id') for c in root}
    dangling=[c.get('id') for c in root if c.get('edge')=='1' and
              (c.get('source') not in ids or c.get('target') not in ids)]
    if dangling: raise RuntimeError(f'Dangling edges: {dangling}')
    output=source.with_name(source.stem+'-原排版修订版.drawio')
    ET.indent(tree,space='  ')
    data=ET.tostring(tree.getroot(),encoding='utf-8',xml_declaration=True)
    ET.fromstring(data)
    output.write_bytes(data)
    changed_geometry=[]
    for c in root:
        id=c.get('id')
        if id in original and c.get('vertex')=='1':
            a=original[id].find('mxGeometry');b=c.find('mxGeometry')
            if a is not None and b is not None and a.attrib!=b.attrib:changed_geometry.append(id)
    item={'file':output.name,'images':sum('shape=image' in c.get('style','') for c in root),
          'dangling_edges':dangling,'geometry_adjustments':changed_geometry,
          'removed_off_canvas_copy':sorted(removed),'original_preserved':True}
    report.append(item)
    print(json.dumps(item,ensure_ascii=False))
(DIR/'原排版修订版-校验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
