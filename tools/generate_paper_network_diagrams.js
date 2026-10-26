const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'paper', 'drawio');
fs.mkdirSync(OUT, { recursive: true });

const esc = (s) => String(s ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;').replaceAll('"', '&quot;');

const P = {
  ink: '#1F2937', muted: '#64748B', tensorFill: '#F8FAFC', tensorStroke: '#475569',
  attnFill: '#DBEAFE', attnStroke: '#2563EB', normFill: '#EDE9FE', normStroke: '#7C3AED',
  mlpFill: '#FEF3C7', mlpStroke: '#D97706', geoFill: '#DCFCE7', geoStroke: '#15803D',
  stateFill: '#FEE2E2', stateStroke: '#DC2626', feedback: '#B45309', cond: '#0F766E',
  panel: '#F8FAFC', panelStroke: '#CBD5E1', white: '#FFFFFF',
};

class Diagram {
  constructor(name, id, width, height) {
    this.name = name; this.id = id; this.width = width; this.height = height;
    this.cells = []; this.seq = 1;
  }
  add(xml) { this.cells.push(xml); }
  style(kind, extra = '') {
    const base = 'whiteSpace=wrap;html=1;fontFamily=Microsoft YaHei;fontSize=12;fontColor=#1F2937;strokeWidth=1.6;shadow=0;';
    const styles = {
      tensor: `shape=cube;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.10;size=10;fillColor=${P.tensorFill};strokeColor=${P.tensorStroke};`,
      attention: `rounded=1;arcSize=10;fillColor=${P.attnFill};strokeColor=${P.attnStroke};`,
      norm: `rounded=1;arcSize=18;fillColor=${P.normFill};strokeColor=${P.normStroke};`,
      mlp: `rounded=1;arcSize=10;fillColor=${P.mlpFill};strokeColor=${P.mlpStroke};`,
      fixed: `rounded=1;arcSize=8;fillColor=${P.geoFill};strokeColor=${P.geoStroke};dashed=1;dashPattern=7 4;`,
      state: `shape=cube;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.10;size=10;fillColor=${P.stateFill};strokeColor=${P.stateStroke};`,
      op: `ellipse;aspect=fixed;fillColor=#FFFFFF;strokeColor=${P.ink};fontSize=17;fontStyle=1;`,
      module: `rounded=1;arcSize=10;fillColor=#FFFFFF;strokeColor=${P.tensorStroke};`,
      panel: `rounded=1;arcSize=8;fillColor=${P.panel};strokeColor=${P.panelStroke};dashed=1;dashPattern=5 4;verticalAlign=top;align=left;spacingLeft=10;spacingTop=6;fontStyle=1;fontSize=13;`,
      title: `text;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=18;fontStyle=1;`,
      subtitle: `text;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=11;fontColor=${P.muted};`,
      label: `text;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=11;`,
      legend: `rounded=1;arcSize=6;fillColor=#FFFFFF;strokeColor=${P.panelStroke};align=left;verticalAlign=middle;spacingLeft=10;fontSize=11;`,
    };
    return base + styles[kind] + extra;
  }
  node(id, value, x, y, w, h, kind = 'module', extra = '') {
    this.add(`        <mxCell id="${id}" value="${esc(value)}" style="${esc(this.style(kind, extra))}" vertex="1" parent="1"><mxGeometry x="${x}" y="${y}" width="${w}" height="${h}" as="geometry"/></mxCell>`);
    return id;
  }
  text(id, value, x, y, w, h, kind = 'label', extra = '') {
    return this.node(id, value, x, y, w, h, kind, extra);
  }
  edge(id, source, target, value = '', kind = 'main', points = []) {
    const variants = {
      main: `edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=${P.ink};strokeWidth=2.2;endArrow=block;endFill=1;fontFamily=Microsoft YaHei;fontSize=10;fontColor=${P.ink};`,
      cond: `edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=${P.cond};strokeWidth=1.5;dashed=1;dashPattern=6 4;endArrow=block;endFill=1;fontFamily=Microsoft YaHei;fontSize=10;fontColor=${P.cond};`,
      feedback: `edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=${P.feedback};strokeWidth=1.8;dashed=1;dashPattern=8 4;endArrow=block;endFill=1;fontFamily=Microsoft YaHei;fontSize=10;fontColor=${P.feedback};`,
    };
    const arr = points.length ? `<Array as="points">${points.map(([x,y]) => `<mxPoint x="${x}" y="${y}"/>`).join('')}</Array>` : '';
    this.add(`        <mxCell id="${id}" value="${esc(value)}" style="${esc(variants[kind])}" edge="1" parent="1" source="${source}" target="${target}"><mxGeometry relative="1" as="geometry">${arr}</mxGeometry></mxCell>`);
  }
  xml() {
    return `<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" modified="2026-09-30T00:00:00.000Z" agent="Codex" version="28.2.5" type="device">
  <diagram id="${this.id}" name="${esc(this.name)}">
    <mxGraphModel dx="${this.width}" dy="${this.height}" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="${this.width}" pageHeight="${this.height}" math="0" shadow="0">
      <root><mxCell id="0"/><mxCell id="1" parent="0"/>
${this.cells.join('\n')}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>\n`;
  }
  save(filename) {
    const p = path.join(OUT, filename);
    fs.writeFileSync(p, this.xml(), 'utf8');
    return p;
  }
}

function legend(d, y, width) {
  const value = '<b>统一标记</b>　▱ 张量　<span style="color:#2563EB">■</span> 注意力　<span style="color:#7C3AED">■</span> AdaLN　<span style="color:#D97706">■</span> MLP/投影　<span style="color:#15803D">□</span> 固定算子　<span style="color:#0F766E">┄</span> 条件注入　<span style="color:#B45309">┄</span> 跨步反馈';
  d.node('legend', value, 40, y, width - 80, 42, 'legend');
}

function fig32() {
  const d = new Diagram('图3-2 控制主干网络', 'fig32-opt', 2200, 820);
  d.text('title', '控制点扩散主干：TrajectoryEncoder + 8 × TrajBlock', 40, 18, 1500, 32, 'title');
  d.text('sub', 'C=48，D=128，4 个注意力头（d_h=32），FFN=512', 40, 52, 900, 22, 'subtitle');
  d.node('panel', 'TrajBlock ℓ（重复 8 次，参数独立）', 650, 135, 1460, 390, 'panel');

  d.node('qt', '<b>Q_t</b><br>[B,48,2]', 40, 270, 145, 66, 'state');
  d.node('enc', '<b>TrajectoryEncoder</b><br>Φ_xy + MLP + PE_1D', 225, 260, 220, 86, 'mlp');
  d.node('x0', '<b>X⁰</b><br>[B,48,128]', 485, 270, 135, 66, 'tensor');

  d.node('xl', '<b>Xˡ</b>', 690, 292, 82, 56, 'tensor');
  d.node('ad1', '<b>AdaLN</b><br>h_t', 805, 286, 115, 68, 'norm');
  d.node('sa', '<b>MHSA</b><br>相对索引偏置', 955, 286, 145, 68, 'attention');
  d.node('p1', '⊕', 1135, 296, 48, 48, 'op');
  d.node('ad2', '<b>AdaLN</b><br>h_t', 1215, 286, 115, 68, 'norm');
  d.node('ca', '<b>Cross-Attn</b><br>Q=X，K/V=C_G', 1365, 286, 165, 68, 'attention');
  d.node('p2', '⊕', 1565, 296, 48, 48, 'op');
  d.node('ad3', '<b>AdaLN</b><br>h_t', 1645, 286, 115, 68, 'norm');
  d.node('ffn', '<b>FFN</b><br>128→512→128', 1795, 286, 155, 68, 'mlp');
  d.node('p3', '⊕', 1985, 296, 48, 48, 'op');
  d.node('xo', '<b>Xˡ⁺¹</b>', 2050, 292, 50, 56, 'tensor');

  d.node('ht', '<b>h_t</b><br>[B,128]', 1040, 86, 130, 54, 'tensor');
  d.node('cg', '<b>C_G</b><br>[B,256,128]', 1415, 86, 150, 54, 'tensor');
  d.node('hctrl', '<b>H_ctrl</b><br>[B,48,128]', 1590, 600, 170, 68, 'tensor');
  d.node('hfb', '<b>h_fb</b><br>[B,48,128]', 1130, 610, 150, 58, 'tensor');
  d.node('gate', '<b>FeedbackFusion</b><br>σ(W[H_ctrl;h_fb]) ⊙ h_fb', 1320, 590, 230, 88, 'mlp');
  d.node('hout', '<b>H̃_ctrl</b><br>[B,48,128]', 1810, 600, 170, 68, 'tensor');

  d.edge('e1','qt','enc'); d.edge('e2','enc','x0'); d.edge('e3','x0','xl');
  d.edge('e4','xl','ad1'); d.edge('e5','ad1','sa'); d.edge('e6','sa','p1');
  d.edge('r1','xl','p1','残差','main',[[790,390],[1159,390]]);
  d.edge('e7','p1','ad2'); d.edge('e8','ad2','ca'); d.edge('e9','ca','p2');
  d.edge('r2','p1','p2','残差','main',[[1160,415],[1589,415]]);
  d.edge('e10','p2','ad3'); d.edge('e11','ad3','ffn'); d.edge('e12','ffn','p3');
  d.edge('r3','p2','p3','残差','main',[[1590,440],[2009,440]]); d.edge('e13','p3','xo');
  d.edge('c1','ht','ad1','','cond'); d.edge('c2','ht','ad2','','cond'); d.edge('c3','ht','ad3','','cond');
  d.edge('c4','cg','ca','K/V','cond');
  d.edge('stack','xo','hctrl','×8 后','main',[[2075,555],[1675,555]]);
  d.edge('fb1','hctrl','gate'); d.edge('fb2','hfb','gate','','feedback'); d.edge('fb3','gate','hout','门控残差','feedback');
  legend(d, 740, 2200);
  return d.save('TrajSafe-Diffuser_图3-2_§3.3控制主干8×TrajBlock-优化版.drawio');
}

function fig33() {
  const d = new Diagram('图3-3 骨架匹配与安全几何网络', 'fig33-opt', 2200, 1040);
  d.text('title','骨架匹配、拓扑选择与生成式安全几何',40,18,1500,32,'title');
  d.text('sub','M=4 条候选骨架，Q=128 个安全查询；实线为特征流，绿色虚线框为无参数几何',40,52,1200,22,'subtitle');
  d.node('p1','A　拓扑匹配',35,105,2130,280,'panel');
  d.node('p2','B　安全几何生成',35,415,2130,470,'panel');

  d.node('skel','<b>Γ_m</b><br>[B,4,L,2]',70,210,140,62,'tensor');
  d.node('se','<b>SkeletonEncoder ×2</b><br>Φ_xy + PE_1D + Transformer',250,195,260,92,'attention');
  d.node('hs','<b>H_S</b><br>[B,4,L,128]',550,210,160,62,'tensor');
  d.node('hc','<b>H̃_ctrl</b><br>[B,48,128]',550,120,160,62,'tensor');
  d.node('match','<b>MatchBlock</b><br>control→skeleton cross-attn',760,188,240,86,'attention');
  d.node('r','<b>R</b><br>[B,4,48,128]',1045,210,170,62,'tensor');
  d.node('topo','<b>TopologyHead</b><br>候选打分',1260,195,190,86,'mlp');
  d.node('mstar','<b>m*</b><br>选中拓扑',1495,210,145,62,'tensor');
  d.node('select','<b>H_S[m*]</b><br>[B,L,128]',1690,210,170,62,'tensor');

  d.node('sq','<b>SafetyQueryHead</b><br>128→512→128',90,535,215,80,'mlp');
  d.node('hg','<b>H_Γ</b><br>[B,128,128]',345,544,165,62,'tensor');
  d.node('prog','<b>s_i=i/(Q−1)</b><br>[B,128]',90,680,180,58,'tensor');
  d.node('curve','<b>CurveDecoder</b><br>Γ_m* + s_i',315,670,190,76,'fixed');
  d.node('center','<b>c</b><br>[B,128,2]',550,680,145,58,'tensor');
  d.node('qe','<b>q_e</b><br>H_Γ + MLP(Φ_xy(c))',745,535,210,80,'mlp');
  d.node('ce','<b>C_E</b><br>[B,1024,128]',745,690,175,62,'tensor');
  d.node('ht','<b>h_t</b><br>[B,128]',965,690,135,62,'tensor');
  d.node('ab','<b>ᾱ_t</b><br>扩散累计系数',1125,680,140,62,'tensor');
  d.node('ell','<b>EllipseGeometry</b><br>中心偏置 Cross-Attn',1005,535,235,80,'attention');
  d.node('hell','<b>H_ell</b><br>[B,128,128]',1285,544,170,62,'tensor');
  d.node('shape','<b>EllipseShapeHead</b><br>AdaLN + MLP',1500,525,210,92,'mlp');
  d.node('s4','<b>shape₄</b><br>[B,128,4]',1755,544,160,62,'tensor');
  d.node('corr','<b>RegionBuilder</b><br>椭圆→半平面→连续凸走廊',1955,525,185,92,'fixed');
  d.node('ruse','<b>R[m*]</b><br>[B,48,128]',720,790,155,62,'tensor');
  d.node('pfh','<b>PathFeatureHead</b><br>128→512→128',900,780,195,82,'mlp');
  d.node('fuse','<b>SafetyControlFusion</b><br>Q=H_path，K/V=H_ell',1320,735,250,86,'attention');
  d.node('hsafe','<b>H_safe^ctrl</b><br>[B,48,128]',1580,745,180,62,'tensor');
  d.node('hpath','<b>H_path</b><br>[B,48,128]',1120,790,170,62,'tensor');

  d.edge('a1','skel','se'); d.edge('a2','se','hs'); d.edge('a3','hs','match'); d.edge('a4','hc','match');
  d.edge('a5','match','r'); d.edge('a6','r','topo'); d.edge('a7','topo','mstar'); d.edge('a8','mstar','select');
  d.edge('b1','select','sq','选中骨架','main',[[1775,450],[195,450]]); d.edge('b2','sq','hg');
  d.edge('b3','mstar','curve','Γ_m*','main',[[1565,395],[410,395],[410,650]]); d.edge('b4','prog','curve'); d.edge('b5','curve','center');
  d.edge('b6','hg','qe'); d.edge('b7','center','qe','位置编码','cond'); d.edge('b8','qe','ell'); d.edge('b9','ce','ell','K/V','cond'); d.edge('b10','ht','ell','AdaLN','cond'); d.edge('b10a','ab','ell','偏置强度','cond');
  d.edge('b11','ell','hell'); d.edge('b12','hell','shape'); d.edge('b13','ht','shape','','cond'); d.edge('b14','shape','s4'); d.edge('b15','s4','corr');
  d.edge('path1','r','ruse','按 m* 选择','main',[[1130,430],[795,430]]); d.edge('path2','ruse','pfh'); d.edge('path3','pfh','hpath');
  d.edge('b16','hell','fuse','K/V','main',[[1370,655],[1445,655]]); d.edge('b17','hpath','fuse','Q'); d.edge('b18','fuse','hsafe');
  legend(d, 940, 2200);
  return d.save('TrajSafe-Diffuser_图3-3_§3.2骨架匹配与安全几何-优化版.drawio');
}

function fig34() {
  const d = new Diagram('图3-4 融合与最终去噪网络', 'fig34-opt', 2100, 850);
  d.text('title','三流特征融合与最终控制点去噪网络',40,18,1500,32,'title');
  d.text('sub','FusionMLP 将控制、路径与安全特征压缩到 D=128；FinalDenoiser 由 3 个独立 TrajBlock 组成',40,52,1500,22,'subtitle');
  d.node('panel','FinalDenoiser ×3',880,130,790,390,'panel');

  d.node('hc','<b>H̃_ctrl</b><br>[B,48,128]',45,160,175,64,'tensor');
  d.node('hp','<b>H_path</b><br>[B,48,128]',45,260,175,64,'tensor');
  d.node('hs','<b>H_safe^ctrl</b><br>[B,48,128]',45,360,175,64,'tensor');
  d.node('cat','<b>Concat</b><br>[B,48,384]',285,245,165,76,'module');
  d.node('fusion','<b>FusionMLP</b><br>LN(3D) → 3D→2D → GELU → 2D→D',505,225,285,116,'mlp');
  d.node('f','<b>F</b><br>[B,48,128]',820,250,130,64,'tensor');

  d.node('ad1','<b>AdaLN</b>',930,230,105,58,'norm');
  d.node('sa','<b>MHSA</b><br>relative bias',1065,220,140,78,'attention');
  d.node('p1','⊕',1230,235,48,48,'op');
  d.node('ad2','<b>AdaLN</b>',1300,230,105,58,'norm');
  d.node('ca','<b>Cross-Attn</b><br>K/V=C_G',1435,220,150,78,'attention');
  d.node('p2','⊕',1600,235,48,48,'op');
  d.node('ad3','<b>AdaLN</b>',1035,365,105,58,'norm');
  d.node('ffn','<b>FFN</b><br>128→512→128',1175,355,155,78,'mlp');
  d.node('p3','⊕',1360,370,48,48,'op');
  d.node('hout','<b>H_clean</b><br>[B,48,128]',1500,362,145,64,'tensor');
  d.node('ht','<b>h_t</b>',1060,105,110,50,'tensor');
  d.node('cg','<b>C_G</b><br>[B,256,128]',1435,95,150,58,'tensor');

  d.node('head','<b>共享 head_p</b><br>Linear 128→2',1730,225,175,78,'mlp');
  d.node('raw','<b>Q_raw</b><br>[B,48,2]',1940,235,125,62,'state');
  d.node('bd','<b>BoundaryDecoder</b><br>固定 S/G 端点',1730,400,175,78,'fixed');
  d.node('out','<b>Q*</b><br>[B,48,2]',1940,410,125,62,'state');
  d.node('sg','<b>S, G</b><br>[B,2,2]',1730,550,145,58,'tensor');

  d.edge('e1','hc','cat'); d.edge('e2','hp','cat'); d.edge('e3','hs','cat'); d.edge('e4','cat','fusion'); d.edge('e5','fusion','f'); d.edge('e6','f','ad1');
  d.edge('e7','ad1','sa'); d.edge('e8','sa','p1'); d.edge('e9','p1','ad2'); d.edge('e10','ad2','ca'); d.edge('e11','ca','p2');
  d.edge('e12','p2','ad3','', 'main', [[1625,325],[1088,325]]); d.edge('e13','ad3','ffn'); d.edge('e14','ffn','p3'); d.edge('e15','p3','hout');
  d.edge('r1','f','p1','残差','main',[[885,330],[1254,330]]); d.edge('r2','p1','p2','残差','main',[[1254,315],[1624,315]]); d.edge('r3','p2','p3','残差','main',[[1624,455],[1384,455]]);
  d.edge('c1','ht','ad1','','cond'); d.edge('c2','ht','ad2','','cond'); d.edge('c3','ht','ad3','','cond'); d.edge('c4','cg','ca','K/V','cond');
  d.edge('e16','hout','head'); d.edge('e17','head','raw'); d.edge('e18','raw','bd'); d.edge('c5','sg','bd','端点条件','cond'); d.edge('e19','bd','out');
  legend(d, 760, 2100);
  return d.save('TrajSafe-Diffuser_图3-4_§3.3融合与最终去噪-优化版.drawio');
}

function fig35() {
  const d = new Diagram('图3-5 闭环安全修正与反馈', 'fig35-opt', 2320, 980);
  d.text('title','推理期闭环安全修正与历史反馈网络',40,18,1500,32,'title');
  d.text('sub','上：冻结走廊约束下的控制点 ALM；下：可学习反馈编码与门控残差。二者通过 Q_safe 与 ΔQ 形成跨步闭环',40,52,1800,22,'subtitle');
  d.node('p1','A　每个有效去噪步：无参数安全修正',35,115,2250,330,'panel');
  d.node('p2','B　下一去噪步：可学习历史反馈',35,475,2250,330,'panel');

  d.node('raw','<b>Q_raw</b><br>[B,48,2]',70,245,150,64,'state');
  d.node('pack','<b>冻结约束包</b><br>Mq≤v',280,145,170,62,'fixed');
  d.node('eval','<b>ConstraintState</b><br>连续 B-spline / Bézier 违规 g',280,235,230,86,'fixed');
  d.node('dual','<b>对偶更新</b><br>λ ← ReLU(λ + ρg)',560,235,210,86,'fixed');
  d.node('prox','<b>原变量近端步</b><br>解析梯度 + Cholesky 求解',820,225,245,106,'fixed');
  d.node('smooth','<b>平滑与步长限制</b><br>kernel=3；曲线位移≤0.20',1115,235,245,86,'fixed');
  d.node('loop','<b>内迭代 ×K</b><br>K≤10，tol=10⁻³',1410,235,190,86,'fixed');
  d.node('safe','<b>Q_safe</b><br>[B,48,2]',1650,245,150,64,'state');
  d.node('ddim','<b>DDIM step</b><br>更新扩散状态',1835,235,150,86,'fixed');
  d.node('hard','<b>Endpoint clamp</b><br>q₀=S，q₄₇=G',2015,235,150,86,'fixed');
  d.node('next','<b>q_{t−1}</b><br>[B,48,2]',2190,245,105,64,'state');

  d.node('delta','<b>ΔQ</b><br>Q_safe−Q_raw',190,600,160,64,'state');
  d.node('valid','<b>valid</b><br>[B]',190,700,120,56,'tensor');
  d.node('cat','<b>Concat</b><br>[Q_safe, ΔQ, valid]<br>[B,48,5]',410,615,190,92,'module');
  d.node('fbenc','<b>FeedbackEncoder</b><br>5→64→128（末层 zero-init）',660,610,235,102,'mlp');
  d.node('hfb','<b>h_fb</b><br>[B,48,128]',945,628,155,64,'tensor');
  d.node('hctrl','<b>H_ctrl</b><br>[B,48,128]',945,525,155,64,'tensor');
  d.node('gate','<b>门控</b><br>σ(Linear[H_ctrl;h_fb]) × valid',1160,600,255,106,'mlp');
  d.node('mul','⊙',1460,625,48,48,'op');
  d.node('add','⊕',1570,625,48,48,'op');
  d.node('hout','<b>H̃_ctrl</b><br>[B,48,128]',1680,617,170,64,'tensor');
  d.node('forward','<b>下一次网络前向</b><br>图 3-2 → 图 3-4',1910,600,210,98,'attention');

  d.edge('a1','raw','eval'); d.edge('a2','pack','eval','Mq≤v','cond'); d.edge('a3','eval','dual'); d.edge('a4','dual','prox'); d.edge('a5','prox','smooth'); d.edge('a6','smooth','loop'); d.edge('a7','loop','safe'); d.edge('a8','safe','ddim'); d.edge('a9','ddim','hard'); d.edge('a10','hard','next');
  d.edge('warm','loop','eval','λ 跨步/内迭代','feedback',[[1505,370],[395,370]]);
  d.edge('b1','safe','cat','Q_safe','feedback',[[1725,465],[505,465]]); d.edge('b2','raw','delta','','feedback',[[145,455],[270,455]]); d.edge('b3','safe','delta','','feedback',[[1725,485],[270,485]]); d.edge('b4','delta','cat'); d.edge('b5','valid','cat');
  d.edge('b6','cat','fbenc'); d.edge('b7','fbenc','hfb'); d.edge('b8','hfb','gate'); d.edge('b9','hctrl','gate'); d.edge('b10','gate','mul'); d.edge('b11','hfb','mul','','feedback'); d.edge('b12','mul','add'); d.edge('b13','hctrl','add','残差','main',[[1020,770],[1594,770]]); d.edge('b14','add','hout'); d.edge('b15','hout','forward');
  d.edge('close','forward','raw','t→t−1','feedback',[[2120,850],[90,850]]);
  legend(d, 890, 2320);
  return d.save('TrajSafe-Diffuser_图3-5_§3.4闭环安全修正与反馈-优化版.drawio');
}

const files = [fig32(), fig33(), fig34(), fig35()];
for (const f of files) console.log(f);
