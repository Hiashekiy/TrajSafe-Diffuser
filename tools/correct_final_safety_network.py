from pathlib import Path
import json,xml.etree.ElementTree as E
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config
out=Path('paper/figures/final-safety-network');p=out/'final-safety-network.drawio';tr=E.parse(p);r=tr.find('.//mxGraphModel/root');cs={z.get('id'):z for z in r}
cfg=load_config('configs/paper_modified.yaml');m=cfg['model'];assert m['path_features']['enabled'] is False
remove={'ctrl','ctrl_caption','cat'}
for z in list(r):
 if z.get('id') in remove or (z.get('edge')=='1' and (z.get('source') in remove or z.get('target') in remove)):r.remove(z)
cs['path'].set('style',cs['ctrl'].get('style'));cs['path_caption'].set('value','<b>Control</b> <i>H<sub>ctrl</sub></i><br>48 × 128')
cs['safe_caption'].set('value','<b>Safety-fused</b> <i>H<sub>fuse</sub></i><br>48 × 128')
cs['norm'].find('mxGeometry').set('x','290')
cs['sca'].find('mxGeometry').set('x','465');cs['sca'].find('mxGeometry').set('width','220')
cs['add'].find('mxGeometry').set('x','740');cs['safe'].find('mxGeometry').set('x','805')
cs['ell'].find('mxGeometry').set('x','490')
cs['mlp'].set('value','Fusion MLP<br>LN(128) → 256<br>GELU → 128');cs['mlp'].find('mxGeometry').set('x','1040');cs['mlp'].find('mxGeometry').set('width','220')
for z in r:
 if z.get('edge')!='1':continue
 s,t=z.get('source'),z.get('target');g=z.find('mxGeometry')
 def pts(seq):
  ar=g.find('Array')
  if ar is not None:g.remove(ar)
  if seq:
   ar=E.SubElement(g,'Array',attrib={'as':'points'})
   for x,y in seq:E.SubElement(ar,'mxPoint',x=str(x),y=str(y))
 if (s,t)==('path','add'):pts([(110,190),(755,190)])
 if (s,t)==('ell','sca'):pts([(650,110),(650,195),(641,195)])
 if (s,t)==('ht','norm'):pts([(150,165),(337.5,165)])
# Single-stream MLP reads the complete residual fusion output.
z=E.SubElement(r,'mxCell',id='safe_to_mlp',parent='1',edge='1',source='safe',target='mlp',value='',style='edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=block;endFill=1;strokeColor=#273344;strokeWidth=1.9;exitX=1;exitY=0.5;entryX=0;entryY=0.5;');E.SubElement(z,'mxGeometry',relative='1',attrib={'as':'geometry'})
cs['kvnote'].find('mxGeometry').set('x','665');cs['kvnote'].find('mxGeometry').set('y','150')
# Short annotation replaces the former concatenation branch.
z=E.SubElement(r,'mxCell',id='fusion_formula',parent='1',vertex='1',value='H<sub>fuse</sub> = H<sub>ctrl</sub> + CrossAttention(AdaLN(H<sub>ctrl</sub>, h<sub>t</sub>), H<sub>ell</sub>)',style='text;html=1;align=center;fontFamily=Times New Roman;fontSize=21;fillColor=none;strokeColor=none;');E.SubElement(z,'mxGeometry',x='260',y='335',width='1050',height='30',attrib={'as':'geometry'})
ids=[z.get('id') for z in r];assert len(ids)==len(set(ids))
for z in r:
 assert 'H_path' not in z.get('value','') and 'H<sub>path' not in z.get('value','') and '384' not in z.get('value','')
 if z.get('edge')=='1':assert z.get('source') in ids and z.get('target') in ids
E.indent(tr,space='  ');tr.write(p,encoding='utf-8',xml_declaration=True)
# Store layout directly from the corrected editable source so it cannot regenerate the old variant.
(out/'final-safety-network.json').write_text(json.dumps({'configuration':'configs/paper_modified.yaml','path_features_enabled':False,'fusion_num_streams':1,'fusion_input_width':128,'editable_source':'final-safety-network.drawio','vertices':[{ 'id':z.get('id'),'value':z.get('value'), 'geometry':dict(z.find('mxGeometry').attrib)} for z in r if z.get('vertex')=='1'], 'edges':[{ 'source':z.get('source'),'target':z.get('target')} for z in r if z.get('edge')=='1']},ensure_ascii=False,indent=2),encoding='utf-8')
(out/'code-verification.md').write_text('''# 最新配置下的代码核对（2026-10-10 修订）

使用当前工作区，而非旧提交。planner.py 与 fusion.py 文件修改时间为 2026-10-08，论文主配置 paper_modified.yaml 为 2026-10-09。

配置链：paper_modified.yaml → ablation_hpath_without.yaml → config_160k4p_c48_oneshot.yaml → config_160k4p_oneshot.yaml → config_160k8p.yaml。通过 load_config 解析实际继承结果：path_features.enabled=false、safety_query.center_encoding=false，C=48、Q=128、D=128、4 heads、3 final blocks。

1. PathFeatureHead 不构建，h_path=None。安全分支保留独立 safety_query_head，用于骨架安全查询；该模块不产生 H_path。
2. planner.py 中 query=h_ctrl，SafetyControlFusion 计算 H_fuse=H_ctrl+CrossAttention(Q=projection(AdaLN(H_ctrl,h_t)), K/V=projection(H_ell))，H_fuse 为 48×128。图中 H_fuse 对应代码变量 h_safe_ctrl / 输出 A_safety。
3. FusionMLP 的 num_streams=1，直接读取 H_fuse；LayerNorm(128)→Linear(128,256)→GELU→Linear(256,128)。无拼接，无 H_path。
4. FinalDenoiser 依然为3个独立参数 TrajBlock，每块包含三个各自带残差的 AdaLN 子层：相对索引偏置自注意力、全局地图交叉注意力、128→512→128 GELU FFN。各 AdaLN 调制投影独立。
5. head_p 为共享 Linear(128,2)。BoundaryDecoder 为固定无训练参数的端点补偿：Q_bdry=Q_raw+w_s*(S-Q_raw[0])+w_g*(G-Q_raw[-1])，两端各4点的权重是 [1,.75,.5,.25]。
6. B 样条解码48个控制点至128个曲线采样点；避障ALM在下游。paper_modified.yaml 还关闭末端硬投影（final_project=false），本图未加入该模块。
7. 历史反馈若启用，已经在上游 H_ctrl 中。图展示有效候选路径；无候选时最终曲线回退 coarse。

上一版错误原因：使用旧 base 配置，未追踪最新论文配置覆盖 path_features 开关。现图已删除 H_path、三路拼接与 LN(384)，恢复实际单流融合。
''',encoding='utf-8')
print('Verified latest configuration:', m['path_features'], 'single-stream input=128; edges=',sum(z.get('edge')=='1' for z in r))

