"""Before/after report for occupied-square filtering, same frozen weights."""
import json
from pathlib import Path
import numpy as np

base=Path('outputs/ablation_hpath/geometry_compare_best_latest')
out=Path('outputs/ablation_hpath/geometry_square_filter_verified')
assert json.loads((out/'status.json').read_text())['stage']=='complete'
old=json.loads((base/'summary.json').read_text()); new=json.loads((out/'summary.json').read_text())
man0=json.loads((base/'manifest.json').read_text()); man1=json.loads((out/'manifest.json').read_text())
assert all(man0['weights'][t]['sha256']==man1['weights'][t]['sha256'] for t in ('best','latest'))
raw0=[json.loads(x) for x in (base/'records.jsonl').read_text().splitlines()]
raw1=[json.loads(x) for x in (out/'records.jsonl').read_text().splitlines()]
key=lambda r:(r['model'],r['seed'],r['index'])
oldrows={key(r):r for r in raw0}; newrows={key(r):r for r in raw1}
assert oldrows.keys()==newrows.keys()
assert all(oldrows[k]['route']['network']['first_choice_idx']==r['route']['network']['first_choice_idx'] for k,r in newrows.items())
lines=['# 完整障碍方格过滤：修正前后测试', '',
 '固定使用原专项测试的第131轮（验证最优）与第132轮（冻结时最新）权重；文件SHA256完全一致。两者均去除Hpath与安全查询中心编码，未重新训练。', '',
 '数据集carla_processed_160k4p_c48，测试集420样本×种子0/1/2；每个权重1260任务，两种选路策略共5040轨迹，三种初始化共967680个基础凸区域。修正前后首次选路完全一致。', '',
 '## 修改', '',
 '将障碍过滤由“中心在分离平面外”改为“整个占据方格在分离平面外”。法向支持半径为|nx|/W+|ny|/H；只有中心最小投影仍在安全半空间外才过滤。选中障碍的平面偏移至少覆盖方格支持半径及数值容差。原safety_margin=0.02（1.6m）保留；本次配置下所选平面的偏移下限不改变原裕度。窗口候选筛选同步改为纳入所有与局部窗口相交的方格；局部边界环继续作为点处理。固定初始化只在样本、中心、形状完全一致时复用几何记录，计数仍按每任务完整统计。', '',
 '20项相关单元测试通过，包含中心已排除但方格仍穿入，以及零裕度与非正方形栅格回归检查。', '',
 '## 基础凸区域安全', '',
 '检查整个多边形与占据方格的正面积交集；沿用原独立SAT/裁剪检查器，GEOS交叉验证。每种初始化、每个权重检查161280个区域。', '',
 '| 权重 | 初始化 | 修正前侵入数（比例） | 修正后侵入数（比例） | 平均面积/m²：前→后 |', '|---|---|---:|---:|---:|']
variants={'predicted':'预测椭圆','fixed_circle':'固定圆','fixed_tangent_ellipse':'固定切向椭圆'}
comparison={}
for tag in ('best','latest'):
 comparison[tag]={}
 for v,name in variants.items():
  a,b=old[tag]['regions'][v],new[tag]['regions'][v]
  rows=[r['regions'][v] for r in raw1 if r['model']==tag]
  comparison[tag][v]={'unsafe_before':a['base_unsafe_total'],'unsafe_after':b['base_unsafe_total'],'total':b['base_total'],'outside_after':sum(r['base_outside_count'] for r in rows),'invalid_after':sum(r['base_invalid_count'] for r in rows),'min_overlap_after':min(r['before_overlap_min'] for r in rows),'bridges_after':sum(r['bridges'] for r in rows),'unsafe_bridges_after':sum(r['bridge_unsafe_count'] for r in rows),'tasks_with_center_outside_after':sum(r['center_max_residual_m']>1e-6 for r in rows),'max_center_residual_m_after':max(r['center_max_residual_m'] for r in rows),'mean_area_before':a['base_area_mean_m2'],'mean_area_after':b['base_area_mean_m2']}
  lines.append('| %s第%d轮 | %s | %d（%.4f%%） | %d（%.4f%%） | %.3f → %.3f |'%('最优' if tag=='best' else '最新',new[tag]['epoch'],name,a['base_unsafe_total'],100*a['base_unsafe_total']/a['base_total'],b['base_unsafe_total'],100*b['base_unsafe_total']/b['base_total'],a['base_area_mean_m2'],b['base_area_mean_m2']))
lines += ['', '## 连通性', '', '| 权重 | 初始化 | 平均相邻交叠：前→后 | 修正后最低相邻交叠 | 修正后桥接总数 | 修正后走廊完成比例 |','|---|---|---:|---:|---:|---:|']
for tag in ('best','latest'):
 for v,name in variants.items():
  a,b=old[tag]['regions'][v],new[tag]['regions'][v]; c=comparison[tag][v]
  lines.append('| %s | %s | %.5f → %.5f | %.5f | %d | %.2f%% |'%(tag,name,a['before_overlap_mean'],b['before_overlap_mean'],c['min_overlap_after'],c['bridges_after'],100*b['constructed_with_bridge']))
lines += ['', '交叠比例=交集面积/较小区域面积，阈值0.10。桥接前后重用相同基础半空间。桥接总数为零时，无法从本次样本判断桥接模块处理间隙的收益。', '', '## 最终轨迹', '',
 '完整16步采样、安全引导和末端投影。轨迹碰撞判据沿用原论文评估：512采样点双线性占据值>0.5或越界；与完整方格的区域检查口径分别报告。', '',
 '| 权重 | 选路策略 | RMSE/m：前→后 | 平滑度/m：前→后 | 碰撞率：前→后 |','|---|---|---:|---:|---:|']
for tag in ('best','latest'):
 for policy in ('network','shortest'):
  a,b=old[tag]['route'][policy],new[tag]['route'][policy]
  lines.append('| %s | %s | %.5f → %.5f | %.5f → %.5f | %.3f%% → %.3f%% |'%(tag,policy,a['rmse_m'],b['rmse_m'],a['smoothness_m'],b['smoothness_m'],100*a['collision'],100*b['collision']))
lines += ['', '## 固定中心诊断', '',
 '中心仍从通路固定采样。本次修正解决方格过滤，不修改中心包含判据。', '', '| 权重 | 初始化 | 修正后存在中心在区域外的任务数/1260 | 最大违反距离/m |','|---|---|---:|---:|']
for tag in ('best','latest'):
 for v,name in variants.items():
  c=comparison[tag][v]
  lines.append('| %s | %s | %d | %.6f |'%(tag,name,c['tasks_with_center_outside_after'],c['max_center_residual_m_after']))
lines += ['', '原始记录见records.jsonl，汇总见summary.json，源码与权重保存在source_snapshot及best.pt/latest.pt。结论范围为本次完整测试及其数值容差。']
(out/'BEFORE_AFTER.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(out/'before_after.json').write_text(json.dumps(comparison,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(comparison,ensure_ascii=False,indent=2))
