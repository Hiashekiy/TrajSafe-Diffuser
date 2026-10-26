"""Add paired comparison and audit counts to a completed geometry experiment."""
import json
from pathlib import Path
import numpy as np

out = Path('outputs/ablation_hpath/geometry_compare_best_latest')
assert json.loads((out/'status.json').read_text())['stage'] == 'complete'
rows = [json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()]
summary = json.loads((out/'summary.json').read_text())
by = {tag: {(r['seed'], r['index']):r for r in rows if r['model']==tag} for tag in ('best','latest')}
assert len(by['best']) == len(by['latest']) == 1260
assert by['best'].keys() == by['latest'].keys()
comparison = {}
lines = ['# 去除 Hpath 模型：四项测试比较', '',
    '使用测试开始时冻结的验证最优第131轮和最新第132轮权重。两个权重均删除Hpath及安全查询的中心坐标编码加法；椭圆中心由所选通路固定采样。训练随后继续，本文中的“最新”仅指冻结时刻。', '',
    '测试集420个样本，生成种子0、1、2，每个权重1260次任务。通路测试共5040条最终轨迹；三种初始化共检查967680个基础凸区域。各对比使用相同样本、噪声和设置。', '',
    '## 1. 通路选择', '',
    '网络策略和最短通路策略均保留相同的构造失败候选回退。RMSE及碰撞率来自完整安全引导和末端投影后的最终轨迹；轨迹碰撞沿用论文评估代码的512点双线性占据值>0.5或越界判据，并非连续曲线与完整方格的严格相交判据。首选标签匹配来自首次安全激活时刻。', '',
    '| 指标 | 最优第131轮 | 最新第132轮 |', '|---|---:|---:|']
for label, policy, key, scale in [('网络首选标签匹配/%','network','top1_label_match',100),('网络最终选路标签匹配/%','network','label_match',100),('网络选路相对最短通路长度','network','length_ratio',1),('网络策略轨迹RMSE/m','network','rmse_m',1),('最短策略轨迹RMSE/m','shortest','rmse_m',1),('网络策略平滑度/m','network','smoothness_m',1),('网络策略轨迹碰撞率/%','network','collision',100),('最短策略轨迹碰撞率/%','shortest','collision',100)]:
    values=[summary[t]['route'][policy][key]*scale for t in ('best','latest')]
    lines.append('| %s | %.5f | %.5f |' % (label,*values))
for policy in ('network','shortest'):
    diff=np.array([by['latest'][k]['route'][policy]['rmse_m']-by['best'][k]['route'][policy]['rmse_m'] for k in by['best']])
    comparison[policy+'_paired_rmse']={'latest_minus_best_mean_m':float(diff.mean()),'latest_lower_count':int((diff<0).sum()),'pairs':len(diff)}
lines += ['', '配对RMSE比较（最新减最优，负数表示最新更低）：'+json.dumps(comparison,ensure_ascii=False), '',
    '## 2. 椭圆初始化', '',
    '预测椭圆、固定圆、固定切向椭圆使用每个权重首选通路上的同一组中心；固定尺度仅从验证集确定。这里比较初始化构成的区域，不把面积大直接解释为性能好。', '',
    '| 初始化 | 最优区域平均面积/m² | 最新区域平均面积/m² |', '|---|---:|---:|']
variants={'predicted':'预测椭圆','fixed_circle':'固定圆','fixed_tangent_ellipse':'固定切向椭圆'}
for v,name in variants.items():
    lines.append('| %s | %.3f | %.3f |'%(name,*[summary[t]['regions'][v]['base_area_mean_m2'] for t in ('best','latest')]))
lines += ['', '有效且选路与参考标签一致的位置上，预测椭圆参数误差：', '', '| 指标 | 最优 | 最新 |', '|---|---:|---:|']
for key,label in [('log_axis_mae','对数轴长平均绝对误差'),('angle_mae_deg','方向平均绝对误差/度'),('label_count','轴长有效位置数'),('direction_count','方向有效位置数')]:
    lines.append('| %s | %.5f | %.5f |'%(label,*[summary[t]['ellipse'][key] for t in ('best','latest')]))
same=[k for k in by['best'] if by['best'][k]['route']['network']['first_choice_idx']==by['latest'][k]['route']['network']['first_choice_idx']]
comparison['same_first_route_pairs']=len(same)
comparison['same_route_predicted_area_latest_minus_best_m2']=float(np.mean([by['latest'][k]['regions']['predicted']['base_area_mean_m2']-by['best'][k]['regions']['predicted']['base_area_mean_m2'] for k in same]))
lines += ['', '两个权重首选通路相同的配对数量为%d/1260；在该子集上，最新预测区域平均面积比最优变化%.3f m²。全体平均还包含选路变化。'%(len(same),comparison['same_route_predicted_area_latest_minus_best_m2']), '',
    '## 3. 凸区域安全', '',
    '按整个占据栅格方格检查多边形正面积侵入，边界接触在数值容差内不计侵入。分离轴检查及交集面积已用独立GEOS实现复核。最终轨迹零碰撞不等于整个区域无障碍。', '',
    '| 初始化 | 最优侵入数/161280 | 最优占比/% | 最新侵入数/161280 | 最新占比/% |', '|---|---:|---:|---:|---:|']
for v,name in variants.items():
    b,l=[summary[t]['regions'][v] for t in ('best','latest')]
    lines.append('| %s | %d | %.4f | %d | %.4f |'%(name,b['base_unsafe_total'],100*b['base_unsafe_total']/b['base_total'],l['base_unsafe_total'],100*l['base_unsafe_total']/l['base_total']))
comparison['audit']={}
for t in ('best','latest'):
    comparison['audit'][t]={}
    for v in variants:
        vr=[r['regions'][v] for r in by[t].values()]
        comparison['audit'][t][v]={'tasks_with_center_outside':sum(r['center_max_residual_m']>1e-6 for r in vr),'max_center_residual_m':max(r['center_max_residual_m'] for r in vr),'max_single_region_overlap_m2':max(r['max_region_intersection_area_m2'] for r in vr),'outside_regions':sum(r['base_outside_count'] for r in vr),'invalid_regions':sum(r['base_invalid_count'] for r in vr),'bridge_total':sum(r['bridges'] for r in vr),'below_overlap_total':sum(r['before_below_threshold'] for r in vr),'min_overlap_all_tasks':min(r['before_overlap_min'] for r in vr)}
lines += ['', '部分固定中心落在其构成的凸区域外，当前构造有效性判据没有强制中心包含。预测椭圆发生该情况的任务数：最优%d/1260，最新%d/1260；最大违反距离分别为%.6f、%.6f m。'%(comparison['audit']['best']['predicted']['tasks_with_center_outside'],comparison['audit']['latest']['predicted']['tasks_with_center_outside'],comparison['audit']['best']['predicted']['max_center_residual_m'],comparison['audit']['latest']['predicted']['max_center_residual_m']), '',
    '保存的侵入案例中，占据方格中心在区域外而方格边缘进入区域；当前以障碍栅格中心点构造半空间，与完整方格安全判据存在差异。逐区域侵入面积相加会重复计算重叠区域，不能当作走廊并集侵入面积。', '',
    '## 4. 区域连通与桥接', '',
    '桥接前后使用完全相同的基础区域。交叠比例定义为交集面积除以较小区域面积，阈值0.10。', '',
    '| 初始化 | 最优平均相邻交叠 | 最新平均相邻交叠 | 最优/最新最低交叠 | 最优/最新桥接总数 |', '|---|---:|---:|---:|---:|']
for v,name in variants.items():
    b,l=[summary[t]['regions'][v] for t in ('best','latest')]
    ba,la=[comparison['audit'][t][v] for t in ('best','latest')]
    lines.append('| %s | %.5f | %.5f | %.5f / %.5f | %d / %d |'%(name,b['before_overlap_mean'],l['before_overlap_mean'],ba['min_overlap_all_tasks'],la['min_overlap_all_tasks'],ba['bridge_total'],la['bridge_total']))
lines += ['', '若所有基础区域已满足阈值且桥接总数为零，本次数据只能支持这些样本连通，无法证明桥接模块在困难间隙中的收益。', '',
    '完整表格及构造记录见REPORT.md、summary.json和records.jsonl。专项区域取首次安全激活时的首选通路，实际采样可使用后备候选，因此区域专项与最终冻结走廊应分开解释。训练并发占用GPU，本次时间不用于推理速度结论。']
(out/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(out/'paired_comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(comparison,ensure_ascii=False,indent=2))

