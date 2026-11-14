import collections,json,statistics,sys
from pathlib import Path
import numpy as np

def summarize(path):
 p=Path(path);rows=[json.loads(x) for x in (p/'results.jsonl').read_text(encoding='utf8').splitlines()];groups=collections.defaultdict(list)
 for r in rows:groups[r['method']].append(r)
 aggregate={}
 for m,rr in groups.items():
  ret=[r for r in rr if r['metrics'] is not None];succ=[r for r in rr if r['planning_success']];t=[r['wall_time'] for r in rr]
  def avg(xs,key):
   values=[r['metrics'][key] for r in xs if r['metrics'].get(key) is not None]
   return statistics.mean(values) if values else None
  aggregate[m]={'attempts':len(rr),'returned':len(ret),'successes':len(succ),'success_rate':len(succ)/len(rr),'no_trajectory_failures':sum(r['trajectory'] is None for r in rr),'invalid_trajectory_count':sum(r['trajectory'] is not None and r['metrics'] is None for r in rr),'returned_collision_count':sum(r['metrics']['collision'] for r in ret),'returned_collision_rate':sum(r['metrics']['collision'] for r in ret)/len(ret) if ret else None,'all_attempt_failure_rate':1-len(succ)/len(rr),'time_mean_s':statistics.mean(t),'time_median_s':statistics.median(t),'time_p95_s':float(np.quantile(t,.95)),'violation_max_mean_m_returned':avg(ret,'common_sdf_violation_max_m'),'rmse_m_returned':avg(ret,'rmse_m'),'nrmse_returned':avg(ret,'nrmse'),'positive_violation_mean_m_returned':avg(ret,'common_sdf_violation_positive_mean_m'),'internal_metric_count':sum(r.get('internal_max_violation_scene') is not None for r in ret),'internal_feasible_returned':sum(r.get('internal_max_violation_scene') is not None and r['internal_max_violation_scene']<=1e-5 for r in ret),'within_10s_success_rate':sum(r['planning_success'] and r['wall_time']<=10 for r in rr)/len(rr),'clearance_m_success':avg(succ,'clearance_success_m'),'detour_success':avg(succ,'detour_success'),'max_curvature_success':avg(succ,'max_curvature_success'),'mean_square_curvature_success':avg(succ,'mean_square_curvature_success'),'failure_reasons':dict(collections.Counter((r['failure_reason'] or '').split(':')[0] for r in rr if not r['planning_success'])),'source':rr[0]['source_id'],'fidelity':rr[0]['fidelity']}
 rng=np.random.default_rng(42);paired={}
 if 'TrajSafe-Diffuser' in groups:
  def scene_rates(rr):
   d=collections.defaultdict(list)
   for r in rr:d[r['scene_id']].append(float(r['planning_success']))
   return {i:statistics.mean(v) for i,v in d.items()}
  ours=scene_rates(groups['TrajSafe-Diffuser'])
  for m,rr in groups.items():
   if m=='TrajSafe-Diffuser':continue
   a=scene_rates(rr);ids=sorted(set(a)&set(ours));diff=np.array([ours[i]-a[i] for i in ids]);episode={r['scene_id']:r['episode_id'] for r in rr};clusters=sorted({episode[i] for i in ids});sums=np.array([sum(diff[j] for j,i in enumerate(ids) if episode[i]==e) for e in clusters]);counts=np.array([sum(episode[i]==e for i in ids) for e in clusters]);pick=rng.integers(0,len(clusters),(10000,len(clusters)));boot=sums[pick].sum(1)/counts[pick].sum(1)
   paired[m]={'n_scenes':len(ids),'n_episode_clusters':len(clusters),'success_delta_ours_minus_method':float(diff.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist()}
 payload={'methods':aggregate,'paired_scene_success':paired,'records':len(rows),'notes':['collision rate conditional on a returned trajectory; missing trajectory is failure, not collision-free','quality metrics conditional on successful samples; denominators reported','all methods use identical occupancy/SDF evaluation and endpoint tolerance','finite sampling does not certify continuous safety','local nonconvex projection and sampled Clarke stationarity are approximate; no global nearest certificate','timing single-scene including actual consumed cached inputs and map/solver initialization, excludes evaluation; original offline skeleton cache preparation unavailable','external methods are task-adapted; negative results cannot be attributed solely to original papers','bootstrap units are whole episodes; 420 test samples come from15 episodes','native point-state feasibility is separate from final B-spline safety; no safety repair after fitting']}
 (p/'aggregate.json').write_text(json.dumps(payload,indent=2),encoding='utf8')
 lines=['# 安全算法分级对比结果','','|Method|Attempts|Returned|Success %|Collision % / returned|No trajectory|Time mean/P95(s)|','|---|---:|---:|---:|---:|---:|---:|']
 for m,a in aggregate.items():
  coll='N/A' if a['returned_collision_rate'] is None else f"{100*a['returned_collision_rate']:.2f}"
  lines.append(f"|{m}|{a['attempts']}|{a['returned']}|{100*a['success_rate']:.2f}|{coll}|{a['no_trajectory_failures']}|{a['time_mean_s']:.3f}/{a['time_p95_s']:.3f}|")
 lines+=['','## Failure reasons','',*[(m+': '+json.dumps(a['failure_reasons'])) for m,a in aggregate.items()],'',*payload['notes']]
 (p/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
if __name__=='__main__':summarize(sys.argv[1])
