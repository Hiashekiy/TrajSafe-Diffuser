"""Base generation control versus full Ours; NOT seven-paper reproduction."""
import json,statistics,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/safety_baselines/results/map_only_vs_ours_20261113'
def main():
 OUT.mkdir(exist_ok=False)
 manifest={'status':'running','scope':'Diffuser map-only vs full Ours 1+8; paper runners pending','completed':[],'protocol':{'scenes':420,'seeds':[0,1,2],'steps':16,'chunk':32}}
 def save():(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
 save()
 try:
  for seed in range(3):
   for method in (['Diffuser','Ours'] if seed%2==0 else ['Ours','Diffuser']):
    tag=f'{method}_s{seed}';out=OUT/f'{tag}.json'
    if method=='Diffuser':cmd=[sys.executable,'-u','experiments/safety_baselines/scripts/eval_map_only.py','--seed',str(seed),'--out',str(out)]
    else:cmd=[sys.executable,'-u','scripts/eval_chapter4.py','--config','experiments/safety_baselines/configs/v4_map16_l1_final8_e100.yaml','--ckpt','experiments/safety_baselines/results/depth_l1_final8_e100_20261112_190325/ckpt/best_task.pt','--method','TrajSafe-Diffuser (1+8)','--guidance','alm','--feedback','on','--final-project','on','--split','test','--samples','0','--chunk','32','--steps','16','--seed',str(seed),'--device','cuda','--out',str(out),'--md',str(out.with_suffix('.md')),'--records',str(out.with_suffix('.jsonl'))]
    manifest['current']=tag;save();print('START',tag,flush=True)
    with (OUT/f'{tag}.log').open('w',encoding='utf8') as log:rc=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if rc:raise RuntimeError(f'{tag} failed exit{rc}')
    manifest['completed'].append(tag);save();print('DONE',tag,flush=True)
  metrics=['rmse_m','collision_rate','sdf_feasible_rate','sdf_mean_violation_m','smoothness_m','sdf_min_distance_m','endpoint_error','runtime_ms_mean']
  runs={m:[json.loads((OUT/f'{m}_s{s}.json').read_text()) for s in range(3)] for m in ['Diffuser','Ours']}
  agg={m:{k:{'mean':statistics.mean(r['aggregate'][k] for r in rr),'std':statistics.stdev(r['aggregate'][k] for r in rr)} for k in metrics} for m,rr in runs.items()}
  records={m:{} for m in runs}
  for m in runs:
   for s in range(3):
    data=[json.loads(l) for l in (OUT/f'{m}_s{s}.jsonl').read_text().splitlines()]
    if len(data)!=420 or len({r['sample_id'] for r in data})!=420:raise ValueError('missing/duplicate scenes')
    records[m][s]={r['sample_id']:r for r in data}
  ids=sorted(records['Diffuser'][0]);diff=[]
  for m in records:
   for s in range(3):
    if sorted(records[m][s])!=ids:raise ValueError('unmatched scene ids')
  for i in ids:diff.append(statistics.mean(records['Ours'][s][i]['rmse_m']-records['Diffuser'][s][i]['rmse_m'] for s in range(3)))
  diff=np.asarray(diff);rng=np.random.default_rng(42);boot=[rng.choice(diff,len(diff),replace=True).mean() for _ in range(10000)]
  paired={'mean_rmse_delta_ours_minus_base':float(diff.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'better_scenes':int((diff<0).sum()),'worse_scenes':int((diff>0).sum()),'unit':'scene after averaging three seeds'}
  report={'aggregates':agg,'paired':paired,'limitations':['two-model comparison only; seven paper algorithms not yet evaluated','both trained100 epochs but only one training seed','map-only inputs vs full Ours skeleton/safety/feedback; system comparison, not isolated safety optimizer ablation','512 evaluation points; no continuous safety proof; no adaptive near-obstacle evaluation yet','legacy seed+chunk-offset shared noise protocol requires fixed chunk32','sdf_min_distance is mean over all returned paths including collisions, not success-conditional clearance','batch-amortized inference timing excludes evaluation-only SDF and is not single-task end-to-end latency']}
  (OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf8')
  lines=['# 新地图基座 Diffuser 与完整 Ours 对照','','双方均为1+8公共框架、100轮训练；420场景×3种子，各1260次推理。','', '|指标|Diffuser|TrajSafe-Diffuser|','|---|---:|---:|']
  for k in metrics:
   a,b=agg['Diffuser'][k],agg['Ours'][k];lines.append(f"|{k}|{a['mean']:.6f} ± {a['std']:.6f}|{b['mean']:.6f} ± {b['std']:.6f}|")
  lines+=['',str(paired),'',*report['limitations']]
  (OUT/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf8');manifest['status']='complete';manifest['current']=None;save();print('COMPLETE',flush=True)
 except Exception as exc:manifest['status']='failed';manifest['error']=repr(exc);save();raise
if __name__=='__main__':main()
