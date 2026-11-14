"""Matched inference protocol for existing full Ours 1+3 and new 1+8."""
import argparse,datetime,hashlib,json,math,os,statistics,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'experiments/safety_baselines'
MODELS={
 'final3':('configs/config_160k4p_c48_v4_map16_l1.yaml','outputs/models/v4_map16_c48_l1/ckpt/best_task.pt'),
 'final8':('experiments/safety_baselines/configs/v4_map16_l1_final8_e100.yaml','experiments/safety_baselines/results/depth_l1_final8_e100_20261112_190325/ckpt/best_task.pt')}
METRICS=['rmse_m','collision_rate','sdf_feasible_rate','sdf_mean_violation_m','smoothness_m','sdf_min_distance_m','online_corridor_membership_rate','endpoint_error','solver_failure_samples','runtime_ms_mean']
def summarize(out):
    runs={m:[json.loads((out/f'{m}_s{s}.json').read_text(encoding='utf8')) for s in range(3)] for m in MODELS}
    aggregates={m:{k:{'mean':statistics.mean(r['aggregate'][k] for r in rr),'std':statistics.stdev(r['aggregate'][k] for r in rr)} for k in METRICS} for m,rr in runs.items()}
    records={}
    for m in MODELS:
        records[m]={}
        for s in range(3):
            rr=[json.loads(line) for line in (out/f'{m}_s{s}.jsonl').read_text(encoding='utf8').splitlines()]
            if len(rr)!=420 or len({r['sample_id'] for r in rr})!=420:raise ValueError('incomplete or duplicate scene records')
            records[m][s]={r['sample_id']:r for r in rr}
    ids=sorted(records['final3'][0])
    for m in MODELS:
        for s in range(3):
            if sorted(records[m][s])!=ids:raise ValueError('scene ids mismatch')
    diffs=np.array([statistics.mean(records['final8'][s][i]['rmse_m']-records['final3'][s][i]['rmse_m'] for s in range(3)) for i in ids])
    if not np.isfinite(diffs).all():raise ValueError('nonfinite RMSE in paired data')
    rng=np.random.default_rng(42)
    boot=np.array([rng.choice(diffs,len(diffs),replace=True).mean() for _ in range(10000)])
    paired={'definition':'final8 minus final3; average 3 seeds per scene before scene bootstrap','n_scenes':420,'mean_m':float(diffs.mean()),'ci95_m':np.quantile(boot,[.025,.975]).tolist(),'better_scenes':int((diffs<0).sum()),'worse_scenes':int((diffs>0).sum())}
    payload={'aggregates':aggregates,'paired_rmse':paired,'records_per_model':1260,
      'limitations':['original final3 trained 200 epochs, selected epoch78; final8 trained100, selected epoch84; training budgets differ','timing is batched per-trajectory inference with chunk32, not single-task latency; use the common evaluator timing boundary','collision=0 at evaluation samples does not prove continuous safety']}
    (out/'comparison.json').write_text(json.dumps(payload,indent=2),encoding='utf8')
    lines=['# Full Ours: final3 vs final8','','420 test scenes x seeds 0/1/2; chunk32, DDIM16, ALM+feedback+final-project. Both models rerun in this session.','','|Metric|1+3 mean +/- seed std|1+8 mean +/- seed std|','|---|---:|---:|']
    for k in METRICS:
        a,b=aggregates['final3'][k],aggregates['final8'][k]
        lines.append(f"|{k}|{a['mean']:.6f} +/- {a['std']:.6f}|{b['mean']:.6f} +/- {b['std']:.6f}|")
    lines+=['',f'Paired RMSE delta: {paired}', '',*payload['limitations']]
    (out/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return payload

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);args=ap.parse_args()
    if Path(args.run_id).name!=args.run_id:raise ValueError('invalid run id')
    out=BASE/'results'/args.run_id;out.mkdir(exist_ok=False)
    manifest={'status':'running','pid':os.getpid(),'protocol':{'split':'test','samples':420,'seeds':[0,1,2],'chunk':32,'steps':16,'guidance':'alm','feedback':'on','final_project':'on'},'models':{},'completed':[]}
    for m,(cfg,ckpt) in MODELS.items():manifest['models'][m]={'config':cfg,'checkpoint':ckpt,'sha256':hashlib.sha256((ROOT/ckpt).read_bytes()).hexdigest()}
    def save():(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    save()
    try:
        # Alternate ordering to reduce systematic load/time drift.
        for seed in range(3):
            for m in (['final3','final8'] if seed%2==0 else ['final8','final3']):
                cfg,ckpt=MODELS[m];tag=f'{m}_s{seed}'
                cmd=[sys.executable,'-u','scripts/eval_chapter4.py','--config',cfg,'--ckpt',ckpt,'--method',f'Full Ours (1+{3 if m=="final3" else 8})','--guidance','alm','--feedback','on','--final-project','on','--split','test','--samples','0','--chunk','32','--steps','16','--seed',str(seed),'--device','cuda','--out',str(out/f'{tag}.json'),'--md',str(out/f'{tag}.md'),'--records',str(out/f'{tag}.jsonl')]
                manifest['current']=tag;save();print('START',tag,flush=True)
                with (out/f'{tag}.log').open('w',encoding='utf8') as log:rc=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                if rc:raise RuntimeError(f'{tag} failed exit {rc}; inspect its log')
                manifest['completed'].append(tag);save();print('DONE',tag,flush=True)
        summarize(out);manifest['status']='complete';manifest['current']=None;save();print('COMPLETE',out,flush=True)
    except Exception as exc:
        manifest['status']='failed';manifest['error']=repr(exc);save();raise
if __name__=='__main__':main()
