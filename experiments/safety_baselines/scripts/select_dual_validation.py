import sys,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from src.datasets.carla_spline_dataset import CarlaSplineDataset
from src.diffusion.schedule import NoiseSchedule
from experiments.safety_baselines.models.map_only import load_baseline
from experiments.safety_baselines.algorithms.runner import run_method
from experiments.safety_baselines.adapters.evaluator import evaluate_curve
base=ROOT/'experiments/safety_baselines';cfg=json.loads((base/'configs/seven.json').read_text());cfg['dual_constraint_scale']=128.0;cfg['state_adapter']='physical_trajectory';cfg['field_kind']='exact';cfg['cd_levels']=64;cfg['cd_extra_steps']=64;cfg['rho_growth']=1.01
model,ck=load_baseline(ROOT/cfg['checkpoint'],'cuda');d=ck['config']['diffusion'];schedule=NoiseSchedule(d['timesteps'],d['beta_schedule'])
ds=CarlaSplineDataset('val',ck['config']['data']['processed_root'],num_controls=48)
# Frozen input-index spread, no test results or GT conditioning.
indices=np.linspace(0,len(ds)-1,10).astype(int).tolist();records=[];wins={}
with threadpool_limits(limits=1):
 for method,param,candidates in [('Primal-Dual','dual_eta',[1e-7,1e-6,1e-5,1e-4]),('ALM','rho',[1e-8,1e-7,1e-6,1e-5])]:
  scored=[]
  for value in candidates:
   local=dict(cfg);local[param]=value;batch=[]
   for idx in indices:
    row=ds[idx];occ=row['occupancy'][0].numpy();cond=row['cond'].numpy();r=run_method(method,model,schedule,occ,cond,int(row['sample_id']),0,local);success=False;rmse=None
    if r['trajectory'] is not None:
     try:
      met=evaluate_curve(model.bspline,r['trajectory'],occ,cond,row['pos'].numpy());success=met['planning_success'];rmse=met['rmse_m']
     except ValueError:rmse=None;success=False
    batch.append({'scene_id':int(row['sample_id']),'success':success,'status':r['status'],'failure':r['failure_reason'],'rmse':rmse})
   n=sum(x['success'] for x in batch);avg=statistics.mean(x['rmse'] for x in batch if x['rmse'] is not None) if any(x['rmse'] is not None for x in batch) else float('inf')
   scored.append((n,-avg,value));records.append({'method':method,'param':param,'value':value,'successes':n,'n':len(batch),'mean_rmse_returned':avg,'records':batch});print(method,param,value,n,avg,flush=True)
  wins[param]=max(scored)[2]
result={'split':'val','indices':indices,'constraint_scale':128.,'scale_rationale':'SDF distance / one grid cell (2/256 scene), dimensionless; physical feasible set unchanged; not a safety-margin change','selection':'maximize val planning successes, then minimize returned RMSE; no test tuning','winners':wins,'candidates':records}
(base/'docs/physical_dual_validation.json').write_text(json.dumps(result,indent=2),encoding='utf8');print('WINNERS',wins,flush=True)
