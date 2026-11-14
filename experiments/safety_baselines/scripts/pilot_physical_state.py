import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
from threadpoolctl import threadpool_limits
from src.datasets.carla_spline_dataset import CarlaSplineDataset
from src.diffusion.schedule import NoiseSchedule
from experiments.safety_baselines.models.map_only import load_baseline
from experiments.safety_baselines.algorithms.runner import run_method,METHODS
from experiments.safety_baselines.adapters.evaluator import evaluate_curve

def main():
 cfg=json.loads((ROOT/'experiments/safety_baselines/configs/seven.json').read_text());model,ck=load_baseline(ROOT/cfg['checkpoint'],'cuda');d=ck['config']['diffusion'];schedule=NoiseSchedule(d['timesteps'],d['beta_schedule'],d['beta_start'],d['beta_end'])
 ds=CarlaSplineDataset('val',ck['config']['data']['processed_root'],num_controls=48);row=ds[0]
 occ=row['occupancy'][0].numpy();cond=row['cond'].numpy();gt=row['pos'].numpy();scene=int(row['sample_id'])
 out=ROOT/'experiments/safety_baselines/results/pilot_seven_val0_physical_integrated_20261113';out.mkdir(exist_ok=False)
 cfg['field_kind']='exact';cfg['state_adapter']='physical_trajectory';cfg['pdm_parameterization']='vp';cfg['cbf_substeps']=16;cfg['cbf_max_point_step']=2/256/4;cfg['timeouts'].update({m:60 for m in ['RoS-diffuser','ReS-diffuser','TVS-diffuser']})
 with threadpool_limits(limits=1):
  for method in METHODS:
   result=run_method(method,model,schedule,occ,cond,scene,0,cfg)
   if result['status']=='ok':result['metrics']=evaluate_curve(model.bspline,result['trajectory'],occ,cond,gt)
   (out/(method+'.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
   print(method,result['status'],result['failure_reason'],round(result['wall_time'],3),result.get('metrics',{}).get('planning_success'),flush=True)
if __name__=='__main__':main()
