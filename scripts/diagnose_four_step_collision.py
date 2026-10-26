import sys,json
from pathlib import Path
import numpy as np,torch
ROOT=Path.cwd();run=ROOT/'outputs/paper_ral/run_20261009_no_projection';snap=run/'source_snapshot'
sys.path.insert(0,str(snap));sys.path.insert(0,str(snap/'scripts'))
from src.utils.config import load_config
from src.utils.checkpoint import load_model
from src.datasets.carla_spline_dataset import CarlaSplineDataset,make_collate
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import sample
from src.diffusion.bspline_alm import constraint_state,bspline_alm_correct
from src.geometry.convex_region import halfspaces_to_vertices
from geometry_audit_utils import obstacle_boxes,audit_polygon
cfg=load_config(str(snap/'configs/A.yaml'));model,_,report=load_model(cfg,str(run/'A/ckpt/best_task.pt'),device='cuda',verbose=False)
assert not report['missing'] and not report['adapted'] and not report['unexpected']
model.assert_shapes=False
D=CarlaSplineDataset('test',cfg['data']['processed_root'],geometry_points=1280);batch=make_collate(D)([D[i] for i in range(288,320)])
args=(model,NoiseSchedule(16,beta_schedule=cfg['diffusion']['beta_schedule']),batch['cond'].cuda(),batch['occupancy'].cuda(),batch['candidate_xy'].cuda(),batch['candidate_mask'].cuda(),batch['candidate_geometry'].cuda(),batch['candidate_geometry_lengths'].cuda())
alm=dict(cfg['alm'],final_project=False,safe_prediction_to_ddim=True,force_feedback_valid_zero=False,guidance='alm')
results=[]
for seed in [1,2]:
 out=sample(*args,device='cuda',steps=4,seed=seed,seed_offset=288,return_trace=True,alm_config=alm,corridor_config=cfg['corridor'],guidance='alm')
 b=3;pack=out['pack'];_,g,mask=constraint_state(out['control'],pack);mx=float(g[b].masked_fill(~mask[b],-torch.inf).max())
 boxes,half=obstacle_boxes(np.asarray(batch['occupancy'][b,0]));unsafe=0;seen=set()
 for i in range(int(pack.num_pieces[b])):
  f=pack.face_mask[b,i].cpu().numpy();a=pack.piece_A[b,i].cpu().numpy()[f];bb=pack.piece_b[b,i].cpu().numpy()[f]
  key=a.tobytes()+bb.tobytes()
  if key in seen:continue
  seen.add(key);check=audit_polygon(halfspaces_to_vertices(a,bb),boxes,half);unsafe+=int(check['colliding_cells']>0)
 z={'seed':seed,'index':291,'validation':out['final_validation'][b],'bezier_max_residual_scene':mx,'bezier_max_residual_m':mx*80,'regions_audited':len(seen),'unsafe_regions':unsafe,'activation_step':int(out['activation_step'][b]),'trace':[],'budget_diagnosis':[]}
 for tr in out['trace']:
  stats=tr.get('alm_stats');z['trace'].append({'t':int(tr['t']),'guided':bool(tr['guided'][b]),'stats':None if not stats else {k:float(v[b]) for k,v in stats.items() if v.numel()==32}})
 raw=out['trace'][-1]['q0_raw'].cuda()
 for budget in [8,20,40,80]:
  q,_,st=bspline_alm_correct(raw,pack,model.bspline,None,alm,inner_steps=budget,max_steps=torch.full((32,),budget,device='cuda',dtype=torch.long))
  from src.diffusion.sampler import dense_validation
  _,gg,mm=constraint_state(q,pack)
  z['budget_diagnosis'].append({'iterations':budget,'bezier_residual_m':80*float(gg[b].masked_fill(~mm[b],-torch.inf).max()),'validation':dense_validation(model,model.bspline,q,pack,batch['occupancy'].cuda(),batch['cond'].cuda(),cfg=alm)[b]})
 results.append(z);print(json.dumps(z),flush=True)
(run/'four_step_collision_diagnosis.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
