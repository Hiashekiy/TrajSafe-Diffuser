import json,torch,numpy as np
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.checkpoint import load_model
from src.datasets.carla_spline_dataset import CarlaSplineDataset,make_collate
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import sample
from src.geometry.convex_region import EllipseRegionBuilder,halfspaces_to_vertices
from scripts.geometry_audit_utils import obstacle_boxes,audit_polygon
stored=torch.load('outputs/ablation_hpath/geometry_compare_best_latest/latest.pt',map_location='cpu',weights_only=False)
cfg=stored['config']; model,_,_=load_model(cfg,stored,device='cuda',verbose=False); model.assert_shapes=False
reg=dict(cfg['alm']); reg.update(cfg['corridor']['region'])
ds=CarlaSplineDataset('test','data/processed/160m-k4p-c48',geometry_points=1280,require_labels=True)
collate=make_collate(ds); sch=NoiseSchedule(16,beta_schedule=cfg['diffusion'].get('beta_schedule','squaredcos_cap_v2'))
original=model.forward_all; cap={}
class Captured(Exception):pass
def wrapped(*a,**kw):
 p=original(*a,**kw)
 if int(a[3][0])==12:
  cap.update(center=p['ellipse']['center'].detach(),shape=p['ellipse']['shape4'].detach())
  raise Captured()
 return p
model.forward_all=wrapped
results=[]
for start,indices in [(224,[229]),(256,[256,259])]:
 b=collate([ds[i] for i in range(start,start+32)])
 occ=b['occupancy'].cuda()
 try:
  sample(model,sch,b['cond'].cuda(),occ,b['candidate_xy'].cuda(),b['candidate_mask'].cuda(),b['candidate_geometry'].cuda(),b['candidate_geometry_lengths'].cuda(),device='cuda',steps=16,seed=0,seed_offset=start,return_trace=False,alm_config=cfg['alm'],corridor_config=cfg['corridor'],guidance='alm')
 except Captured:pass
 for index in indices:
  i=index-start
  A,bb,m,v=EllipseRegionBuilder(occ[i:i+1],reg).build_from_ellipse(cap['center'][i:i+1],cap['shape'][i:i+1])
  boxes,h=obstacle_boxes(b['occupancy'][i,0].numpy()); audits=[]
  for a,z,mask in zip(A[0].cpu().numpy(),bb[0].cpu().numpy(),m[0].cpu().numpy()):
   audits.append(audit_polygon(halfspaces_to_vertices(a[mask],z[mask]),boxes,h))
  results.append({'index':index,'regions':len(audits),'unsafe_regions':sum(a['colliding_cells']>0 for a in audits),'overlap_m2':sum(a['overlap_area_scene2'] for a in audits)*6400})
print(json.dumps(results))
Path('outputs/ablation_hpath/geometry_square_filter_verified/targeted_window_regression.json').write_text(json.dumps(results,indent=2))
