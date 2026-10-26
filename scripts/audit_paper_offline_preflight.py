import json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.geometry.convex_region import halfspaces_to_vertices
from scripts.geometry_audit_utils import obstacle_boxes,audit_polygon
root=Path('data/processed/160m-k4p-c48'); out=Path('outputs/paper_ral');out.mkdir(exist_ok=True)
result={}
for split in ['train','val']:
 p=root/split
 a=np.load(p/'alm_cell_a.npy',mmap_mode='r'); b=np.load(p/'alm_cell_b.npy',mmap_mode='r');valid=np.load(p/'alm_cell_valid.npy',mmap_mode='r');occ=np.load(p/'occupancy.npy',mmap_mode='r')
 unsafe=total=0;example=None
 for idx in np.linspace(0,len(a)-1,32,dtype=int):
  boxes,h=obstacle_boxes(np.asarray(occ[idx]).squeeze())
  for region in np.flatnonzero(valid[idx]):
   mask=np.isfinite(b[idx,region]) & (np.linalg.norm(a[idx,region],axis=-1)>1e-8)
   poly=halfspaces_to_vertices(a[idx,region,mask],b[idx,region,mask]);r=audit_polygon(poly,boxes,h)
   total+=1
   if r['colliding_cells']:
    unsafe+=1
    if example is None:example={'index':int(idx),'region':int(region),'area_m2':r['overlap_area_scene2']*6400}
 result[split]={'samples':32,'regions':total,'unsafe_regions':unsafe,'example':example}
 print(split,result[split],flush=True)
(out/'offline_label_preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
