import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from src.datasets.carla_spline_dataset import CarlaSplineDataset
p=ROOT/'experiments/safety_baselines/configs/protocol.json'
if p.exists():raise FileExistsError('protocol already frozen')
ds=CarlaSplineDataset('test',str(ROOT/'data/processed/160m-k4p-c48'),num_controls=48)
# Input-only stratification by free-space fraction, then evenly-spaced ranks.
features=[]
for i in range(len(ds)):
 r=ds[i];features.append((float(1-r['occupancy'].mean()),int(r['sample_id']),i))
ranked=sorted(features);s1=[ranked[int(k)][2] for k in np.linspace(0,len(ds)-1,20)]
remaining=sorted(set(range(len(ds)))-set(s1),key=lambda i:hashlib.sha256(str(int(ds[i]['sample_id'])).encode()).hexdigest())
s2=sorted(s1+remaining[:80]);all_ids=[int(ds[i]['sample_id']) for i in range(len(ds))]
protocol={'description':'input-only frozen test strata; no method outcome used','scene_ids':all_ids,'S1':{'indices':s1,'seeds':[0]},'S2':{'indices':s2,'seeds':[0,1,2]},'S3':{'indices':list(range(len(ds))),'seeds':[0,1,2]},'scene_hash':hashlib.sha256(json.dumps(all_ids).encode()).hexdigest()}
p.write_text(json.dumps(protocol,indent=2),encoding='utf8');print('frozen',len(s1),len(s2),len(ds))
