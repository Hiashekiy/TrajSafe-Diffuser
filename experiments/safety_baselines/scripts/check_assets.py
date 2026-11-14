import sys,json,platform
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE/'.deps312'),str(ROOT)]
import numpy as np
import torch,scipy
from experiments.safety_baselines.adapters.model_adapter import load_locked_model
out={'versions':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__},'cuda_available':torch.cuda.is_available(),'models':{}}
for method in ['Diffuser','TrajSafe-Diffuser']:
    try:
        model,schedule,cfg,report=load_locked_model(method)
        out['models'][method]={'status':'compatible','config':cfg,'load_report':report,'parameters':sum(p.numel() for p in model.parameters()),'alpha_bar':schedule.alphas_cumprod.tolist()}
    except ValueError as exc:
        out['models'][method]={'status':'blocked','error':str(exc)}
p=ROOT/'data/processed/160m-k4p-c48/test/sample_id.npy'
ids=np.load(p,allow_pickle=False)
out['test']={'count':len(ids),'unique_ids':len(set(ids.tolist())),'scene_ids':ids.tolist()}
(HERE/'docs/locked_assets.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=str),encoding='utf8')
print({k:v['status'] for k,v in out['models'].items()})
print('Asset validation saved; blocked models must not be evaluated')
