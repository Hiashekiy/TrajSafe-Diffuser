"""Strict read-only model loading; never accept silently adapted weights."""
from src.utils.config import load_config, num_controls
from src.utils.checkpoint import load_model
from src.diffusion.schedule import NoiseSchedule

ASSETS={
 'Diffuser':('experiments/safety_baselines/configs/map_only_l1_final8.yaml','experiments/safety_baselines/results/map_only_l1_final8_e100_20261112_212929/ckpt/best_task.pt'),
 'TrajSafe-Diffuser':('experiments/safety_baselines/configs/v4_map16_l1_final8_e100.yaml','experiments/safety_baselines/results/depth_l1_final8_e100_20261112_190325/ckpt/best_task.pt')}

def load_locked_model(method,device='cpu'):
    config,checkpoint=ASSETS[method]
    cfg=load_config(config)
    if num_controls(cfg)!=48:raise ValueError('expected 48 controls')
    if method=='Diffuser':
        from experiments.safety_baselines.models.map_only import load_baseline
        model,ck=load_baseline(checkpoint,device);cfg=ck['config'];report={'missing':[],'adapted':[],'unexpected':[]}
    else:
        model,_,report=load_model(cfg,checkpoint,device=device,verbose=False)
    if report and any(report.get(k) for k in ('missing','adapted','unexpected')):
        raise ValueError('checkpoint/config mismatch: '+str({k:report.get(k) for k in ('arch','missing','adapted','unexpected')}))
    if method=='Diffuser' and (getattr(model,'safety_features_enabled',False) or getattr(model,'feedback_enabled',False)):
        raise ValueError('baseline must not use learned safety features or feedback')
    model.eval()
    d=cfg['diffusion']
    schedule=NoiseSchedule(d['timesteps'],beta_schedule=d['beta_schedule'],
        beta_start=d.get('beta_start',1e-4),beta_end=d.get('beta_end',.02))
    return model,schedule,cfg,report
