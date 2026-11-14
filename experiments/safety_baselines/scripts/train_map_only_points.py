"""Train the map-only direct-point baseline from scratch."""
import argparse,datetime,hashlib,json,os,random,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.utils.config import load_config
from src.utils.seed import set_seed
from experiments.safety_baselines.models.point_data import make_point_loader
from src.diffusion.schedule import NoiseSchedule
from experiments.safety_baselines.models.point_losses import point_x0_loss,point_smoothness_loss,boundary_point_loss
from experiments.safety_baselines.models.map_only_points import MapOnlyPointDiffuser

def loss_batch(model,schedule,batch,cfg,device):
    # Only point targets, maps and endpoints are loaded.
    q=batch['pos'].to(device);occ=batch['occupancy'].to(device);cond=batch['cond'].to(device)
    t=torch.randint(0,schedule.num_timesteps,(len(q),),device=device)
    a=schedule.sqrt_alphas_cumprod[t].float()[:,None,None];s=schedule.sqrt_one_minus_alphas_cumprod[t].float()[:,None,None]
    qt=a*q+s*torch.randn_like(q);out=model(qt,occ,cond,t);l=cfg['loss']
    ws,wg=model.boundary_decoder.weights(q.shape[1],device=device,dtype=q.dtype)
    raw={'point':point_x0_loss(out['trajectory_raw'],q),'coarse':point_x0_loss(out['coarse_raw'],q),
      'smooth':point_smoothness_loss(out['trajectory_raw'],q,acc_weight=l['smooth_acc_weight'],jerk_weight=l['smooth_jerk_weight']),
      'boundary':boundary_point_loss(out['trajectory_raw'],q,cond,ws,wg)+l['boundary_coarse_weight']*boundary_point_loss(out['coarse_raw'],q,cond,ws,wg)}
    total=sum(l['lambda_'+k]*v for k,v in raw.items())
    if not torch.isfinite(total):raise RuntimeError('nonfinite loss')
    return total,raw,out

@torch.no_grad()
def validate(model,schedule,loader,cfg,device):
    model.eval();values=[]
    for i,batch in enumerate(loader):
        if i>=cfg['train']['val_batches']:break
        total,raw,out=loss_batch(model,schedule,batch,cfg,device)
        p=out['trajectory'];gt=batch['pos'].to(device);occ=batch['occupancy'].to(device)
        rmse=((p-gt).square().sum(-1).mean(-1).sqrt()*cfg['data']['scene_to_meter']).mean()
        R=occ.shape[-1];ix=torch.floor((p[...,0]+1)*R/2).long();iy=torch.floor((p[...,1]+1)*R/2).long()
        inside=(ix>=0)&(iy>=0)&(ix<R)&(iy<R)
        hit=(occ[torch.arange(len(p),device=device)[:,None],0,iy.clamp(0,R-1),ix.clamp(0,R-1)]>.5)|~inside
        values.append({'total':float(total),'curve_rmse_m':float(rmse),'point_collision_rate':float(hit.float().mean())})
    if not values:raise RuntimeError('empty validation')
    avg={k:sum(v[k] for v in values)/len(values) for k in values[0]}
    avg['task_score']=avg['curve_rmse_m']+cfg['data']['scene_to_meter']*avg['point_collision_rate'];return avg

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);ap.add_argument('--config',default='experiments/safety_baselines/configs/map_only_points_l1_final8_e100.json');ap.add_argument('--epochs',type=int);ap.add_argument('--limit',type=int);ap.add_argument('--max-batches',type=int);args=ap.parse_args()
    if Path(args.run_id).name!=args.run_id:raise ValueError('invalid run-id')
    out=ROOT/'experiments/safety_baselines/results'/args.run_id;out.mkdir(exist_ok=False);(out/'ckpt').mkdir()
    cfg=load_config(args.config)
    if args.epochs is not None:cfg['train']['epochs']=args.epochs
    cfg['train']['ckpt_dir']=str(out/'ckpt');cfg['baseline_run']={'limit':args.limit,'max_batches':args.max_batches,'fresh_training':True}
    (out/'config_snapshot.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
    device='cuda';set_seed(cfg['env']['seed']);torch.backends.cudnn.benchmark=cfg['train'].get('cudnn_benchmark',False)
    model=MapOnlyPointDiffuser(cfg).to(device);schedule=NoiseSchedule(**{'num_timesteps':cfg['diffusion']['timesteps'],'beta_schedule':cfg['diffusion']['beta_schedule'],'beta_start':cfg['diffusion']['beta_start'],'beta_end':cfg['diffusion']['beta_end']}).to(device)
    loaders={};sizes={}
    for split in ['train','val']:
        loaders[split],ds=make_point_loader(split,cfg['data']['processed_root'],batch_size=cfg['data']['batch_size'],shuffle=split=='train',num_workers=cfg['data']['num_workers'] if split=='train' else 0,limit=args.limit,num_points=model.num_points);sizes[split]=len(ds)
    optim=torch.optim.AdamW(model.parameters(),lr=cfg['train']['lr'],weight_decay=cfg['train']['weight_decay'])
    summary={'status':'training','pid':os.getpid(),'architecture':model.architecture,'params':sum(p.numel() for p in model.parameters()),'epochs_requested':cfg['train']['epochs'],'epochs_done':0,'history':[],'sizes':sizes,'input_fields':['occupancy','cond','trajectory_t','t'],'best_task_score':None,'best_task_epoch':None}
    manifest={'status':'training','pid':os.getpid(),'python':sys.executable,'seed':cfg['env']['seed'],'architecture':model.architecture,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'versions':{'torch':torch.__version__,'numpy':np.__version__},'device':torch.cuda.get_device_name(),'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'experiments/safety_baselines/models/map_only_points.py',ROOT/'experiments/safety_baselines/models/point_data.py',ROOT/'experiments/safety_baselines/models/point_losses.py']}}
    def write():
        (out/'training_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf8');(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    def save(name,epoch):
        target=out/'ckpt'/name;temp=target.with_suffix('.tmp')
        torch.save({'architecture':model.architecture,'model_state':model.state_dict(),'optimizer_state':optim.state_dict(),'epoch':epoch,'config':cfg,'rng_state':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'numpy':np.random.get_state(),'python':random.getstate()}},temp);os.replace(temp,target)
    write();print(json.dumps({k:v for k,v in summary.items() if k!='history'}),flush=True);started=time.time();best_val=float('inf')
    try:
        for epoch in range(cfg['train']['epochs']):
            model.train();total_sum=0;count=0
            for step,batch in enumerate(loaders['train']):
                if args.max_batches is not None and step>=args.max_batches:break
                optim.zero_grad(set_to_none=True);loss,raw,_=loss_batch(model,schedule,batch,cfg,device);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train']['grad_clip'],error_if_nonfinite=True);optim.step();total_sum+=float(loss.detach());count+=1
                if (step+1)%cfg['train']['log_interval']==0:print(f'epoch={epoch} step={step+1}/{len(loaders["train"])} loss={total_sum/count:.6f}',flush=True)
            if not count:raise RuntimeError('empty training')
            val=validate(model,schedule,loaders['val'],cfg,device)
            if val['total']<best_val:best_val=val['total'];save('best.pt',epoch)
            if summary['best_task_score'] is None or val['task_score']<summary['best_task_score']:
                summary['best_task_score']=val['task_score'];summary['best_task_epoch']=epoch;save('best_task.pt',epoch)
            save('latest.pt',epoch)
            if (epoch+1)%cfg['train']['save_every']==0:save(f'epoch_{epoch+1}.pt',epoch)
            summary['epochs_done']=epoch+1;summary['elapsed_seconds']=time.time()-started;summary['history'].append({'epoch':epoch,'train_total':total_sum/count,'val':val,'seconds':summary['elapsed_seconds']});write();print(f'epoch={epoch} val={val} seconds={summary["elapsed_seconds"]:.1f}',flush=True)
            if summary['elapsed_seconds']>cfg['train']['max_hours']*3600:break
        manifest['status']=summary['status']='complete' if summary['epochs_done']==cfg['train']['epochs'] else 'budget_reached';write()
    except BaseException as exc:
        manifest['status']=summary['status']='failed';manifest['error']=repr(exc);write();raise
if __name__=='__main__':main()
