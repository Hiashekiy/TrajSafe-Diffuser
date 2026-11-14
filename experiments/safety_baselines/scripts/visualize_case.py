"""Phase F: per-scene visualisation and unified metrics.

For one validation scene it writes

    <out>/case_<idx>_trajectories.png   occupancy map + start/goal + every method's
                                        delivered polyline, collisions marked
    <out>/case_<idx>_denoise.gif        the DDIM reverse chain, frame by frame
    <out>/case_<idx>_denoise.csv        per-step metrics of the reverse chain
    <out>/case_<idx>_record.json        full structured record (config, seeds, metrics)

and prints the unified metric table for the same scene. Every method is scored by
the same evaluator (`adapters/point_evaluator.py`); Ours is out of scope here.
"""
import argparse,csv,io,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import pick_times
from experiments.safety_baselines.models.map_only_points import load_point_baseline,pin_endpoints
from experiments.safety_baselines.models.point_data import PointDataset
from experiments.safety_baselines.adapters.point_evaluator import evaluate_points
from experiments.safety_baselines.adapters.constraints import MapConstraint
from experiments.safety_baselines.algorithms.runner import run_method,initial_noise,METHODS

def plot_scene(occ,cond,curves,gt,path,title):
    fig,ax=plt.subplots(figsize=(6,6),dpi=130)
    ax.imshow(occ,cmap='gray_r',origin='lower',extent=[-1,1,-1,1],alpha=.85)
    ax.plot(gt[:,0],gt[:,1],'--',color='tab:gray',lw=1.2,label='GT')
    ax.plot(cond[:,0],cond[:,1],'o',color='tab:green',ms=7,label='start/goal')
    for name,p,style in curves:
        ax.plot(p[:,0],p[:,1],style,lw=1.4,label=name)
        hit=np.array([occ[min(int((y+1)*occ.shape[0]/2),occ.shape[0]-1),min(int((x+1)*occ.shape[1]/2),occ.shape[1]-1)]
                      for x,y in p]).astype(bool)
        if hit.any():ax.plot(p[hit,0],p[hit,1],'x',color='red',ms=5)
    ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_title(title,fontsize=9)
    ax.legend(fontsize=6,loc='lower right');ax.set_xticks([]);ax.set_yticks([])
    fig.tight_layout();fig.savefig(path);plt.close(fig)

def plot_frame(occ,points,frame,path,title):
    fig,ax=plt.subplots(figsize=(4,4),dpi=110)
    ax.imshow(occ,cmap='gray_r',origin='lower',extent=[-1,1,-1,1],alpha=.85)
    ax.plot(points[:,0],points[:,1],'-o',color='tab:blue',ms=1.5,lw=1)
    ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title,fontsize=7)
    fig.tight_layout();fig.savefig(path);plt.close(fig);return path

def denoise_frames(model,schedule,occ_t,cond_t,occ,init,steps,outdir,prefix):
    """Reverse chain with per-step metrics; returns rows and a GIF."""
    torch.manual_seed(0)
    points=pin_endpoints(torch.as_tensor(np.asarray(init,dtype=np.float32),device=cond_t.device).reshape(1,128,2),cond_t)
    memory=model.encode_scene(occ_t);times=pick_times(schedule.num_timesteps,steps)
    pairs=[(t,t-1) for t in range(schedule.num_timesteps-1,-1,-1)] if times is None else \
        [(times[i],times[i-1]) for i in range(len(times)-1,0,-1)]+[(times[0],-1)]
    rows=[];frames=[]
    for k,(t,nxt) in enumerate(pairs):
        with torch.no_grad():
            clean=model.forward_from_memory(points,memory,cond_t,torch.full((1,),t,device=cond_t.device,dtype=torch.long))['trajectory']
        p=clean[0].cpu().numpy()
        m=evaluate_points(p,occ,cond_t[0].cpu().numpy(),p)   # rmse vs itself -> use metrics only
        f=plot_frame(occ,p,k,f'{outdir}/{prefix}_f{k:03d}.png',f't={t} nfe={k+1}')
        frames.append(f)
        rows.append({'frame':k,'t':int(t),'next':int(nxt),'min_clearance_m':m['min_clearance_m'],
            'vertex_collision':m['vertex_collision'],'path_collision':m['collision'],
            'path_length_ratio':m['path_length_ratio'],'max_turning_deg':m['max_turning_deg']})
        if nxt<0:points=clean
        else:
            a=schedule.sqrt_alphas_cumprod.to(points).view(-1);s=schedule.sqrt_one_minus_alphas_cumprod.to(points).view(-1)
            points=a[nxt]*clean+s[nxt]*(points-a[t]*clean)/s[t]
        points=pin_endpoints(points,cond_t)
    imgs=[Image.open(f) for f in frames];imgs[0].save(f'{outdir}/{prefix}.gif',save_all=True,append_images=imgs[1:],duration=90,loop=0)
    for f in frames:Path(f).unlink()
    return rows,points[0].cpu().numpy()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='experiments/safety_baselines/configs/seven_native_points_vertexall_t256.json')
    ap.add_argument('--indices',default='0,81')
    ap.add_argument('--methods',default='Diffuser,Projected,ReS-diffuser,PDM')
    ap.add_argument('--ddim-steps',type=int,default=256)
    ap.add_argument('--out-dir',default='experiments/safety_baselines/results/phase_f_visualization')
    a=ap.parse_args();cfg=json.loads((ROOT/a.config).read_text(encoding='utf8'))
    out=ROOT/a.out_dir;out.mkdir(parents=True,exist_ok=True)
    model,ck=load_point_baseline(ROOT/cfg['checkpoint'],'cuda')
    d=ck['config']['diffusion'];schedule=NoiseSchedule(d['timesteps'],d['beta_schedule'],d['beta_start'],d['beta_end']).to('cuda')
    ds=PointDataset('val',ck['config']['data']['processed_root'])
    summary=[]
    for index in [int(v) for v in a.indices.split(',')]:
        row=ds[index];occ=row['occupancy'][0].numpy();cond=row['cond'].numpy();gt=row['pos'].numpy()
        occ_t=row['occupancy'][None].cuda();cond_t=row['cond'][None].cuda()
        init=initial_noise(index,0,128);init[[0,-1]]=cond
        curves=[];record={'index':index,'config':a.config,'checkpoint':cfg['checkpoint'],'seed':0,
            'init_noise_seed':int(__import__('hashlib').sha256(f'{index}|0|initial'.encode()).hexdigest()[:8],16),'methods':{}}
        # DDIM reverse chain + gif/csv
        rows,final=denoise_frames(model,schedule,occ_t,cond_t,occ,init,a.ddim_steps,out,f'case_{index}_denoise')
        with (out/f'case_{index}_denoise.csv').open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
        m=evaluate_points(final,occ,cond,gt)
        curves.append((f'Diffuser-{a.ddim_steps}',final,'-'))
        record['methods']['Diffuser']=m
        for method in [x for x in a.methods.split(',') if x!='Diffuser']:
            r=run_method(method,model,schedule,occ,cond,index,0,cfg,init_noise=init)
            if r['trajectory'] is None:
                record['methods'][method]={'status':r['status'],'failure_reason':r['failure_reason']};continue
            p=np.asarray(r['trajectory']);mm=evaluate_points(p,occ,cond,gt)
            record['methods'][method]=mm
            curves.append((method,p,'-' if len(curves)%2 else '--'))
            (out/f'case_{index}_{method}_steps.json').write_text(json.dumps(r['per_step_trace'][:2000],indent=1),encoding='utf8')
        plot_scene(occ,cond,curves,gt,out/f'case_{index}_trajectories.png',
            f'case {index}: DDIM-{a.ddim_steps} vs safety loops')
        (out/f'case_{index}_record.json').write_text(json.dumps(record,indent=2),encoding='utf8')
        for k,v in record['methods'].items():
            if 'collision' in v:
                print('%-14s vtx %-5s seg %-5s path %-5s succ %-5s minclear %6.2f m len %.2f turn %5.1f'%(
                    k,v['vertex_collision'],v['segment_only_collision'],v['collision'],v['planning_success'],
                    v['min_clearance_m'],v['path_length_ratio'],v['max_turning_deg']))
            else:print('%-14s %s %s'%(k,v['status'],v.get('failure_reason')))
        summary.append(record)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('wrote',out)

if __name__=='__main__':main()
