"""Phase A: base generator audit — DDIM step counts, checkpoint choice, raw vs decoded.

Runs the same initial noise vector through the DDIM sampler at several step counts
(and optionally several checkpoints) and reports the independent evaluator metrics
per case, plus model-internal diagnostics:

* NFE (network forward evaluations) and wall time;
* x0 prediction error at a set of noise levels (raw vs boundary-decoded output);
* score norm / direction statistics.

Nothing here changes any paper algorithm; this only measures the shared generator.
"""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.diffusion.schedule import NoiseSchedule
from experiments.safety_baselines.models.map_only_points import load_point_baseline,pin_endpoints
from experiments.safety_baselines.models.point_data import PointDataset
from experiments.safety_baselines.adapters.point_evaluator import evaluate_points
from experiments.safety_baselines.adapters.point_sampler import sample_points
from experiments.safety_baselines.algorithms.runner import initial_noise

def score_diagnostics(model,schedule,occ_t,cond_t,gt_t,levels):
    """x0 error and score statistics at given noise levels (single clean-data draw)."""
    out=[]
    g=torch.Generator(device=occ_t.device).manual_seed(0)
    for t in levels:
        noise=torch.randn(gt_t.shape,generator=g,device=gt_t.device)
        a=schedule.sqrt_alphas_cumprod[t];s=schedule.sqrt_one_minus_alphas_cumprod[t]
        x=a*gt_t+s*noise
        pred=model(x,occ_t,cond_t,torch.full((len(x),),t,device=gt_t.device,dtype=torch.long))
        raw=pred['trajectory_raw'];dec=pred['trajectory']
        eps_raw=(x-a*raw)/s;eps_dec=(x-a*dec)/s
        s_raw=-eps_raw/s;s_dec=-eps_dec/s
        cos=torch.nn.functional.cosine_similarity(eps_raw.reshape(len(x),-1),noise.reshape(len(x),-1),dim=1)
        out.append({'t':int(t),'sigma':float((s/a).cpu()),
            'x0_rmse_raw_m':float(torch.sqrt(((raw-gt_t)**2).sum(-1).mean())*80),
            'x0_rmse_decoded_m':float(torch.sqrt(((dec-gt_t)**2).sum(-1).mean())*80),
            'eps_rmse_raw':float(torch.sqrt(((eps_raw-noise)**2).mean())),
            'eps_rmse_decoded':float(torch.sqrt(((eps_dec-noise)**2).mean())),
            'eps_cosine_raw':float(cos.mean()),
            'score_norm_raw_median':float(s_raw.reshape(len(x),-1).norm(dim=1).median()),
            'score_norm_decoded_median':float(s_dec.reshape(len(x),-1).norm(dim=1).median()),
            'pred_raw_finite':bool(torch.isfinite(raw).all()),'pred_decoded_finite':bool(torch.isfinite(dec).all())})
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',default='experiments/safety_baselines/results/map_only_points_t256_e100/ckpt/best_task.pt')
    ap.add_argument('--indices',default='0,81,163,244,326,407,489,570,652,733')
    ap.add_argument('--steps',default='16,64,128,256')
    ap.add_argument('--out-dir',default='experiments/safety_baselines/results/phase_a_base_model')
    ap.add_argument('--tag',default='best_task')
    a=ap.parse_args()
    out=ROOT/a.out_dir;out.mkdir(parents=True,exist_ok=True)
    model,ck=load_point_baseline(ROOT/a.checkpoint,'cuda')
    d=ck['config']['diffusion'];schedule=NoiseSchedule(d['timesteps'],d['beta_schedule'],d['beta_start'],d['beta_end']).to('cuda')
    ds=PointDataset('val',ck['config']['data']['processed_root'])
    rows=[]
    for index in [int(v) for v in a.indices.split(',')]:
        row=ds[index];occ=row['occupancy'][0].numpy();cond=row['cond'].numpy();gt=row['pos'].numpy()
        occ_t=row['occupancy'][None].cuda();cond_t=row['cond'][None].cuda();gt_t=row['pos'][None].cuda()
        init=initial_noise(index,0,128);init[[0,-1]]=cond
        for steps in [int(v) for v in a.steps.split(',')]:
            torch.cuda.synchronize();t0=time.perf_counter()
            q=sample_points(model,schedule,occ_t,cond_t,seed=0,steps=steps,init=init)
            torch.cuda.synchronize();secs=time.perf_counter()-t0
            p=q[0].cpu().numpy();m=evaluate_points(p,occ,cond,gt)
            nfe=min(steps,d['timesteps'])
            rows.append({'checkpoint':a.tag,'index':index,'steps':steps,'nfe':nfe,'seconds':secs,
                'seconds_per_nfe':secs/max(nfe,1),**m})
            print(a.tag,index,'steps',steps,'vtx',m['vertex_collision'],'path',m['collision'],
                'succ',m['planning_success'],'rmse %.1f'%m['rmse_m'],'%.2fs'%secs,flush=True)
        levels=[0,max(d['timesteps']//4,d['timesteps']//8),d['timesteps']//2,3*d['timesteps']//4,d['timesteps']-1]
        for r in score_diagnostics(model,schedule,occ_t,cond_t,gt_t,levels):rows.append({'checkpoint':a.tag,'index':index,'score':r})
    agg={}
    for tag in sorted({r['checkpoint'] for r in rows}):
        for steps in sorted({r['steps'] for r in rows if 'steps' in r}):
            rr=[r for r in rows if r.get('checkpoint')==tag and r.get('steps')==steps]
            n=len(rr)
            agg['%s|steps=%d'%(tag,steps)]={'n':n,
                'vertex_collisions':int(sum(r['vertex_collision'] for r in rr)),
                'segment_only_collisions':int(sum(r['segment_only_collision'] for r in rr)),
                'path_collisions':int(sum(r['collision'] for r in rr)),
                'successes':int(sum(r['planning_success'] for r in rr)),
                'rmse_m_mean':float(np.mean([r['rmse_m'] for r in rr])),
                'min_clearance_m_mean':float(np.mean([r['min_clearance_m'] for r in rr])),
                'path_length_ratio_mean':float(np.mean([r['path_length_ratio'] for r in rr])),
                'max_turning_deg_mean':float(np.mean([r['max_turning_deg'] for r in rr])),
                'mean_second_diff_m_mean':float(np.mean([r['mean_second_difference_m'] for r in rr])),
                'nfe':int(rr[0]['nfe']),'seconds_mean':float(np.mean([r['seconds'] for r in rr]))}
    (out/('sampler_steps_%s.jsonl'%a.tag)).write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf8')
    (out/('sampler_steps_%s_summary.json'%a.tag)).write_text(json.dumps(agg,indent=2),encoding='utf8')
    print(json.dumps(agg,indent=2));print('wrote',out)

if __name__=='__main__':main()
