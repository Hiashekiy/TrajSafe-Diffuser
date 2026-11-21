"""Paired cold-field benchmark: deployed diffusion field search versus plain A*.
Both methods use the bundled package's same four-neighbour inflated grid.
GPU time is synchronized; no ellipse/ALM or subsequent DDIM steps are timed.
"""
from pathlib import Path
import argparse, csv, json, platform, sys, time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.datasets.carla_spline_dataset import make_loader
from src.diffusion.sampler_v5 import _build_safe_grids
from src.diffusion.schedule import NoiseSchedule
from src.models.factory import build_model
from src.utils.config import load_config, num_controls, num_safety_queries
from src.grid_search.field_search import guidance_class


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--limit',type=int,default=420)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--out',default='outputs/benchmark_field_vs_astar')
    args=ap.parse_args()
    out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
    cfg=load_config(str(ROOT/'configs/config_v5_c32q128.yaml'))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    GF=guidance_class()
    model=build_model(cfg).to(device)
    ck=torch.load(ROOT/'outputs/models/v5_c32q128/ckpt/final_best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model'],strict=True); model.eval()
    sched=NoiseSchedule(cfg['diffusion']['timesteps'],beta_schedule=cfg['diffusion'].get('beta_schedule','squaredcos_cap_v2')).to(device)
    loader,_=make_loader('test',cfg['data']['processed_root'],batch_size=1,shuffle=False,num_workers=0,limit=args.limit,require_labels=False,num_controls=num_controls(cfg),num_safety_queries=num_safety_queries(cfg))
    field_opts={k:cfg['grid_search']['field_guided'][k] for k in ['weight','lam','distance','source_method','num_samples','method']}
    search_opts={k:cfg['grid_search']['field_guided'][k] for k in ['alpha','gamma','adaptive','clearance_beta','clearance_mode','clearance_threshold']}
    def sync():
        if device=='cuda': torch.cuda.synchronize()
    raw_rows=[]; case_rows=[]
    with torch.no_grad():
        for idx,batch in enumerate(loader):
            cond=batch['cond'].to(device); occ=batch['occupancy'].to(device)
            grids,spec,grid_ms=_build_safe_grids(occ,cfg)
            safe=grids[0][0]; inflated=(~safe.free).astype(np.uint8)
            ends=cond[0].cpu().numpy()
            start,goal=[tuple(np.rint(spec.world_to_cell(p)[::-1]).astype(int)) for p in ends]
            torch.manual_seed(idx)
            q=torch.randn(1,num_controls(cfg),2,device=device)
            q[:,0]=cond[:,0]; q[:,-1]=cond[:,1]
            t=int(sched.num_timesteps-1)
            tb=torch.full((1,),t,device=device,dtype=torch.long)
            ab=sched.sqrt_alphas_cumprod[t].expand(1)
            heading=batch.get('heading'); hv=batch.get('heading_valid')
            kw=dict(active_path=None,feedback_delta=None,feedback_valid=None,
                    heading=None if heading is None else heading.to(device),
                    heading_valid=None if hv is None else hv.to(device))
            # Warm network and library imports before any measured samples.
            if idx==0:
                for _ in range(5): model(q,occ,cond,tb,ab,**kw)
                tmp=GF(inflated,**field_opts)
                guide=model(q,occ,cond,tb,ab,**kw)['final'][0].cpu().numpy()
                tmp.build(spec.world_to_cell(guide)[:,::-1]); tmp.search(start,goal,**search_opts); tmp.astar(start,goal)
                sync()
            net_times=[]
            for rep in range(args.repeats):
                sync(); t0=time.perf_counter()
                prediction=model(q,occ,cond,tb,ab,**kw)
                guide=prediction['final'][0].cpu().numpy(); sync()
                net_times.append((time.perf_counter()-t0)*1000)
            guide_rc=spec.world_to_cell(guide)[:,::-1]
            runs=[]
            for rep in range(args.repeats):
                pair={}
                # Alternate order to reduce systematic warm-cache bias.
                order=['guided','astar'] if (idx+rep)%2==0 else ['astar','guided']
                for method in order:
                    t0=time.perf_counter()
                    gf=GF(inflated,**field_opts)
                    init_ms=(time.perf_counter()-t0)*1000
                    build_ms=0.; fallback=False
                    if method=='guided':
                        t1=time.perf_counter()
                        try: gf.build(guide_rc)
                        except RuntimeError: fallback=True
                        build_ms=(time.perf_counter()-t1)*1000
                        res=gf.astar(start,goal) if fallback else gf.search(start,goal,**search_opts)
                        if res['status']!='success':
                            fallback=True; res=gf.astar(start,goal)
                    else: res=gf.astar(start,goal)
                    wall_ms=(time.perf_counter()-t0)*1000
                    assert res['status']=='success', (idx,method,res['status'])
                    assert gf.is_valid(res['path'],start,goal)['valid']
                    pair[method]=dict(wall_ms=wall_ms,init_ms=init_ms,build_ms=build_ms,
                                      preprocess_ms=res['preprocess_ms'],queue_ms=res['search_ms'],
                                      expanded=res['expanded'],steps=res['length'],fallback=fallback)
                rr={'sample':int(batch['sample_id'][0]),'index':idx,'repeat':rep,'grid_ms':grid_ms,'network_ms':net_times[rep]}
                for method,v in pair.items(): rr.update({method+'_'+k:x for k,x in v.items()})
                rr['guided_with_network_ms']=rr['guided_wall_ms']+rr['network_ms']
                rr['guided_with_map_network_ms']=rr['guided_with_network_ms']+grid_ms
                rr['astar_with_map_ms']=rr['astar_wall_ms']+grid_ms
                runs.append(rr); raw_rows.append(rr)
            row={k:(float(np.median([r[k] for r in runs])) if isinstance(v,(int,float,np.number)) else v) for k,v in runs[0].items() if k!='repeat'}
            row['sample']=int(batch['sample_id'][0]); row['index']=idx
            row['length_ratio']=row['guided_steps']/max(1,row['astar_steps'])
            row['guided_search_faster']=row['guided_wall_ms']<row['astar_wall_ms']
            case_rows.append(row)
            if (idx+1)%20==0: print(f'{idx+1}/{len(loader)} paired cases',flush=True)
    for filename,rows in [('runs.csv',raw_rows),('per_case.csv',case_rows)]:
        with open(out/filename,'w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    def stats(key):
        a=np.array([r[key] for r in case_rows],dtype=float)
        return dict(mean=float(a.mean()),median=float(np.median(a)),p90=float(np.percentile(a,90)),max=float(a.max()))
    summary=dict(n=len(case_rows),repeats=args.repeats,device=torch.cuda.get_device_name() if device=='cuda' else 'CPU',cpu=platform.processor(),torch=torch.__version__,field_parameters=field_opts,search_parameters=search_opts,protocol='same footprint-inflated 4-neighbour grid, unit edge cost, Manhattan A*; independent fresh field/grid per repeat; alternating method order; 5 GPU warmups; per-case medians across repeats; one first-step diffusion forward, synchronized including D2H; map inflation shared and timed once; excludes later DDIM, ellipses, ALM, resampling, dataset IO and model loading',success_guided=len(case_rows),success_astar=len(case_rows),metrics={k:stats(k) for k in ['grid_ms','network_ms','guided_init_ms','guided_build_ms','guided_preprocess_ms','guided_queue_ms','guided_wall_ms','astar_init_ms','astar_queue_ms','astar_wall_ms','guided_with_network_ms','guided_with_map_network_ms','astar_with_map_ms','guided_expanded','astar_expanded','length_ratio']},guided_faster_fraction=float(np.mean([r['guided_search_faster'] for r in case_rows])),fallback_cases=int(sum(r['guided_fallback']>0 for r in case_rows)))
    summary['astar_over_guided_median_time']=summary['metrics']['astar_wall_ms']['median']/summary['metrics']['guided_wall_ms']['median']
    summary['astar_over_guided_with_network_median_time']=summary['metrics']['astar_wall_ms']['median']/summary['metrics']['guided_with_network_ms']['median']
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
