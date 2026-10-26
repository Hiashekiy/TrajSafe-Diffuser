"""Compare best/latest modified models on route, ellipse, region and connection tests."""
from pathlib import Path
import argparse, copy, datetime, hashlib, json, shutil, sys, time, traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def write_json(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def freeze(out, reuse_run=None, model_run=None):
    base=Path(model_run).resolve() if model_run else ROOT/'outputs/ablation_hpath/without_hpath'
    snap=out/'source_snapshot'
    source_base=(Path(reuse_run).resolve()/'source_snapshot') if reuse_run else ((base.parent/'source_snapshot') if model_run else base/'source_snapshot')
    shutil.copytree(source_base/'src',snap/'src',ignore=shutil.ignore_patterns('__pycache__'))
    (snap/'scripts').mkdir()
    for name in ['eval_chapter4.py','geometry_audit_utils.py']:
        src=(source_base/'scripts'/name) if name=='eval_chapter4.py' else ROOT/'scripts'/name
        shutil.copy2(src,snap/'scripts'/name)
    shutil.copy2(__file__,snap/'scripts'/Path(__file__).name)
    if reuse_run is not None:
        previous=Path(reuse_run).resolve()
        # Preserve the old architecture/evaluation source and exact weights;
        # only replace the region constructor for the before/after experiment.
        shutil.copy2(ROOT/'src/geometry/convex_region.py',snap/'src/geometry/convex_region.py')
        weights={}
        for tag in ['best','latest']:
            src=previous/(tag+'.pt')
            ck=torch.load(src,map_location='cpu',weights_only=False)
            assert ck['config']['model']['path_features']['enabled'] is False
            assert ck['config']['model']['safety_query']['center_encoding'] is False
            dest=out/(tag+'.pt'); shutil.copy2(src,dest)
            weights[tag]={'epoch':int(ck['epoch'])+1,'path':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
        return snap,weights
    weights={}
    for tag,name in [('best','best_task.pt'),('latest','latest.pt')]:
        src=base/'ckpt'/name
        for attempt in range(6):
            try:
                ck=torch.load(src,map_location='cpu',weights_only=False)
                break
            except Exception:
                if attempt==5: raise
                time.sleep(.5)
        if not model_run and tag=='latest' and int(ck['epoch'])+1==weights['best']['epoch']:
            print('Best and latest coincide; waiting for the next completed epoch before freezing latest',flush=True)
            for wait_step in range(150):
                time.sleep(1)
                try:
                    newer=torch.load(src,map_location='cpu',weights_only=False)
                except Exception:
                    continue
                if int(newer['epoch'])+1>weights['best']['epoch']:
                    ck=newer
                    break
            else:
                raise RuntimeError('Latest has not advanced; best/latest are identical, no distinct comparison available')
        cfg=ck['config']
        assert cfg['model']['path_features']['enabled'] is False
        assert cfg['model']['safety_query']['center_encoding'] is False
        dest=out/(tag+'.pt')
        torch.save({k:ck[k] for k in ['model_state','config','epoch']},dest)
        weights[tag]={'epoch':int(ck['epoch'])+1,'path':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    return snap,weights


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--samples',type=int,default=0)
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    ap.add_argument('--out',required=True)
    ap.add_argument('--reuse-run',help='Use exact frozen best/latest weights; replace only convex_region.py')
    ap.add_argument('--model-run',help='Completed modified-model training directory (ckpt child)')
    ap.add_argument('--processed',help='Dataset root for this experiment')
    args=ap.parse_args()
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',{'stage':'freezing'})
    snap,weights=freeze(out,args.reuse_run,args.model_run)
    sys.path.insert(0,str(snap)); sys.path.insert(0,str(snap/'scripts'))
    from src.utils.config import load_config
    from src.utils.checkpoint import load_model
    from src.datasets.carla_spline_dataset import CarlaSplineDataset,make_collate
    from src.diffusion.schedule import NoiseSchedule
    from src.diffusion.sampler import sample
    from src.geometry.sdf_field import build_occupancy_sdf
    from src.geometry.convex_region import EllipseRegionBuilder,halfspaces_to_vertices
    from src.geometry.safety_corridor import build_safety_corridor,overlap_ratio,polygon_area
    from eval_chapter4 import evaluate_controls
    from geometry_audit_utils import obstacle_boxes,audit_polygon
    cfg=torch.load(weights['best']['path'],map_location='cpu',weights_only=False)['config']
    dsroot=Path(args.processed).resolve() if args.processed else ROOT/'data/processed/160m-k4p-c48'
    # Freeze baseline scales from validation labels, before reading test metrics.
    labels=np.load(dsroot/'val/ellipse_shape4_gt.npy',mmap_mode='r')
    valid=np.load(dsroot/'val/shape_valid.npy').astype(bool)
    logs=np.asarray(labels[valid,:2],dtype=np.float64)
    scales={'a':float(np.exp(np.median(logs[:,0]))),'b':float(np.exp(np.median(logs[:,1]))),
            'radius':float(np.exp(np.median(logs.mean(axis=1)))),'source':'median validation valid labels'}
    manifest={'weights':weights,'seeds':args.seeds,'samples_per_seed':args.samples or 420,
              'dataset':str(dsroot),'scales':scales,'steps':16,'warmup':3,
              'geometry_capture':'first step after 3 warmup forwards (t=12), first-choice route; no center prediction',
              'route_policies':['network ranking with corridor fallback','shortest ranking with corridor fallback'],
              'geometry_variants':['predicted','fixed_circle','fixed_tangent_ellipse'],
              'safety_check':'independent polygon versus occupied pixel squares SAT, positive-area intersections; tolerance 1e-9 scene',
              'timing':'descriptive timing only; not a controlled speed benchmark',
              'reuse_run':args.reuse_run,
              'fixed_geometry_cache':'exact sample, centre and shape byte equality; cached timing excluded from means',
              'source_sha256':{str(p.relative_to(snap)):hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.rglob('*') if p.is_file()}}
    write_json(out/'manifest.json',manifest)
    print('FROZEN',json.dumps(weights),'SCALES',scales,flush=True)
    ds=CarlaSplineDataset('test',str(dsroot),geometry_points=1280,limit=args.samples or None,require_labels=True)
    collate=make_collate(ds)
    schedule=NoiseSchedule(16,beta_schedule=cfg['diffusion'].get('beta_schedule','squaredcos_cap_v2'))
    region_cfg=dict(cfg['alm']); region_cfg.update(cfg['corridor']['region'])
    s2m=80.; chunk=32
    obstacle_cache={}
    fixed_geometry_cache={}
    all_records=[]
    for tag in ['best','latest']:
        stored=torch.load(weights[tag]['path'],map_location='cpu',weights_only=False)
        model,_,report=load_model(stored['config'],stored,device='cuda',verbose=False)
        assert not report['missing'] and not report['adapted'] and not report['unexpected']
        assert model.path_feature_head is None and model.ellipse_geometry.mlp_c is None
        model.assert_shapes=False
        original=model.forward_all
        for seed in args.seeds:
            for start in range(0,len(ds),chunk):
                batch=collate([ds[i] for i in range(start,min(start+chunk,len(ds)))])
                occ=batch['occupancy'].cuda(); cond=batch['cond'].cuda()
                xy=batch['candidate_xy'].cuda(); mask=batch['candidate_mask'].cuda()
                gamma=batch['candidate_geometry'].cuda(); lens=batch['candidate_geometry_lengths'].cuda()
                lengths=(gamma[:, :, 1:]-gamma[:,:,:-1]).norm(dim=-1)
                lengthmask=torch.arange(lengths.shape[-1],device='cuda')[None,None]< (lens[:,:,None]-1)
                route_lengths=(lengths*lengthmask).sum(-1).masked_fill(~mask,float('inf'))
                shortest=route_lengths.argmin(-1)
                captures=[]
                def wrapped(*aa,**kw):
                    if active_policy=='shortest':
                        requested=kw.get('select_index')
                        kw['select_index']=shortest if requested is None else torch.where(requested<0,shortest,requested)
                    prediction=original(*aa,**kw)
                    if active_policy=='shortest':
                        prediction['topo']=dict(prediction['topo'])
                        prediction['topo']['pi']=torch.softmax(-route_lengths,dim=-1)
                    if active_policy=='network' and not captures and int(aa[3][0])==12:
                        captures.append({key:prediction['ellipse'][key].detach().cpu().numpy()
                                         for key in ['center','shape4','a','b','theta','progress']})
                        captures[0]['idx']=prediction['selected_idx'].detach().cpu().numpy()
                        captures[0]['pi']=prediction['topo']['pi'].detach().cpu().numpy()
                    return prediction
                model.forward_all=wrapped
                route_rows=[{} for _ in range(len(cond))]
                for active_policy in ['network','shortest']:
                    t0=time.perf_counter()
                    result=sample(model,schedule,cond,occ,xy,mask,gamma,lens,device='cuda',steps=16,seed=seed,seed_offset=start,
                                  return_trace=False,alm_config=cfg['alm'],corridor_config=cfg['corridor'],guidance='alm')
                    sdf=build_occupancy_sdf(occ,scene_to_meter=s2m,map_margin=float(cfg['alm'].get('map_margin',0.)))
                    metrics=evaluate_controls(model.bspline,result['control'],occ,cond,batch['pos'].cuda(),sdf,result.get('pack'),batch,s2m)
                    # Use the same dense evaluation basis as the manuscript.
                    from src.diffusion.sampler import _dense_decode
                    curve=_dense_decode(model.bspline,result['control'],512)
                    pathlength=(curve[:,1:]-curve[:,:-1]).norm(dim=-1).sum(-1)*s2m
                    for b in range(len(cond)):
                        idx=int(result['selected_idx'][b]); corridor=result['corridors'][b]
                        route_rows[b][active_policy]={
                            'selected_idx':idx,'selected_length_m':float(route_lengths[b,idx])*s2m,
                            'length_ratio':float(route_lengths[b,idx]/route_lengths[b,shortest[b]]),
                            'label_match':bool(idx==int(batch['topology_best'][b])),
                            'collision':bool(metrics['collision'][b]),'rmse_m':float(metrics['rmse_m'][b]),
                            'smoothness_m':float(metrics['smoothness_m'][b]),'trajectory_length_m':float(pathlength[b]),
                            'bridges':int(corridor['bridge_cell_count']) if corridor else 0,
                            'corridor_attempts':int(result['activation_info']['attempts'][b]),
                            'corridor_failure':result['activation_info']['failure_reason'][b],
                            'first_choice_idx':int(captures[0]['idx'][b]) if active_policy=='network' else int(shortest[b]),
                            'ranking_ce':float(-np.log(max(captures[0]['pi'][b,int(batch['topology_best'][b])],1e-12))) if active_policy=='network' else None}
                    del result
                model.forward_all=original
                cap=captures[0]
                for b in range(len(cond)):
                    index=start+b
                    if index not in obstacle_cache:
                        obstacle_cache[index]=obstacle_boxes(batch['occupancy'][b,0].numpy())
                    boxes,half=obstacle_cache[index]
                    builder=EllipseRegionBuilder(occ[b:b+1],region_cfg)
                    center=cap['center'][b]; anchors=cap['progress'][b]
                    selected=int(cap['idx'][b]); fullgamma=gamma[b,selected].cpu().numpy(); glen=int(lens[b,selected])
                    row={'model':tag,'epoch':weights[tag]['epoch'],'seed':seed,'index':index,'route':route_rows[b],'ellipse':{},'regions':{}}
                    row['route']['network']['top1_label_match']=bool(selected==int(batch['topology_best'][b]))
                    good=(batch['shape_valid'][b].numpy().astype(bool)) & (selected==int(batch['topology_best'][b]))
                    gt=batch['ellipse_shape4_gt'][b].numpy()
                    if np.any(good):
                        diff=cap['shape4'][b]-gt
                        gt_ratio=np.exp(gt[:,0]-gt[:,1])
                        oriented=good & (gt_ratio>=1.1)
                        dot=(cap['shape4'][b,:,2:]*gt[:,2:]).sum(-1)
                        row['ellipse']['label_count']=int(good.sum())
                        row['ellipse']['log_axis_mae']=float(np.abs(diff[good,:2]).mean())
                        row['ellipse']['direction_count']=int(oriented.sum())
                        row['ellipse']['angle_mae_deg']=float(np.mean(.5*np.arccos(np.clip(dot[oriented],-1,1)))*180/np.pi) if np.any(oriented) else None
                    else:
                        row['ellipse'].update(label_count=0,direction_count=0,log_axis_mae=None,angle_mae_deg=None)
                    row['ellipse'].update(a_mean_m=float(cap['a'][b].mean()*s2m),b_mean_m=float(cap['b'][b].mean()*s2m))
                    direction=np.gradient(center,axis=0)
                    theta=np.arctan2(direction[:,1],direction[:,0])
                    circle=np.tile([np.log(scales['radius']),np.log(scales['radius']),1.,0.],(128,1))
                    tangent=np.column_stack((np.full(128,np.log(scales['a'])),np.full(128,np.log(scales['b'])),np.cos(2*theta),np.sin(2*theta)))
                    for variant,shape in [('predicted',cap['shape4'][b]),('fixed_circle',circle),('fixed_tangent_ellipse',tangent)]:
                        cache_key=(index,variant,center.tobytes(),shape.tobytes())
                        if variant!='predicted' and cache_key in fixed_geometry_cache:
                            row['regions'][variant]=dict(fixed_geometry_cache[cache_key])
                            row['regions'][variant].update(base_build_ms=None,connection_ms=None,cache_hit=True)
                            continue
                        t0=time.perf_counter()
                        tensors=builder.build_from_ellipse(torch.as_tensor(center,device='cuda',dtype=torch.float32)[None],torch.as_tensor(shape,device='cuda',dtype=torch.float32)[None])
                        A,bb,m,v=tensors
                        aa=A[0].cpu().numpy().astype(float); bounds=bb[0].cpu().numpy().astype(float); masks=m[0].cpu().numpy().astype(bool)
                        polys=[halfspaces_to_vertices(a[mk],bv[mk]) for a,bv,mk in zip(aa,bounds,masks)]
                        geom_ms=(time.perf_counter()-t0)*1000
                        areas=[polygon_area(p)*s2m*s2m for p in polys]
                        ratios=[overlap_ratio(polys[i],polys[i+1]) for i in range(len(polys)-1)]
                        checks=[audit_polygon(p,boxes,half) for p in polys]
                        # Cross-check the first unsafe polygon of each model/variant
                        # with an independent GEOS implementation and save full precision evidence.
                        evidence=out/(tag+'_'+variant+'_safety_example.json')
                        unsafe=[i for i,c in enumerate(checks) if c['colliding_cells']>0]
                        if unsafe and not evidence.exists():
                            from shapely.geometry import Polygon, box
                            i=unsafe[0]; poly=np.asarray(polys[i]); low,high=poly.min(0),poly.max(0)
                            near=boxes[np.all(boxes+half>low,axis=1)&np.all(boxes-half<high,axis=1)]
                            geos=Polygon(poly)
                            geos_area=sum(geos.intersection(box(c[0]-half,c[1]-half,c[0]+half,c[1]+half)).area for c in near)
                            assert abs(geos_area-checks[i]['overlap_area_scene2'])<1e-8
                            write_json(evidence,{'model':tag,'variant':variant,'sample':index,'seed':seed,'region':i,
                                       'polygon':poly.tolist(),'nearby_obstacle_centers':near.tolist(),'pixel_half_width':half,
                                       'sat_area_m2':checks[i]['overlap_area_scene2']*s2m*s2m,'geos_area_m2':geos_area*s2m*s2m})
                        residual=[float(np.max(a[mk]@c-bv[mk])) for a,bv,mk,c in zip(aa,bounds,masks,center)]
                        class CachedBuilder:
                            def __init__(self,base,data): self.base,self.data=base,data
                            def __getattr__(self,key): return getattr(self.base,key)
                            def build_from_ellipse(self,*args,**kwargs): return self.data
                        t0=time.perf_counter()
                        closed=build_safety_corridor(CachedBuilder(builder,tensors),center,shape,anchors,gamma=fullgamma,gamma_lengths=glen,config=cfg['corridor'])
                        connect_ms=(time.perf_counter()-t0)*1000
                        bridges=[audit_polygon(c.polygon,boxes,half) for c in closed.cells if c.source=='bridge']
                        row['regions'][variant]={
                            'base_count':len(polys),'base_area_mean_m2':float(np.mean(areas)),'base_area_p05_m2':float(np.percentile(areas,5)),
                            'base_unsafe_count':sum(c['colliding_cells']>0 for c in checks),
                            'base_outside_count':sum(c['outside'] for c in checks),'base_invalid_count':sum(c['invalid'] for c in checks),
                            'obstacle_intersection_area_m2':sum(c['overlap_area_scene2'] for c in checks)*s2m*s2m,
                            'max_region_intersection_area_m2':max(c['overlap_area_scene2'] for c in checks)*s2m*s2m,
                            'center_max_residual_m':max(residual)*s2m,
                            'before_overlap_mean':float(np.mean(ratios)),'before_overlap_min':float(min(ratios)),
                            'before_below_threshold':sum(x+1e-12<cfg['corridor']['min_overlap_ratio'] for x in ratios),
                            'after_overlap_mean':float(np.mean(closed.overlap_ratio)) if closed.valid else None,
                            'after_overlap_min':float(min(closed.overlap_ratio)) if closed.valid else None,
                            'bridges':closed.bridge_cell_count,'bridge_unsafe_count':sum(c['colliding_cells']>0 or c['outside'] or c['invalid'] for c in bridges),
                            'failure_reason':closed.failure_reason,'constructed_without_bridge':all(x+1e-12>=cfg['corridor']['min_overlap_ratio'] for x in ratios) and bool(v.all()),
                            'constructed_with_bridge':closed.valid,'base_build_ms':geom_ms,'connection_ms':connect_ms,'cache_hit':False}
                        if variant!='predicted':
                            fixed_geometry_cache[cache_key]=dict(row['regions'][variant])
                    all_records.append(row)
                    with (out/'records.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
                print('PROGRESS',tag,'epoch',weights[tag]['epoch'],'seed',seed,'samples',min(start+chunk,len(ds)),'/',len(ds),flush=True)
                write_json(out/'status.json',{'stage':'testing','model':tag,'seed':seed,'samples_done':min(start+chunk,len(ds)),'samples_total':len(ds)})
        del model,stored
        torch.cuda.empty_cache()
    summarize(out,all_records,manifest)
    write_json(out/'status.json',{'stage':'complete','report':str(out/'REPORT.md')})
    print('COMPLETE',out,flush=True)


def summarize(out,records,manifest):
    summaries={}
    def avg(values):
        vals=[v for v in values if v is not None]
        return float(np.mean(vals)) if vals else None
    for tag in ['best','latest']:
        rows=[r for r in records if r['model']==tag]
        sm={'n':len(rows),'epoch':manifest['weights'][tag]['epoch'],'route':{},'regions':{},'ellipse':{}}
        for policy in ['network','shortest']:
            sm['route'][policy]={k:avg([r['route'][policy][k] for r in rows]) for k in rows[0]['route'][policy] if k not in ['corridor_failure']}
        for variant in ['predicted','fixed_circle','fixed_tangent_ellipse']:
            sm['regions'][variant]={k:avg([r['regions'][variant][k] for r in rows]) for k in rows[0]['regions'][variant] if k!='failure_reason'}
            sm['regions'][variant]['base_unsafe_total']=sum(r['regions'][variant]['base_unsafe_count'] for r in rows)
            sm['regions'][variant]['base_total']=sum(r['regions'][variant]['base_count'] for r in rows)
            sm['regions'][variant]['failures']={reason:sum(r['regions'][variant]['failure_reason']==reason for r in rows) for reason in set(r['regions'][variant]['failure_reason'] for r in rows) if reason}
        for metric,count in [('log_axis_mae','label_count'),('angle_mae_deg','direction_count')]:
            n=sum(r['ellipse'][count] for r in rows)
            sm['ellipse'][count]=n
            sm['ellipse'][metric]=sum((r['ellipse'][metric] or 0)*r['ellipse'][count] for r in rows)/n if n else None
        summaries[tag]=sm
    write_json(out/'summary.json',summaries)
    lines=['# 修改模型：最优与最新权重的四项专项比较','',
           '两个模型均已删除 Hpath 和安全查询中的中心编码加法；中心仍沿所选通路固定采样。',
           '最优/最新指开始测试时冻结的权重：第 %d / %d 轮。'%(summaries['best']['epoch'],summaries['latest']['epoch']),
           '每个模型 %d 个任务，生成种子 %s。'%(summaries['best']['n'],manifest['seeds']),
           '相同固定初始化几何按样本、中心和形状的字节一致性复用；耗时仅对实际构建项求均值。',
           '模型选择依据验证任务分数；测试不用于选模型。耗时仅描述本次运行，不与论文空载时间比较。','',
           '## 1 通路选择','',
           '网络排名与最短通路排名都保留相同的走廊构造失败候选回退。Top1 标签匹配统计首次安全激活步骤；其他轨迹指标统计完整闭环最终输出。','',
           '| 模型 | 选择策略 | 通路长度/m | 相对最短长度 | 轨迹长度/m | RMSE/m | 平滑度/m | 碰撞率/% | 标签匹配/% |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for tag in summaries:
        for policy in ['network','shortest']:
            r=summaries[tag]['route'][policy]
            lines.append('| %s | %s | %.3f | %.4f | %.3f | %.4f | %.5f | %.3f | %.2f |'%(tag,policy,r['selected_length_m'],r['length_ratio'],r['trajectory_length_m'],r['rmse_m'],r['smoothness_m'],100*r['collision'],100*r['label_match']))
    lines+=['','## 2 椭圆初始化','',
            '三种初始化共享首次激活时网络首选通路及完全相同中心。固定尺度由验证集有效标签中位数确定：'+str(manifest['scales']),
            '标签误差只统计所选通路与标签通路一致、且标签有效的位置；角度误差使用半周期差，排除标签长短轴比<1.1的近圆形椭圆。','',
            '| 模型 | 初始化 | 平均区域面积/m² | 区域面积P05/m²（每任务P05的均值） | 桥接后走廊完成/% | 平均基础构造/ms（本次运行） |',
            '|---|---|---:|---:|---:|---:|']
    for tag in summaries:
        for variant,r in summaries[tag]['regions'].items():
            lines.append('| %s | %s | %.3f | %.3f | %.2f | %.2f |'%(tag,variant,r['base_area_mean_m2'],r['base_area_p05_m2'],100*r['constructed_with_bridge'],r['base_build_ms']))
    lines+=['','标签误差：'+json.dumps({k:v['ellipse'] for k,v in summaries.items()},ensure_ascii=False),'',
            '## 3 凸区域独立安全检查','',
            '使用多边形与占据栅格方格的分离轴检查，并计算正面积交集；检查整个区域，不只是顶点。边界接触在固定数值容差内不计正面积侵入。',
            '| 模型 | 初始化 | 检查区域数 | 侵入障碍区域数 | 平均每任务侵入面积/m² | 平均每任务越界区域数 | 桥接不安全区域/任务 |',
            '|---|---|---:|---:|---:|---:|---:|']
    for tag in summaries:
        for variant,r in summaries[tag]['regions'].items():
            lines.append('| %s | %s | %d | %d | %.8f | %.3f | %.3f |'%(tag,variant,r['base_total'],r['base_unsafe_total'],r['obstacle_intersection_area_m2'],r['base_outside_count'],r['bridge_unsafe_count']))
    lines+=['','## 4 相邻区域衔接','',
            '桥接前后共享同一组基础凸区域；交叠比例=交集面积/较小区域面积，阈值固定0.10。桥接后交叠量只汇总成功构建的走廊，失败原因保留在summary.json。','',
            '| 模型 | 初始化 | 桥接前平均交叠 | 桥接前每任务最小交叠均值 | 不足阈值的间隙/任务 | 桥接后平均交叠 | 桥接后每任务最小交叠均值 | 桥接数/任务 |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    def fmt(x): return '—' if x is None else '%.4f'%x
    for tag in summaries:
        for variant,r in summaries[tag]['regions'].items():
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s |'%(tag,variant,*[fmt(r[k]) for k in ['before_overlap_mean','before_overlap_min','before_below_threshold','after_overlap_mean','after_overlap_min','bridges']]))
    lines+=['','首次激活几何专项与实际采样冻结走廊并非始终相同：采样可切换到后备候选。所有原始记录包含最终候选、首次候选、构造失败原因与回退次数，避免混淆首选质量和系统回退效果。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    try: main()
    except Exception:
        traceback.print_exc()
        if '--out' in sys.argv:
            target=ROOT/sys.argv[sys.argv.index('--out')+1]
            if target.exists(): write_json(target/'status.json',{'stage':'failed','traceback':traceback.format_exc()})
        raise
