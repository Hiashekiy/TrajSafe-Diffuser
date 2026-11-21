"""Render real V5 search / geometry / correction traces (no synthetic geometry)."""
from pathlib import Path
import sys, csv, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.datasets.carla_spline_dataset import make_loader, make_collate
from src.diffusion.sampler_v5 import sample_v5, _build_safe_grids
from src.diffusion.alm_local import collision_mask
from src.diffusion.schedule import NoiseSchedule
from src.models.factory import build_model
from src.utils.config import load_config, num_controls, num_safety_queries
from src.grid_search.field_search import guidance_class

OUT=ROOT/'outputs/field_search_visualization'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
cfg=load_config(str(ROOT/'configs/config_v5_c32q128.yaml'))
_,ds=make_loader('test',cfg['data']['processed_root'],batch_size=1,shuffle=False,num_workers=0,require_labels=False,num_controls=num_controls(cfg),num_safety_queries=num_safety_queries(cfg))
collate=make_collate(ds)
device='cuda' if torch.cuda.is_available() else 'cpu'
model=build_model(cfg).to(device)
ck=torch.load(ROOT/'outputs/models/v5_c32q128/ckpt/final_best.pt',map_location='cpu',weights_only=False)
model.load_state_dict(ck['model'],strict=True)
sched=NoiseSchedule(cfg['diffusion']['timesteps'],beta_schedule=cfg['diffusion'].get('beta_schedule','squaredcos_cap_v2')).to(device)
rows=list(csv.DictReader(open(ROOT/'outputs/field_search_test420/eval_test.csv',encoding='utf-8')))
# Select actual corrections, with substantial pre-correction collisions.
rows=sorted(enumerate(rows),key=lambda ir:max([float(s['coll_frac']) for s in json.loads(ir[1]['step_trace']) if s['alm_ok']] or [-1]),reverse=True)
colors={'guide':'#D55E00','search':'#0072B2','raw':'#D55E00','safe':'#009E73','final':'#CC79A7'}
manifest=[]
for idx,row in rows:
    batch=collate([ds[idx]])
    result=sample_v5(model,sched,batch,cfg,device=device,seed=idx)
    candidates=[s for s in result['trace'] if (s.get('alm') or {}).get('alm_ok') and s.get('safe_curve') is not None and (s.get('alm') or {}).get('cells')]
    if not candidates: continue
    st=max(candidates,key=lambda s:s['collision'][0]['frac'])
    occ=batch['occupancy'][0,0].numpy()
    grids,spec,_=_build_safe_grids(batch['occupancy'],cfg)
    safe=grids[0][0]
    cond=batch['cond'][0].numpy()
    guide=result['trace'][0]['raw_curve'][0]
    raw=st['raw_curve'][0]; corrected=st['safe_curve'][0]
    final=result['final'][0].detach().cpu().numpy()
    gf=guidance_class()((~safe.free).astype(np.uint8),lam=.01,weight='quadratic',distance='manhattan',source_method='drop',num_samples=64)
    gf.build(spec.world_to_cell(guide)[:,::-1])
    sr=gf.search(tuple(np.rint(spec.world_to_cell(cond[0])[::-1]).astype(int)),tuple(np.rint(spec.world_to_cell(cond[1])[::-1]).astype(int)),alpha=.15,gamma=1.,adaptive=False,clearance_beta=.10,clearance_mode='quadratic_hinge',clearance_threshold=3.)
    search=spec.cell_to_world(np.asarray(sr['path'])[:,::-1])
    ell=st['ellipse']; cells=st['alm']['cells'][0]
    sid=int(batch['sample_id'][0]); ncell=len(cells)
    xy=np.concatenate([guide,search,raw,corrected,final,cond])
    pix=spec.world_to_cell(xy)
    lo=np.maximum(pix.min(0)-18,[-.5,-.5]); hi=np.minimum(pix.max(0)+18,[spec.width-.5,spec.height-.5])
    center=(lo+hi)/2; span=max(hi-lo); lo=center-span/2; hi=center+span/2
    fig,axs=plt.subplots(2,4,figsize=(19,10),layout='constrained')
    axs=axs.ravel()
    titles=['① 原始地图与首轮引导路径','② 智能体足迹膨胀后的搜索地图','③ 归一化引导场与有效源点','④ 净空距离场与搜索路径','⑤ 网络预测的安全椭圆','⑥ 实际安全走廊与 ALM 修正','⑦ 未修正 / 修正后 / 最终输出','⑧ 四步去噪的原始预测']
    def line(ax,p,color,ls='-',lw=2):
        q=spec.world_to_cell(p); ax.plot(q[:,0],q[:,1],color=color,ls=ls,lw=lw,zorder=5)
    def base(ax,title):
        ax.imshow(occ,origin='lower',cmap=ListedColormap(['#FAFAFA','#343C48']),vmin=0,vmax=1,interpolation='nearest')
        ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1]); ax.set_aspect('equal')
        ax.set_title(title,loc='left',fontsize=11,pad=9)
        ax.set_xticks([]); ax.set_yticks([])
        q=spec.world_to_cell(cond)
        ax.scatter(q[0,0],q[0,1],marker='o',s=42,c='#009E73',edgecolors='white',zorder=10)
        ax.scatter(q[1,0],q[1,1],marker='*',s=100,c='#CC79A7',edgecolors='white',zorder=10)
        for spine in ax.spines.values(): spine.set_color('#D4D9DF')
    for ax,title in zip(axs,titles): base(ax,title)
    line(axs[0],guide,colors['guide'],'--')
    added=(occ==0)&~safe.free
    axs[1].imshow(np.ma.masked_where(~added,added),origin='lower',cmap=ListedColormap(['#E69F00']),alpha=.65,interpolation='nearest')
    line(axs[1],guide,colors['guide'],'--',1.5)
    fld=gf.normalised()['F_hat']
    im=axs[2].imshow(np.ma.masked_where(~safe.free,fld),origin='lower',cmap='viridis',vmin=0,vmax=1,alpha=.92)
    src=gf.sources['positions_rc']; axs[2].scatter(src[:,1],src[:,0],s=10,c='white',edgecolors='#333333',linewidths=.3,zorder=6)
    line(axs[2],guide,'#E69F00','--',1)
    fig.colorbar(im,ax=axs[2],fraction=.045,pad=.015,label='归一化引导值')
    distance=gf.clearance().distance()
    im=axs[3].imshow(np.ma.masked_where(~safe.free,distance),origin='lower',cmap='cividis',vmin=0,vmax=12)
    line(axs[3],search,colors['search'],lw=2.2)
    fig.colorbar(im,ax=axs[3],fraction=.045,pad=.015,label='净空（格，≥12截断）')
    tt=np.linspace(0,2*np.pi,65)
    for j in range(len(ell['center'][0])):
        a,b,theta=[ell[k][0,j] for k in ['a','b','theta']]
        c=ell['center'][0,j]; u=a*np.cos(tt); v=b*np.sin(tt)
        p=c+np.stack([np.cos(theta)*u-np.sin(theta)*v,np.sin(theta)*u+np.cos(theta)*v],-1)
        line(axs[4],p,'#56B4E9',lw=.65)
    line(axs[4],raw,colors['raw'],'--',1.6)
    for poly in cells:
        p=spec.world_to_cell(np.asarray(poly)); axs[5].fill(p[:,0],p[:,1],color='#56B4E9',alpha=.13,edgecolor='#0072B2',lw=.5)
    line(axs[5],raw,colors['raw'],'--',1.5); line(axs[5],corrected,colors['safe'],lw=2)
    line(axs[6],raw,colors['raw'],'--',2.2); line(axs[6],corrected,colors['safe'],lw=2.4); line(axs[6],final,colors['final'],':',2)
    pal=['#999999','#E69F00','#0072B2','#CC79A7']
    for step,col in zip(result['trace'],pal): line(axs[7],step['raw_curve'][0],col,'--' if step['step']<4 else '-',1.6)
    axs[7].legend([Line2D([],[],color=c,lw=2) for c in pal],[f'第{k}步原始预测' for k in range(1,5)],fontsize=8,loc='lower left',framealpha=.85)
    hit,_=collision_mask(batch['occupancy'][0],raw)
    p=spec.world_to_cell(raw[hit]); axs[6].scatter(p[:,0],p[:,1],marker='x',s=22,c='#D55E00',zorder=9)
    fig.suptitle(f'真实样本 {sid}  |  场引导搜索 → 椭圆 → 安全走廊 → 闭环路径生成\n展示第 {st["step"]} 步修正：原始碰撞点 {int(hit.sum())}/{len(hit)}，实际走廊 {ncell} 个凸单元，ALM 验证通过',fontsize=15)
    handles=[Line2D([],[],color=colors['guide'],ls='--',lw=2),Line2D([],[],color=colors['search'],lw=2),Line2D([],[],color=colors['safe'],lw=2),Line2D([],[],color=colors['final'],ls=':',lw=2)]
    fig.legend(handles,['引导 / 未修正预测','场引导搜索路径','该步 ALM 修正结果','四步最终输出'],loc='outside lower center',ncol=4,fontsize=11)
    name=f'pipeline_{sid}'
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas_box = fig.bbox
    for text in [*fig.texts, *[ax.title for ax in axs]]:
        box = text.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= canvas_box.x1+1 and box.y1 <= canvas_box.y1+1, text.get_text()
    fig.savefig(OUT/f'{name}.png',dpi=180)
    # Keep vector exports for editing and paper assembly.
    fig.savefig(OUT/f'{name}.pdf')
    fig.savefig(OUT/f'{name}.svg')
    plt.close(fig)
    np.savez_compressed(OUT/f'{name}_data.npz',occupancy=occ,inflated_occupancy=~safe.free,guidance=guide,guidance_field=gf.field,clearance=distance,search_path=search,raw_curve=raw,corrected_curve=corrected,final_curve=final,ellipse_center=ell['center'][0],ellipse_a=ell['a'][0],ellipse_b=ell['b'][0],ellipse_theta=ell['theta'][0])
    manifest.append(dict(sample=sid,index=idx,seed=idx,step=st['step'],raw_collision_points=int(hit.sum()),corridor_cells=ncell,alm_ok=True,final_source=result['final_source'][0],note='椭圆与走廊来自该步实际推理；引导场按同图同参数复建；距离单位为格。'))
    print(json.dumps(manifest[-1],ensure_ascii=False),flush=True)
    if len(manifest)==3: break
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(OUT)
