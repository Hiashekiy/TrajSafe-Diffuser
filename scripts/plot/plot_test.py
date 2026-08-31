"""Sample with the mixed-scene model on a given maze and plot."""
import argparse, os, sys, numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.datasets.data_io import load_maze
from src.datasets.mixed_dataset import EXTENT
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import sample
from src.models.planner import Planner
from src.utils.checkpoint import load_checkpoint
from src.geometry.sdf_utils import sample_sdf_torch
from src.utils.metrics import interp_trajectory
from src.utils.visualization import draw_traj, set_map_limits

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--maze", default="umaze")
    ap.add_argument("--ckpt", default="outputs/ckpt/best.pt")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed; omit for a different random case set / sampling noise each run")
    args=ap.parse_args()
    cfg=load_config(args.config)
    if args.seed is not None:
        set_seed(args.seed)
    case_rng=np.random.default_rng(args.seed)
    device="cuda" if torch.cuda.is_available() else "cpu"
    norm,occ,sdf,cond=load_maze(args.maze, split="test", n=args.n, rng=case_rng)
    cond_t=torch.as_tensor(cond,dtype=torch.float32).to(device)
    # build planner with mixed extent/norm
    geom=dict(cfg["geometry"]); geom["extent"]=list(EXTENT)
    model=Planner(cfg["model"], geom, norm).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    schedule=NoiseSchedule(cfg["diffusion"]["timesteps"],beta_schedule=cfg["diffusion"]["beta_schedule"],beta_start=cfg["diffusion"]["beta_start"],beta_end=cfg["diffusion"]["beta_end"]).to(device)
    map_t=torch.as_tensor(occ,dtype=torch.float32).to(device)[None,None]
    x0,_=sample(model,map_t,schedule,cond_t,args.n,device=device,steps=args.steps)
    with torch.no_grad():
        t0=torch.zeros(x0.shape[0],device=device,dtype=torch.long)
        pred=model(x0,t0,map_t,cond=cond_t)
    mins=np.asarray(norm.mins[2:4]); maxs=np.asarray(norm.maxs[2:4]); eps=norm.eps
    def unnorm(p): return (np.asarray(p,dtype=float)+1.0)/2.0*(maxs-mins+eps)+mins
    pos=unnorm(x0.cpu().numpy()[:,:,2:4])
    cw=unnorm(pred["ellipse_center"].cpu().numpy())
    r1=pred["ellipse_radii"][...,0].cpu().numpy(); r2=pred["ellipse_radii"][...,1].cpu().numpy(); th=pred["ellipse_theta"].cpu().numpy()
    # collision on dense path via shared sdf
    sdf_t=torch.as_tensor(sdf,dtype=torch.float32).to(device)[None,None]
    coll=[]
    for i in range(args.n):
        dense=interp_trajectory(pos[i],interp_steps=8)
        d=sample_sdf_torch(sdf_t.expand(1,-1,-1,-1), torch.as_tensor(dense,dtype=torch.float32).to(device)[None], EXTENT).cpu().numpy()[0]
        coll.append(float(np.mean(d<=0.0)))
    ncol = 2
    nrow = max(1, (args.n + 1) // 2)
    fig,axes=plt.subplots(nrow,ncol,figsize=(6*ncol,6*nrow))
    axes=np.array(axes).reshape(-1)
    for ax,i in zip(axes.ravel(),range(args.n)):
        ax.imshow(occ, origin="lower", extent=(EXTENT[0],EXTENT[1],EXTENT[2],EXTENT[3]), cmap="gray_r", alpha=0.9)
        vel = x0.cpu().numpy()[i, :, 4:6]   # normalized velocity, used as direction arrows
        draw_traj(ax, pos[i], velocities=vel, marker_every=0, arrow_every=0)
        for j in range(0,len(pos[i]),8):
            if not np.isfinite(r1[i][j]+r2[i][j]) or r1[i][j]<=0: continue
            e=Ellipse((cw[i][j][0],cw[i][j][1]),2*r1[i][j],2*r2[i][j],angle=np.degrees(th[i][j]),fill=False,edgecolor="tab:red",lw=1.0,alpha=0.7)
            ax.add_patch(e)
        ax.set_title(f"{args.maze} #{i} coll={coll[i]:.3f}", fontsize=10)
        ax.set_aspect("equal"); set_map_limits(ax, EXTENT); ax.legend(fontsize=6)
    for k in range(args.n, len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"mixed-scene model on {args.maze} ([0,8]^2)", fontsize=13)
    fig.tight_layout()
    os.makedirs("outputs",exist_ok=True)
    out=f"outputs/test_{args.maze}.png"
    fig.savefig(out,dpi=120,bbox_inches="tight")
    print("saved",out,"coll",coll,"seed",args.seed)

if __name__=="__main__":
    main()
