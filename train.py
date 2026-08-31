"""Sample-level mixed training on the mixed [0,8]^2 dataset."""
import argparse, os, sys, time
import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import Logger
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.datasets.normalization import load_normalization
from src.datasets.mixed_dataset import make_loader, EXTENT
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.conditioning import apply_endpoint_condition
from src.models.planner import Planner
from src.losses.total_loss import total_loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--log-interval", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config(args.config)
    env = cfg["env"]; train_cfg = cfg["train"]; loss_cfg = cfg["loss"]; model_cfg = cfg["model"]; diff_cfg = cfg["diffusion"]
    set_seed(int(env["seed"]))
    device = "cuda" if torch.cuda.is_available() and env.get("device", "cuda") == "cuda" else "cpu"
    print(f"[train] device={device}", flush=True)
    base = "data/processed/mixed"
    norm, _ = load_normalization(os.path.join(base, "normalization.json"))
    train_loader = make_loader(os.path.join(base, "train"), train_cfg["batch_size"], True, train_cfg.get("num_workers",0))
    val_loader = make_loader(os.path.join(base, "val"), train_cfg["batch_size"], False, 0)
    print(f"[train] train={len(train_loader.dataset)} val={len(val_loader.dataset)}", flush=True)
    schedule = NoiseSchedule(diff_cfg["timesteps"], beta_schedule=diff_cfg["beta_schedule"], beta_start=diff_cfg["beta_start"], beta_end=diff_cfg["beta_end"]).to(device)
    model = Planner(model_cfg, cfg["geometry"], norm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    epochs = args.epochs if args.epochs is not None else int(train_cfg["epochs"])
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    ckpt_dir = train_cfg.get("ckpt_dir", "outputs/ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = Logger(os.path.join(ckpt_dir, "train.log"))
    best_val = float("inf"); best_ckpt = os.path.join(ckpt_dir, "best.pt")
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        d = load_checkpoint(args.resume, model, optimizer, map_location=device)
        start_epoch = d.get("epoch", 0) + 1
        logger.info(f"resumed from {args.resume} at epoch {start_epoch}")
    n_b = len(train_loader)
    for epoch in range(start_epoch, epochs):
        t0 = time.time(); losses=[]
        model.train()
        for bi, batch in enumerate(train_loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            t = torch.randint(0, schedule.num_timesteps, (batch["traj"].shape[0],), device=device)
            H = batch["traj"].shape[1]
            x_t = schedule.q_sample(batch["traj"].float().double(), t).float()
            x_t = apply_endpoint_condition(x_t, batch["cond"], H, obs_start=2, obs_end=4)
            out = model(x_t, t, batch["map_tensor"], cond=batch["cond"], extent=EXTENT, state_norm=norm)
            out["x0_pred"] = apply_endpoint_condition(out["x0_pred"], batch["cond"], H, obs_start=2, obs_end=4)
            loss = total_loss(out, batch, batch["map_tensor"], batch["sdf_tensor"], EXTENT, norm, loss_cfg, cfg["geometry"], device=device, warmup=(epoch < warmup_epochs))
            optimizer.zero_grad(); loss["total"].backward()
            if train_cfg.get("grad_clip"): torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
            optimizer.step()
            losses.append({k: v.item() for k, v in loss.items() if torch.is_tensor(v) and v.requires_grad})
            if (bi+1) % args.log_interval == 0 or bi+1 == n_b:
                logger.info(f"epoch {epoch+1}/{epochs} batch {bi+1}/{n_b} total={loss['total'].item():.3f} t={time.time()-t0:.1f}s")
        mean = {k: float(np.mean([l[k] for l in losses])) for k in losses[0]} if losses else {}
        logger.info(f"epoch {epoch+1}/{epochs} loss={mean} time={time.time()-t0:.1f}s")
        model.eval(); vls=[]
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                # eval at a fixed mid-noise timestep so validation is deterministic
                # (random t per batch makes val total fluctuate wildly).
                t = torch.full((batch["traj"].shape[0],), int(schedule.num_timesteps // 2), device=device)
                H = batch["traj"].shape[1]
                x_t = schedule.q_sample(batch["traj"].float().double(), t).float()
                x_t = apply_endpoint_condition(x_t, batch["cond"], H, obs_start=2, obs_end=4)
                out = model(x_t, t, batch["map_tensor"], cond=batch["cond"], extent=EXTENT, state_norm=norm)
                out["x0_pred"] = apply_endpoint_condition(out["x0_pred"], batch["cond"], H, obs_start=2, obs_end=4)
                loss = total_loss(out, batch, batch["map_tensor"], batch["sdf_tensor"], EXTENT, norm, loss_cfg, cfg["geometry"], device=device, warmup=(epoch < warmup_epochs))
                vls.append({k: v.item() for k, v in loss.items() if torch.is_tensor(v)})
        vmean = {k: float(np.mean([l[k] for l in vls])) for k in vls[0]} if vls else {}
        logger.info(f"val   loss={vmean}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if epoch >= warmup_epochs and vmean.get("total", float("inf")) < best_val:
            best_val = vmean["total"]; save_checkpoint(best_ckpt, model, optimizer, epoch=epoch+1)
            logger.info(f"save best ckpt {best_ckpt} (val total {best_val:.4f})")
        if (epoch+1) % 10 == 0 or epoch == epochs-1:
            save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt"), model, optimizer, epoch=epoch+1)
            logger.info(f"save ckpt epoch_{epoch+1}.pt")
        if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
