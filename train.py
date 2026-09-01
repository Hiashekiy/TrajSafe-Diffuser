"""Train the zero-sum bridge diffusion planner on the scene-normalized dataset."""
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
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.zerosum import compute_z0
from src.models.planner import Planner
from src.losses.total_loss import total_loss
from src.losses.gradnorm import GradNorm
from src.losses.auto_weighting import Lagrangian
from src.datasets.scene_dataset import make_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--log-interval", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config(args.config)
    env = cfg["env"]; train_cfg = cfg["train"]; loss_cfg = cfg["loss"]
    model_cfg = cfg["model"]; diff_cfg = cfg["diffusion"]
    set_seed(int(env["seed"]))
    device = "cuda" if torch.cuda.is_available() and env.get("device", "cuda") == "cuda" else "cpu"
    print(f"[train] device={device}", flush=True)

    base = "data/processed_scene"
    train_loader = make_loader(os.path.join(base, "train"), train_cfg["batch_size"], True,
                               train_cfg.get("num_workers", 0))
    val_loader = make_loader(os.path.join(base, "val"), train_cfg["batch_size"], False, 0)
    print(f"[train] train={len(train_loader.dataset)} val={len(val_loader.dataset)}", flush=True)

    schedule = NoiseSchedule(diff_cfg["timesteps"], beta_schedule=diff_cfg["beta_schedule"],
                             beta_start=diff_cfg["beta_start"], beta_end=diff_cfg["beta_end"]).to(device)
    model = Planner(model_cfg, cfg["geometry"], None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]),
                                  weight_decay=float(train_cfg["weight_decay"]))
    gradnorm = GradNorm(["L_p", "L_param", "L_iou", "L_smooth"])
    shared_params = list(model.safety_fusion.mlp.parameters())
    lag_col = Lagrangian(delta=loss_cfg.get("collision_delta", 0.0),
                         eta=loss_cfg.get("collision_eta", 0.05), init=loss_cfg.get("collision_lag_init", 0.1),
                         max_lam=loss_cfg.get("lambda_collision_max", 0.3))
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
        if "gradnorm" in d:
            gradnorm.load_state_dict(d["gradnorm"])
        if "lag_col" in d:
            lag_col.load_state_dict(d["lag_col"])
        logger.info(f"resumed from {args.resume} at epoch {start_epoch}")
    n_b = len(train_loader)
    T = schedule.num_timesteps
    H = model.horizon
    for epoch in range(start_epoch, epochs):
        t0 = time.time(); losses = []
        model.train()
        for bi, batch in enumerate(train_loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            B = batch["pos"].shape[0]
            t = torch.randint(0, T, (B,), device=device)
            z0_gt, _, _, _ = compute_z0(batch["pos"], batch["cond"])
            z_t = schedule.q_sample_zero_sum(z0_gt, t)
            out = model(z_t, t, batch["map_tensor"], batch["cond"])
            loss = total_loss(out, batch, loss_cfg, device=device,
                              warmup=(epoch < warmup_epochs), epoch=epoch,
                              gradnorm=gradnorm, shared_params=shared_params, lag_col=lag_col)
            if not torch.isfinite(loss["total"]):
                logger.info(f"epoch {epoch+1}/{epochs} batch {bi+1}/{n_b} non-finite total={loss['total'].item()} -- skip")
                continue
            optimizer.zero_grad(); loss["total"].backward()
            if train_cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
            optimizer.step()
            losses.append({k: v.item() for k, v in loss.items() if torch.is_tensor(v) and v.requires_grad})
            if (bi + 1) % args.log_interval == 0 or bi + 1 == n_b:
                logger.info(f"epoch {epoch+1}/{epochs} batch {bi+1}/{n_b} total={loss['total'].item():.3f} t={time.time()-t0:.1f}s")
        mean = {k: float(np.mean([l[k] for l in losses])) for k in losses[0]} if losses else {}
        logger.info(f"epoch {epoch+1}/{epochs} loss={mean} time={time.time()-t0:.1f}s")
        model.eval(); vls = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                t = torch.full((batch["pos"].shape[0],), int(T // 2), device=device)
                z0_gt, _, _, _ = compute_z0(batch["pos"], batch["cond"])
                z_t = schedule.q_sample_zero_sum(z0_gt, t)
                out = model(z_t, t, batch["map_tensor"], batch["cond"])
                loss = total_loss(out, batch, loss_cfg, device=device,
                                  warmup=(epoch < warmup_epochs), epoch=epoch)
                if not torch.isfinite(loss["total"]):
                    continue
                vls.append({k: v.item() for k, v in loss.items() if torch.is_tensor(v)})
        vmean = {k: float(np.mean([l[k] for l in vls])) for k in vls[0]} if vls else {}
        logger.info(f"val   loss={vmean}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if epoch >= warmup_epochs and vmean.get("total", float("inf")) < best_val:
            best_val = vmean["total"]; save_checkpoint(best_ckpt, model, optimizer, epoch=epoch + 1,
                                                       extra={"gradnorm": gradnorm.state_dict(), "lag_col": lag_col.state_dict()})
            logger.info(f"save best ckpt {best_ckpt} (val total {best_val:.4f})")
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt"), model, optimizer, epoch=epoch + 1,
                            extra={"gradnorm": gradnorm.state_dict(), "lag_col": lag_col.state_dict()})
            logger.info(f"save ckpt epoch_{epoch+1}.pt")
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
