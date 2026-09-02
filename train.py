"""Train the zero-sum bridge diffusion planner on the scene-normalized dataset.

Sequential three-phase curriculum:
  --phase traj     : train trajectory only (L_traj)
  --phase ellipse  : load a trajectory checkpoint, freeze trajectory backbone, train
                     EllipseAggregator + EllipseHead (+ optional local-decoder micro-tune)
  --phase joint    : V5 safety fine-tuning.  Freeze low-level encoders, only train
                     trajectory/ellipse generators; loss =
                     lambda_smooth*L_smooth + L_E + gated AL safety + gated J_guide.
"""
import argparse, os, sys, time, random
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
from src.diffusion.zerosum import compute_z0, compute_base
from src.models.planner import Planner
from src.losses.total_loss import total_loss
from src.guidance.consensus_guidance import apply_consensus_guidance_unrolled, guidance_weight
from src.datasets.scene_dataset import make_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--weights-only", action="store_true",
                    help="load only model weights and reset optimizer/epoch")
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--phase", default="joint", choices=["traj", "ellipse", "joint"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    env = cfg["env"]; train_cfg = cfg["train"]; loss_cfg = cfg["loss"]
    model_cfg = cfg["model"]; diff_cfg = cfg["diffusion"]
    set_seed(int(env["seed"]))
    device = "cuda" if torch.cuda.is_available() and env.get("device", "cuda") == "cuda" else "cpu"
    phase = args.phase
    print(f"[train] device={device} phase={phase}", flush=True)

    base = "data/processed_scene"
    train_loader = make_loader(os.path.join(base, "train"), train_cfg["batch_size"], True,
                               train_cfg.get("num_workers", 0))
    val_loader = make_loader(os.path.join(base, "val"), train_cfg["batch_size"], False, 0)
    print(f"[train] train={len(train_loader.dataset)} val={len(val_loader.dataset)}", flush=True)

    schedule = NoiseSchedule(diff_cfg["timesteps"], beta_schedule=diff_cfg["beta_schedule"],
                             beta_start=diff_cfg["beta_start"], beta_end=diff_cfg["beta_end"]).to(device)
    model = Planner(model_cfg, cfg["geometry"], None).to(device)

    epochs = args.epochs if args.epochs is not None else int(train_cfg["epochs"])

    # ---- per-phase learning rates ----
    traj_lr = float(train_cfg.get("traj_lr", train_cfg["lr"]))
    ellipse_lr = float(train_cfg.get("ellipse_lr", train_cfg["lr"]))
    local_decoder_lr = float(train_cfg.get("local_decoder_lr", 0.1 * ellipse_lr))
    joint_lr = float(cfg.get("joint_finetune", {}).get("lr", 1e-5))

    # ---- build optimizer + set trainable state per phase ----
    if phase == "ellipse":
        backbone_names = ["scene_encoder", "trajectory_encoder", "local_sampler",
                          "point_scene_attention", "safety_fusion", "trajectory_decoder",
                          "residual_head", "map_pos_embed", "type_embed"]
        for name in backbone_names:
            m = getattr(model, name)
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()
        # keep the trajectory backbone deterministic; only micro-tune the local decoder
        # (the last up-conv blocks + out_conv) which contain no dropout.
        ellipse_params = []
        for p in model.ellipse_aggregator.parameters():
            p.requires_grad_(True); ellipse_params.append(p)
        for p in model.ellipse_head.parameters():
            p.requires_grad_(True); ellipse_params.append(p)
        for p in model.ellipse_pe_embed.parameters():
            p.requires_grad_(True); ellipse_params.append(p)
        local_decoder_params = []
        for up in model.scene_encoder.ups[-2:]:
            for p in up.parameters():
                p.requires_grad_(True); local_decoder_params.append(p)
        for p in model.scene_encoder.out_conv.parameters():
            p.requires_grad_(True); local_decoder_params.append(p)
        groups = [{"params": ellipse_params, "lr": ellipse_lr},
                  {"params": local_decoder_params, "lr": local_decoder_lr}]
        optimizer = torch.optim.AdamW(groups, lr=ellipse_lr,
                                      weight_decay=float(train_cfg["weight_decay"]))
        model.ellipse_enabled = True
    elif phase == "joint":
        # V5: freeze low-level encoders, only tune trajectory/ellipse generators
        # with a small uniform LR.  If joint_loss.keep_ellipse_loss is false, the
        # ellipse branch is frozen too (it is already Phase-2 quality) and only the
        # trajectory generator + AL safety are fine-tuned.
        joint_cfg = cfg.get("joint_finetune", {})
        keep_ellipse = bool(cfg.get("joint_loss", {}).get("keep_ellipse_loss", True))
        train_names = list(joint_cfg.get("train", [
            "safety_fusion", "trajectory_decoder", "residual_head",
            "ellipse_aggregator", "ellipse_head", "ellipse_pe_embed",
        ]))
        if not keep_ellipse:
            train_names = [n for n in train_names
                           if n not in ("ellipse_aggregator", "ellipse_head", "ellipse_pe_embed")]
        for p in model.parameters():
            p.requires_grad_(False)
        for name in train_names:
            m = getattr(model, name)
            for p in m.parameters():
                p.requires_grad_(True)
        if not keep_ellipse:
            # keep the ellipse branch deterministic at Phase-2 quality
            model.ellipse_aggregator.eval(); model.ellipse_head.eval(); model.ellipse_pe_embed.eval()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=joint_lr,
                                      weight_decay=float(train_cfg["weight_decay"]))
        model.ellipse_enabled = True
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        model.ellipse_enabled = (phase != "traj")
        optimizer = torch.optim.AdamW(model.parameters(), lr=traj_lr,
                                      weight_decay=float(train_cfg["weight_decay"]))

    def set_train_state():
        if phase == "ellipse":
            # trajectory backbone stays in eval (deterministic p_hat); local decoder
            # (no dropout) and ellipse module train.
            model.trajectory_encoder.eval()
            model.point_scene_attention.eval()
            model.safety_fusion.eval()
            model.trajectory_decoder.eval()
            model.residual_head.eval()
            model.scene_encoder.eval()
            for up in model.scene_encoder.ups[-2:]:
                up.train()
            model.scene_encoder.out_conv.train()
            model.ellipse_aggregator.train()
            model.ellipse_head.train()
            model.ellipse_pe_embed.train()
        elif phase == "joint":
            # V5: frozen encoders stay deterministic; only generators train.
            keep_ellipse = bool(cfg.get("joint_loss", {}).get("keep_ellipse_loss", True))
            for name in ["scene_encoder", "trajectory_encoder", "point_scene_attention"]:
                getattr(model, name).eval()
            for name in ["safety_fusion", "trajectory_decoder", "residual_head"]:
                getattr(model, name).train()
            if keep_ellipse:
                for name in ["ellipse_aggregator", "ellipse_head", "ellipse_pe_embed"]:
                    getattr(model, name).train()
            else:
                for name in ["ellipse_aggregator", "ellipse_head", "ellipse_pe_embed"]:
                    getattr(model, name).eval()
        else:
            model.train()

    # ---- ellipse cfg (weights + joint ramp) ----
    ellipse_cfg = dict(cfg.get("ellipse_loss", {}))
    joint_cfg = dict(cfg.get("joint_ellipse", {}))
    ellipse_cfg.update(joint_cfg)
    if phase == "joint" and ellipse_cfg.get("ramp", True):
        ratio = float(ellipse_cfg.get("ramp_ratio", 0.2))
        ellipse_cfg["ramp_epochs"] = max(1, int(round(epochs * ratio)))

    # ---- V5 Phase 3 safety / joint cfg ----
    al_cfg = dict(cfg.get("segment_safety", {}))
    if phase == "joint" and al_cfg.get("enabled", True):
        ratio = float(al_cfg.get("epoch_ramp_ratio", 0.2))
        al_cfg["epoch_ramp_epochs"] = max(1, int(round(epochs * ratio)))
        al_state = {"dual": float(al_cfg.get("dual_init", 0.1))}
    else:
        al_cfg = {}
        al_state = None
    joint_loss_cfg = dict(cfg.get("joint_loss", {}))
    guidance_cfg = dict(cfg.get("consensus_guidance", {}))

    def _ckpt_extra():
        if phase == "joint" and al_state is not None:
            return {"al_dual": float(al_state["dual"])}
        return None

    base_ckpt = train_cfg.get("ckpt_dir", "outputs/ckpt")
    ckpt_dir = os.path.join(base_ckpt, phase)   # 每个阶段独立目录，避免互相覆盖
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = Logger(os.path.join(ckpt_dir, "train.log"))
    best_val = float("inf"); best_ckpt = os.path.join(ckpt_dir, "best.pt")
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        # A checkpoint from another curriculum phase is a weight initialization,
        # not an optimizer resume: parameter groups (and trainable parameters)
        # differ between phases.  Checkpoint directories are phase-specific.
        resume_phase = os.path.basename(os.path.dirname(os.path.normpath(args.resume)))
        phase_switch = resume_phase in {"traj", "ellipse", "joint"} and resume_phase != phase
        weights_only = args.weights_only or phase_switch or phase == "ellipse"
        if weights_only:
            d = load_checkpoint(args.resume, model, map_location=device)
            # A phase switch starts its own schedule and epoch counter.  The
            # ellipse phase historically keeps the source epoch for logging,
            # but its optimizer is still intentionally reset.
            start_epoch = 0 if (args.weights_only or phase_switch) else d.get("epoch", 0) + 1
            logger.info(f"loaded model weights from {args.resume}; optimizer reset, "
                        f"starting epoch {start_epoch}")
        else:
            d = load_checkpoint(args.resume, model, optimizer, map_location=device)
            start_epoch = d.get("epoch", 0) + 1
            logger.info(f"resumed from {args.resume} at epoch {start_epoch}")
        # Restore the AL dual variable so safety penalty continuity survives resume.
        if phase == "joint" and al_state is not None and "al_dual" in d:
            al_state["dual"] = float(d["al_dual"])
    n_b = len(train_loader)
    T = schedule.num_timesteps
    for epoch in range(start_epoch, epochs):
        t0 = time.time(); losses = []
        set_train_state()
        for bi, batch in enumerate(train_loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            B = batch["pos"].shape[0]
            t = torch.randint(0, T, (B,), device=device)
            z0_gt, _, _, _ = compute_z0(batch["pos"], batch["cond"])
            z_t = schedule.q_sample_zero_sum(z0_gt, t)
            out = model(z_t, t, batch["map_tensor"], batch["cond"])

            # V5: J_guide is NOT a loss.  During training it is applied as a
            # gradient correction on z0 (unrolled), and the losses are then
            # computed on the guided trajectory.
            out_eff = out
            # Training-side unrolled guidance: only on a random fraction of steps
            # (guidance_cfg.training_step_frac) to cut double-backward cost.
            use_guidance_step = random.random() < float(guidance_cfg.get("training_step_frac", 0.5))                 if guidance_cfg else False
            if phase == "joint" and guidance_cfg and guidance_cfg.get("enabled", True) \
                    and out.get("ellipse_center") is not None and use_guidance_step:
                tt = t
                w_G = guidance_weight(tt, T,
                                      guidance_cfg.get("start_t_ratio", 0.40),
                                      guidance_cfg.get("full_t_ratio", 0.10))
                if float(w_G.max().item()) > 0.0:
                    start = batch["cond"][:, 0]
                    goal = batch["cond"][:, 1]
                    N = batch["pos"].shape[1] - 1
                    base = compute_base(goal - start, N)
                    z0g, posg, _gs = apply_consensus_guidance_unrolled(
                        out["z0_pred"], base, start,
                        out["ellipse_center"], out["ellipse_radii"],
                        out["ellipse_theta"], guidance_cfg, tt, T)
                    out_eff = dict(out)
                    out_eff["z0_pred"] = z0g
                    out_eff["pos_pred"] = posg

            loss = total_loss(out_eff, batch, loss_cfg, device=device, phase=phase, epoch=epoch,
                              ellipse_cfg=ellipse_cfg, diffusion_t=t, num_timesteps=T,
                              al_state=al_state, al_cfg=al_cfg,
                              joint_loss_cfg=joint_loss_cfg)
            if not torch.isfinite(loss["total"]):
                logger.info(f"epoch {epoch+1}/{epochs} batch {bi+1}/{n_b} non-finite total={loss['total'].item()} -- skip")
                continue
            optimizer.zero_grad(); loss["total"].backward()
            if train_cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
            optimizer.step()
            if phase == "joint" and al_state is not None:
                interval = int(al_cfg.get("dual_update_interval", 10))
                if (bi + 1) % interval == 0 and loss.get("al_w_active", 0.0) > 0.0:
                    mv = loss.get("al_mean_V")
                    if mv is not None:
                        dual = float(al_state["dual"])
                        al_state["dual"] = float(min(
                            float(al_cfg.get("dual_max", 10.0)),
                            max(0.0, dual + float(al_cfg.get("dual_lr", 0.1)) * float(mv.item()))))
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
                loss = total_loss(out, batch, loss_cfg, device=device, phase=phase, epoch=epoch,
                                  ellipse_cfg=ellipse_cfg, diffusion_t=t, num_timesteps=T,
                                  al_state=None, al_cfg=al_cfg,
                                  joint_loss_cfg=joint_loss_cfg)
                if not torch.isfinite(loss["total"]):
                    continue
                vls.append({k: v.item() for k, v in loss.items() if torch.is_tensor(v)})
        vmean = {k: float(np.mean([l[k] for l in vls])) for k in vls[0]} if vls else {}
        logger.info(f"val   loss={vmean}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if vmean.get("total", float("inf")) < best_val:
            best_val = vmean["total"]
            save_checkpoint(best_ckpt, model, optimizer, epoch=epoch + 1,
                            extra=_ckpt_extra())
            logger.info(f"save best ckpt {best_ckpt} (val total {best_val:.4f})")
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt"), model,
                            optimizer, epoch=epoch + 1, extra=_ckpt_extra())
            logger.info(f"save ckpt epoch_{epoch+1}.pt")
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
