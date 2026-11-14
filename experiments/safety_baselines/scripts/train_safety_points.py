"""Train the native-point safety model (S-C48's recipe on a 128-point state).

Mirrors `scripts/train_map_only_points.py`: T=256, 100 epochs, AdamW + cosine, batch 16,
bf16 autocast, gradient clipping, and the same checkpoint trio (best / best_task / latest).
`best_task` uses the SAME selector the earlier runs used (`val_rmse_m + 80 * vertex
collision rate`) so the numbers stay comparable, and it also screens on the *continuous*
curve collision (the selector that Phase A showed to be better aligned with sampling).

Smoke:  --limit 64 --epochs 1 --batch-size 8
"""
import argparse, json, math, os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.diffusion.schedule import NoiseSchedule
from experiments.safety_baselines.models.safety_points import SafetyPointDiffuser, pin_endpoints
from experiments.safety_baselines.models.safety_point_data import make_safety_point_loader
from experiments.safety_baselines.models.safety_point_losses import compose_losses


def cosine_warmup(step, total, warmup=200):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def evaluate(model, schedule, loader, device, chunk=32, eval_batches=None):
    """Denoiser-side val metrics: x0 loss, waypoint RMSE (m), vertex-collision rate."""
    model.eval()
    tot = {'loss': 0.0, 'rmse': 0.0, 'hit': 0.0, 'n': 0}
    for bi, batch in enumerate(loader):
        if eval_batches is not None and bi >= eval_batches:
            break
        b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        t = torch.randint(0, schedule.num_timesteps, (b['curve_gt'].shape[0],), device=device)
        a = schedule.sqrt_alphas_cumprod[t].float()[:, None, None]
        s = schedule.sqrt_one_minus_alphas_cumprod[t].float()[:, None, None]
        x_t = a * b['curve_gt'] + s * torch.randn_like(b['curve_gt'])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x_t, b['occupancy'], b['cond'], t,
                        schedule.sqrt_alphas_cumprod[t].float(), b['candidate_xy'],
                        b['candidate_mask'], b['geometry'], b['geometry_lengths'],
                        select_index=b['topology_best'])
        pred = out['points'].float()
        loss = torch.nn.functional.mse_loss(pred, b['curve_gt'])
        rmse = torch.sqrt(((pred - b['curve_gt']) ** 2).sum(-1).mean(-1)) * 80.0
        R = b['occupancy'].shape[-1]
        ix = torch.floor((pred[..., 0] + 1) * R / 2).long().clamp(0, R - 1)
        iy = torch.floor((pred[..., 1] + 1) * R / 2).long().clamp(0, R - 1)
        occ = b['occupancy'][:, 0]
        hit = (occ[torch.arange(len(pred), device=device)[:, None], iy, ix] > .5)
        nb = len(pred)
        tot['loss'] += float(loss) * nb; tot['rmse'] += float(rmse.sum())
        tot['hit'] += float(hit.float().mean(dim=1).sum()); tot['n'] += nb
    model.train()
    n = max(1, tot['n'])
    hit_rate = tot['hit'] / n
    rmse = tot['rmse'] / n
    return {'x0_loss': tot['loss'] / n, 'x0_rmse_m': rmse, 'vertex_collision_rate': hit_rate,
            'task_score': rmse + 80.0 * hit_rate, 'n': tot['n']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='experiments/safety_baselines/configs/safety_points_t256.json')
    ap.add_argument('--out', default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--val-batches', type=int, default=None)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    cfg = json.loads(open(os.path.join(ROOT, args.config), encoding='utf8').read())
    if args.epochs: cfg['train']['epochs'] = args.epochs
    if args.batch_size: cfg['train']['batch_size'] = args.batch_size
    if args.val_batches: cfg['train']['val_batches'] = args.val_batches
    if args.smoke: cfg['train'].update(epochs=1, batch_size=8, val_batches=2)
    run = args.out or cfg['train']['ckpt_dir'].replace('/ckpt', '')
    run = os.path.join(ROOT, run) if not os.path.isabs(run) else run
    ck = os.path.join(run, 'ckpt'); os.makedirs(ck, exist_ok=True)
    json.dump(cfg, open(os.path.join(run, 'config_snapshot.json'), 'w', encoding='utf8'), indent=2)

    device = 'cuda'
    torch.manual_seed(cfg['train'].get('seed', 0)); np.random.seed(cfg['train'].get('seed', 0))
    torch.backends.cuda.matmul.allow_tf32 = True
    d = cfg['diffusion']; schedule = NoiseSchedule(d['timesteps'], d['beta_schedule'],
                                                   d.get('beta_start', 1e-4), d.get('beta_end', .02)).to(device)
    model = SafetyPointDiffuser(cfg['model'], cfg.get('ellipse'),
                                num_points=cfg['data']['num_points']).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print('model params %.4f M' % (n_par / 1e6), flush=True)

    loaders = {}
    for split in ('train', 'val'):
        loaders[split], _ = make_safety_point_loader(split, cfg['data']['processed_root'],
                                                  cfg['train']['batch_size'], split == 'train',
                                                  num_workers=cfg['train'].get('num_workers', 0),
                                                  limit=args.limit,
                                                  geometry_points=cfg['data'].get('geometry_points', 512))
    opt = AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train'].get('weight_decay', 0.01))
    steps_per_epoch = len(loaders['train'])
    total = max(1, cfg['train']['epochs'] * steps_per_epoch)
    sched = LambdaLR(opt, lambda st: cosine_warmup(st, total, cfg['train'].get('warmup_steps', 200)))
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    hist = []; best_val = float('inf'); best_task = float('inf'); best_epoch = -1; task_epoch = -1
    step = 0
    for ep in range(cfg['train']['epochs']):
        model.train(); t0 = time.perf_counter(); acc = {}; nb = 0
        for batch in loaders['train']:
            b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            t = torch.randint(0, schedule.num_timesteps, (b['curve_gt'].shape[0],), device=device)
            a = schedule.sqrt_alphas_cumprod[t].float()[:, None, None]
            s = schedule.sqrt_one_minus_alphas_cumprod[t].float()[:, None, None]
            x_t = a * b['curve_gt'] + s * torch.randn_like(b['curve_gt'])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(x_t, b['occupancy'], b['cond'], t,
                            schedule.sqrt_alphas_cumprod[t].float(), b['candidate_xy'],
                            b['candidate_mask'], b['geometry'], b['geometry_lengths'],
                            select_index=b['topology_best'])
                loss, parts = compose_losses(cfg['loss'], out, b, b['cond'], b['occupancy'])
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite loss at epoch %d' % ep)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train'].get('grad_clip', 1.0))
            opt.step(); sched.step(); step += 1
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + v
            acc['total'] = acc.get('total', 0.0) + float(loss.detach())
            nb += 1
        train_stats = {k: v / max(1, nb) for k, v in acc.items()}
        val = evaluate(model, schedule, loaders['val'], device, eval_batches=cfg['train'].get('val_batches', 24))
        rec = {'epoch': ep, 'seconds': time.perf_counter() - t0, 'lr': sched.get_last_lr()[0],
               **{'train_' + k: v for k, v in train_stats.items()}, **val}
        hist.append(rec)
        torch.save({'model_state': model.state_dict(), 'epoch': ep, 'config': cfg},
                   os.path.join(ck, 'latest.pt'))
        if val['x0_loss'] < best_val:
            best_val = val['x0_loss']
            torch.save({'model_state': model.state_dict(), 'epoch': ep, 'config': cfg},
                       os.path.join(ck, 'best.pt'))
        if val['task_score'] < best_task:
            best_task = val['task_score']; task_epoch = ep
            torch.save({'model_state': model.state_dict(), 'epoch': ep, 'config': cfg},
                       os.path.join(ck, 'best_task.pt'))
        if ep % cfg['train'].get('save_every', 5) == 0 or ep == cfg['train']['epochs'] - 1:
            torch.save({'model_state': model.state_dict(), 'epoch': ep, 'config': cfg},
                       os.path.join(ck, 'epoch_%d.pt' % ep))
        print('ep %3d %.1fs total %.4f x0 %.4f | val x0 %.4f rmse %.2f m hit %.3f task %.2f'
              % (ep, rec['seconds'], train_stats.get('total', 0), train_stats.get('points', 0),
                 val['x0_loss'], val['x0_rmse_m'], val['vertex_collision_rate'], val['task_score']),
              flush=True)
        json.dump({'model': cfg['model'], 'diffusion': d, 'data': cfg['data'],
                   'loss_weights': cfg['loss'], 'epochs_done': ep + 1,
                   'epochs_requested': cfg['train']['epochs'], 'params_m': n_par / 1e6,
                   'best_val_total': best_val, 'best_task_score': best_task,
                   'best_task_epoch': task_epoch, 'history': hist},
                  open(os.path.join(run, 'training_summary.json'), 'w', encoding='utf8'), indent=2)
    print('done. best_task epoch', task_epoch, 'score %.3f' % best_task, flush=True)


if __name__ == '__main__':
    main()
