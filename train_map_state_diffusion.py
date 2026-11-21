"""Train and evaluate a map-conditioned B-spline diffusion model."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.diffusion.schedule import NoiseSchedule
from src.models.map_state_diffusion import MapStateDiffusion


def load_split(root, split, device, limit=None):
    names = ('occupancy', 'conditions', 'control_gt', 'heading', 'heading_valid', 'curve_gt')
    arrays = [np.load(root / split / f'{name}.npy', mmap_mode='r') for name in names]
    count = min(len(arrays[0]), limit) if limit is not None else len(arrays[0])
    return tuple(torch.from_numpy(a[:count].copy()).to(device) for a in arrays)


def batches(split, order, batch_size):
    for idx in order.split(batch_size):
        occ, ends, target, heading, valid, curve = (a[idx] for a in split)
        yield (occ[:, None].float().contiguous(memory_format=torch.channels_last),
               ends.float(), target.float(), heading.float(), valid.bool(), curve.float())


def pin_endpoints(state, ends):
    return torch.cat([ends[:, :1], state[:, 1:-1], ends[:, 1:]], dim=1)


def sample(model, schedule, occupancy, ends, heading, valid, noise):
    """Deterministic DDIM from one supplied noise tensor, with a cached CNN pass."""
    memory = model.encode_map(occupancy)
    state = pin_endpoints(noise, ends)
    signal = schedule.sqrt_alphas_cumprod.float()
    sigma = schedule.sqrt_one_minus_alphas_cumprod.float()
    for step in reversed(range(schedule.num_timesteps)):
        ts = torch.full((len(state),), step, dtype=torch.long, device=state.device)
        clean = model.denoise(state, memory, ends, ts, heading, valid)
        if step == schedule.num_timesteps - 1:
            first = clean
        if step:
            noise_est = (state - signal[step] * clean) / sigma[step]
            state = pin_endpoints(signal[step - 1] * clean + sigma[step - 1] * noise_est, ends)
    with torch.autocast(state.device.type, enabled=False):
        first_curve = model.bspline.decode_controls(first)
        final_curve = model.bspline.decode_controls(clean)
    # The clamped B-spline has these endpoints analytically. Restore their
    # exact values after the CUDA matrix product's reduced-precision rounding.
    return pin_endpoints(first_curve, ends), pin_endpoints(final_curve, ends)


def collision_any(trajectory, occupancy):
    h, w = occupancy.shape[-2:]
    cells = ((trajectory + 1) * 0.5 * trajectory.new_tensor([w, h])).floor().long()
    x, y = cells.unbind(-1)
    outside = (x < 0) | (x >= w) | (y < 0) | (y >= h)
    occupied = occupancy[torch.arange(len(x), device=x.device)[:, None], 0,
                         y.clamp(0, h - 1), x.clamp(0, w - 1)] > 0
    return (outside | occupied).any(dim=-1)


@torch.inference_mode()
def evaluate(model, schedule, split, batch_size, use_amp, *, limit=None):
    model.eval()
    device = split[0].device
    count = min(len(split[0]), limit) if limit else len(split[0])
    order = torch.randperm(len(split[0]), generator=torch.Generator().manual_seed(43))[:count].to(device)
    generator = torch.Generator(device=device).manual_seed(44)
    totals = torch.zeros(5, device=device)
    for occ, ends, target, heading, valid, curve in batches(split, order, batch_size):
        noise = torch.randn(target.shape, device=device, generator=generator)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            first, final = sample(model, schedule, occ, ends, heading, valid, noise)
        first_error = (first - curve).square().sum(-1).mean(-1).sqrt()
        final_error = (final - curve).square().sum(-1).mean(-1).sqrt()
        endpoint_error = (final[:, [0, -1]] - ends).norm(dim=-1).max()
        totals[:4] += torch.stack([first_error.sum(), final_error.sum(),
                                   collision_any(first, occ).sum(), collision_any(final, occ).sum()])
        totals[4] = torch.maximum(totals[4], endpoint_error)
    first_rmse, final_rmse, first_collision, final_collision = (totals[:4] / count).tolist()
    return dict(samples=count, first_rmse=first_rmse, rmse=final_rmse,
                first_collision_rate=first_collision, collision_rate=final_collision,
                max_endpoint_error=totals[4].item())


def save_checkpoint(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def evaluate_checkpoints(config, output, model, schedule, device, use_amp):
    test = load_split(Path(config['data_root']), 'test', device)
    summaries = {}
    for name in ('latest', 'best'):
        ckpt = torch.load(output / f'{name}.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(ckpt['model'])
        metrics = evaluate(model, schedule, test, config['train']['batch_size'], use_amp)
        summaries[name] = dict(epoch=ckpt['epoch'], **metrics)
    (output / 'test.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/config_map_state_diffusion.yaml')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8-sig'))
    train = config['train']
    device = torch.device(args.device)
    use_amp = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    torch.manual_seed(config['seed'])
    torch.set_float32_matmul_precision('high')
    output = args.output or Path(config['output'] + ('_smoke' if args.smoke else ''))
    output.mkdir(parents=True, exist_ok=True)
    latest = output / 'latest.pt'
    if latest.exists() and not args.resume and not args.evaluate_only:
        raise FileExistsError(f'{latest} already exists; pass --resume to continue')
    model = MapStateDiffusion(config).to(device, memory_format=torch.channels_last)
    diffusion = config['diffusion']
    schedule = NoiseSchedule(diffusion['timesteps'], diffusion['beta_schedule']).to(device)
    if args.evaluate_only:
        evaluate_checkpoints(config, output, model, schedule, device, use_amp)
        return
    train_set = load_split(Path(config['data_root']), 'train', device, 32 if args.smoke else None)
    val_set = load_split(Path(config['data_root']), 'val', device, 8 if args.smoke else None)
    epochs = 1 if args.smoke else train['epochs']
    optimizer = torch.optim.AdamW(model.parameters(), lr=train['lr'],
                                  weight_decay=train['weight_decay'], fused=device.type == 'cuda')
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=train['min_lr'])
    best = float('inf')
    start_epoch = 1
    if args.resume:
        previous = torch.load(latest, map_location='cpu', weights_only=True)
        model.load_state_dict(previous['model'])
        optimizer.load_state_dict(previous['optimizer'])
        decay.load_state_dict(previous['scheduler'])
        best = previous['best']
        start_epoch = previous['epoch'] + 1
    signal = schedule.sqrt_alphas_cumprod.float()
    sigma = schedule.sqrt_one_minus_alphas_cumprod.float()
    print(f'train={len(train_set[0])} val={len(val_set[0])} epochs={epochs} device={device} bf16={use_amp} '
          f'parameters={sum(p.numel() for p in model.parameters()):,}', flush=True)
    for epoch in range(start_epoch, epochs + 1):
        start = time.perf_counter()
        model.train()
        torch.manual_seed(config['seed'] + epoch)
        order = torch.randperm(len(train_set[0]), device=device)
        weighted_loss = torch.zeros((), device=device)
        for occ, ends, target, heading, valid, _ in batches(train_set, order, train['batch_size']):
            t = torch.randint(0, schedule.num_timesteps, (len(target),), device=device)
            noisy = signal[t, None, None] * target + sigma[t, None, None] * torch.randn_like(target)
            noisy = pin_endpoints(noisy, ends)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                prediction = model(noisy, occ, ends, t, heading, valid)
            loss = F.mse_loss(prediction[:, 1:-1], target[:, 1:-1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            weighted_loss += loss.detach() * len(target)
        lr = optimizer.param_groups[0]['lr']
        decay.step()
        record = dict(epoch=epoch, loss=weighted_loss.item()/len(train_set[0]), lr=lr)
        if not np.isfinite(record['loss']):
            raise FloatingPointError(record)
        if epoch == 1 or epoch % train['eval_every'] == 0 or epoch == epochs:
            record['validation'] = evaluate(model, schedule, val_set, train['batch_size'], use_amp, limit=256)
        score = record.get('validation', {}).get('rmse', float('inf'))
        improved = score < best
        best = min(best, score)
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          scheduler=decay.state_dict(), config=config, epoch=epoch, best=best)
        if improved:
            save_checkpoint(output / 'best.pt', checkpoint)
        save_checkpoint(latest, checkpoint)
        record['seconds'] = round(time.perf_counter() - start, 2)
        with (output / 'train.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
    if not args.smoke:
        evaluate_checkpoints(config, output, model, schedule, device, use_amp)


if __name__ == '__main__':
    main()
