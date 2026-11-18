"""Straight-line skeleton experiment (plain DDIM, no ALM).

Run: python scripts/sample_straight_skeleton.py
Each run chooses a split uniformly, then a row uniformly within that split.
Only conditions and occupancy are loaded; cached searched paths are never read.
"""
from pathlib import Path
import argparse
from datetime import datetime
import json
import secrets
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/config_160k4p_c48_v4_map16.yaml')
    parser.add_argument('--ckpt', help='Default: config train.ckpt_dir/best_task.pt')
    parser.add_argument('--processed-root', help='Override config data.processed_root')
    parser.add_argument('--split', choices=['random', 'train', 'val', 'test'], default='random')
    parser.add_argument('--index', type=int, help='Row within selected split (default random)')
    parser.add_argument('--seed', type=int, help='Reproduce selection and diffusion noise')
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--device', default=None)
    parser.add_argument('--ellipse-stride', type=int, default=1)
    parser.add_argument('--out', default='outputs/straight_skeleton')
    args = parser.parse_args()
    if args.steps < 2 or args.ellipse_stride < 1:
        parser.error('--steps must be >= 2 and --ellipse-stride must be >= 1')

    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.utils.config import load_config, num_safety_queries
    from src.utils.checkpoint import load_model
    from src.diffusion.schedule import NoiseSchedule
    from src.diffusion.sampler import sample
    from sample import draw_ellipses, to_px

    def resolve(path):
        path = Path(path)
        return path if path.is_absolute() else ROOT / path

    cfg = load_config(args.config)
    ckpt = resolve(args.ckpt or str(Path(cfg['train']['ckpt_dir']) / 'best_task.pt'))
    if not ckpt.is_file():
        raise FileNotFoundError(f'Checkpoint missing: {ckpt}; specify --ckpt')
    data_root = resolve(args.processed_root or cfg['data']['processed_root'])
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    rng = np.random.default_rng(seed)
    split = str(rng.choice(['train', 'val', 'test'])) if args.split == 'random' else args.split
    split_dir = data_root / split
    conditions = np.load(split_dir / 'conditions.npy', mmap_mode='r')
    occupancy = np.load(split_dir / 'occupancy.npy', mmap_mode='r')
    if len(conditions) == 0 or len(conditions) != len(occupancy):
        raise ValueError(f'Empty or inconsistent split: {split_dir}')
    index = int(rng.integers(len(conditions))) if args.index is None else args.index
    if not 0 <= index < len(conditions):
        parser.error(f'--index must be in [0, {len(conditions)})')
    cond_np = np.array(conditions[index], dtype=np.float32).reshape(2, 2)
    occ_np = np.array(occupancy[index], dtype=np.float32)
    if occ_np.ndim != 2 or occ_np.shape[0] != occ_np.shape[1]:
        raise ValueError(f'Expected square occupancy map, got {occ_np.shape}')
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    model, checkpoint, report = load_model(cfg, str(ckpt), device=device)
    if report and any(report.get(k) for k in ('adapted', 'missing', 'unexpected')):
        raise RuntimeError('Checkpoint/config mismatch; use the config belonging to this checkpoint')
    model.eval()
    # Supply ONE valid straight candidate in both representations. This removes
    # topology selection and all information from precomputed skeleton searches.
    q = num_safety_queries(cfg)
    if q < 2:
        raise ValueError('At least two skeleton points are required')
    alpha = torch.linspace(0, 1, q, device=device)[None, None, :, None]
    cond = torch.from_numpy(cond_np)[None].to(device)
    line = (1 - alpha) * cond[:, None, :1] + alpha * cond[:, None, 1:]
    occ = torch.from_numpy(occ_np)[None, None].to(device)
    schedule = NoiseSchedule(cfg['diffusion']['timesteps'], beta_schedule=cfg['diffusion'].get(
        'beta_schedule', 'squaredcos_cap_v2')).to(device)
    alm = dict(cfg.get('alm') or {}, enabled=False, final_project=False)
    print(f'[straight] split={split} index={index} seed={seed} device={device}', flush=True)
    with torch.inference_mode():
        result = sample(model, schedule, cond, occ, line,
                        torch.ones((1, 1), dtype=torch.bool, device=device), line,
                        torch.full((1, 1), q, dtype=torch.long, device=device),
                        device=device, steps=args.steps, seed=seed,
                        guidance='none', alm_config=alm,
                        corridor_config=dict(cfg.get('corridor') or {}, enabled=False))
    if result['alm_status'] != ['disabled'] or result['feedback']['enabled']:
        raise RuntimeError('Expected completely uncorrected sampling')
    arrays = {key: result[key][0].detach().cpu().numpy() for key in (
        'p', 'control', 'ellipse_center', 'ellipse_a', 'ellipse_b', 'ellipse_theta')}
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f'{split}_{index}_s{seed}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    prefix = out_dir / tag
    np.savez_compressed(str(prefix) + '.npz', **arrays, cond=cond_np,
                        occupancy=occ_np, skeleton=line[0, 0].cpu().numpy())
    res = occ_np.shape[0]
    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    ax.imshow(occ_np, origin='lower', cmap='gray_r', interpolation='nearest')
    ax.plot(*to_px(arrays['p'], res).T, color='#d62728', lw=1.8,
            label='Generated trajectory (no ALM)', zorder=4)
    draw_ellipses(ax, arrays['ellipse_center'], arrays['ellipse_a'],
                  arrays['ellipse_b'], arrays['ellipse_theta'], res,
                  stride=args.ellipse_stride, color='#2878b5')
    ax.plot([], [], color='#2878b5', lw=0.7, label='Predicted ellipses')
    start, goal = to_px(cond_np, res)
    ax.scatter(*start, c='#21a354', marker='o', s=90, edgecolors='black', label='Start', zorder=6)
    ax.scatter(*goal, c='#ff9900', marker='*', s=180, edgecolors='black', label='Goal', zorder=6)
    ax.set(xlim=(-0.5, res - 0.5), ylim=(-0.5, res - 0.5), aspect='equal',
           title=f'Straight skeleton | {split} #{index} | seed={seed} | no ALM')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(str(prefix) + '.png')
    plt.close(fig)
    metadata = dict(config=str(resolve(args.config)), checkpoint=str(ckpt),
                    processed_root=str(data_root), split=split, index=index,
                    seed=seed, steps=args.steps, device=device,
                    skeleton='straight_start_to_goal', guidance='none',
                    alm_enabled=False, final_project=False, feedback=False,
                    ellipse_stride=args.ellipse_stride)
    for key in ('episode_id', 'sample_id'):
        path = split_dir / f'{key}.npy'
        if path.is_file():
            metadata[key] = int(np.load(path, mmap_mode='r')[index])
    Path(str(prefix) + '.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved: {prefix}.png', flush=True)


if __name__ == '__main__':
    main()

