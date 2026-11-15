"""Plot the unconstrained base and the seven corrected paths on their maps.

Two scenes: one where the base is already safe, one where it collides. All eight rows use
the same seed / paired initial noise as the comparison table, so what you see is exactly
the run behind `docs/seven_safety_points_report.md`.
"""
import argparse, json, os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.diffusion.schedule import NoiseSchedule
from experiments.safety_baselines.models.safety_points import SafetyPointDiffuser
from experiments.safety_baselines.models.safety_point_data import make_safety_point_loader
from experiments.safety_baselines.algorithms.runner import run_method, METHODS, initial_seed
from experiments.safety_baselines.adapters.unified_curve_eval import evaluate_curve, polyline_at_u

CLEAN = {'color': 'tab:blue', 'lw': 1.6}
BAD = {'color': 'tab:red', 'lw': 1.3}


@torch.no_grad()
def sample_diffuser(model, schedule, batch, device, seed=0):
    T = schedule.num_timesteps
    B = batch['cond'].shape[0]
    torch.manual_seed(seed)
    x = torch.randn(B, model.num_points, 2, device=device)
    x[:, 0] = batch['cond'][:, 0]; x[:, -1] = batch['cond'][:, 1]
    a = schedule.sqrt_alphas_cumprod.to(device).float()
    s = schedule.sqrt_one_minus_alphas_cumprod.to(device).float()
    for t in range(T - 1, -1, -1):
        tv = torch.full((B,), t, device=device, dtype=torch.long)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x, batch['occupancy'], batch['cond'], tv, a[tv], batch['candidate_xy'],
                        batch['candidate_mask'], batch['geometry'], batch['geometry_lengths'])
        clean = out['points'].float()
        x = clean if t == 0 else a[t - 1] * clean + s[t - 1] * (x - a[t] * clean) / s[t]
        x[:, 0] = batch['cond'][:, 0]; x[:, -1] = batch['cond'][:, 1]
    return x[0].cpu().numpy()


def panel(ax, occ, gt, cond, curves, title):
    ax.imshow(occ, cmap='gray_r', origin='lower', extent=[-1, 1, -1, 1], alpha=.85)
    ax.plot(gt[:, 0], gt[:, 1], '--', color='tab:gray', lw=1.0)
    ax.plot(cond[:, 0], cond[:, 1], 'o', color='tab:green', ms=5)
    for name, p, bad in curves:
        st = BAD if bad else CLEAN
        ax.plot(p[:, 0], p[:, 1], '-', color=st['color'], lw=st['lw'])
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='experiments/safety_baselines/configs/seven_safety_points_t256.json')
    ap.add_argument('--indices', default='0,163')
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--override', action='append', default=[])          # 61001 (base safe), 74014 (base collides)
    ap.add_argument('--out-dir', default='experiments/safety_baselines/results/seven_safety_paths')
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(ROOT, a.config), encoding='utf8'))
    for item in a.override:
        k, _, v = item.partition('='); cfg[k] = json.loads(v)
    mcfg = json.load(open(os.path.join(ROOT, cfg['model_config']), encoding='utf8'))
    device = 'cuda'
    d = mcfg['diffusion']
    schedule = NoiseSchedule(d['timesteps'], d['beta_schedule'], d.get('beta_start', 1e-4), d.get('beta_end', .02)).to(device)
    model = SafetyPointDiffuser(mcfg['model'], mcfg.get('ellipse'), num_points=mcfg['data']['num_points']).to(device)
    st = torch.load(os.path.join(ROOT, cfg['checkpoint']), map_location='cpu', weights_only=False)
    model.load_state_dict(st['model_state']); model.eval()
    idx = [int(v) for v in a.indices.split(',')]
    loader, _ = make_safety_point_loader(a.split, mcfg['data']['processed_root'], 1, False, indices=idx)
    out = os.path.join(ROOT, a.out_dir); os.makedirs(out, exist_ok=True)
    methods = ['Diffuser-ddpm'] + METHODS
    u = np.linspace(0., 1., 512)
    for k, batch in enumerate(loader):
        b = {kk: v.to(device) for kk, v in batch.items()}
        scene = int(b['sample_id'][0])
        occ = b['occupancy'][0, 0].cpu().numpy().astype(bool)
        cond = b['cond'][0].cpu().numpy()
        gt = b['curve_gt'][0].cpu().numpy()
        gt_curve = polyline_at_u(gt, u)
        cand = {kk: b[kk] for kk in ('candidate_xy', 'candidate_mask', 'geometry', 'geometry_lengths')}
        init = np.random.default_rng(initial_seed(scene, 0)).normal(size=(model.num_points, 2))
        init[[0, -1]] = cond
        store = {}
        for method in methods:
            t0 = time.perf_counter()
            if method == 'Diffuser-ddpm':
                from experiments.safety_baselines.scripts.run_seven_safety_points import sample_ddpm
                p, _ = sample_ddpm(model, schedule, b, device, seed=0,
                            extra_steps=int(cfg.get('cd_extra_steps', 0)))
            else:
                r = run_method(method, model, schedule, occ, cond, scene, 0, cfg, init_noise=init,
                               task_extra={'candidates': cand})
                p = None if r['trajectory'] is None else np.asarray(r['trajectory'], dtype=float)
            secs = time.perf_counter() - t0
            if p is None:
                store[method] = {'points': None, 'collision': None}
                print(scene, method, 'FAILED', str(r['failure_reason'])[:110], flush=True)
                continue
            m = evaluate_curve(polyline_at_u(p, u), occ, cond, gt_curve, u)
            store[method] = {'points': p, 'collision': m['curve_collision'],
                             'clear': m['min_clearance_m'], 'rmse': m['rmse_m'],
                             'len': m['path_length_ratio'], 'secs': secs}
            print('%6d %-14s path %-5s clear %6.2f len %5.2f %.1fs'
                  % (scene, method, m['curve_collision'], m['min_clearance_m'], m['path_length_ratio'], secs), flush=True)
        # per-method grid
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=110)
        for ax, method in zip(axes.ravel(), methods):
            s = store[method]
            if s['points'] is None:
                panel(ax, occ, gt, cond, [], method + ' (failed)')
                continue
            title = '%s  %s  clear %.1f m  len %.2f' % (
                method, 'COLLIDES' if s['collision'] else 'ok', s['clear'], s['len'])
            panel(ax, occ, gt, cond, [(method, s['points'], s['collision'])], title)
        fig.suptitle('scene %d 鈥?grey dashed = ground truth, green = start/goal, red = colliding path' % scene, fontsize=10)
        if a.override:
            fig.suptitle(fig._suptitle.get_text() + '   |   ' + ', '.join(a.override), fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(out, 'scene_%d_grid.png' % scene)); plt.close(fig)
        # all curves on one map
        fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
        ax.imshow(occ, cmap='gray_r', origin='lower', extent=[-1, 1, -1, 1], alpha=.85)
        ax.plot(gt[:, 0], gt[:, 1], '--', color='tab:gray', lw=1.2, label='GT')
        ax.plot(cond[:, 0], cond[:, 1], 'o', color='tab:green', ms=7, label='start/goal')
        for method in methods:
            s = store[method]
            if s['points'] is None:
                continue
            ax.plot(s['points'][:, 0], s['points'][:, 1], '-', lw=1.2, alpha=.9,
                    label=method + (' (X)' if s['collision'] else ''))
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=6, loc='lower right')
        fig.tight_layout(); fig.savefig(os.path.join(out, 'scene_%d_overlay.png' % scene)); plt.close(fig)
        np.savez_compressed(os.path.join(out, 'scene_%d_paths.npz' % scene),
                            **{m: store[m]['points'] for m in methods if store[m]['points'] is not None})
    print('wrote', out)


if __name__ == '__main__':
    main()
