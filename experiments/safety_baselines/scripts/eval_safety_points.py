"""Unified evaluation of the native-point safety model on the frozen 20 val scenes.

DDIM over the full schedule (T=256), x0 parameterisation, endpoints pinned, argmax(pi)
topology at inference -- i.e. exactly the sampler the seven baselines will use. The
delivered object is the 128-point polyline, scored by `unified_curve_eval.evaluate_curve`
on a 512-point grid (the same evaluator used for the map-only base and for S-C48).
"""
import argparse, json, os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)
import numpy as np
import torch

from src.diffusion.schedule import NoiseSchedule
from experiments.safety_baselines.models.safety_points import SafetyPointDiffuser, pin_endpoints
from experiments.safety_baselines.models.safety_point_data import make_safety_point_loader
from experiments.safety_baselines.adapters.unified_curve_eval import evaluate_curve, polyline_at_u

INDICES = [0, 81, 163, 244, 326, 407, 489, 570, 652, 733, 815, 896, 978, 1059, 1141,
           1222, 1304, 1385, 1467, 1549]


@torch.no_grad()
def sample_ddim(model, schedule, batch, device, steps=None, seed=0):
    """Plain DDIM in point space; returns (points [B,128,2], nfe)."""
    T = schedule.num_timesteps
    times = None if (steps is None or steps >= T) else \
        sorted(set(torch.linspace(0, T - 1, steps).long().tolist()) | {0, T - 1})
    pairs = ([(t, t - 1) for t in range(T - 1, -1, -1)] if times is None
             else [(times[i], times[i - 1]) for i in range(len(times) - 1, 0, -1)] + [(times[0], -1)])
    B = batch['cond'].shape[0]
    torch.manual_seed(seed)
    x = torch.randn(B, model.num_points, 2, device=device)
    x = pin_endpoints(x, batch['cond'])
    a = schedule.sqrt_alphas_cumprod.to(device).float()
    s = schedule.sqrt_one_minus_alphas_cumprod.to(device).float()
    nfe = 0
    for t, nxt in pairs:
        tv = torch.full((B,), int(t), device=device, dtype=torch.long)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x, batch['occupancy'], batch['cond'], tv, a[tv],
                        batch['candidate_xy'], batch['candidate_mask'],
                        batch['geometry'], batch['geometry_lengths'])
        nfe += 1
        clean = out['points'].float()
        x = clean if nxt < 0 else a[nxt] * clean + s[nxt] * (x - a[t] * clean) / s[t]
        x = pin_endpoints(x, batch['cond'])
    return x, nfe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='experiments/safety_baselines/results/safety_points_t256/ckpt/best.pt')
    ap.add_argument('--config', default='experiments/safety_baselines/configs/safety_points_t256.json')
    ap.add_argument('--split', default='val')
    ap.add_argument('--indices', default=','.join(str(i) for i in INDICES))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--eval-points', type=int, default=512)
    ap.add_argument('--out-dir', default='experiments/safety_baselines/results/safety_points_eval')
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(ROOT, a.config), encoding='utf8'))
    device = 'cuda'
    d = cfg['diffusion']
    schedule = NoiseSchedule(d['timesteps'], d['beta_schedule'], d.get('beta_start', 1e-4), d.get('beta_end', .02)).to(device)
    model = SafetyPointDiffuser(cfg['model'], cfg.get('ellipse'), num_points=cfg['data']['num_points']).to(device)
    state = torch.load(os.path.join(ROOT, a.checkpoint), map_location='cpu', weights_only=False)
    model.load_state_dict(state['model_state']); model.eval()
    print('checkpoint epoch', state.get('epoch'), 'params %.4f M' % (sum(p.numel() for p in model.parameters()) / 1e6))
    idx = [int(v) for v in a.indices.split(',')]
    loader, ds = make_safety_point_loader(a.split, cfg['data']['processed_root'], 1, False,
                                          limit=None, indices=idx)
    u = np.linspace(0., 1., a.eval_points)
    out = os.path.join(ROOT, a.out_dir); os.makedirs(out, exist_ok=True)
    rows = []
    for k, batch in enumerate(loader):
        b = {kk: v.to(device) for kk, v in batch.items()}
        t0 = time.perf_counter()
        pts, nfe = sample_ddim(model, schedule, b, device, steps=a.steps, seed=a.seed + k)
        torch.cuda.synchronize(); secs = time.perf_counter() - t0
        p = pts[0].cpu().numpy()
        occ = b['occupancy'][0, 0].cpu().numpy().astype(bool)
        cond = b['cond'][0].cpu().numpy()
        gt = b['curve_gt'][0].cpu().numpy()
        gt_curve = polyline_at_u(gt, u)
        curve = polyline_at_u(p, u)
        m = evaluate_curve(curve, occ, cond, gt_curve, u)
        m.update(scene=int(b['sample_id'][0]), index=int(b['index'][0]), method='safety-point',
                 checkpoint=os.path.basename(a.checkpoint), nfe=nfe, seconds=secs,
                 vertex_collision_polyline=bool(evaluate_curve(p, occ, cond, gt, np.linspace(0, 1, len(p)))['curve_vertex_collision']))
        rows.append(m)
        print('%6d path %-5s vtx %-5s succ %-5s clear %6.2f m rmse %5.2f %.1fs'
              % (m['scene'], m['curve_collision'], m['vertex_collision_polyline'], m['planning_success'],
                 m['min_clearance_m'], m['rmse_m'], secs), flush=True)
    n = len(rows)
    agg = {'n': n,
           'path_collisions': int(sum(r['curve_collision'] for r in rows)),
           'vertex_collisions_polyline': int(sum(r['vertex_collision_polyline'] for r in rows)),
           'successes': int(sum(r['planning_success'] for r in rows)),
           'min_clearance_m_mean': float(np.mean([r['min_clearance_m'] for r in rows])),
           'violation_max_m': float(np.max([r['violation_max_m'] for r in rows])),
           'rmse_m_mean': float(np.mean([r['rmse_m'] for r in rows])),
           'path_length_ratio_mean': float(np.mean([r['path_length_ratio'] for r in rows])),
           'max_curvature_1pm_mean': float(np.mean([r['max_curvature_1pm'] for r in rows])),
           'mean_second_diff_m_mean': float(np.mean([r['mean_second_difference_m'] for r in rows])),
           'mean_turning_deg_mean': float(np.mean([r['mean_turning_deg'] for r in rows])),
           'nfe': int(rows[0]['nfe']), 'seconds_mean': float(np.mean([r['seconds'] for r in rows]))}
    tag = os.path.basename(a.checkpoint).replace('.pt', '')
    json.dump({'checkpoint': a.checkpoint, 'config': a.config, 'seed': a.seed,
               'steps': a.steps or cfg['diffusion']['timesteps'], 'aggregate': agg, 'rows': rows},
              open(os.path.join(out, 'eval_%s.json' % tag), 'w', encoding='utf8'), indent=2)
    print(json.dumps(agg, indent=2))
    print('wrote', out)


if __name__ == '__main__':
    main()
