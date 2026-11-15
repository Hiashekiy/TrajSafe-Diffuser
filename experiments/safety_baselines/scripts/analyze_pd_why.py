"""Why does the per-vertex safety correction not reduce path collisions?

Two questions, one run.

1. Is the integration doing its job?  For Primal-Dual we record, at every reverse step,
   the worst violation before and after the correction (`violation_before/after`, scene
   units). If the correction works, `after < before` on the steps where anything is violated.

2. Why does that not make the *path* safe?  The delivered object is a 128-point polyline and
   the evaluator measures its resampled curve. So we count separately, for the same path:
       bad_vertices  -- waypoints inside an occupied cell
       bad_segments  -- segments whose interior passes through an occupied cell
   A correction that clears every waypoint but leaves segments crossing is the textbook
   "vertex-safe / curve-unsafe" gap; that is what the numbers below separate.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)

from experiments.safety_baselines.adapters.unified_curve_eval import evaluate_curve, polyline_at_u
from experiments.safety_baselines.algorithms.runner import initial_seed, run_method
from experiments.safety_baselines.models.safety_point_data import make_safety_point_loader
from experiments.safety_baselines.models.safety_points import SafetyPointDiffuser
from src.diffusion.schedule import NoiseSchedule


def geometry(query, path, samples=8):
    """(bad waypoints, bad segments, worst penetration) using the task's own barrier.

    `query(points) -> (h, grad)` is `task.map.query`, i.e. the exact distance field the
    correction optimises against -- not a hand-rolled grid index, so waypoint and segment
    counts are on the same footing as the constraint the method sees. h < 0 means inside.
    """
    def h(points):
        value = query(np.asarray(points, dtype=float))[0]
        return np.asarray(value, dtype=float).reshape(-1)
    hv = h(path)
    bad_v = int((hv < 0).sum())
    bad_s, worst = 0, 0.0
    for i in range(len(path)-1):
        t = np.linspace(0., 1., samples + 2)[1:-1, None]
        inner = path[i]*(1-t) + path[i+1]*t
        hi = h(inner)
        if (hi < 0).any():
            bad_s += 1
            worst = max(worst, float(-hi.min()))
    return bad_v, bad_s, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='experiments/safety_baselines/configs/seven_safety_points_t256.json')
    ap.add_argument('--indices', default=','.join(str(i) for i in range(8)))
    ap.add_argument('--methods', default='Diffuser-ddpm,Primal-Dual,RoS-diffuser,TVS-diffuser')
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(ROOT, a.config), encoding='utf8'))
    mcfg = json.load(open(os.path.join(ROOT, cfg['model_config']), encoding='utf8'))
    d = mcfg['diffusion']
    device = 'cuda'
    schedule = NoiseSchedule(d['timesteps'], d['beta_schedule'],
                             d.get('beta_start', 1e-4), d.get('beta_end', .02)).to(device)
    model = SafetyPointDiffuser(mcfg['model'], mcfg.get('ellipse'), num_points=mcfg['data']['num_points']).to(device)
    st = torch.load(os.path.join(ROOT, cfg['checkpoint']), map_location='cpu', weights_only=False)
    model.load_state_dict(st['model_state']); model.eval()

    idx = [int(v) for v in a.indices.split(',')]
    methods = [m for m in a.methods.split(',') if m]
    loader, _ = make_safety_point_loader('test', mcfg['data']['processed_root'], 1, False, indices=idx)
    u = np.linspace(0., 1., 512)
    print('%8s %-14s %5s %5s %7s %8s %8s %9s %7s' %
          ('scene', 'method', 'badV', 'badS', 'worst_seg', 'corr_tot', 'viol_drop', 'viol_mean', 'len'), flush=True)
    for k, batch in enumerate(loader):
        b = {kk: v.to(device) for kk, v in batch.items()}
        scene = int(b['sample_id'][0])
        occ = b['occupancy'][0, 0].cpu().numpy().astype(bool)
        cond = b['cond'][0].cpu().numpy()
        gt = polyline_at_u(b['curve_gt'][0].cpu().numpy(), u)
        cand = {kk: b[kk] for kk in ('candidate_xy', 'candidate_mask', 'geometry', 'geometry_lengths')}
        init = np.random.default_rng(initial_seed(scene, 0)).normal(size=(model.num_points, 2))
        init[[0, -1]] = cond
        from experiments.safety_baselines.scripts.run_seven_safety_points import sample_ddpm
        from experiments.safety_baselines.adapters.safety_point_task import SafetyPointTask
        task = SafetyPointTask(model, schedule, occ, cond, cfg['constraint_points'], cfg.get('radius_scene', 0.), cfg.get('field_kind', 'exact'), candidates=cand)
        for method in methods:
            if method == 'Diffuser-ddpm':
                path, _ = sample_ddpm(model, schedule, b, device, seed=0,
                                      extra_steps=int(cfg.get('cd_extra_steps', 0)))
                tr = []
                status = 'ok'
            else:
                r = run_method(method, model, schedule, occ, cond, scene, 0, cfg, init_noise=init,
                               task_extra={'candidates': cand})
                status = r['status']
                if status != 'ok':
                    print('%8d %-14s FAILED %s' % (scene, method, r['failure_reason'][:40]), flush=True)
                    continue
                path = np.asarray(r['trajectory'], dtype=float)
                tr = r['per_step_trace'] or []
            badv, bads, worst = geometry(task.map.query, path)
            m = evaluate_curve(polyline_at_u(path, u), occ, cond, gt, u)
            if tr and 'violation_before' in tr[0]:
                vb = np.array([x.get('violation_before', np.nan) for x in tr], dtype=float)
                va = np.array([x.get('violation_after', np.nan) for x in tr], dtype=float)
                ok = np.isfinite(vb) & np.isfinite(va) & (vb > 0)
                drop = float(np.nanmean(vb[ok] - va[ok])) if ok.any() else float('nan')
                vmean = float(np.nanmean(vb[ok])) if ok.any() else float('nan')
                corr = float(np.nansum([x.get('correction_norm', 0.) for x in tr]))
            else:
                drop = vmean = corr = float('nan')
            print('%8d %-14s %5d %5d %7.2f %8.2f %8.4f %9.4f %7.3f' %
                  (scene, method, badv, bads, worst, corr, drop, vmean, m['path_length_ratio']), flush=True)


if __name__ == '__main__':
    main()
