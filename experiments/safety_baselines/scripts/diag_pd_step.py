"""One-step anatomy of the Primal-Dual correction: which quantity sets its magnitude?

Prints, for a handful of levels of one scene: the raw violation (scene units and metres),
the multiplier, the Jacobian row sum, and the resulting correction norm -- next to the
actual spacing between neighbouring waypoints, which is the scale the correction should
nudge, not jump over.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)

from experiments.safety_baselines.algorithms.official import constrained_diffusers as official
from experiments.safety_baselines.algorithms.official.posterior import (
    is_noiseless, level_ladder, posterior_coefficients)
from experiments.safety_baselines.algorithms.runner import initial_seed
from experiments.safety_baselines.models.safety_point_data import make_safety_point_loader
from experiments.safety_baselines.models.safety_points import SafetyPointDiffuser
from src.diffusion.schedule import NoiseSchedule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='experiments/safety_baselines/configs/seven_safety_points_t256.json')
    ap.add_argument('--index', type=int, default=0)
    ap.add_argument('--steps', type=int, default=12)
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

    loader, _ = make_safety_point_loader('test', mcfg['data']['processed_root'], 1, False, indices=[a.index])
    batch = next(iter(loader))
    b = {k: v.to(device) for k, v in batch.items()}
    scene = int(b['sample_id'][0])
    occ = b['occupancy'][0, 0].cpu().numpy().astype(bool)
    cond = b['cond'][0].cpu().numpy()
    cand = {k: b[k] for k in ('candidate_xy', 'candidate_mask', 'geometry', 'geometry_lengths')}

    from experiments.safety_baselines.adapters.safety_point_task import SafetyPointTask
    task = SafetyPointTask(model, schedule, occ, cond, cfg['constraint_points'], cfg.get('radius_scene', 0.),
                           cfg.get('field_kind', 'exact'), candidates=cand)
    s = float(cfg['dual_constraint_scale']); eta = float(cfg['dual_eta'])
    print('scene %d   scale=%.4g  eta=%.4g  eta*scale^2=%.4g (official lr)' % (scene, s, eta, eta*s*s))
    print('constraint rows = %d   free coords D = %d' % (len(task.constraints(np.zeros(task.D))[0]), task.D))

    rng = np.random.default_rng(initial_seed(scene, 0))
    x = np.asarray(task.initial(rng)[0], dtype=float)
    ab = np.asarray(task.ab); c1, c2, var = posterior_coefficients(ab, task.betas)
    lam = np.zeros((1, len(task.constraints(x)[0])))
    ladder = level_ladder(len(ab), int(cfg.get('cd_extra_steps', 0)))
    print('%6s %9s %9s %9s %9s %9s' % ('t', '|g|max_m', 'vio_max', 'lam_max', '|corr|', 'spacing_m'))
    taken = 0
    for t in ladder:
        x0 = np.asarray(task.clean(x, t)[0], dtype=float)
        mean = (c1[t]*x0 + c2[t]*x)[None]
        g, J = task.constraints(x)
        g = np.asarray(g, dtype=float)
        pts = task.unpack(x)
        spacing = float(np.median(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        before = mean.copy()
        vio = (-g*s)[None]
        grad = (-np.asarray(J, dtype=float)*s)[None]
        mean, lam = official.primal_dual(mean, vio, grad, lam, eta)
        corr = float(np.linalg.norm(mean - before))
        std = 0.0 if is_noiseless(t) else float(np.sqrt(var[t]))
        x = mean[0] + std*rng.normal(size=task.D)
        if taken < a.steps:
            print('%6d %9.3f %9.4f %9.2e %9.2e %9.3f' % (t, g.max()*80., g.max(), lam.max(), corr, spacing*80.))
            taken += 1


if __name__ == '__main__':
    main()
