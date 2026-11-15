"""Diagnose Primal-Dual only: is its 2.8x path distortion a magnitude problem or an oscillation?

For each scene it runs the unconstrained DDPM baseline and Primal-Dual on the *same* ladder and
the same seed, then reports

  len      -- curve length ratio of the delivered path (PD vs baseline)
  lam_max  -- final multiplier magnitude (0.0 means the correction never engaged)
  corr_tot -- summed ||mean_after - mean_before|| over all steps, in scene units
  step_tot -- summed ||x_{t-1} - x_t||, i.e. how far the state actually travelled
  flip     -- fraction of steps with cos(correction_t, correction_{t-1}) < 0
  tail     -- fraction of corr_tot spent in the noiseless t=1 repeats

Reading it:
  * flip near 0 and corr_tot tiny      -> the multiplier never grows (lr too small / units)
  * flip well above 0.5                -> the push reverses direction: zig-zag, not magnitude
  * tail large                         -> the drift is produced by the repeated t=1 steps
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
from experiments.safety_baselines.algorithms.official.posterior import (
    is_noiseless, level_ladder, posterior_coefficients)
from experiments.safety_baselines.algorithms.runner import initial_seed, run_method
from experiments.safety_baselines.models.safety_point_data import make_safety_point_loader
from experiments.safety_baselines.models.safety_points import SafetyPointDiffuser, pin_endpoints
from src.diffusion.schedule import NoiseSchedule


@torch.no_grad()
def sample_baseline(model, schedule, batch, device, seed, extra_steps):
    """The same ladder with the correction removed (mirror of run_seven_safety_points.sample_ddpm)."""
    T = schedule.num_timesteps
    B = batch['cond'].shape[0]
    torch.manual_seed(seed)
    x = pin_endpoints(torch.randn(B, model.num_points, 2, device=device), batch['cond'])
    a = schedule.sqrt_alphas_cumprod.to(device).float()
    c1, c2, var = posterior_coefficients(schedule.alphas_cumprod.cpu().numpy(),
                                         schedule.betas.cpu().numpy())
    for t in level_ladder(T, extra_steps):
        tv = torch.full((B,), t, device=device, dtype=torch.long)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x, batch['occupancy'], batch['cond'], tv, a[tv], batch['candidate_xy'],
                        batch['candidate_mask'], batch['geometry'], batch['geometry_lengths'])
        clean = out['points'].float()
        mean = c1[t]*clean + c2[t]*x
        x = mean if is_noiseless(t) else mean + float(np.sqrt(var[t]))*torch.randn_like(x)
        x = pin_endpoints(x, batch['cond'])
    return x[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='experiments/safety_baselines/configs/seven_safety_points_t256.json')
    ap.add_argument('--split', default='test')
    ap.add_argument('--indices', default='0,3,9,15,20,31,44,57,70,101')
    ap.add_argument('--method', default='Primal-Dual')
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(ROOT, a.config), encoding='utf8'))
    mcfg = json.load(open(os.path.join(ROOT, cfg['model_config']), encoding='utf8'))
    d = mcfg['diffusion']
    device = 'cuda'
    schedule = NoiseSchedule(d['timesteps'], d['beta_schedule'],
                             d.get('beta_start', 1e-4), d.get('beta_end', .02)).to(device)
    model = SafetyPointDiffuser(mcfg['model'], mcfg.get('ellipse'), num_points=mcfg['data']['num_points']).to(device)
    state = torch.load(os.path.join(ROOT, cfg['checkpoint']), map_location='cpu', weights_only=False)
    model.load_state_dict(state['model_state']); model.eval()

    idx = [int(v) for v in a.indices.split(',')]
    loader, _ = make_safety_point_loader(a.split, mcfg['data']['processed_root'], 1, False, indices=idx)
    u = np.linspace(0., 1., 512)
    extra = int(cfg.get('cd_extra_steps', 0))
    print('%8s %6s %6s %8s %9s %9s %5s %6s' % ('scene', 'len_base', 'len_PD', 'lam_max',
                                               'corr_tot', 'step_tot', 'flip', 'tail'), flush=True)
    for k, batch in enumerate(loader):
        b = {kk: v.to(device) for kk, v in batch.items()}
        scene = int(b['sample_id'][0])
        occ = b['occupancy'][0, 0].cpu().numpy().astype(bool)
        cond = b['cond'][0].cpu().numpy()
        gt = polyline_at_u(b['curve_gt'][0].cpu().numpy(), u)
        cand = {kk: b[kk] for kk in ('candidate_xy', 'candidate_mask', 'geometry', 'geometry_lengths')}
        init = np.random.default_rng(initial_seed(scene, 0)).normal(size=(model.num_points, 2))
        init[[0, -1]] = cond

        base = sample_baseline(model, schedule, b, device, 0, extra)
        m_base = evaluate_curve(polyline_at_u(base, u), occ, cond, gt, u)

        r = run_method(a.method, model, schedule, occ, cond, scene, 0, cfg, init_noise=init,
                       task_extra={'candidates': cand})
        if r['status'] != 'ok':
            print('%8d %6.2f %8s %s' % (scene, m_base['path_length_ratio'], 'FAILED', r['failure_reason'][:50]), flush=True)
            continue
        path = np.asarray(r['trajectory'], dtype=float)
        m_pd = evaluate_curve(polyline_at_u(path, u), occ, cond, gt, u)
        tr = r['per_step_trace']
        lam_max = max(x['lambda_max'] for x in tr)
        corr = np.array([x['correction_norm'] for x in tr])
        step = np.array([x['step_norm'] for x in tr])
        cos = np.array([x['correction_cos_prev'] for x in tr], dtype=float)
        valid = np.isfinite(cos)
        flip = float(np.mean(cos[valid] < 0)) if valid.any() else float('nan')
        noisy = np.array([not x['noiseless'] for x in tr])
        tail = float(corr[~noisy].sum()/max(corr.sum(), 1e-12))
        print('%8d %6.2f %6.2f %8.3f %9.2f %9.2f %5.2f %6.2f'
              % (scene, m_base['path_length_ratio'], m_pd['path_length_ratio'], lam_max,
                 corr.sum(), step.sum(), flip, tail), flush=True)


if __name__ == '__main__':
    main()
