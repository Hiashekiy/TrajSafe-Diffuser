"""Run the seven safety baselines on the native-point safety base (safety_points_t256).

Rows: the unconstrained DDIM-256 baseline plus the seven loops. The state is the 126 free
waypoints, so `NativePointTask` / `SegmentPointTask` are used unchanged -- no control-space
mapping, no new solver. Evaluation is the shared 512-point curve evaluator.

The DDIM row uses the SAME initial noise as the loops (paired protocol), and its sample is
produced with the model directly (no safety processing).
"""
import argparse, json, os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)
import numpy as np
import torch

from src.diffusion.schedule import NoiseSchedule
from experiments.safety_baselines.models.safety_points import SafetyPointDiffuser, pin_endpoints
from experiments.safety_baselines.models.safety_point_data import make_safety_point_loader
from experiments.safety_baselines.algorithms.runner import run_method, METHODS, initial_seed
from experiments.safety_baselines.algorithms.official.posterior import (
    is_noiseless, level_ladder, posterior_coefficients)
from experiments.safety_baselines.adapters.unified_curve_eval import evaluate_curve, polyline_at_u

INDICES = [0, 81, 163, 244, 326, 407, 489, 570, 652, 733, 815, 896, 978, 1059, 1141,
           1222, 1304, 1385, 1467, 1549]


@torch.no_grad()
def sample_diffuser(model, schedule, batch, device, seed=0):
    T = schedule.num_timesteps
    B = batch['cond'].shape[0]
    torch.manual_seed(seed)
    x = pin_endpoints(torch.randn(B, model.num_points, 2, device=device), batch['cond'])
    a = schedule.sqrt_alphas_cumprod.to(device).float()
    s = schedule.sqrt_one_minus_alphas_cumprod.to(device).float()
    nfe = 0
    for t in range(T - 1, -1, -1):
        tv = torch.full((B,), t, device=device, dtype=torch.long)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x, batch['occupancy'], batch['cond'], tv, a[tv], batch['candidate_xy'],
                        batch['candidate_mask'], batch['geometry'], batch['geometry_lengths'])
        nfe += 1
        clean = out['points'].float()
        x = clean if t == 0 else a[t - 1] * clean + s[t - 1] * (x - a[t] * clean) / s[t]
        x = pin_endpoints(x, batch['cond'])
    return x[0].cpu().numpy(), nfe


@torch.no_grad()
def sample_ddpm(model, schedule, batch, device, seed=0, extra_steps=64):
    """Unconstrained baseline on the SAME official ladder the seven safety loops walk.

    `sample_diffuser` above is DDIM-256; the seven loops now follow the release's ancestral
    chain (`reversed(range(-extra, T))`, no noise at t=1). Comparing them against a DDIM
    sample mixes two samplers, so this row reproduces the chain with the correction removed
    — same number of forward passes, same noise draws keyed by seed.
    """
    T = schedule.num_timesteps
    B = batch['cond'].shape[0]
    torch.manual_seed(seed)
    x = pin_endpoints(torch.randn(B, model.num_points, 2, device=device), batch['cond'])
    a = schedule.sqrt_alphas_cumprod.to(device).float()
    c1, c2, var = posterior_coefficients(schedule.alphas_cumprod.cpu().numpy(),
                                         schedule.betas.cpu().numpy())
    nfe = 0
    for t in level_ladder(T, extra_steps):
        tv = torch.full((B,), t, device=device, dtype=torch.long)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x, batch['occupancy'], batch['cond'], tv, a[tv], batch['candidate_xy'],
                        batch['candidate_mask'], batch['geometry'], batch['geometry_lengths'])
        nfe += 1
        clean = out['points'].float()
        mean = c1[t] * clean + c2[t] * x
        x = mean if is_noiseless(t) else mean + float(np.sqrt(var[t])) * torch.randn_like(x)
        x = pin_endpoints(x, batch['cond'])
    return x[0].cpu().numpy(), nfe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='experiments/safety_baselines/configs/seven_safety_points_t256.json')
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--override', action='append', default=[],
                    help='KEY=VALUE applied to the config, e.g. --override sd_tvs_level_scene=0.05')
    ap.add_argument('--indices', default=','.join(str(i) for i in INDICES))
    ap.add_argument('--methods', default=','.join(['Diffuser'] + METHODS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-points', type=int, default=512)
    ap.add_argument('--out-dir', default='experiments/safety_baselines/results/seven_safety_points')
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(ROOT, a.config), encoding='utf8'))
    for item in a.override:
        key, _, value = item.partition('=')
        cfg[key] = json.loads(value)          # numbers/bools/null parse, bare words stay strings
    if a.override: print('overrides', a.override, flush=True)
    mcfg = json.load(open(os.path.join(ROOT, cfg['model_config']), encoding='utf8'))
    device = 'cuda'
    d = mcfg['diffusion']
    schedule = NoiseSchedule(d['timesteps'], d['beta_schedule'], d.get('beta_start', 1e-4), d.get('beta_end', .02)).to(device)
    model = SafetyPointDiffuser(mcfg['model'], mcfg.get('ellipse'), num_points=mcfg['data']['num_points']).to(device)
    state = torch.load(os.path.join(ROOT, cfg['checkpoint']), map_location='cpu', weights_only=False)
    model.load_state_dict(state['model_state']); model.eval()
    print('base epoch', state.get('epoch'), 'params %.4f M' % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    idx = [int(v) for v in a.indices.split(',')]
    loader, _ = make_safety_point_loader(a.split, mcfg['data']['processed_root'], 1, False, indices=idx)
    u = np.linspace(0., 1., a.eval_points)
    out = os.path.join(ROOT, a.out_dir); os.makedirs(out, exist_ok=True)
    methods = [m for m in a.methods.split(',') if m]
    rows = []
    for k, batch in enumerate(loader):
        b = {kk: v.to(device) for kk, v in batch.items()}
        scene = int(b['sample_id'][0]); index = int(b['index'][0])
        occ = b['occupancy'][0, 0].cpu().numpy().astype(bool)
        cond = b['cond'][0].cpu().numpy()
        gt = b['curve_gt'][0].cpu().numpy()
        gt_curve = polyline_at_u(gt, u)
        cand = {kk: b[kk] for kk in ('candidate_xy', 'candidate_mask', 'geometry', 'geometry_lengths')}
        init = np.random.default_rng(initial_seed(scene, a.seed)).normal(size=(model.num_points, 2))
        init[[0, -1]] = cond
        for method in methods:
            t0 = time.perf_counter()
            if method == 'Diffuser':
                p, nfe = sample_diffuser(model, schedule, b, device, seed=a.seed + k)
                status, reason, trace = 'ok', None, []
            elif method == 'Diffuser-ddpm':
                p, nfe = sample_ddpm(model, schedule, b, device, seed=a.seed + k,
                                     extra_steps=int(cfg.get('cd_extra_steps', 0)))
                status, reason, trace = 'ok', None, []
            else:
                r = run_method(method, model, schedule, occ, cond, scene, a.seed, cfg, init_noise=init,
                               task_extra={'candidates': cand})
                status, reason, trace = r['status'], r['failure_reason'], r['per_step_trace'] or []
                p = None if r['trajectory'] is None else np.asarray(r['trajectory'], dtype=float)
                nfe = r['model_forward_calls']
            secs = time.perf_counter() - t0
            if p is None:
                rec = {'scene': scene, 'index': index, 'method': method, 'status': status,
                       'failure_reason': reason, 'seconds': secs, 'nfe': nfe, 'curve_collision': None}
                rows.append(rec)
                print('%6d %-14s FAILED %s (%.1fs)' % (scene, method, reason, secs), flush=True)
                continue
            m = evaluate_curve(polyline_at_u(p, u), occ, cond, gt_curve, u)
            m.update(scene=scene, index=index, method=method, status=status, failure_reason=reason,
                     seconds=secs, nfe=nfe, steps=len(trace))
            rows.append(m)
            print('%6d %-14s path %-5s succ %-5s clear %6.2f m rmse %5.2f nfe %3d %.1fs'
                  % (scene, method, m['curve_collision'], m['planning_success'],
                     m['min_clearance_m'], m['rmse_m'], nfe, secs), flush=True)
        with open(os.path.join(out, 'results.jsonl'), 'a', encoding='utf8') as f:
            for r in rows:
                if r['scene'] == scene:
                    f.write(json.dumps(r) + '\n')
    agg = {}
    for method in methods:
        rr = [r for r in rows if r['method'] == method]; ok = [r for r in rr if r.get('curve_collision') is not None]
        agg[method] = {'n': len(rr), 'returned': len(ok),
                       'path_collisions': int(sum(r['curve_collision'] for r in ok)),
                       'successes': int(sum(r['planning_success'] for r in ok)),
                       'min_clearance_m_mean': float(np.mean([r['min_clearance_m'] for r in ok])) if ok else None,
                       'violation_max_m': float(np.max([r['violation_max_m'] for r in ok])) if ok else None,
                       'rmse_m_mean': float(np.mean([r['rmse_m'] for r in ok])) if ok else None,
                       'path_length_ratio_mean': float(np.mean([r['path_length_ratio'] for r in ok])) if ok else None,
                       'max_curvature_1pm_mean': float(np.mean([r['max_curvature_1pm'] for r in ok])) if ok else None,
                       'seconds_mean': float(np.mean([r['seconds'] for r in rr])),
                       'nfe_mean': float(np.mean([r['nfe'] for r in rr]))}
    json.dump({'config': a.config, 'base_checkpoint': cfg['checkpoint'], 'seed': a.seed,
               'indices': idx, 'aggregate': agg}, open(os.path.join(out, 'summary.json'), 'w', encoding='utf8'), indent=2)
    print(json.dumps(agg, indent=2))
    print('wrote', out)


if __name__ == '__main__':
    main()
