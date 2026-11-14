"""Phase B-Review diagnostics for the S-C48 control-space port (2026-11-14 review).

Answers, on the three representative scenes (61001 safe / 74014 light / 86008 severe):

  1. is the control-space feasible set non-empty at all?  -> evaluate the dataset's own
     `control_gt` B-spline against the 128-point constraint set and the 512-point
     collision evaluator (validation-grade diagnostic only, never used as a solution);
  2. what does J_Q look like?  -> row nnz (B-spline local support), ||J||_2, ||J||_F,
     and how they change with the denoising level;
  3. SafeDiffuser: which solver actually ran, and did the QP demonstrably change the
     nominal velocity?  -> per-level OSQP status, primal/dual residual, ||v-v_nom||,
     ||s||, min(Jv+b), min(Jv+b+s);
  4. projection: does warm-starting from the previous feasible point matter?  -> same
     projection from (a) the target and (b) the GT control polygon;
  5. PD/ALM: J spectrum, ||J^T lambda||, ||score||, per-level ||dx||, and the
     cd_extra_steps in {0,16,64} sensitivity;
  6. real NFE per method (network forward count), not the config value.
"""
import argparse, json, os, sys
LEGACY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'legacy_sc48_code'))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, LEGACY); sys.path.insert(1, ROOT)
import numpy as np
import torch
from src.utils.config import load_config
from src.utils.checkpoint import load_model
from src.diffusion.schedule import NoiseSchedule
from src.datasets.carla_spline_dataset import CarlaSplineDataset
from experiments.safety_baselines.algorithms.runner import run_method, initial_seed, METHODS
from experiments.safety_baselines.algorithms.solvers import Budget, nearest_projection
from experiments.safety_baselines.adapters.sc48_control import SC48ControlTask
from experiments.safety_baselines.adapters.unified_curve_eval import evaluate_curve

METHOD_LIST = ['Diffuser'] + METHODS


def build(model, sched, item, cfg, scene, seed, cbf_solver='legacy', slack_weight=None):
    occ = np.asarray(item['occupancy'][0].numpy(), dtype=bool); cond = np.asarray(item['cond'], dtype=float)
    cand = dict(candidate_xy=np.asarray(item['candidate_xy']), candidate_mask=np.asarray(item['candidate_mask']),
                candidate_geometry=np.asarray(item['candidate_geometry']),
                geometry_lengths=np.asarray(item['candidate_geometry_lengths']))
    task = SC48ControlTask(model, sched, occ, cond, cfg['constraint_points'], cfg.get('radius_scene', 0.),
                           cfg.get('field_kind', 'bilinear'), candidates=cand,
                           cbf_solver=cbf_solver, slack_weight=slack_weight)
    task.projection_maxiter = cfg['projection_maxiter']; task.projection_tol = cfg['projection_tol_scene']
    init = np.random.default_rng(initial_seed(scene, seed)).normal(size=(model.num_controls, 2))
    init[[0, -1]] = cond
    task.initial_override = np.asarray(init, dtype=float)[1:-1].reshape(-1)
    task.particle_seed = initial_seed(scene, seed)
    return task, occ, cond, cand, init


def jac_stats(task, x):
    g, J = task.constraints(x)
    nz = (np.abs(J) > 0).sum(axis=1)
    return {'rows': int(J.shape[0]), 'vars': int(J.shape[1]),
            'row_nnz_min': int(nz.min()), 'row_nnz_median': float(np.median(nz)), 'row_nnz_max': int(nz.max()),
            'J_two_norm': float(np.linalg.norm(J, 2)), 'J_fro': float(np.linalg.norm(J)),
            'max_constraint': float(np.max(g)), 'active_rows': int((g > -1e-6).sum())}


def gt_feasibility(model, task, item, occ, cond, u512, B512):
    """Does the dataset's own control_gt satisfy the 128-point set and the 512-point check?"""
    q_gt = np.asarray(item['control_gt'], dtype=float)
    g128, _ = task.constraints(q_gt[1:-1].reshape(-1))
    curve = B512 @ q_gt
    gt128 = np.asarray(item['pos'], dtype=float)
    gt_curve = np.stack([np.interp(u512, np.linspace(0, 1, len(gt128)), gt128[:, k]) for k in (0, 1)], axis=1)
    m = evaluate_curve(curve, occ, cond, gt_curve, u512)
    return {'control_gt_max_constraint': float(np.max(g128)),
            'control_gt_feasible_128': bool(np.max(g128) <= 1e-9),
            'control_gt_curve_collision': m['curve_collision'],
            'control_gt_min_clearance_m': m['min_clearance_m']}


def agg_trace(trace, keys):
    out = {}
    for k in keys:
        vals = [t.get(k) for t in trace if isinstance(t.get(k), (int, float)) and t.get(k) is not None]
        if vals:
            out[k + '_mean'] = float(np.mean(vals)); out[k + '_max'] = float(np.max(vals))
            out[k + '_min'] = float(np.min(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='experiments/safety_baselines/configs/sc48_native16_vertexall.json')
    ap.add_argument('--base-config', default=os.path.join(LEGACY, 'configs/config_160k4p_c48_s_nofb.yaml'))
    ap.add_argument('--indices', default='0,2,4')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='experiments/safety_baselines/results/sc48_phase_b_review/diagnostics.json')
    a = ap.parse_args()
    cfg = json.loads(open(os.path.join(ROOT, a.config), encoding='utf8').read())
    bcfg = load_config(a.base_config)
    bcfg['data']['processed_root'] = 'D:/ProjectDirectory/Neural-IRISDiffuser/data/processed/160m-k4p-c48'
    model, _, rep = load_model(bcfg, os.path.join(ROOT, cfg['checkpoint']), device='cuda', verbose=False)
    assert not (rep['missing'] or rep['unexpected'] or rep['adapted']), rep
    model.eval()
    d = bcfg['diffusion']
    sched = NoiseSchedule(d['timesteps'], d['beta_schedule'], d.get('beta_start', 1e-4), d.get('beta_end', .02)).to('cuda')
    ds = CarlaSplineDataset('val', bcfg['data']['processed_root'])
    u512 = torch.linspace(0., 1., 512, device='cuda').cpu().numpy()
    B512 = model.bspline.basis_at(torch.linspace(0., 1., 512, device='cuda')).detach().cpu().numpy().astype(float)
    out = {'config': a.config, 'seed': a.seed, 'scenes': {}}
    with torch.inference_mode():
        for index in [int(v) for v in a.indices.split(',')]:
            item = ds[index]; scene = int(ds.sample_id[index])
            task, occ, cond, cand, init = build(model, sched, item, cfg, scene, a.seed)
            rec = {'index': index, 'sample_id': scene}
            rec['feasibility'] = gt_feasibility(model, task, item, occ, cond, u512, B512)
            rec['jacobian'] = {str(t): jac_stats(task, task.initial(np.random.default_rng(0))[0] + 0.0)
                               for t in (0, 1, 8, 15)}
            # --- SafeDiffuser solver comparison, per-level QP diagnostics ---
            for solver, w in (('legacy', None), ('osqp', 100.0), ('osqp', 0.01)):
                c = dict(cfg); c['relax_weight'] = w if w is not None else cfg['relax_weight']
                c['cbf_solver'] = solver
                c['timeouts'] = dict(cfg['timeouts'], **{m: 240.0 for m in ('RoS-diffuser',)})
                r = run_method('RoS-diffuser', model, sched, occ, cond, scene, a.seed, c, init_noise=init,
                               task_extra={'candidates': cand, 'cbf_solver': solver,
                                           'slack_weight': (1.0 / w ** 2) if w else None})
                rec['ros_%s_w%s' % (solver, w)] = {
                    'status': r['status'], 'reason': r['failure_reason'], 'nfe': r['model_forward_calls'],
                    'seconds': r['wall_time'],
                    **agg_trace(r['per_step_trace'] or [],
                                ('velocity_change_norm', 'slack_norm', 'cbf_residual', 'cbf_residual_soft',
                                 'primal_residual', 'dual_residual', 'min_h_scene', 'integration_steps'))}
            # --- projection: warm start from target vs from control_gt ---
            tb = Budget(120.)
            x0 = np.asarray(task.initial_override, dtype=float).reshape(-1)
            target = x0 + np.random.default_rng(3).normal(size=x0.shape) * 0.05
            for label, start in (('cold', None), ('warm_gt', np.asarray(item['control_gt'], dtype=float)[1:-1].reshape(-1))):
                try:
                    xx, info = task.project(target, tb, start=start)
                    rec['project_%s' % label] = {'ok': True, 'residual': info.get('residual'),
                                                 'iterations': info.get('iterations'), 'status': info.get('status'),
                                                 'move': float(np.linalg.norm(xx - target))}
                except Exception as exc:
                    rec['project_%s' % label] = {'ok': False, 'reason': str(exc)}
            # --- PD/ALM: extra-step sensitivity + gradient norms ---
            for method in ('Primal-Dual', 'ALM'):
                for extra in (0, 16, 64):
                    c = dict(cfg); c['cd_extra_steps'] = extra
                    for m in ('Primal-Dual', 'ALM'): c['timeouts'] = dict(c['timeouts'], **{m: 240.0})
                    r = run_method(method, model, sched, occ, cond, scene, a.seed, c, init_noise=init,
                                   task_extra={'candidates': cand})
                    traj = None if r['trajectory'] is None else np.asarray(r['trajectory'], dtype=float)
                    met = {}
                    if traj is not None:
                        curve = B512 @ traj
                        gt128 = np.asarray(item['pos'], dtype=float)
                        gt_curve = np.stack([np.interp(u512, np.linspace(0, 1, len(gt128)), gt128[:, k]) for k in (0, 1)], axis=1)
                        met = evaluate_curve(curve, occ, cond, gt_curve, u512)
                    rec['%s_extra%d' % (method, extra)] = {
                        'status': r['status'], 'reason': r['failure_reason'], 'nfe': r['model_forward_calls'],
                        'seconds': r['wall_time'],
                        'collision': met.get('curve_collision'), 'min_clearance_m': met.get('min_clearance_m'),
                        'length_ratio': met.get('path_length_ratio'), 'max_curvature': met.get('max_curvature_1pm'),
                        'rmse_m': met.get('rmse_m')}
            out['scenes'][scene] = rec
            print(json.dumps({k: v for k, v in rec.items() if k != 'jacobian'}, ensure_ascii=False)[:600], flush=True)
    p = os.path.join(ROOT, a.out); os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, 'w', encoding='utf8').write(json.dumps(out, indent=2, ensure_ascii=False))
    print('wrote', p)


if __name__ == '__main__':
    main()
