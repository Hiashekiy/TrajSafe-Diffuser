"""Is the level-0 projection failure a solver-travel problem or a genuinely far target?

The warm start cannot help Projected/PDM because the very FIRST level has no previous
feasible point, and that is where they abort. This script separates the two hypotheses on
one scene:

  H1 solver cannot travel : a feasible spline DOES exist close to the target, the local
                            SQP just cannot get there -> a two-stage move (per-point
                            free-space clip in CURVE space, then a least-squares fit back
                            to controls) should land in the feasible set;
  H2 target is genuinely far : even the two-stage move cannot reach feasibility, i.e. the
                            nearest feasible spline is far in control space.

The two-stage move is a DIAGNOSTIC ONLY (it is not a paper algorithm); it is never used as
a solution, only to decide which hypothesis holds.

Also reports the projection's own displacement and the residual plateaus, so "stalled"
versus "converged" is distinguishable.
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
from src.geometry.bspline import numpy_fit_curve_to_controls
from experiments.safety_baselines.algorithms.runner import initial_seed
from experiments.safety_baselines.algorithms.solvers import Budget
from experiments.safety_baselines.adapters.sc48_control import SC48ControlTask
from experiments.safety_baselines.adapters.constraints import MapConstraint


def clip_curve_to_free(points, occ, margin=0.004):
    """Move each curve point out of the occupancy along the SDF gradient (paper-free step)."""
    field = MapConstraint(occ, extent=2., radius=0.)
    h, grad = field.query(points)
    out = np.asarray(points, dtype=float).copy()
    bad = h < margin
    out[bad] = out[bad] + grad[bad] * (margin - h[bad])[:, None]
    return out, int(bad.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='experiments/safety_baselines/configs/sc48_native16_vertexall.json')
    ap.add_argument('--base-config', default=os.path.join(LEGACY, 'configs/config_160k4p_c48_s_nofb.yaml'))
    ap.add_argument('--indices', default='0,163,407')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='experiments/safety_baselines/results/sc48_phase_b_review/reachability.json')
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
    out = {'config': a.config, 'seed': a.seed, 'scenes': {}}
    with torch.inference_mode():
        for index in [int(v) for v in a.indices.split(',')]:
            item = ds[index]; scene = int(ds.sample_id[index])
            occ = np.asarray(item['occupancy'][0].numpy(), dtype=bool); cond = np.asarray(item['cond'], dtype=float)
            cand = dict(candidate_xy=np.asarray(item['candidate_xy']), candidate_mask=np.asarray(item['candidate_mask']),
                        candidate_geometry=np.asarray(item['candidate_geometry']),
                        geometry_lengths=np.asarray(item['candidate_geometry_lengths']))
            task = SC48ControlTask(model, sched, occ, cond, cfg['constraint_points'], 0., 'bilinear', candidates=cand)
            task.projection_maxiter = cfg['projection_maxiter']; task.projection_tol = cfg['projection_tol_scene']
            rec = {'index': index, 'sample_id': scene, 'levels': {}}
            for t in (15, 8, 0):
                # the state a CD/PDM loop would try to project at this level: the model's own
                # noisy sample of the clean prediction plus the level's noise scale
                init = np.random.default_rng(initial_seed(scene, a.seed)).normal(size=(model.num_controls, 2))
                init[[0, -1]] = cond
                x = np.asarray(init, dtype=float)[1:-1].reshape(-1)
                clean = task.clean(x, t)[0]
                target = clean + np.random.default_rng(7).normal(size=clean.shape) * np.sqrt(1 - task.ab[t]) / np.sqrt(task.ab[t]) * 0.1
                r = {}
                # (a) the local SQP that Projected/PDM actually call
                try:
                    xx, info = task.project(target, Budget(120.), start=None)
                    r['sqp'] = {'ok': True, 'residual': info.get('residual'), 'iterations': info.get('iterations'),
                                'move': float(np.linalg.norm(xx - target))}
                except Exception as exc:
                    r['sqp'] = {'ok': False, 'reason': str(exc)}
                # (b) two-stage diagnostic: clip in CURVE space, fit controls, then project
                curve = (task.B @ task.unpack(target))
                clipped, nbad = clip_curve_to_free(curve, occ)
                q_fit = numpy_fit_curve_to_controls(model.bspline.knots.detach().cpu().numpy(),
                                                    model.num_controls, model.bspline.degree,
                                                    clipped, cond[0], cond[1])
                x_fit = np.asarray(q_fit, dtype=float)[1:-1].reshape(-1)
                g_fit, _ = task.constraints(x_fit)
                try:
                    xx2, info2 = task.project(x_fit, Budget(120.), start=x_fit)
                    r['two_stage'] = {'clipped_points': nbad, 'fit_max_constraint': float(np.max(g_fit)),
                                      'fit_feasible': bool(np.max(g_fit) <= 1e-9),
                                      'polish_ok': True, 'polish_residual': info2.get('residual'),
                                      'distance_fit_to_target': float(np.linalg.norm(x_fit - target))}
                except Exception as exc:
                    r['two_stage'] = {'clipped_points': nbad, 'fit_max_constraint': float(np.max(g_fit)),
                                      'fit_feasible': bool(np.max(g_fit) <= 1e-9), 'polish_ok': False,
                                      'polish_reason': str(exc),
                                      'distance_fit_to_target': float(np.linalg.norm(x_fit - target))}
                rec['levels'][str(t)] = r
                sqp_msg = str(r['sqp'].get('residual', r['sqp'].get('reason', '')))
                print(scene, 't=%d' % t, 'sqp', r['sqp'].get('ok'), sqp_msg[:70],
                      '| two_stage fit_max=%.4f feasible=%s dist=%.3f' % (r['two_stage']['fit_max_constraint'],
                                                                         r['two_stage']['fit_feasible'],
                                                                         r['two_stage']['distance_fit_to_target']), flush=True)
            out['scenes'][scene] = rec
    p = os.path.join(ROOT, a.out); os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, 'w', encoding='utf8').write(json.dumps(out, indent=2, ensure_ascii=False))
    print('wrote', p)


if __name__ == '__main__':
    main()
