"""Original Chapter-4 per-method timing benchmark.

All five comparison methods are timed on the complete 420-sample test split.
TrajSafe acceleration is measured separately by the original interleaved
``scripts/ch4_perf_stride.py`` benchmark so that step reduction and region sharing
are not reimplemented here.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, '.')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                'scripts'))
import torch  # noqa: E402

from eval_chapter4 import DENSE_POINTS, evaluate_controls  # noqa: E402
from src.datasets.carla_spline_dataset import (CarlaSplineDataset,  # noqa: E402
                                               make_collate)
from src.diffusion.sampler import sample  # noqa: E402
from src.diffusion.schedule import NoiseSchedule  # noqa: E402
from src.geometry.sdf_field import build_occupancy_sdf  # noqa: E402
from src.utils.checkpoint import load_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402

A_CFG = 'configs/config_160k4p_c48_oneshot.yaml'
A_CKPT = 'outputs/models/oneshot_k4p_c48/ckpt/best_task.pt'
D_CFG = 'configs/config_160k4p_c48_d_std.yaml'
D_CKPT = 'outputs/models/oneshot_k4p_c48_d/ckpt/best_task.pt'
N_TIMING = 420


def frozen(mode):
    path = 'outputs/chapter4/sweep_%s.json' % mode
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)['winner']


def load(cfg_path, ckpt):
    cfg = load_config(cfg_path)
    model, _, report = load_model(cfg, ckpt, device='cuda', verbose=False)
    if report and (report['missing'] or report['adapted'] or report['unexpected']):
        raise SystemExit('checkpoint/config mismatch for %s' % ckpt)
    model.assert_shapes = False          # dev-time check, costs a sync per forward
    return cfg, model


def run(cfg, model, ds, collate, steps, guidance, sdf_extra=None, n=None,
        measure_metrics=True, want_pack=False):
    s2m = float(cfg['data'].get('scene_to_meter', 80.0))
    sch = NoiseSchedule(16, beta_schedule=cfg['diffusion'].get(
        'beta_schedule', 'squaredcos_cap_v2'))
    alm = dict(cfg.get('alm') or {})
    alm['guidance'] = guidance
    # NOTE: the four external processors read their knobs from sdf_config, NOT
    # from alm_config (the corridor ALM reads alm_config) -- passing them to the
    # wrong dict would silently run the default parameters.
    sdf_cfg = dict(sdf_extra or {})
    if guidance == 'alm':
        alm.update(sdf_extra or {})
    total = len(ds) if n is None else min(n, len(ds))
    wall = cpu = 0.0
    phase = dict(network_s=0.0, corridor_s=0.0, final_projection_s=0.0)
    metrics = []
    lp = 0
    for start in range(0, total, 32):
        rows = list(range(start, min(start + 32, total)))
        batch = collate([ds[i] for i in rows])
        args = (model, sch, batch['cond'].cuda(), batch['occupancy'].cuda(),
                batch['candidate_xy'].cuda(), batch['candidate_mask'].cuda(),
                batch['candidate_geometry'].cuda(),
                batch['candidate_geometry_lengths'].cuda())
        kw = dict(device='cuda', steps=steps, seed=0, seed_offset=start,
                  return_trace=False, alm_config=alm,
                  corridor_config=cfg.get('corridor'), guidance=guidance,
                  sdf_config=sdf_cfg)
        sample(*args, **kw)
        torch.cuda.synchronize()
        w0, c0 = time.perf_counter(), time.process_time()
        out = sample(*args, **kw)
        torch.cuda.synchronize()
        wall += time.perf_counter() - w0
        cpu += time.process_time() - c0
        for key in phase:
            phase[key] += out['timing'][key]
        if out.get('final_projection') is not None:
            lp += sum(1 for s in out['final_projection']['hard_project_status']
                      if s.startswith('lp_'))
        if measure_metrics:
            sdf = build_occupancy_sdf(batch['occupancy'].cuda(),
                                      scene_to_meter=s2m)
            metrics.append(evaluate_controls(
                model.bspline, out['control'], batch['occupancy'].cuda(),
                batch['cond'].cuda(), batch['pos'].cuda(), sdf,
                out.get('pack') if want_pack else None, batch, s2m))
    row = {'wall_ms': 1000.0 * wall / total, 'cpu_ms': 1000.0 * cpu / total,
           'network_ms': 1000.0 * phase['network_s'] / total,
           'corridor_ms': 1000.0 * phase['corridor_s'] / total,
           'final_projection_ms': 1000.0 * phase['final_projection_s'] / total,
           'n': total, 'lp_samples': lp}
    if measure_metrics:
        coll = sum(1 for m in metrics for c in m['collision'] if bool(c))
        feas = [float(v) for m in metrics for v in m['sdf_feasible']]
        rmse = [float(v) for m in metrics for v in m['rmse_m']]
        smooth = [float(v) for m in metrics for v in m['smoothness_m']]
        viol = [float(v) for m in metrics for v in m['sdf_violation_m']]
        row.update(collisions=coll, n_metrics=len(feas),
                   feasible_rate=sum(feas) / len(feas),
                   rmse_m=sum(rmse) / len(rmse),
                   smoothness_m=sum(smooth) / len(smooth),
                   mean_violation_m=sum(viol) / len(viol))
    return row


def main() -> int:
    a_cfg, a_model = load(A_CFG, A_CKPT)
    d_cfg, d_model = load(D_CFG, D_CKPT)

    print('=== A. 空载各方法计时（%d 条测试样本,16 步）' % N_TIMING, flush=True)
    ds_a = CarlaSplineDataset('test', a_cfg['data']['processed_root'],
                              geometry_points=1280, limit=N_TIMING)
    coll_a = make_collate(ds_a)
    ds_d = CarlaSplineDataset('test', d_cfg['data']['processed_root'],
                              geometry_points=1280, limit=N_TIMING)
    coll_d = make_collate(ds_d)
    rows = [
        ('Diffusion', d_cfg, d_model, ds_d, coll_d, 'none', {}),
        ('Penalty-guided', d_cfg, d_model, ds_d, coll_d, 'penalty', frozen('penalty')),
        ('Projected', d_cfg, d_model, ds_d, coll_d, 'projection', frozen('projection')),
        ('Primal-Dual', d_cfg, d_model, ds_d, coll_d, 'primal_dual', frozen('primal_dual')),
        ('SDF-ALM', d_cfg, d_model, ds_d, coll_d, 'alm_sdf', frozen('alm_sdf')),
        ('TrajSafe-Diffuser', a_cfg, a_model, ds_a, coll_a, 'alm',
         {'final_project': False}),
    ]
    for name, cfg, model, ds, coll, guidance, extra in rows:
        r = run(cfg, model, ds, coll, 16, guidance, extra, measure_metrics=False)
        print('RESULT_A %-26s wall %7.1f | cpu %7.1f | net %6.1f | '
              'opt %7.1f | proj %6.1f | LP %d/%d'
              % (name, r['wall_ms'], r['cpu_ms'], r['network_ms'],
                 r['corridor_ms'], r['final_projection_ms'], r['lp_samples'],
                 r['n']), flush=True)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
