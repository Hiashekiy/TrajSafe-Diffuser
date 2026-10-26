"""Original interleaved benchmark for the two existing acceleration knobs.

No acceleration algorithm is implemented here.  The script only exercises the
existing sampler's ``steps`` argument and the existing safety-corridor
``region_stride`` argument.  The four paper configurations are interleaved by
chunk so that CPU noise affects them equally:

    (16 steps, stride 1), (4 steps, stride 1),
    (16 steps, stride 8), (4 steps, stride 8).

Terminal hard projection is disabled because it is no longer part of the
reported method.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, '.')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'scripts'))
import torch  # noqa: E402

from eval_chapter4 import evaluate_controls  # noqa: E402
from src.datasets.carla_spline_dataset import (CarlaSplineDataset,  # noqa: E402
                                               make_collate)
from src.diffusion.sampler import sample  # noqa: E402
from src.diffusion.schedule import NoiseSchedule  # noqa: E402
from src.geometry.sdf_field import build_occupancy_sdf  # noqa: E402
from src.utils.checkpoint import load_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402

CFG = 'configs/config_160k4p_c48_oneshot.yaml'
CKPT = 'outputs/models/oneshot_k4p_c48/ckpt/best_task.pt'
CASES = [(16, 1), (4, 1), (16, 8), (4, 8)]
CHUNK = 32
#: optional smoke-test limit (argv[1]); 0 = the whole test split
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0


class Acc:
    def __init__(self):
        self.wall = self.cpu = 0.0
        self.corr = self.proj = self.net = self.guard = 0.0
        self.n = self.coll = self.feas = self.feas_n = 0
        self.rmse = self.smooth = 0.0
        self.cells = self.bridges = self.fails = self.seen = 0

    def add(self, wall, cpu, out, metrics):
        self.wall += wall
        self.cpu += cpu
        self.n += len(metrics['collision'])
        t = out['timing']
        self.corr += t['corridor_s']
        self.proj += t['final_projection_s']
        self.net += t['network_s']
        self.guard += t['guard_s']
        for corridor in out['corridors']:
            self.seen += 1
            if not corridor:
                self.fails += 1
                continue
            self.cells += int(corridor.get('base_cell_count') or 0)
            self.bridges += int(corridor.get('bridge_cell_count') or 0)
        self.coll += sum(1 for c in metrics['collision'] if bool(c))
        for f in metrics['sdf_feasible']:
            self.feas_n += 1
            self.feas += int(bool(f))
        self.rmse += sum(float(v) for v in metrics['rmse_m'])
        self.smooth += sum(float(v) for v in metrics['smoothness_m'])


def main() -> int:
    cfg = load_config(CFG)
    ds = CarlaSplineDataset('test', cfg['data']['processed_root'],
                            geometry_points=1280, limit=LIMIT or None)
    collate = make_collate(ds)
    s2m = float(cfg['data'].get('scene_to_meter', 80.0))
    sch = NoiseSchedule(16, beta_schedule=cfg['diffusion'].get(
        'beta_schedule', 'squaredcos_cap_v2'))
    model, _, _ = load_model(cfg, CKPT, device='cuda', verbose=False)
    model.assert_shapes = False
    sdf_margin = float((cfg.get('sdf') or {}).get(
        'map_margin', cfg['alm'].get('map_margin', 0.0)))

    def corridor_cfg(k):
        # ``sample`` forwards this dict to ``build_safety_corridor(config=...)``,
        # so the knob belongs at the top level of the corridor section.
        c = dict(cfg.get('corridor') or {})
        c['region_stride'] = k
        return c

    def build_args(batch):
        return (model, sch, batch['cond'].cuda(), batch['occupancy'].cuda(),
                batch['candidate_xy'].cuda(), batch['candidate_mask'].cuda(),
                batch['candidate_geometry'].cuda(),
                batch['candidate_geometry_lengths'].cuda())

    def run(args, steps, k, start, batch):
        alm = dict(cfg['alm'])
        alm['guidance'] = 'alm'
        alm['final_project'] = False
        torch.cuda.synchronize()
        w0, c0 = time.perf_counter(), time.process_time()
        out = sample(*args, device='cuda', steps=steps, seed=0,
                     seed_offset=start, return_trace=False, alm_config=alm,
                     corridor_config=corridor_cfg(k), guidance='alm')
        torch.cuda.synchronize()
        wall, cpu = time.perf_counter() - w0, time.process_time() - c0
        sdf = build_occupancy_sdf(batch['occupancy'].cuda(),
                                  scene_to_meter=s2m, map_margin=sdf_margin)
        metrics = evaluate_controls(model.bspline, out['control'],
                                    batch['occupancy'].cuda(),
                                    batch['cond'].cuda(), batch['pos'].cuda(),
                                    sdf, None, batch, s2m)
        return wall, cpu, out, metrics

    accs = {case: Acc() for case in CASES}
    n_chunk = (len(ds) + CHUNK - 1) // CHUNK
    first = None
    for ci, start in enumerate(range(0, len(ds), CHUNK)):
        rows = list(range(start, min(start + CHUNK, len(ds))))
        batch = collate([ds[i] for i in rows])
        args = build_args(batch)
        if first is None:                       # one warm-up per paper case
            first = batch
            for steps, k in CASES:
                run(args, steps, k, start, batch)
        for steps, k in CASES:
            wall, cpu, out, metrics = run(args, steps, k, start, batch)
            accs[(steps, k)].add(wall, cpu, out, metrics)
        if ci % 3 == 0:
            print('progress chunk %d/%d' % (ci + 1, n_chunk), flush=True)

    base = None
    for steps, k in CASES:
        a = accs[(steps, k)]
        n = max(a.n, 1)
        wall = a.wall * 1000 / n
        if base is None:
            base = wall
        expected_cells = 128 if k == 1 else 17
        actual_cells = a.cells / max(a.seen, 1)
        if abs(actual_cells - expected_cells) > 1e-6:
            raise RuntimeError('steps=%d stride=%d expected %d base cells, got %.1f'
                               % (steps, k, expected_cells, actual_cells))
        print('RESULT steps=%d stride=%d  wall %6.1f ms/traj (%.2fx) | cpu %6.1f | '
              'corridor %5.1f | finalProj %5.1f | net %5.1f | guard %4.1f | '
              'cells %.1f bridges %.2f fails %d | coll %d/%d | feas %6.2f%% | '
              'rmse %.2f | smooth %.3f'
              % (steps, k, wall, base / wall, a.cpu * 1000 / n, a.corr * 1000 / n,
                 a.proj * 1000 / n, a.net * 1000 / n, a.guard * 1000 / n,
                 a.cells / max(a.seen, 1), a.bridges / max(a.seen, 1), a.fails,
                 a.coll, a.feas_n, 100 * a.feas / max(a.feas_n, 1),
                 a.rmse / max(a.feas_n, 1), a.smooth / max(a.feas_n, 1)),
              flush=True)
    print('ALL DONE', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
