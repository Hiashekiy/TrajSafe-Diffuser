"""A/B the batched polygon extraction, back to back in one process.

``vertices_pairwise_batch`` is monkey-patched to a loop of individual
``halfspaces_to_vertices`` calls, which is EXACTLY the pre-batching call pattern
(one call per base cell, one per consecutive intersection set).  Both arms then
run on the same samples with the same machine state, so the ratio is meaningful
even though the absolute numbers depend on CPU load.
"""
import sys
import time

sys.path.insert(0, '.')
import torch  # noqa: E402

import src.geometry.safety_corridor as SC  # noqa: E402
from src.datasets.carla_spline_dataset import (CarlaSplineDataset,  # noqa: E402
                                               make_collate)
from src.diffusion.sampler import sample  # noqa: E402
from src.diffusion.schedule import NoiseSchedule  # noqa: E402
from src.geometry.convex_region import halfspaces_to_vertices  # noqa: E402
from src.utils.checkpoint import load_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 32


def slow_batch(A_list, b_list, **kwargs):
    """The pre-batching behaviour: one polygon extraction per set."""
    return [halfspaces_to_vertices(a, b) for a, b in zip(A_list, b_list)]


def main() -> int:
    fast = SC.vertices_pairwise_batch
    cfg = load_config('configs/config_160k4p_c48_oneshot.yaml')
    ds = CarlaSplineDataset('test', cfg['data']['processed_root'],
                            geometry_points=1280, limit=N)
    coll = make_collate(ds)
    sch = NoiseSchedule(16, beta_schedule=cfg['diffusion'].get(
        'beta_schedule', 'squaredcos_cap_v2'))
    model, _, _ = load_model(cfg, 'outputs/models/oneshot_k4p_c48/ckpt/best_task.pt',
                             device='cuda', verbose=False)
    model.assert_shapes = False
    alm = dict(cfg['alm'])
    alm['guidance'] = 'alm'
    kw = dict(device='cuda', steps=16, seed=0, seed_offset=0, return_trace=False,
              alm_config=alm, corridor_config=cfg.get('corridor'), guidance='alm')

    results = {}
    for tag in ('slow', 'fast', 'slow', 'fast'):
        SC.vertices_pairwise_batch = slow_batch if tag == 'slow' else fast
        times = []
        valid = 0
        for start in range(0, len(ds), 16):
            rows = list(range(start, min(start + 16, len(ds))))
            batch = coll([ds[i] for i in rows])
            args = (model, sch, batch['cond'].cuda(), batch['occupancy'].cuda(),
                    batch['candidate_xy'].cuda(), batch['candidate_mask'].cuda(),
                    batch['candidate_geometry'].cuda(),
                    batch['candidate_geometry_lengths'].cuda())
            sample(*args, **kw)                      # warm
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = sample(*args, **kw)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) / len(rows))
            valid += sum(1 for c in out['corridors'] if c and c.get('valid'))
        results.setdefault(tag, []).append(
            (min(times), sum(times) / len(times), valid))
    for tag in ('slow', 'fast'):
        best = min(v[0] for v in results[tag])
        valid = max(v[2] for v in results[tag])
        print('RESULT %-5s best %.1f ms/traj | mean %.1f ms/traj | corridors valid %d/%d'
              % (tag, best * 1000, 1000 * sum(v[1] for v in results[tag]) / len(results[tag]),
                 valid, N), flush=True)
    a = min(v[0] for v in results['slow'])
    b = min(v[0] for v in results['fast'])
    print('RESULT speedup (per-cell calls -> batched): %.2fx' % (a / b), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
