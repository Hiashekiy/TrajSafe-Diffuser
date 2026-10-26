"""对比不同去噪步数的轨迹质量（离线回放，不需要 CARLA）。

    python scripts/debug/compare_steps.py --run outputs/scenarios/sweep/v40 --steps 4 8 16
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def metrics(path_xy):
    p = np.asarray(path_xy, float)
    d1 = np.diff(p, axis=0)
    ds = np.linalg.norm(d1, axis=1)
    d2 = np.diff(p, 2, axis=0)
    # 曲率
    a = d1[:-1]
    b = d1[1:]
    cross = np.abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])
    denom = np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-9)
    ang = np.arccos(np.clip(cross/np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-9), -1, 1))
    curv = ang/np.maximum(ds[:-1], 1e-6)
    # 横向摆动（相对首尾连线）
    v = p[-1]-p[0]
    v = v/np.linalg.norm(v)
    n = np.array([-v[1], v[0]])
    lat = (p-p[0]) @ n
    return dict(rms2=float(np.sqrt((d2**2).sum(1).mean())),
                curv_max=float(curv.max()),
                radius_min=float(1.0/max(curv.max(), 1e-9)),
                lat_wander=float(np.ptp(lat)),
                length=float(ds.sum()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--steps', type=int, nargs='+', default=[4, 8, 16])
    ap.add_argument('--out')
    args = ap.parse_args(argv)

    from src.carla_bridge.frame import SceneFrame
    from src.carla_bridge.planner_adapter import PlannerAdapter

    run = Path(args.run)
    plan = np.load(run/'plan.npz', allow_pickle=True)
    frame = SceneFrame(np.asarray(plan['anchor'], float), float(plan['yaw']))
    cond = np.asarray(plan['condition'], np.float32)
    occ, road = plan['occupancy'], plan['road']
    goal = np.asarray(plan['goal'], float)

    fig, ax = plt.subplots(figsize=(6.4, 8.0), dpi=170)
    colours = {4: '#d1495b', 8: '#edae49', 16: '#00798c'}
    results = {}
    for steps in args.steps:
        adapter = PlannerAdapter(dict(
            model_id='K4P_c48_oneshot:best_task',
            config='configs/config_160k4p_c48_oneshot.yaml',
            dataset='160m-k4p-c48', device='cuda', steps=steps,
            region_stride=1, corridor_margin_m=1.0,
            corridor_axis_max_scene=0.035, visualize=True))
        t0 = time.perf_counter()
        out = adapter(dict(id=0, time=0.0, frame=frame, occupancy=occ, road=road,
                           condition=cond, goal=goal, seed=43))
        ms = 1000*(time.perf_counter()-t0)
        xy = out['path'].xy
        m = metrics(xy)
        m['plan_ms'] = ms
        m['alm'] = out['alm_status']
        results[steps] = m
        ax.plot(xy[:, 0], xy[:, 1], color=colours.get(steps, 'k'), lw=2.0,
                label='steps=%d  rms2=%.3f  Rmin=%.1fm  wander=%.1fm  %.0fms'
                      % (steps, m['rms2'], m['radius_min'], m['lat_wander'], ms))
        print('steps=%-3d rms2=%.4f curv_max=%.3f Rmin=%.1fm wander=%.2fm  %.0f ms'
              % (steps, m['rms2'], m['curv_max'], m['radius_min'], m['lat_wander'], ms), flush=True)

    ax.scatter(goal[0], goal[1], marker='X', s=110, color='#b03a2e', zorder=5, label='goal')
    ax.scatter(xy[0, 0], xy[0, 1], marker='o', s=90, facecolor='none',
               edgecolor='#22303c', zorder=5, label='ego')
    ax.set_aspect('equal')
    ax.grid(alpha=.25, lw=.6)
    ax.set_xlabel('x / m')
    ax.set_ylabel('y / m')
    ax.legend(fontsize=8, loc='best')
    ax.set_title('DDIM 步数 vs 轨迹质量（同一场景、同一 seed）', fontsize=10)
    out_png = Path(args.out or (run/'compare_steps.png'))
    fig.savefig(out_png, bbox_inches='tight', facecolor='white')
    print('saved', out_png)


if __name__ == '__main__':
    main()
