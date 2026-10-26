"""Render one closed-loop planning frame: corridor, control polygon, raw vs safe curve.

    python scripts/debug/plot_one_frame.py --run demo_fast_205421 --index 6 \
        --out outputs/closed_loop/frame_inspect.png --window 70
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.wkb import loads as wkb_loads

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def scene_to_world(frame, xy):
    return frame.world(np.asarray(xy, dtype=float).reshape(-1, 2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='目录名，如 demo_fast_205421')
    ap.add_argument('--root', default='outputs/closed_loop')
    ap.add_argument('--index', type=int, default=6, help='第几个快照（按文件名排序）')
    ap.add_argument('--out', required=True)
    ap.add_argument('--window', type=float, default=70., help='绘图窗口边长（米）')
    ap.add_argument('--forward', type=float, default=0., help='窗口中心的前向偏移（米）')
    ap.add_argument('--config', default='configs/carla_online_closed_loop.yaml')
    args = ap.parse_args(argv)

    import yaml
    from src.carla_bridge.frame import SceneFrame
    from src.carla_bridge.planner_adapter import PlannerAdapter

    run = Path(args.root)/args.run
    snaps = sorted((run/'snapshots').glob('*.npz'))
    snap = snaps[min(args.index, len(snaps)-1)]
    z = np.load(snap, allow_pickle=True)
    frame = SceneFrame(np.asarray(z['anchor'], float), float(z['yaw']))
    occ, road, cond = z['occupancy'], z['road'], np.asarray(z['condition'], np.float32)

    cfg = yaml.safe_load((ROOT/args.config).read_text(encoding='utf8'))
    planner = PlannerAdapter(dict(cfg['planner'], visualize=True))
    request = dict(id=0, time=0., frame=frame, occupancy=occ, road=road,
                   road_area=wkb_loads(z['road_geometry_wkb'].tobytes()), condition=cond,
                   goal=frame.world(cond[1]), body_margin_m=1.2, interior_body_margin_m=1.55,
                   seed=cfg['scenario']['seed'])
    result = planner(request)
    viz = result['viz']

    # ---- ego-local display frame (forward = up)
    ego = frame.origin
    yaw = frame.yaw
    c, s = np.cos(yaw), np.sin(yaw)

    def to_local(xy):
        """-> (lateral, forward): forward is the frame x axis, lateral is frame y."""
        d = np.asarray(xy, float).reshape(-1, 2)-ego
        fwd = c*d[:, 0]+s*d[:, 1]
        lat = -s*d[:, 0]+c*d[:, 1]
        return np.stack([lat, fwd], axis=1)

    def L(scene_xy):
        return to_local(scene_to_world(frame, scene_xy))

    lim = args.window/2
    fig, ax = plt.subplots(figsize=(11, 11), dpi=130)
    ax.set_facecolor('#101216')
    ax.tick_params(colors='#dddddd')
    for sp in ax.spines.values(): sp.set_color('#888888')

    # occupancy (free = light) and approved lane band
    gy, gx = np.mgrid[0:occ.shape[0], 0:occ.shape[1]]
    pts = np.stack([(gx/(occ.shape[1]-1))*2-1, (gy/(occ.shape[0]-1))*2-1], axis=-1).reshape(-1, 2)
    local = L(pts).reshape(occ.shape[0], occ.shape[1], 2)
    ax.pcolormesh(local[:, :, 0], local[:, :, 1], np.where(occ == 0, 1.0, 0.0),
                  cmap='Greys', shading='auto', vmin=-0.35, vmax=1.0, zorder=1)
    ax.contour(local[:, :, 0], local[:, :, 1], (road == 0).astype(float), levels=[0.5],
               colors='#4a90d9', linewidths=0.9, zorder=2)

    # corridor cells
    for poly in viz['corridor']['polygons']:
        ax.add_patch(MplPolygon(L(poly), closed=True, facecolor='#ffb428', alpha=.16,
                                edgecolor='#ffb428', linewidth=1.0, zorder=3))
    # skeleton candidates + selected
    cand = viz['candidates']
    for i, ok in enumerate(cand['mask']):
        if ok:
            ax.plot(*L(cand['xy'][i]).T, color='#9aa0a6', lw=.8, alpha=.75, zorder=4)
    sel = viz['selected']
    n = int(cand['lengths'][sel])
    if n > 1:
        ax.plot(*L(cand['geometry'][sel][:n]).T, color='#4aa3ff', lw=2.0, zorder=5,
                label='selected topology')

    # curves and control polygons
    raw = L(viz['raw_curve']); safe = L(viz['safe_curve'])
    rawq = L(viz['raw_controls']); safeq = L(viz['safe_controls'])
    ax.plot(*raw.T, color='#ff60d0', lw=1.6, ls='--', zorder=6, label='network raw (before ALM)')
    ax.plot(*safe.T, color='#25e0ff', lw=2.4, zorder=7, label='after ALM (executed)')
    ax.plot(*rawq.T, color='#ff60d0', lw=.9, alpha=.6, zorder=6)
    ax.plot(*safeq.T, color='#25e0ff', lw=1.0, alpha=.85, zorder=7)
    ax.scatter(*safeq.T, s=26, color='#25c86a', zorder=8, label='control points (C=48)')
    ax.scatter(*safeq[0], s=90, marker='o', facecolor='none', edgecolor='#2ecc71', zorder=9)
    ax.scatter(*safeq[-1], s=90, marker='X', color='#ff5b5b', zorder=9)

    # ego footprint
    ego_local = np.array([[2.35, 1.05], [2.35, -1.05], [-2.35, -1.05], [-2.35, 1.05]])
    ax.add_patch(MplPolygon(ego_local, closed=True, facecolor='#ff9a2e', alpha=.9, zorder=10))

    ax.set_xlim(-lim, lim); ax.set_ylim(args.forward-lim, args.forward+lim)
    ax.set_aspect('equal')
    ax.set_xlabel('lateral / m', color='#dddddd'); ax.set_ylabel('forward / m', color='#dddddd')
    steps = viz['steps']
    ax.set_title('%s  frame %s\ncorridor %d cells, min overlap %.2f, topology #%d, '
                 'ALM steps %s' % (args.run, snap.stem, viz['corridor']['cells'],
                                   viz['corridor']['overlap_min'] or 0., sel,
                                   [round(s['delta'], 4) for s in steps]),
                 fontsize=9, color='#eeeeee')
    ax.grid(alpha=.12)
    ax.legend(loc='upper left', fontsize=8, framealpha=.2, labelcolor='w')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight', facecolor='#0b0d10')
    print('saved', out)

    meta = out.with_suffix('.json')
    meta.write_text(json.dumps(dict(run=args.run, snapshot=snap.stem,
                                    corridor_cells=viz['corridor']['cells'],
                                    steps=steps,
                                    controls_safe=viz['safe_controls'].tolist(),
                                    controls_raw=viz['raw_controls'].tolist()), indent=1), encoding='utf8')
    print('saved', meta)


if __name__ == '__main__':
    main()
