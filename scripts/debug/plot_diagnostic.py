"""占据图 + 骨架候选 + 选中拓扑 + 椭圆 + 走廊 + 轨迹 + 终点（诊断图）。"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon as MplPolygon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out')
    ap.add_argument('--forward', type=float, nargs=2, default=[-35., 75.])
    ap.add_argument('--lateral', type=float, nargs=2, default=[-25., 25.])
    args = ap.parse_args(argv)

    from src.carla_bridge.frame import SceneFrame
    run = Path(args.run)
    z = np.load(run/'plan.npz', allow_pickle=True)
    occ, road = z['occupancy'], z['road']
    frame = SceneFrame(np.asarray(z['anchor'], float), float(z['yaw']))
    curve, goal = z['curve_world'], z['goal']

    def local(points):
        d = np.asarray(points, float).reshape(-1, 2)-frame.origin
        c, s = np.cos(frame.yaw), np.sin(frame.yaw)
        return np.stack([-s*d[:, 0]+c*d[:, 1], c*d[:, 0]+s*d[:, 1]], axis=1)

    res = occ.shape[0]
    gx, gy = np.meshgrid(np.arange(res), np.arange(res))
    scene = np.stack([(gx/(res-1))*2-1, (gy/(res-1))*2-1], axis=-1).reshape(-1, 2)
    grid_local = local(frame.world(scene)).reshape(res, res, 2)

    fig, ax = plt.subplots(figsize=(11, 11), dpi=125)
    ax.set_facecolor('#0d0f12')
    ax.pcolormesh(grid_local[:, :, 0], grid_local[:, :, 1], np.where(occ == 0, 0.0, 1.0),
                  cmap='Greys', shading='auto', vmin=-0.2, vmax=1.0)
    ax.contour(grid_local[:, :, 0], grid_local[:, :, 1], (road == 0).astype(float),
               levels=[0.5], colors='#2f6fd0', linewidths=1.0)

    # 骨架候选（灰）与选中拓扑（蓝）
    cand_xy = z['cand_xy'] if 'cand_xy' in z else np.zeros((0, 2))
    cand_mask = z['cand_mask'] if 'cand_mask' in z else np.zeros(0, bool)
    if cand_xy.ndim == 3:
        for i, ok in enumerate(cand_mask):
            if ok:
                ax.plot(*local(cand_xy[i]).T, color='#7f858c', lw=1.0, alpha=.85)
    sel = int(z['selected']) if 'selected' in z else -1
    geom, glen = z['cand_geom'], z['cand_len']
    if sel >= 0 and geom.ndim == 3 and len(glen) > sel and glen[sel] > 1:
        dense = geom[sel][:int(glen[sel])]
        ax.plot(*local(dense).T, color='#4aa3ff', lw=2.2, label='selected topology')

    # 走廊格子（琥珀）
    if 'corridors' in z:
        for poly in z['corridors']:
            pts = local(np.asarray(poly))
            ax.add_patch(MplPolygon(pts, closed=True, facecolor='#ffb428', alpha=.12,
                                    edgecolor='#ffb428', linewidth=.9, zorder=4))

    # 椭圆（青色轮廓）
    if 'ellipse_center' in z and z['ellipse_center'].size:
        ctr = local(np.asarray(z['ellipse_center']).reshape(-1, 2))
        ab = np.asarray(z['ellipse_ab']).reshape(-1, 2)
        for (cx, cy), (a, b) in zip(ctr, ab):
            ax.add_patch(Ellipse((cx, cy), width=2*float(b), height=2*float(a),
                                 angle=0.0, facecolor='none', edgecolor='#25c8ff',
                                 linewidth=1.0, alpha=.9, zorder=5))

    ax.plot(*local(curve).T, color='#25e0ff', lw=2.6, zorder=7, label='planned trajectory')
    ax.scatter(0, 0, s=110, marker='o', facecolor='none', edgecolor='#ff9a2e',
               linewidths=2.2, zorder=8, label='ego start')
    g = local(goal[None])[0]
    ax.scatter(*g, s=150, marker='X', color='#ff4d4d', zorder=8, label='goal')
    ax.annotate('GOAL', g, textcoords='offset points', xytext=(8, 8), color='#ff4d4d',
                fontsize=10, weight='bold')

    ax.set_xlim(*args.lateral)
    ax.set_ylim(*args.forward)
    ax.set_aspect('equal')
    ax.set_xlabel('lateral / m', color='#cccccc')
    ax.set_ylabel('forward / m', color='#cccccc')
    ax.tick_params(colors='#bbbbbb')
    for sp in ax.spines.values():
        sp.set_color('#555555')
    ax.legend(loc='upper left', fontsize=8, labelcolor='w', framealpha=.25)
    ax.grid(alpha=.1, color='#666666')
    ax.set_title('occupancy | grey=skeleton candidates  blue=selected  amber=corridor  cyan=ellipses  '
                 'thick cyan=trajectory', color='#eeeeee', fontsize=9)
    out = Path(args.out or (run/'diagnostic.png'))
    fig.savefig(out, bbox_inches='tight', facecolor='#0b0d10')
    print('saved', out)


if __name__ == '__main__':
    main()
