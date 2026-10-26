"""规划结果诊断图：占据图 + 骨架 + 椭圆 + 凸区域(走廊) + 轨迹 + 终点。

    python scripts/debug/show_plan.py --run outputs/scenarios/S1_20261025_003532
"""
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
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)

    from src.carla_bridge.frame import SceneFrame
    run = Path(args.run)
    z = np.load(run/'plan.npz', allow_pickle=True)
    occ, road = z['occupancy'], z['road']
    frame = SceneFrame(np.asarray(z['anchor'], float), float(z['yaw']))

    def loc(points):
        d = np.asarray(points, float).reshape(-1, 2)-frame.origin
        c, s = np.cos(frame.yaw), np.sin(frame.yaw)
        return np.stack([-s*d[:, 0]+c*d[:, 1], c*d[:, 0]+s*d[:, 1]], axis=1)

    def locw(points):          # scene -> world -> 自车局部
        return loc(frame.world(np.asarray(points, float).reshape(-1, 2)))

    res = occ.shape[0]
    gx, gy = np.meshgrid(np.arange(res), np.arange(res))
    scene = np.stack([(gx/(res-1))*2-1, (gy/(res-1))*2-1], axis=-1).reshape(-1, 2)
    gl = loc(frame.world(scene)).reshape(res, res, 2)

    fig, axes = plt.subplots(1, 2, figsize=(17, 9), dpi=110,
                             gridspec_kw=dict(width_ratios=[1.15, 1]))
    for ax, lim in zip(axes, [((-40, 40), (-40, 40)), ((-12, 12), (-5, 62))]):
        ax.set_facecolor('#0e1014')
        ax.pcolormesh(gl[:, :, 0], gl[:, :, 1], np.where(occ == 0, 0.0, 1.0),
                      cmap='Greys', shading='auto', vmin=-0.15, vmax=1.0, zorder=1)
        ax.contour(gl[:, :, 0], gl[:, :, 1], (road == 0).astype(float), levels=[0.5],
                   colors='#3b78d8', linewidths=.9, zorder=2)

        # 骨架候选（灰）与选中拓扑（蓝）
        cand_xy, cand_mask = z['cand_xy'], z['cand_mask']
        for i, ok in enumerate(cand_mask):
            if ok:
                ax.plot(*loc(cand_xy[i]).T, color='#8b9098', lw=1.1, alpha=.9,
                        zorder=3, label='skeleton candidates' if i == 0 else None)
        sel = int(z['selected'])
        geom, glen = z['cand_geom'], z['cand_len']
        if sel >= 0 and geom.ndim == 3 and glen[sel] > 1:
            dense = loc(geom[sel][:int(glen[sel])])
            ax.plot(*dense.T, color='#3fa9ff', lw=2.4, ls='--', zorder=4, label='selected topology')

        # 凸区域（走廊）
        for k, poly in enumerate(z['corridors']):
            pts = loc(np.asarray(poly))
            ax.add_patch(MplPolygon(pts, closed=True, facecolor='#ffb028', alpha=.10,
                                    edgecolor='#ff8f00', linewidth=1.6, zorder=5,
                                    label='convex corridor cells' if k == 0 else None))

        # 椭圆（沿路径的安全椭圆）
        ctr = locw(z['ellipse_center'])
        ab = np.asarray(z['ellipse_ab']).reshape(-1, 2)
        for k, ((cx, cy), (a, b)) in enumerate(zip(ctr, ab)):
            if k % 6:                       # 抽稀：每 6 个锚点画一个（半轴仅 0.2 m，画满会糊成一片）
                continue
            ax.add_patch(Ellipse((cx, cy), 2*float(b), 2*float(a), facecolor='#00e5ff',
                                 edgecolor='#00b8d4', linewidth=.8, alpha=.85, zorder=6))

        # 轨迹
        ax.plot(*loc(z['curve_world']).T, color='#28e070', lw=3.0, zorder=8, label='planned trajectory')
        ax.scatter(0, 0, s=130, marker='o', facecolor='none', edgecolor='#ff9a2e', linewidths=2.4,
                   zorder=9, label='ego start')
        g = loc(z['goal'][None])[0]
        ax.scatter(*g, s=190, marker='X', color='#ff3b3b', zorder=9, label='goal')
        ax.annotate('GOAL', g, textcoords='offset points', xytext=(10, 10), color='#ff5b5b',
                    fontsize=11, weight='bold')
        ax.set_xlim(*lim[0])
        ax.set_ylim(*lim[1])
        ax.set_aspect('equal')
        ax.set_xlabel('lateral / m', color='#cccccc')
        ax.set_ylabel('forward / m', color='#cccccc')
        ax.tick_params(colors='#bbbbbb')
        for sp in ax.spines.values():
            sp.set_color('#555555')
        ax.grid(alpha=.1, color='#777777')

    axes[0].set_title('整幅 160×160 m 输入（模型看到的就是这个）', color='#eeeeee', fontsize=11)
    axes[1].set_title('局部放大：骨架 / 椭圆 / 走廊 / 轨迹 / 障碍', color='#eeeeee', fontsize=11)
    axes[1].legend(loc='upper right', fontsize=7.5, labelcolor='w', framealpha=.3)
    fig.savefig(Path(args.out or (run/'plan_result.png')), bbox_inches='tight', facecolor='#0b0d10')
    print('saved', Path(args.out or (run/'plan_result.png')))


if __name__ == '__main__':
    main()
