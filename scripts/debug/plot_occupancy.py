"""把某一帧真正喂给模型的占据图画出来（占据 + 车道 mask + 规划轨迹 + 起终点）。

    python scripts/debug/plot_occupancy.py --run outputs/scenarios/S1_20261025_002513
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out')
    args = ap.parse_args(argv)

    from src.carla_bridge.frame import SceneFrame
    run = Path(args.run)
    z = np.load(run/'plan.npz', allow_pickle=True)
    occ, road = z['occupancy'], z['road']
    frame = SceneFrame(np.asarray(z['anchor'], float), float(z['yaw']))
    curve, goal = z['curve_world'], z['goal']

    def local(world_xy):
        d = np.asarray(world_xy, float).reshape(-1, 2)-frame.origin
        c, s = np.cos(frame.yaw), np.sin(frame.yaw)
        return np.stack([-s*d[:, 0]+c*d[:, 1], c*d[:, 0]+s*d[:, 1]], axis=1)   # (lateral, forward)

    res = occ.shape[0]
    gx, gy = np.meshgrid(np.arange(res), np.arange(res))
    scene = np.stack([(gx/(res-1))*2-1, (gy/(res-1))*2-1], axis=-1).reshape(-1, 2)
    world = frame.world(scene).reshape(res, res, 2)
    local_pts = local(world.reshape(-1, 2)).reshape(res, res, 2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), dpi=130)
    for ax, title in zip(axes, ('occupancy fed to the model (white = free)', 'occupancy + planned trajectory')):
        ax.set_facecolor('#0d0f12')
        # 自由=浅灰，禁止=深灰，车道 mask 用蓝线框
        ax.pcolormesh(local_pts[:, :, 0], local_pts[:, :, 1], np.where(occ == 0, 0.0, 1.0),
                      cmap='Greys', shading='auto', vmin=-0.2, vmax=1.0)
        ax.contour(local_pts[:, :, 0], local_pts[:, :, 1], (road == 0).astype(float),
                   levels=[0.5], colors='#3b7dd8', linewidths=1.0)
        ax.set_title(title, color='#e8e8e8', fontsize=11)
        ax.set_xlabel('lateral / m', color='#cccccc')
        ax.set_ylabel('forward / m', color='#cccccc')
        ax.tick_params(colors='#bbbbbb')
        for sp in ax.spines.values():
            sp.set_color('#666666')
        ax.set_xlim(-45, 45)
        ax.set_ylim(-45, 45)
        ax.set_aspect('equal')

    axes[1].plot(*local(curve).T, color='#25e0ff', lw=2.2, label='planned trajectory')
    axes[1].scatter(0, 0, s=90, marker='o', facecolor='none', edgecolor='#ff9a2e',
                    linewidths=2, label='ego start')
    axes[1].scatter(*local(goal[None])[0], s=110, marker='X', color='#ff5b5b', label='goal')
    axes[1].legend(loc='upper left', fontsize=8, labelcolor='w', framealpha=.2)

    free = float((occ == 0).mean())
    in_lane = float((road == 0).mean())
    along = local(curve)
    free_on_path = float(np.mean([occ[int(np.clip((local([[p[0], p[1]]])[0][1]+80)/0.625, 0, res-1)),
                                      int(np.clip((local([[p[0], p[1]]])[0][0]+80)/0.625, 0, res-1))] == 0
                                 for p in curve[::4]]))
    fig.suptitle('occupancy diagnostic | free area %.1f%% | lane area %.1f%% | path points on free %.1f%%'
                 % (100*free, 100*in_lane, 100*free_on_path), color='#f0f0f0', fontsize=11)
    out = Path(args.out or (run/'occupancy_diagnostic.png'))
    fig.savefig(out, bbox_inches='tight', facecolor='#0b0d10')
    print('saved', out)
    print('自由面积 %.3f 车道面积 %.3f 轨迹自由率 %.3f' % (free, in_lane, free_on_path))


if __name__ == '__main__':
    main()
