"""把 CARLA 俯视原图 + 规划结果合成论文级插图（标注全部在 matplotlib 里画）。

    python scripts/render_figure.py --run outputs/scenarios/<run> --out outputs/scenarios/S1_fig.png
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

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---- 论文配色（低饱和，打印友好）----
C_CORR_F, C_CORR_E = '#8fb3d9', '#3a6ea5'
C_BAND_F, C_BAND_E = '#8fd19e', '#2f7d4f'
C_PATH = '#f5f5f5'
C_OBST = '#c0392b'
C_START, C_GOAL = '#e8871e', '#c0392b'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out')
    ap.add_argument('--dpi', type=int, default=200)
    ap.add_argument('--band-clip', type=float, default=7.5)
    args = ap.parse_args(argv)

    from shapely.geometry import Polygon as SPoly, MultiPolygon as SMulti, LineString
    from shapely.ops import unary_union
    from src.carla_bridge.frame import SceneFrame

    run = Path(args.run)
    view = np.load(run/'view.npz', allow_pickle=True)
    plan = np.load(run/'plan.npz', allow_pickle=True)
    rgb = view['rgb']
    inv = np.asarray(view['inverse'], float)
    ground = float(view['ground_z'])
    half_w = float(view['half_width'])
    frame = SceneFrame(np.asarray(plan['anchor'], float), float(plan['yaw']))
    path = plan['curve_world']
    goal = plan['goal']
    metrics = json.loads((run/'summary.json').read_text(encoding='utf8'))

    h, w = rgb.shape[:2]

    def px(xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        hom = np.column_stack([xy, np.full(len(xy), ground), np.ones(len(xy))])
        cam = hom @ inv.T
        f = w/2.0
        return np.column_stack([w/2 + f*cam[:, 1]/cam[:, 0], h/2 - f*cam[:, 2]/cam[:, 0]])

    def scene_px(poly_scene):
        return px(frame.world(np.asarray(poly_scene)))

    fig, ax = plt.subplots(figsize=(9.2, 7.4), dpi=args.dpi)
    ax.imshow(rgb)
    ax.set_axis_off()

    # 1) 安全走廊：格子与路径±clip 带求交后合并成一条轮廓
    band_clip = LineString(path).buffer(args.band_clip, cap_style=2, join_style=2)
    parts = []
    for poly in plan['corridors']:
        sp = SPoly(frame.world(np.asarray(poly)))
        if not sp.is_valid:
            sp = sp.buffer(0)
        inter = sp.intersection(band_clip)
        if not inter.is_empty:
            parts.extend(inter.geoms if isinstance(inter, SMulti) else [inter])
    if parts:
        sil = unary_union(parts)
        for part in (sil.geoms if isinstance(sil, SMulti) else [sil]):
            ax.add_patch(MplPolygon(scene_px(part.exterior.coords[:-1]), closed=True,
                                    facecolor=C_CORR_F, alpha=.22, edgecolor=C_CORR_E,
                                    linewidth=1.1, zorder=3))

    # 2) 车身扫掠带
    from src.carla_bridge.path_profile import Path as GeomPath
    geom = GeomPath.build(path)
    half = half_w + 0.15
    left, right, cross = [], [], []
    for s in np.arange(0.0, geom.s[-1], 1.0):
        p, yaw = geom.pose(float(s))
        n = np.array([-np.sin(yaw), np.cos(yaw)])
        left.append(p + n*half)
        right.append(p - n*half)
    for s in np.arange(0.0, geom.s[-1], 5.0):
        p, yaw = geom.pose(float(s))
        n = np.array([-np.sin(yaw), np.cos(yaw)])
        cross.append((p + n*half, p - n*half))
    left, right = np.array(left), np.array(right)
    band = np.vstack([left, right[::-1]])
    ax.add_patch(MplPolygon(px(band), closed=True, facecolor=C_BAND_F, alpha=.34,
                            edgecolor='none', zorder=4))
    for line in (left, right):
        ax.plot(*px(line).T, color=C_BAND_E, lw=1.6, zorder=5, solid_capstyle='round')
    for a, b in cross:
        seg = px(np.array([a, b]))
        ax.plot(*seg.T, color=C_BAND_E, lw=.7, alpha=.75, zorder=5)

    # 3) 规划轨迹
    ax.plot(*px(path).T, color=C_PATH, lw=2.0, zorder=6, solid_capstyle='round')

    # 4) 障碍物
    for corners in view['obstacle_corners']:
        ax.add_patch(MplPolygon(px(np.asarray(corners)), closed=True, facecolor='none',
                                edgecolor=C_OBST, linewidth=1.8, zorder=7))

    # 5) 起点 / 终点
    for point, colour, label in ((path[0], C_START, 'START'), (goal, C_GOAL, 'GOAL')):
        p2 = px(point[None])[0]
        ax.plot(*p2, marker='o', ms=11, mfc='none', mec=colour, mew=2.2, zorder=8)
        ax.annotate(label, p2, textcoords='offset points', xytext=(13, -6), color=colour,
                    fontsize=11, weight='bold', zorder=9)

    # 6) 比例尺（10 m）
    p0 = px(path[len(path)//2][None])[0]
    p1 = px((path[len(path)//2] + np.array([10.0, 0.0]))[None])[0]
    bar_px = float(np.linalg.norm(p1[:2]-p0[:2]))
    x0, y0 = w*0.055, h*0.93
    ax.add_patch(Rectangle((x0, y0), bar_px, 9, facecolor='white', edgecolor='#333', lw=1, zorder=10))
    ax.text(x0, y0-14, '10 m', color='white', fontsize=10, zorder=10,
            bbox=dict(boxstyle='square,pad=0.15', fc='#00000066', ec='none'))

    # 7) 图例与标题
    handles = [Line2D([], [], color=C_CORR_F, lw=7, alpha=.55, label='safety corridor'),
               Line2D([], [], color=C_BAND_F, lw=7, alpha=.75, label='body sweep (2 m wide body)'),
               Line2D([], [], color=C_PATH, lw=2.2, label='planned trajectory'),
               Line2D([], [], color=C_OBST, lw=2.0, label='static obstacle')]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.015),
              ncol=4, frameon=False, fontsize=9.5)
    fp = metrics['footprint']
    ax.set_title('%s  %s\nplan %.0f ms   corridor %s cells   clearance %s m   footprint %s'
                 % (metrics['scenario'], metrics['name'], metrics['plan_ms'],
                    metrics['corridor_cells'], fp['min_clearance_m'],
                    'PASS' if fp['passed'] else 'FAIL'),
                 fontsize=10.5, pad=10)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    print('limits', ax.get_xlim(), ax.get_ylim(), 'imgsize', rgb.shape)
    out = Path(args.out or (run/'figure.png'))
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    print('saved', out)


if __name__ == '__main__':
    main()
