"""出图：左 = 2D 示意图（论文方法图风格），右 = CARLA 实景（极简标注）。

    python scripts/render_schematic.py --run outputs/scenarios/<run> --out outputs/scenarios/S1_schematic.png
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
from matplotlib.patches import Polygon, Circle, FancyArrow
from matplotlib.lines import Line2D
from shapely.geometry import Polygon as SPoly, MultiPolygon as SMulti, LineString
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
    'font.size': 9,
    'axes.linewidth': .8,
})

INK = '#22303c'
ROAD = '#e9ecef'
ROAD_EDGE = '#c7ced6'
LANE = '#b9c2cc'
C_CORR, C_CORR_E = '#9dc3e6', '#3f7fb5'
C_SWEEP, C_SWEEP_E = '#a8dbbe', '#2e7d5b'
C_PATH = '#12395c'
C_OBST, C_OBST_E = '#f2c6c2', '#b03a2e'
C_EGO, C_EGO_E = '#ffffff', '#22303c'
C_GOAL = '#b03a2e'


def rect(centre, yaw, hl, hw):
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = np.array([c, s]), np.array([-s, c])
    return np.array([centre+dx*hl+dy*hw, centre+dx*hl-dy*hw,
                     centre-dx*hl-dy*hw, centre-dx*hl+dy*hw])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out')
    ap.add_argument('--margin', type=float, default=9.0)
    ap.add_argument('--dpi', type=int, default=220)
    args = ap.parse_args(argv)

    from src.carla_bridge.frame import SceneFrame
    from src.carla_bridge.path_profile import Path as GeomPath

    run = Path(args.run)
    plan = np.load(run/'plan.npz', allow_pickle=True)
    metrics = json.loads((run/'summary.json').read_text(encoding='utf8'))
    frame = SceneFrame(np.asarray(plan['anchor'], float), float(plan['yaw']))
    path = plan['curve_world']
    goal = plan['goal']
    geom = GeomPath.build(path)

    # 自车局部坐标（x 沿车道，y 横向）
    yaw0 = float(np.asarray(plan["yaw"]))      # 用车道方向，而不是轨迹前两点

    def local(pts):
        d = np.asarray(pts, float).reshape(-1, 2)-path[0]
        c, s = np.cos(-yaw0), np.sin(-yaw0)
        return np.stack([c*d[:, 0]-s*d[:, 1], s*d[:, 0]+c*d[:, 1]], axis=1)

    # 道路区域（用 road mask 的边界）
    road = plan['road']
    res = road.shape[0]
    road_poly = None
    try:
        import cv2
        mask = ((road == 0).astype(np.uint8))*255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        if cnts:
            gx, gy = np.meshgrid(np.arange(res), np.arange(res))
            scene = np.stack([(gx/(res-1))*2-1, (gy/(res-1))*2-1], axis=-1)
            world = frame.world(scene.reshape(-1, 2)).reshape(res, res, 2)
            ii, jj = cnts[0][:, 0, 1], cnts[0][:, 0, 0]
            road_poly = world[ii, jj]
    except Exception:
        road_poly = None

    half_w = float(np.load(run/'view.npz', allow_pickle=True)['half_width']) if (run/'view.npz').exists() else 1.0

    fig = plt.figure(figsize=(12.4, 5.6), dpi=args.dpi)
    ax = fig.add_axes([0.05, 0.10, 0.42, 0.80])
    ax2 = fig.add_axes([0.50, 0.10, 0.47, 0.80])

    # ---------------- 左：2D 示意图 ----------------
    ax.set_facecolor('white')
    if road_poly is not None:
        ax.add_patch(Polygon(local(road_poly), closed=True, facecolor=ROAD,
                             edgecolor=ROAD_EDGE, linewidth=1.0, zorder=1))

    corr_parts = []
    band_clip = LineString(path).buffer(7.0, cap_style=2, join_style=2)
    for poly in plan['corridors']:
        sp = SPoly(frame.world(np.asarray(poly)))
        if not sp.is_valid:
            sp = sp.buffer(0)
        inter = sp.intersection(band_clip)
        if not inter.is_empty:
            corr_parts.extend(inter.geoms if isinstance(inter, SMulti) else [inter])
    if corr_parts:
        sil = unary_union(corr_parts)
        for part in (sil.geoms if isinstance(sil, SMulti) else [sil]):
            ax.add_patch(Polygon(local(part.exterior.coords[:-1]), closed=True,
                                 facecolor=C_CORR, alpha=.55, edgecolor=C_CORR_E,
                                 linewidth=1.0, zorder=2))

    half = half_w + 0.15
    left, right, cross = [], [], []
    for s in np.arange(0.0, geom.s[-1], .5):
        p, yaw = geom.pose(float(s))
        n = np.array([-np.sin(yaw), np.cos(yaw)])
        left.append(p+n*half)
        right.append(p-n*half)
    for s in np.arange(0.0, geom.s[-1], 4.0):
        p, yaw = geom.pose(float(s))
        n = np.array([-np.sin(yaw), np.cos(yaw)])
        cross.append((p+n*half, p-n*half))
    ax.add_patch(Polygon(local(np.vstack([left, right[::-1]])), closed=True,
                         facecolor=C_SWEEP, alpha=.80, edgecolor=C_SWEEP_E,
                         linewidth=1.1, zorder=3))
    for a, b in cross:
        seg = local(np.array([a, b]))
        ax.plot(seg[:, 0], seg[:, 1], color=C_SWEEP_E, lw=.6, alpha=.8, zorder=4)
    ax.plot(*local(path).T, color=C_PATH, lw=2.2, solid_capstyle='round', zorder=5)

    for corners in plan['corridors'][:0]:
        pass
    obst = np.load(run/'view.npz', allow_pickle=True)['obstacle_corners'] if (run/'view.npz').exists() else []
    for corners in obst:
        ax.add_patch(Polygon(local(np.asarray(corners)), closed=True, facecolor=C_OBST,
                             edgecolor=C_OBST_E, linewidth=1.3, zorder=6))

    ax.add_patch(Polygon(local(rect(path[0], yaw0, 2.1, 1.0)), closed=True,
                         facecolor=C_EGO, edgecolor=C_EGO_E, linewidth=1.5, zorder=7))
    ax.annotate('EGO', local(path[0][None])[0]+np.array([-3.6, 0]), color=INK,
                fontsize=9, weight='bold', ha='center', va='center')
    gp = local(goal[None])[0]
    ax.add_patch(Circle(gp, .55, facecolor='none', edgecolor=C_GOAL, lw=1.6, zorder=7))
    ax.plot(*gp, marker='x', ms=6, color=C_GOAL, zorder=7)
    ax.annotate('GOAL', gp+np.array([.9, .4]), color=C_GOAL, fontsize=9, weight='bold')

    ax.set_aspect('equal')
    p0, p1 = local(path[0][None])[0], local(path[-1][None])[0]
    lo = np.minimum(p0, p1)-args.margin
    hi = np.maximum(p0, p1)+args.margin
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1]-2.0, hi[1]+2.0)
    ax.set_xlabel('沿车道方向 / m')
    ax.set_ylabel('横向 / m')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(alpha=.18, lw=.6)
    ax.set_axisbelow(True)
    ax.set_title('(a) 规划结果示意', fontsize=10.5, color=INK, loc='left')

    # ---------------- 右：CARLA 实景 ----------------
    view = np.load(run/'view.npz', allow_pickle=True)
    rgb = view['rgb']
    inv = np.asarray(view['inverse'], float)
    ground = float(view['ground_z'])
    h, w = rgb.shape[:2]

    def px(xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        hom = np.column_stack([xy, np.full(len(xy), ground), np.ones(len(xy))])
        cam = hom @ inv.T
        f = w/2.0
        return np.column_stack([w/2 + f*cam[:, 1]/cam[:, 0], h/2 - f*cam[:, 2]/cam[:, 0]])

    ax2.imshow(rgb)
    ax2.set_axis_off()
    ax2.set_xlim(0, w)
    ax2.set_ylim(h, 0)
    ax2.plot(*px(path).T, color='#ffffff', lw=2.4, solid_capstyle='round', zorder=4)
    for corners in obst:
        ax2.add_patch(Polygon(px(np.asarray(corners)), closed=True, facecolor='none',
                              edgecolor='#ff5a4d', linewidth=1.8, zorder=5))
    ax2.add_patch(Polygon(px(rect(path[0], yaw0, 2.1, 1.0)), closed=True, facecolor='none',
                          edgecolor='#6fd3ff', linewidth=1.8, zorder=5))
    ax2.set_title('(b) CARLA 实景（俯视）', fontsize=10.5, color=INK, loc='left')

    handles = [Line2D([], [], color=C_CORR, lw=8, alpha=.7, label='安全走廊（凸区域序列）'),
               Line2D([], [], color=C_SWEEP, lw=8, alpha=.9, label='车身扫掠带'),
               Line2D([], [], color=C_PATH, lw=2.2, label='规划轨迹'),
               Line2D([], [], marker='s', color='w', mfc=C_OBST, mec=C_OBST_E,
                      ms=11, label='静止障碍车'),
               Line2D([], [], marker='s', color='w', mfc=C_EGO, mec=C_EGO_E,
                      ms=11, label='自车')]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.005))
    fp = metrics['footprint']
    fig.suptitle('%s  %s     规划 %.0f ms ｜ 走廊 %s 格 ｜ 最小间距 %s m ｜ 车身检查 %s'
                 % (metrics['scenario'], metrics['name'], metrics['plan_ms'],
                    metrics['corridor_cells'], fp['min_clearance_m'],
                    '通过' if fp['passed'] else '未通过'),
                 fontsize=11, color=INK, y=0.965)
    out = Path(args.out or (run/'schematic.png'))
    fig.savefig(out, facecolor='white', bbox_inches='tight')
    print('saved', out)


if __name__ == '__main__':
    main()
