"""骨架层排查：自由空间连通分量 / 骨架图连通 / 形态学修补后能否出候选。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, 'E:/CARLA_0.9.16/PythonAPI/carla')
import numpy as np
import cv2
import carla
import dashboard_app as d
from scipy.ndimage import label
from src.carla_bridge.planner_adapter import PlannerAdapter

w = d.carla_world()
c = d.cfg()
cmap = w.get_map()
ad = PlannerAdapter(c['planner'])

for idx in (12, 50, 60, 101):
    sp = cmap.get_spawn_points()[idx]
    wp = cmap.get_waypoint(sp.location)
    ego = np.array([wp.transform.location.x, wp.transform.location.y])
    yaw = float(wp.transform.rotation.yaw)
    probe = wp
    for _ in range(33):
        nxt = [q for q in probe.next(2.0) if q.lane_id == probe.lane_id]
        if not nxt:
            break
        probe = nxt[0]
    goal = np.array([probe.transform.location.x, probe.transform.location.y])
    d.STATE['region'] = dict(cx=float(sp.location.x), cy=float(sp.location.y))
    d.STATE['obs_actors'] = []
    frame, occ, road, ra, bodies, chain, alt = d.build_scene(w, ego, yaw)

    free = (occ == 0)
    lab, ncomp = label(free, structure=np.ones((3, 3)))
    ge = frame.grid(ego.reshape(1, 2))[0]
    gg = frame.grid(goal.reshape(1, 2))[0]
    ie = lab[int(ge[1]), int(ge[0])]
    ig = lab[int(gg[1]), int(gg[0])]
    sizes = np.bincount(lab.ravel())
    big = [(k, int(sizes[k])) for k in range(1, ncomp+1) if sizes[k] > 100]
    print('=== spawn %d  dist %.0f' % (idx, float(np.linalg.norm(goal-ego))))
    print('    free %.3f  连通分量 %d (大于100格: %s)' % (free.mean(), ncomp, big[:6]))
    print('    ego 在分量 %d (大小 %d)，goal 在分量 %d (大小 %d)  %s' % (
        ie, sizes[ie] if ie > 0 and ie < len(sizes) else 0,
        ig, sizes[ig] if ig > 0 and ig < len(sizes) else 0,
        '同一块' if ie == ig and ie > 0 else '★不同块'))

    cond = frame.scene(np.array([ego, goal])).astype(np.float32)
    for name, o in (('原始', occ),
                    ('闭运算3x3', cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))),
                    ('闭运算5x5', cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)))):
        try:
            res = ad(dict(id=0, time=0.0, frame=frame, occupancy=o, road=1-o,
                          condition=cond, goal=goal, seed=43))
            print('    %-10s OK  len %.1f  走廊 %s' % (
                name, float(res['path'].s[-1]), (res['corridor'] or {}).get('num_cells')))
        except ValueError as exc:
            print('    %-10s FAIL %s' % (name, exc))
