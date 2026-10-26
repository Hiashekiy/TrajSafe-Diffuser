"""扫全图找弯道 / 路口 / 分叉路段，供复杂场景使用。

    python scripts/debug/find_curve_junction.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, 'E:/CARLA_0.9.16/PythonAPI/carla')


def main():
    import carla
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    cmap = world.get_map()
    print('MAP', cmap.name, flush=True)

    rows = []
    for idx, sp in enumerate(cmap.get_spawn_points()):
        w0 = cmap.get_waypoint(sp.location)
        chain, ok = [w0], True
        junction, fork = False, False
        for _ in range(35):                      # 70 m
            nxt = chain[-1].next(2.0)
            if not nxt:
                ok = False
                break
            if len(nxt) > 1:
                fork = True
            if chain[-1].is_junction:
                junction = True
            same = [w for w in nxt if w.lane_id == chain[-1].lane_id]
            chain.append((same or nxt)[0])
        if not ok or len(chain) < 30:
            continue
        yaw = np.array([w.transform.rotation.yaw for w in chain])
        d = np.abs((np.diff(yaw)+180) % 360-180)
        curvature_deg = float(d.sum())
        rows.append((curvature_deg, int(junction), int(fork), idx,
                     w0.road_id, w0.lane_id, round(float(w0.s), 1)))

    rows.sort(reverse=True)
    print('=== 弯度最大的 14 段（度, 路口, 分叉, spawn, road, lane, s）===')
    for r in rows[:14]:
        print('  %6.1f  j=%d f=%d  spawn=%-4d road=%-4s lane=%-3s s=%s' % r)
    print('=== 路口路段（前 14）===')
    for r in [x for x in rows if x[1]][:14]:
        print('  %6.1f  spawn=%-4d road=%-4s lane=%-3s s=%s' % (r[0], r[3], r[4], r[5], r[6]))
    print('=== 分叉路段（前 14）===')
    for r in [x for x in rows if x[2]][:14]:
        print('  %6.1f  spawn=%-4d road=%-4s lane=%-3s s=%s' % (r[0], r[3], r[4], r[5], r[6]))


if __name__ == '__main__':
    main()
