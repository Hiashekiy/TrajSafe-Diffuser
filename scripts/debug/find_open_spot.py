"""扫全图找"头顶无遮挡"的直路段（用 cast_ray 检测天桥/建筑）。

    python scripts/debug/find_open_spot.py --length 45
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, 'E:/CARLA_0.9.16/PythonAPI/carla')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--length', type=float, default=45.0)
    ap.add_argument('--step', type=float, default=5.0)
    ap.add_argument('--limit', type=int, default=120)
    args = ap.parse_args(argv)

    import carla
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    cmap = world.get_map()
    print('MAP', cmap.name, flush=True)

    rows = []
    for idx, sp in enumerate(cmap.get_spawn_points()[:args.limit]):
        w0 = cmap.get_waypoint(sp.location)
        if w0.get_left_lane() is None or w0.get_left_lane().lane_type.name != 'Driving':
            continue
        chain, ok = [w0], True
        for _ in range(int(args.length/2.0)):
            nxt = chain[-1].next(2.0)
            if len(nxt) != 1 or chain[-1].get_left_lane() is None:
                ok = False
                break
            chain.append(nxt[0])
        if not ok:
            continue
        # 直线度：首尾航向差
        yaw0 = chain[0].transform.rotation.yaw
        yaw1 = chain[-1].transform.rotation.yaw
        d_yaw = abs((yaw1-yaw0+180) % 360-180)
        if d_yaw > 8:
            continue
        # 头顶遮挡检测：从 +40 m 往下打射线，命中点不是路面就算遮挡
        blocked, labels = 0, {}
        for w in chain[::max(1, int(args.step/2.0))]:
            p = w.transform.location
            res = world.cast_ray(carla.Location(x=p.x, y=p.y, z=p.z+40),
                                 carla.Location(x=p.x, y=p.y, z=p.z+2.0))
            if len(res) == 2 and res[0]:
                label = res[1]
                labels[label] = labels.get(label, 0)+1
                if label != 'Road':
                    blocked += 1
        rows.append((blocked, d_yaw, idx, w0.road_id, w0.lane_id, round(float(w0.s), 1),
                     round(p.x, 1), round(p.y, 1), labels))

    rows.sort()
    print('blocked  d_yaw  spawn  road  lane    s        x        y    hits')
    for r in rows[:14]:
        print('%5d   %5.1f  %5d  %4s  %4s  %6.1f  %7.1f  %7.1f  %s'
              % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]))
    if not rows:
        print('  没有满足条件的路段')


if __name__ == '__main__':
    main()
