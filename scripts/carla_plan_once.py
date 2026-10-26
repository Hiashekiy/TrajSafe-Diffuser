"""CARLA -> occupancy grid -> one TrajSafe trajectory. No control, no safety gating.

只做三件事：
  1. 从 CARLA 取真值：车道多边形、车身位置、障碍物
  2. 栅格化成占据图（frame.py + occupancy.py）
  3. 调用论文方法 sample()（4 步 DDIM + 冻结凸走廊 + 控制点 ALM），输出轨迹

    python scripts/carla_plan_once.py --config configs/carla_plan.yaml --repeats 5
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def xy(waypoint):
    loc = waypoint.transform.location
    return np.array([loc.x, loc.y])


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT/'configs/carla_plan.yaml'))
    p.add_argument('--repeats', type=int, default=5, help='重复规划次数（取 min/median）')
    p.add_argument('--steps', type=int, help='覆盖 planner.steps')
    p.add_argument('--region-stride', type=int, help='覆盖 planner.region_stride')
    p.add_argument('--horizon', type=float, help='覆盖 map.horizon_m')
    p.add_argument('--road', type=int)
    p.add_argument('--lane', type=int)
    p.add_argument('--s', type=float)
    p.add_argument('--out', help='输出目录')
    p.add_argument('--no-obstacle', action='store_true')
    p.add_argument('--snapshot', help='回放已保存的 CARLA 快照 npz（occupancy/road/condition/anchor/yaw），不连接仿真器')
    return p


def main(argv=None):
    args = parse().parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf8'))
    if args.steps:
        cfg['planner']['steps'] = args.steps
    if args.region_stride is not None:
        cfg['planner']['region_stride'] = args.region_stride
    if args.horizon:
        cfg['map']['horizon_m'] = args.horizon
    if args.no_obstacle:
        cfg['obstacle']['enabled'] = False

    sys.path.insert(0, cfg['carla']['python_api'])
    from src.carla_bridge.frame import SceneFrame
    from src.carla_bridge.occupancy import raster, road_quads
    from src.carla_bridge.dynamic_objects import Body
    from src.carla_bridge.planner_adapter import PlannerAdapter

    if args.snapshot:
        # 回放已保存的 CARLA 快照：跳过仿真器，只跑占据图 + 规划
        z = np.load(args.snapshot, allow_pickle=True)
        frame = SceneFrame(np.asarray(z['anchor'], float), float(z['yaw']))
        occupancy, road_mask = z['occupancy'], z['road']
        condition = np.asarray(z['condition'], np.float32)
        goal_xy = frame.world(condition[1])
        road = lane = None
        s0 = None
        chain = []
    else:
        import carla
        client = carla.Client(cfg['carla']['host'], cfg['carla']['port'])
        client.set_timeout(cfg['carla']['timeout_s'])
        world = client.get_world()
        if world.get_map().name.rsplit('/', 1)[-1] != cfg['carla']['town']:
            world = client.load_world(cfg['carla']['town'])
        cmap = world.get_map()

        # ---- 1. 路由（固定路段 + 沿车道取 waypoint 链）
        m = cfg['map']
        road = args.road if args.road is not None else m['road_id']
        lane = args.lane if args.lane is not None else m['lane_id']
        s0 = args.s if args.s is not None else m['s']
        ego_wp = cmap.get_waypoint_xodr(road, lane, s0)
        if ego_wp is None:
            raise RuntimeError('FIXED_ROAD_INVALID: road=%s lane=%s s=%s' % (road, lane, s0))
        chain = [ego_wp]
        while (len(chain)-1)*2.0 < m['route_length_m']:
            nxt = chain[-1].next(2.0)
            if len(nxt) != 1:
                raise RuntimeError('ROAD_CHAIN_BROKEN at %.1f m' % ((len(chain)-1)*2.0))
            chain.append(nxt[0])

        ego_xy = xy(ego_wp)
        yaw = np.deg2rad(ego_wp.transform.rotation.yaw)
        frame = SceneFrame(ego_xy, yaw)
        i_goal = min(len(chain)-1, int(round(m['horizon_m']/2.0)))
        goal_xy = xy(chain[i_goal])

        # ---- 2. 占据图（车道多边形 + 障碍物脚印，膨胀 inflation_cells 格）
        quads = road_quads(chain)
        if m.get('left_lane'):
            quads = quads + road_quads([w.get_left_lane() for w in chain])
        bodies = []
        if cfg['obstacle']['enabled']:
            o = cfg['obstacle']
            j = min(len(chain)-1, int(round(o['distance_m']/2.0)))
            w = chain[j]
            right = w.transform.get_right_vector()
            centre = xy(w) + o['lateral_offset_m']*np.array([right.x, right.y])
            bodies.append(Body(-1, centre, np.deg2rad(w.transform.rotation.yaw),
                               np.zeros(2), o['half_length_m'], o['half_width_m']))
        occupancy, road_mask = raster(frame, quads, bodies, m['inflation_cells'])
        condition = frame.scene(np.array([ego_xy, goal_xy])).astype(np.float32)

    # ---- 3. 规划
    adapter = PlannerAdapter(cfg['planner'])
    request = dict(id=0, time=0.0, frame=frame, occupancy=occupancy, road=road_mask,
                   condition=condition, goal=goal_xy, seed=43)

    timings, result = [], None
    for k in range(max(1, args.repeats)):
        t0 = time.perf_counter()
        result = adapter(request)
        timings.append(1000*(time.perf_counter()-t0))

    out = Path(args.out or cfg['output']['dir'])
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(
        road=dict(road_id=road, lane_id=lane, s=s0, chain_waypoints=len(chain)),
        frame=dict(origin=frame.origin.tolist(), yaw=float(frame.yaw)),
        goal=goal_xy.tolist(), horizon_m=float(np.linalg.norm(goal_xy-frame.origin)),
        obstacle=cfg['obstacle'] if cfg['obstacle']['enabled'] else None,
        occupancy_free_ratio=float((occupancy == 0).mean()),
        planner=dict(steps=cfg['planner']['steps'],
                     region_stride=cfg['planner'].get('region_stride'),
                     model_id=cfg['planner']['model_id']),
        latency_ms=dict(each=[round(t, 1) for t in timings],
                        minimum=round(min(timings), 1),
                        median=round(float(np.median(timings)), 1),
                        maximum=round(max(timings), 1)),
        internal_timing_ms={k: round(v*1000, 1) for k, v in result['timing'].items()},
        alm_status=result['alm_status'], guided=bool(result['guided']),
        topology=int(result['topology']),
        corridor_cells=result['corridor'].get('num_cells'),
        corridor_min_overlap=result['corridor'].get('min_overlap'),
        controls=result['controls'].tolist(),
        curve_world=result['path'].xy.tolist(),
    )
    (out/'plan_once.json').write_text(json.dumps(payload, indent=1), encoding='utf8')
    np.savez_compressed(out/'plan_once.npz', occupancy=occupancy, road=road_mask,
                        condition=condition, anchor=frame.origin, yaw=frame.yaw,
                        controls=result['controls'], curve_world=result['path'].xy,
                        goal=goal_xy)

    print(json.dumps({k: payload[k] for k in ('horizon_m', 'occupancy_free_ratio',
                                              'corridor_cells', 'topology', 'alm_status',
                                              'latency_ms', 'internal_timing_ms')},
                     ensure_ascii=False, indent=1))
    print('saved', out/'plan_once.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
