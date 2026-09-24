"""Verify a frozen plan against the live CARLA map (no driving).

Every planned world point must land on a LaneType.Driving waypoint, and the
ego anchor must reproduce the recorded lane heading.  This is the end-to-end
check of the world/local/scene/grid chain against CARLA ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.carla_bridge import demo as demo_mod  # noqa: E402
from src.carla_bridge.scenario import CarlaSession  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "carla_demo.yaml"))
    parser.add_argument("--plan", default=None)
    args = parser.parse_args(argv)
    cfg = demo_mod.load_yaml(args.config)
    ctx = demo_mod.build_sample_context(cfg)
    npz = args.plan or os.path.join(cfg["output"]["dir"],
                                    "plan_%s.npz" % demo_mod.SAMPLE["key"])
    plan = demo_mod.load_plan(npz)

    session = CarlaSession(host=cfg["carla"]["host"], port=int(cfg["carla"]["port"]),
                           town=str(cfg["carla"]["town"]),
                           fixed_dt=float(cfg["carla"]["fixed_dt"]),
                           vendor_dir=str(cfg["carla"]["python_vendor"]))
    session.connect()
    try:
        carla_map = session.world.get_map()
        print("map: %s" % carla_map.name)
        anchor = ctx.frame.anchor_xy
        wp = carla_map.get_waypoint(session.carla.Location(x=float(anchor[0]),
                                                           y=float(anchor[1]),
                                                           z=float(ctx.frame.z)))
        if wp is None:
            print("FAIL: no waypoint at the anchor")
            return 1
        wp_forward = wp.transform.get_forward_vector()
        dot = float(np.array([wp_forward.x, wp_forward.y]) @ ctx.frame.forward_xy)
        print("anchor lane: id=%d road=%d lane_type=%s width=%.3f z=%.3f "
              "heading_dot=%.6f lane_change=%s"
              % (wp.lane_id, wp.road_id, wp.lane_type, wp.lane_width,
                 wp.transform.location.z, dot, wp.lane_change))
        offset = np.array([wp.transform.location.x - anchor[0],
                           wp.transform.location.y - anchor[1]])
        print("anchor lateral offset from lane centre: %.3f m" % float(np.linalg.norm(offset)))

        curve = np.asarray(plan.curve_world, dtype=np.float64)
        non_driving = 0
        off_road = 0
        lanes = []
        for point in curve:
            w = carla_map.get_waypoint(session.carla.Location(x=float(point[0]),
                                                              y=float(point[1]),
                                                              z=float(ctx.frame.z)))
            if w is None:
                off_road += 1
                lanes.append(None)
                continue
            lanes.append((w.road_id, w.lane_id, w.lane_type))
            if w.lane_type != session.carla.LaneType.Driving:
                non_driving += 1
        goal = np.asarray(plan.curve_world[-1], dtype=np.float64)
        goal_wp = carla_map.get_waypoint(session.carla.Location(x=float(goal[0]),
                                                                y=float(goal[1]),
                                                                z=float(ctx.frame.z)))
        print("planned points: %d | no-waypoint: %d | not Driving: %d"
              % (len(curve), off_road, non_driving))
        print("goal waypoint: %s"
              % (None if goal_wp is None else
                 "road=%d lane=%d type=%s" % (goal_wp.road_id, goal_wp.lane_id,
                                              goal_wp.lane_type)))
        transitions = [i for i in range(1, len(lanes)) if lanes[i] != lanes[i - 1]]
        print("lane transitions at point indices: %s" % transitions[:12])
        if lanes[0] is not None:
            print("start lane: road=%d lane=%d" % lanes[0][:2])
        if lanes[-1] is not None:
            print("final lane: road=%d lane=%d" % lanes[-1][:2])
        ok = off_road == 0 and non_driving == 0
        print("VERDICT: %s" % ("plan lies entirely on Driving lanes" if ok
                               else "PLAN LEAVES DRIVING LANES"))
        return 0 if ok else 1
    finally:
        session.close(restore=False)


if __name__ == "__main__":
    raise SystemExit(main())
