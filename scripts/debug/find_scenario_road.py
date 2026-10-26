"""在 CARLA 当前地图里找一个"有合理通路"的路段，供 S1/S2/S3 场景配置使用。

    python scripts/debug/find_scenario_road.py --length 140 --horizon 55
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--length', type=float, default=140.0)
    ap.add_argument('--horizon', type=float, default=55.0)
    ap.add_argument('--limit', type=int, default=60)
    args = ap.parse_args(argv)

    import carla
    from src.carla_bridge.frame import SceneFrame
    from src.carla_bridge.occupancy import raster, road_quads, connected
    from src.carla_bridge.dynamic_objects import Body
    from src.carla_bridge.skeleton_query import query_graph
    from src.geometry.skeleton_paths import CandidateConfig, generate_candidates

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    cmap = world.get_map()
    print('MAP', cmap.name, flush=True)

    sys.path.insert(0, str(ROOT/'diffusion-dashboard'))
    from engine_carla import Engine
    eng = Engine('cuda', config_path=str(ROOT/'configs/config_160k4p_c48_oneshot.yaml'),
                 dataset='160m-k4p-c48')
    cand_cfg = CandidateConfig(raw_k=16)

    def wp_xy(w):
        return np.array([w.transform.location.x, w.transform.location.y])

    results = []
    for idx, sp in enumerate(cmap.get_spawn_points()[:args.limit]):
        w0 = cmap.get_waypoint(sp.location)
        if w0.get_left_lane() is None:
            continue
        chain, ok = [w0], True
        steps = int(args.length/2.0)
        for _ in range(steps):
            nxt = chain[-1].next(2.0)
            if len(nxt) != 1:
                ok = False
                break
            chain.append(nxt[0])
        if not ok:
            continue
        frame = SceneFrame(wp_xy(chain[0]), np.deg2rad(chain[0].transform.rotation.yaw))
        quads = road_quads(chain)
        left = [w.get_left_lane() for w in chain]
        left = [w for w in left if w is not None]
        if len(left) >= 2:
            quads += road_quads(left)
        i_goal = int(round(args.horizon/2.0))
        obstacle = Body(-1, wp_xy(chain[int(round(30.0/2.0))]),
                        np.deg2rad(chain[15].transform.rotation.yaw), np.zeros(2), 2.2, 0.9)
        occ, road = raster(frame, quads, [obstacle], 4)
        start, goal = wp_xy(chain[0]), wp_xy(chain[i_goal])
        free = bool(frame.free(occ, np.array([start, goal])).all())
        conn = connected(frame, occ, start, goal) if free else False
        n_cand = 0
        if conn:
            graph = eng.get_graph(eng.occupancy_key(occ), occ)
            q = query_graph(graph, *frame.scene(np.array([start, goal])).astype(np.float32))
            n_cand = generate_candidates(q, *frame.scene(np.array([start, goal])).astype(np.float32), cand_cfg).num_valid
        if free and conn and n_cand > 0:
            results.append((n_cand, idx, w0.road_id, w0.lane_id, round(float(w0.s), 1), round(sp.location.x,1), round(sp.location.y,1)))

    results.sort(reverse=True)
    print('可行路段（候选数, spawn_idx, road, lane, s, x, y）:')
    for r in results[:12]:
        print('  cand=%d spawn=%-3d road=%-4s lane=%-3s s=%-7s (%.1f, %.1f)' % r)
    if not results:
        print('  没有找到同时满足 free + connected + 有候选的路段')


if __name__ == '__main__':
    main()
