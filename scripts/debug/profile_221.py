"""把一次规划里那 221 ms 摊开：cProfile 一份 planner(request) 调用。"""
import cProfile
import io
import pstats
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.carla_bridge.frame import SceneFrame
from src.carla_bridge.planner_adapter import PlannerAdapter
from shapely.wkb import loads as wkb


def main():
    cfg = yaml.safe_load((ROOT/'configs/carla_plan.yaml').read_text(encoding='utf8'))
    z = np.load(ROOT/'outputs/closed_loop/demo_fast_205421/snapshots/24162717.npz', allow_pickle=True)
    frame = SceneFrame(np.asarray(z['anchor'], float), float(z['yaw']))
    cond = np.asarray(z['condition'], np.float32)
    adapter = PlannerAdapter(cfg['planner'])
    request = dict(id=0, time=0.0, frame=frame, occupancy=z['occupancy'], road=z['road'],
                   road_area=wkb(z['road_geometry_wkb'].tobytes()), condition=cond,
                   goal=frame.world(cond[1]), seed=43)
    adapter(request)                      # warm-up (cudnn autotune, caches)
    pr = cProfile.Profile()
    pr.enable()
    adapter(request)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(26)
    print('\n'.join(s.getvalue().splitlines()[:36]))


if __name__ == '__main__':
    main()
