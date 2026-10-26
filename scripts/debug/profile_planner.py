"""Where the online planning milliseconds actually go (one request, real snapshot)."""
import sys, time, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import yaml

snapshot = ROOT/'outputs/closed_loop/demo_full_20261024/snapshots/14498348.npz'
z = np.load(snapshot, allow_pickle=True)
from src.carla_bridge.frame import SceneFrame
from shapely.wkb import loads as wkb_loads
from src.carla_bridge.dynamic_objects import Body
from src.carla_bridge.planner_adapter import PlannerAdapter

cfg = yaml.safe_load((ROOT/'configs/carla_online_closed_loop.yaml').read_text(encoding='utf8'))
planner = PlannerAdapter(cfg['planner'])
frame = SceneFrame(np.asarray(z['anchor'], float), float(z['yaw']))
condition = np.asarray(z['condition'], np.float32)
request = dict(id=1, time=0., frame=frame, occupancy=z['occupancy'], road=z['road'],
               road_area=wkb_loads(z['road_geometry_wkb'].tobytes()), condition=condition,
               goal=frame.world(condition[1]), body_margin_m=1.2, interior_body_margin_m=1.55, seed=43)

# stage-by-stage, CUDA synchronised so no async kernel hides behind Python
import torch
from src.diffusion.sampler import sample
from src.geometry.skeleton_paths import generate_candidates
from src.carla_bridge.skeleton_query import query_graph

engine = planner.engine
occ = request['occupancy']
t0 = time.perf_counter()
graph = engine.get_graph(engine.occupancy_key(occ), occ)
t_graph = time.perf_counter()-t0
t0 = time.perf_counter(); query_graph(graph, *condition); t_query = time.perf_counter()-t0
t0 = time.perf_counter()
cands = generate_candidates(graph, condition[0], condition[1], planner.online_cand_cfg)
t_cand = time.perf_counter()-t0
t0 = time.perf_counter(); packed = engine._pack_candidates(cands, condition); t_pack = time.perf_counter()-t0

for i in range(3):
    t0 = time.perf_counter()
    result = planner(request)
    torch.cuda.synchronize()
    print(f'planner(request) full chain #{i}: {1000*(time.perf_counter()-t0):.0f} ms  (worker reports {result and ""!s}'
          f'alm={result["alm_status"]} guided={result["guided"]} topology={result["topology"]})')

tim_ms = {k: round(v*1000, 1) for k, v in result['timing'].items()}
print('stages measured here:')
print(f'  skeleton graph build     {1000*t_graph:7.1f} ms')
print(f'  graph query              {1000*t_query:7.1f} ms')
print(f'  candidate generation     {1000*t_cand:7.1f} ms   valid={cands.num_valid}/{cands.num_slots}')
print(f'  pack for network         {1000*t_pack:7.1f} ms')
print('sampler-reported timing:', json.dumps(tim_ms))
print('per-step ALM correction:', [round(s['delta'], 4) for s in result['viz']['steps']] if result.get('viz') else 'viz off')
