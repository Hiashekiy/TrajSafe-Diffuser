"""Split the online planning budget: network forward, corridor, LP projection, rest."""
import sys, time
from pathlib import Path
import numpy as np, torch, yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.carla_bridge.frame import SceneFrame
from src.carla_bridge.planner_adapter import PlannerAdapter
from shapely.wkb import loads as wkb_loads
from src.diffusion.bspline_alm import bspline_hard_project
from src.carla_bridge.skeleton_query import query_graph
from src.geometry.skeleton_paths import generate_candidates

cfg = yaml.safe_load((ROOT/'configs/carla_online_closed_loop.yaml').read_text(encoding='utf8'))
z = np.load(ROOT/'outputs/closed_loop/demo_full_20261024/snapshots/14498348.npz', allow_pickle=True)
frame = SceneFrame(np.asarray(z['anchor'], float), float(z['yaw']))
condition = np.asarray(z['condition'], np.float32)
request = dict(id=1, time=0., frame=frame, occupancy=z['occupancy'], road=z['road'],
               road_area=wkb_loads(z['road_geometry_wkb'].tobytes()), condition=condition,
               goal=frame.world(condition[1]), body_margin_m=1.2, interior_body_margin_m=1.55, seed=43)
planner = PlannerAdapter(dict(cfg['planner'], visualize=False))
planner(request); torch.cuda.synchronize()

def sync_time(fn, n=3):
    fn(); torch.cuda.synchronize()
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter(); out = fn(); torch.cuda.synchronize(); best = min(best, time.perf_counter()-t0)
    return best*1000, out

# the model needs the real argument set: reuse the sampler's own call, timed
from src.diffusion.sampler import sample
engine = planner.engine
device = planner.cfg['device']
occ1 = torch.as_tensor(z['occupancy'][None, None], dtype=torch.float32, device=device)
cond1 = torch.as_tensor(condition[None], dtype=torch.float32, device=device)
graph = engine.get_graph(engine.occupancy_key(z['occupancy']), z['occupancy'])
cands = generate_candidates(graph, condition[0], condition[1], planner.online_cand_cfg)
packed = engine._pack_candidates(cands, condition)
alm = dict(planner.model_cfg['alm']); corr = dict(planner.model_cfg['corridor'])
from src.diffusion.sampler import effective_warmup_steps
alm['warmup_reverse_steps'] = effective_warmup_steps(alm.get('warmup_reverse_steps', 3), 4)

def run(batch):
    return sample(planner.model, engine.schedule, cond1.repeat(batch,1,1), occ1.repeat(batch,1,1,1),
                  packed['xy'].repeat(batch,1,1,1), packed['mask'].repeat(batch,1),
                  packed['geometry'].repeat(batch,1,1,1), packed['geometry_lengths'].repeat(batch,1),
                  device=device, steps=4, seed=43, return_trace=False, alm_config=alm, corridor_config=corr)

t_single, out = sync_time(lambda: run(1), 3)
t_batch, _ = sync_time(lambda: run(32), 2)
stages = {k: round(v*1000, 1) for k, v in out['timing'].items()}
print(f'full sample(), batch=1  : {t_single:6.0f} ms   stages={stages}')
print(f'full sample(), batch=32 : {t_batch:6.0f} ms   -> {t_batch/32:5.1f} ms per trajectory')
print(f'amortisation factor     : {t_single*32/t_batch:5.1f}x')

# --- LP hard projection alone
pack = out.get('pack')
q = out['control'][:1]
try:
    t_lp, _ = sync_time(lambda: bspline_hard_project(q, pack[0], planner.model.bspline,
                        alm.get('final_project_tol', 1e-9)), 3)
    print(f'bspline_hard_project    : {t_lp:6.1f} ms  (single control polygon, batch=1)')
except Exception as exc:
    print('hard_project probe failed:', type(exc).__name__, exc)
