"""验证：用左图那套“车道中心线粗线栅格”当模型输入，候选通道能不能出来。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, 'E:/CARLA_0.9.16/PythonAPI/carla')
import math
import numpy as np
import cv2
import dashboard_app as d
import carla_scenario_sim as css
from src.carla_bridge.frame import SceneFrame
from src.carla_bridge.planner_adapter import PlannerAdapter

w = d.carla_world()
c = d.cfg()
adapter = PlannerAdapter(c['planner'])
cmap = w.get_map()
w0 = cmap.get_waypoint(cmap.get_spawn_points()[45].location)
ego = np.array([w0.transform.location.x, w0.transform.location.y])
yaw = float(w0.transform.rotation.yaw)
probe, chain = w0, [w0]
for _ in range(80):
    nxt = [x for x in probe.next(2.0) if x.lane_id == probe.lane_id]
    if not nxt:
        break
    probe = nxt[0]
    chain.append(probe)


def lane_raster(frame):
    """车道中心线按真实宽度画粗线（与左侧占用图同一套）。"""
    res = frame.resolution
    cell = frame.size/res
    free = np.zeros((res, res), np.uint8)
    SHIFT = 4
    for wps, hw in d.topology_segments(w):
        pts = d.wp_xy(wps)
        sel = pts[(np.abs(pts[:, 0]-frame.origin[0]) < frame.size*0.72)
                  & (np.abs(pts[:, 1]-frame.origin[1]) < frame.size*0.72)]
        if len(sel) < 2:
            continue
        g = np.rint(frame.grid(sel)*(1 << SHIFT)).astype(np.int32)
        th = max(1, int(math.ceil(2.0*hw/cell)))
        cv2.polylines(free, [g], False, 1, th, cv2.LINE_8, shift=SHIFT)
    free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return 1-free


def trial(n_pts, mode):
    g = np.array([chain[min(len(chain)-1, n_pts)].transform.location.x,
                  chain[min(len(chain)-1, n_pts)].transform.location.y])
    centre = ego if mode == 'ego' else (ego+g)/2.0
    frame = SceneFrame(np.asarray(centre, float), np.deg2rad(yaw))
    occ = lane_raster(frame)
    cond = frame.scene(np.array([ego, g])).astype(np.float32)
    print('--- dist %.0f %s free %.3f ego %s goal %s' % (
        float(np.linalg.norm(g-ego)), mode, float((occ == 0).mean()),
        cond[0].round(2), cond[1].round(2)))
    try:
        res = adapter(dict(id=0, time=0.0, frame=frame, occupancy=occ, road=1-occ,
                           condition=cond, goal=g, seed=43))
        print('    OK len %.1f alm %s cells %s' % (
            float(res['path'].s[-1]), res['alm_status'],
            (res['corridor'] or {}).get('num_cells')))
    except ValueError as exc:
        print('    FAIL', exc)


for n in (32, 50, 62):
    trial(n, 'ego')
trial(62, 'mid')
