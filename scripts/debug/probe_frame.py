"""对照实验：同一 ego/goal，只改输入帧中心，看候选通道是否产生。"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, 'E:/CARLA_0.9.16/PythonAPI/carla')
import carla
import dashboard_app as d
from src.carla_bridge.frame import SceneFrame
from src.carla_bridge.occupancy import raster
from src.carla_bridge.planner_adapter import PlannerAdapter
import carla_scenario_sim as css

w = d.carla_world()
c = d.cfg()
sps = w.get_map().get_spawn_points()
sp = sps[45]
ego_wp = w.get_map().get_waypoint(sp.location)
ego = np.array([ego_wp.transform.location.x, ego_wp.transform.location.y])
yaw = float(ego_wp.transform.rotation.yaw)
# 沿车道向前 65 m 作为终点
probe, chain = ego_wp, [ego_wp]
for _ in range(40):
    nxt = [x for x in probe.next(2.0) if x.lane_id == probe.lane_id]
    if not nxt:
        break
    probe = nxt[0]
    chain.append(probe)
goal = np.array([chain[min(len(chain)-1, 32)].transform.location.x,
                 chain[min(len(chain)-1, 32)].transform.location.y])
print('ego', ego.round(1), 'yaw', round(yaw, 1), 'goal', goal.round(1),
      'dist', round(float(np.linalg.norm(goal-ego)), 1))
print('chain pts', len(chain))

adapter = PlannerAdapter(c['planner'])


def run(label, centre):
    frame = SceneFrame(np.asarray(centre, float), 0.0 if label.startswith('区域') else np.deg2rad(yaw))
    quads = d.frame_quads(w, frame, frame.size*0.70)
    occ, road = raster(frame, quads, [], int(c['grid'].get('inflation_cells', 0)))
    cond = frame.scene(np.array([ego, goal])).astype(np.float32)
    print('---', label, 'free %.3f' % float((occ == 0).mean()),
          'ego scene', cond[0].round(2), 'goal scene', cond[1].round(2))
    try:
        res = adapter(dict(id=0, time=0.0, frame=frame, occupancy=occ, road=road,
                           condition=cond, goal=goal, seed=43))
        print('   OK  alm=%s cells=%s len=%.1f' % (
            res['alm_status'], (res['corridor'] or {}).get('num_cells'),
            float(res['path'].s[-1])))
    except ValueError as exc:
        print('   FAIL', exc)


# A: 论文原样 —— 帧中心 = 自车，帧朝向 = 车头
run('自车中心', ego)
# B: 现在的做法 —— 帧中心 = 区域中心（自车偏在一侧），朝向世界 x
region = np.array([ego[0]+30.0, ego[1]-20.0])
run('区域中心', region)
# C: 帧中心 = 自车，但朝向世界 x（只差朝向）
run('自车中心-朝向0', ego)


def run_chain(label, centre, frame_yaw, lanes_each_side=2):
    """论文脚本的做法：自车链条 + 同向侧车道。"""
    from src.carla_bridge.occupancy import road_quads
    frame = SceneFrame(np.asarray(centre, float), frame_yaw)
    chain = [ego_wp]
    back, probe = [], ego_wp
    while len(back)*2.0 < 40.0:
        prv = probe.previous(2.0)
        if len(prv) != 1:
            break
        probe = prv[0]
        back.append(probe)
    chain = list(reversed(back))+[ego_wp]
    i_ego = len(chain)-1
    chain = css.extend_forward(chain, i_ego, 140.0)
    quads = road_quads(chain)
    sign = np.sign(ego_wp.lane_id) or 1.0

    def same(wp):
        return wp is not None and wp.lane_type.name == 'Driving' and np.sign(wp.lane_id) == sign

    for side in (+1, -1):
        for k in range(1, lanes_each_side+1):
            lane = []
            for wp in chain:
                x = wp
                for _ in range(k):
                    x = x.get_left_lane() if side > 0 else x.get_right_lane()
                    if not same(x):
                        x = None
                        break
                if x is not None:
                    lane.append(x)
            if len(lane) >= 2:
                for run in d.build_runs(lane):
                    quads += road_quads(run)
    occ, road = raster(frame, quads, [], int(c['grid'].get('inflation_cells', 0)))
    cond = frame.scene(np.array([ego, goal])).astype(np.float32)
    print('---', label, 'free %.3f' % float((occ == 0).mean()),
          'quads %d' % len(quads), 'ego', cond[0].round(2), 'goal', cond[1].round(2))
    try:
        res = adapter(dict(id=0, time=0.0, frame=frame, occupancy=occ, road=road,
                           condition=cond, goal=goal, seed=43))
        print('   OK  alm=%s cells=%s len=%.1f' % (
            res['alm_status'], (res['corridor'] or {}).get('num_cells'),
            float(res['path'].s[-1])))
    except ValueError as exc:
        print('   FAIL', exc)


run_chain('链条-自车中心-车头朝向', ego, np.deg2rad(yaw))
run_chain('链条-自车中心-朝向0', ego, 0.0)
run_chain('链条-自车中心-侧车4条', ego, np.deg2rad(yaw), lanes_each_side=4)
