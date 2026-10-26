"""起点位姿锚定：让轨迹在 s=0 处的位置与切向严格等于自车位姿。

做法：把前 L 米用三次 Hermite 曲线重建（起点 = 自车中心 + 自车朝向；
L 处的位置与切向沿用原轨迹），再拼回剩余轨迹并重新做车身检查。
"""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
g = root/'scripts/carla_scenario_sim.py'
s = g.read_text(encoding='utf8')

helper = '''
def anchor_start(path_xy, start_xy, start_yaw, length=4.0, step=0.2):
    """Rebuild the first ``length`` metres as a cubic Hermite that matches the ego pose."""
    from src.carla_bridge.path_profile import Path as _Path
    nm = 3
    L = min(length, float(_Path.build(path_xy).s[-1])*0.25)
    if L <= 0.5:
        return path_xy, 0.0
    geom = _Path.build(path_xy)
    p1, yaw1 = geom.pose(L)
    m0 = np.array([np.cos(start_yaw), np.sin(start_yaw)])
    m1 = np.array([np.cos(yaw1), np.sin(yaw1)])
    turn = abs((yaw1-start_yaw+np.pi) % (2*np.pi) - np.pi)
    if turn < np.deg2rad(6.0):
        return path_xy, 0.0
    pts = []
    for t in np.arange(0.0, 1.0, step/L):
        h00 = 2*t**3-3*t**2+1
        h10 = t**3-2*t**2+t
        h01 = -2*t**3+3*t**2
        h11 = t**3-t**2
        pts.append(h00*np.asarray(start_xy) + h10*L*m0 + h01*p1 + h11*L*m1)
    pts.append(p1)
    keep = geom.xy[geom.s > L]
    return np.vstack([np.array(pts), keep]), float(np.rad2deg(turn))

'''

old = 'def extend_forward(chain, i_ego, need_m):'
assert old in s
s = s.replace(old, helper.lstrip('\n')+old, 1)

old2 = '''        plan_ms = 1000*(time.perf_counter()-t0)
        path = result['path']'''
new2 = '''        plan_ms = 1000*(time.perf_counter()-t0)
        path = result['path']
        if cfg['planner'].get('anchor_start', True):
            from src.carla_bridge.path_profile import Path as _Path
            ego_yaw = np.deg2rad(ego_wp.transform.rotation.yaw)
            new_xy, turned = anchor_start(path.xy, ego_xy, ego_yaw)
            if turned:
                path = _Path.build(new_xy)
                summary['start_anchor_deg'] = round(turned, 1)
                print('起点锚定：原切向与车身相差 %.1f°，已重建前 4 m' % turned, flush=True)'''
assert old2 in s
s = s.replace(old2, new2)
g.write_text(s, encoding='utf8')
print('runner: 起点位姿锚定已加')
