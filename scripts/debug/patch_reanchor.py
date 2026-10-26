"""重拟合/平滑之后再做一次起点锚定，使曲线起点严格等于自车中心。"""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
g = root/'scripts/carla_scenario_sim.py'
s = g.read_text(encoding='utf8')

old = '''    yaw0 = float(np.arctan2(*(geom.xy[1]-geom.xy[0])[::-1]))
    turn = abs((yaw0-start_yaw+np.pi) % (2*np.pi) - np.pi)   # 比较【起始切向】与车身朝向
    if turn < np.deg2rad(6.0):
        return path_xy, 0.0'''
new = '''    yaw0 = float(np.arctan2(*(geom.xy[1]-geom.xy[0])[::-1]))
    turn = abs((yaw0-start_yaw+np.pi) % (2*np.pi) - np.pi)   # 比较【起始切向】与车身朝向
    offset = float(np.linalg.norm(geom.xy[0]-np.asarray(start_xy, float)))
    if turn < np.deg2rad(6.0) and offset < 0.10:
        return path_xy, 0.0
    turn = float(np.rad2deg(max(turn, 0.0))) if offset < 0.10 else max(float(np.rad2deg(turn)), 0.1)'''
assert old in s
s = s.replace(old, new)

old2 = '''    keep = geom.xy[geom.s > L]
    return np.vstack([np.array(pts), keep]), float(np.rad2deg(turn))'''
new2 = '''    keep = geom.xy[geom.s > L]
    return np.vstack([np.array(pts), keep]), float(turn)'''
assert old2 in s
s = s.replace(old2, new2)

old3 = '''        smooth_iters = int(cfg['planner'].get('smooth_iters', 0) or 0)'''
new3 = '''        # 平滑/重拟合可能把端点拉走，这里再锚定一次，保证起点 = 自车中心
        from src.carla_bridge.path_profile import Path as _PathA
        re_xy, re_turn = anchor_start(path.xy, ego_xy, np.deg2rad(ego_wp.transform.rotation.yaw))
        if re_turn:
            path = _PathA.build(re_xy)
            summary['re_anchor_deg'] = round(re_turn, 1)
            print('平滑后再次锚定起点：切向差 %.1f°' % re_turn, flush=True)
        smooth_iters = int(cfg['planner'].get('smooth_iters', 0) or 0)'''
assert old3 in s
s = s.replace(old3, new3)
g.write_text(s, encoding='utf8')
print('起点二次锚定已加')
