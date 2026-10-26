"""拟合回退到 pin_and_spline（只钉起点），去掉终点切向约束。"""
from pathlib import Path

p = Path('scripts/dashboard_app.py')
s = p.read_text(encoding='utf8')

old = """    sample_m = float(c['planner'].get('pin_sample_m', 12.0))
    end_gain = float(c['planner'].get('pin_end_gain', 0.6))
    pin, turn, dev = pin_both_ends(raw, ego_xy, np.deg2rad(ego_yaw), goal_xy, goal_yaw,
                                   sample_m=sample_m, end_gain=end_gain)
    path = GeomPath.build(pin)"""
new = """    sample_m = float(c['planner'].get('pin_sample_m', 12.0))
    pin, turn, dev = pin_and_spline(raw, ego_xy, np.deg2rad(ego_yaw), sample_m=sample_m)
    path = GeomPath.build(pin)"""
assert old in s
s = s.replace(old, new, 1)

old2 = """        fixed = push_clear(pin, bodies, radius=0.65)
        pin, turn, dev = pin_both_ends(fixed, ego_xy, np.deg2rad(ego_yaw),
                                       goal_xy, goal_yaw,
                                       sample_m=max(4.0, sample_m*0.6), end_gain=end_gain)
        path = GeomPath.build(pin)"""
new2 = """        fixed = push_clear(pin, bodies, radius=0.65)
        pin, turn, dev = pin_and_spline(fixed, ego_xy, np.deg2rad(ego_yaw),
                                        sample_m=max(4.0, sample_m*0.6))
        path = GeomPath.build(pin)"""
assert old2 in s
s = s.replace(old2, new2, 1)

p.write_text(s, encoding='utf8')
print('reverted ok')
