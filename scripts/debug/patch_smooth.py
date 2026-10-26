"""对规划轨迹做曲率平滑：锁住起点若干点，其余做拉普拉斯平滑，并报告与原轨迹的偏差。"""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
g = root/'scripts/carla_scenario_sim.py'
s = g.read_text(encoding='utf8')

helper = '''
def smooth_path(path_xy, iters=60, lam=0.45, pin_points=12):
    """Curvature smoothing: pin the first ``pin_points`` (start pose) and the last point."""
    p = np.asarray(path_xy, float).copy()
    n = len(p)
    if n < pin_points+3 or iters <= 0:
        return path_xy, 0.0
    anchor = p[:pin_points].copy()
    for _ in range(iters):
        q = p.copy()
        q[1:-1] = (1.0-lam)*p[1:-1] + lam*0.5*(p[:-2]+p[2:])
        q[:pin_points] = anchor
        q[-1] = p[-1]
        p = q
    dev = float(np.max(np.linalg.norm(p-np.asarray(path_xy, float), axis=1)))
    return p, dev

'''

anchor = 'def anchor_start('
assert anchor in s
s = s.replace(anchor, helper.lstrip('\n')+anchor, 1)

old = '''            if turned:
                path = _Path.build(new_xy)
                summary['start_anchor_deg'] = round(turned, 1)
                print('起点锚定：原切向与车身相差 %.1f°，已重建前 4 m' % turned, flush=True)'''
new = '''            if turned:
                path = _Path.build(new_xy)
                summary['start_anchor_deg'] = round(turned, 1)
                print('起点锚定：原切向与车身相差 %.1f°，已重建前 4 m' % turned, flush=True)
        smooth_iters = int(cfg['planner'].get('smooth_iters', 0) or 0)
        if smooth_iters:
            from src.carla_bridge.path_profile import Path as _Path2
            sm_xy, dev = smooth_path(path.xy, iters=smooth_iters,
                                     lam=float(cfg['planner'].get('smooth_lambda', 0.45)))
            if dev > 0:
                path = _Path2.build(sm_xy)
                summary['smooth_deviation_m'] = round(dev, 3)
                print('曲率平滑：%d 次迭代，与原始规划最大偏差 %.3f m' % (smooth_iters, dev), flush=True)'''
assert old in s
s = s.replace(old, new)
g.write_text(s, encoding='utf8')

cfg = root/'configs/scenarios/S1_lane_block.yaml'
t = cfg.read_text(encoding='utf8')
if 'smooth_iters' not in t:
    t = t.replace('  corridor_axis_max_scene: 0.035',
                  '  corridor_axis_max_scene: 0.035\n  smooth_iters: 60       # 曲率平滑迭代（锁起点位姿）\n  smooth_lambda: 0.45')
cfg.write_text(t, encoding='utf8')
print('平滑已接入：runner + config')
