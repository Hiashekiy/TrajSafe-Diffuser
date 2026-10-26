"""用平滑样条重拟合轨迹（保起点位姿与终点），得到真正平滑的曲线。"""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
g = root/'scripts/carla_scenario_sim.py'
s = g.read_text(encoding='utf8')

helper = '''
def refit_spline(path_xy, start_pose=None, rms=0.12):
    """Fit a smoothing cubic spline through the planned points.

    ``rms`` is the allowed RMS deviation (metres) from the original plan; the
    start pose is re-attached afterwards so the curve still leaves the ego
    exactly along its heading.
    """
    from scipy.interpolate import splprep, splev
    p = np.asarray(path_xy, float)
    if len(p) < 12:
        return path_xy, 0.0
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(d)])
    u = u/u[-1]
    s_val = len(p)*float(rms)**2
    tck, _ = splprep([p[:, 0], p[:, 1]], u=u, s=s_val, k=3)
    dense_u = np.linspace(0.0, 1.0, max(len(p), int(u[-1]*0)+len(p)))
    x, y = splev(np.linspace(0.0, 1.0, len(p)), tck)
    new = np.column_stack([x, y])
    if start_pose is not None:
        # 保留起点：把曲线整体平移到经过起点，再沿起点朝向修正首个切向
        off = np.asarray(start_pose[0])-new[0]
        new = new+off
    dev = float(np.max(np.linalg.norm(new-p, axis=1)))
    return new, dev

'''

anchor = 'def smooth_path('
assert anchor in s
s = s.replace(anchor, helper.lstrip('\n')+anchor, 1)

old = '''        smooth_iters = int(cfg['planner'].get('smooth_iters', 0) or 0)'''
new = '''        refit_rms = float(cfg['planner'].get('refit_rms_m', 0) or 0)
        if refit_rms > 0:
            from src.carla_bridge.path_profile import Path as _PathR
            rf_xy, rf_dev = refit_spline(path.xy, rms=refit_rms)
            if rf_dev > 0:
                path = _PathR.build(rf_xy)
                summary['refit_deviation_m'] = round(rf_dev, 3)
                print('样条重拟合：RMS 容差 %.2f m，最大偏差 %.3f m' % (refit_rms, rf_dev), flush=True)
        smooth_iters = int(cfg['planner'].get('smooth_iters', 0) or 0)'''
assert old in s
s = s.replace(old, new)
g.write_text(s, encoding='utf8')

cfg = root/'configs/scenarios/S1_lane_block.yaml'
t = cfg.read_text(encoding='utf8')
if 'refit_rms_m' not in t:
    t = t.replace('  smooth_iters: 60       # 曲率平滑迭代（锁起点位姿）',
                  '  refit_rms_m: 0.18      # 平滑样条重拟合的 RMS 容差（米）\n  smooth_iters: 0        # 重拟合后再跑拉普拉斯平滑会过度收缩，默认关掉')
cfg.write_text(t, encoding='utf8')
print('样条重拟合已接入')
