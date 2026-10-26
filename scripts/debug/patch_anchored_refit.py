"""用「前置虚拟点」把起点位姿融进一次样条拟合，避免任何拼接折角。"""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
g = root/'scripts/carla_scenario_sim.py'
s = g.read_text(encoding='utf8')

helper = '''
def refit_anchored(path_xy, start_xy, start_yaw, rms=0.18, lead_m=6.0, lead_n=6):
    """One-shot C2 spline fit that leaves the ego exactly along its heading.

    A synthetic lead-in (points behind the ego along -heading) is appended before
    fitting; afterwards the curve is trimmed at the ego centre.  Because the fit
    is global there is no joint and therefore no curvature discontinuity.
    """
    from scipy.interpolate import splprep, splev
    p = np.asarray(path_xy, float)
    if len(p) < 12:
        return path_xy, 0.0, 0.0
    start = np.asarray(start_xy, float)
    m0 = np.array([np.cos(start_yaw), np.sin(start_yaw)])
    lead = np.array([start - m0*(lead_m*(i+1)/lead_n) for i in range(lead_n)])[::-1]
    # 原轨迹从 3 m 之后开始接入，避免与前置点在起点附近打架
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s_arc = np.concatenate([[0.0], np.cumsum(d)])
    body = p[s_arc > 3.0]
    pts = np.vstack([lead, start[None], body])
    dd = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(dd)])
    u = u/u[-1]
    tck, _ = splprep([pts[:, 0], pts[:, 1]], u=u, s=len(pts)*float(rms)**2, k=3)
    uu = np.linspace(0.0, 1.0, 900)
    x, y = splev(uu, tck)
    dense = np.column_stack([x, y])
    # 从"最接近自车中心"的位置起截断
    i0 = int(np.argmin(np.linalg.norm(dense-start, axis=1)))
    trimmed = dense[i0:]
    if len(trimmed) < 5:
        return path_xy, 0.0, 0.0
    tangent = np.arctan2(*(trimmed[min(3, len(trimmed)-1)]-trimmed[0])[::-1])
    dev = float(np.max(np.linalg.norm(trimmed[::5]-p[np.minimum(
        np.arange(0, len(trimmed), 5)*len(p)//len(trimmed), len(p)-1)], axis=1))) if len(p) else 0.0
    return trimmed, float(np.rad2deg(abs((tangent-start_yaw+np.pi) % (2*np.pi)-np.pi))), dev

'''

anchor = 'def anchor_start('
assert anchor in s
s = s.replace(anchor, helper.lstrip('\n')+anchor, 1)

old = '''        plan_ms = 1000*(time.perf_counter()-t0)
        path = result['path']'''
new = '''        plan_ms = 1000*(time.perf_counter()-t0)
        path = result['path']
        ar_rms = float(cfg['planner'].get('anchored_refit_rms', 0) or 0)
        if ar_rms > 0:
            from src.carla_bridge.path_profile import Path as _PathAR
            ar_xy, ar_turn, ar_dev = refit_anchored(
                path.xy, ego_xy, np.deg2rad(ego_wp.transform.rotation.yaw), rms=ar_rms)
            if ar_turn < 90.0 and len(ar_xy) > 10:
                path = _PathAR.build(ar_xy)
                summary.update(refit_tangent_deg=round(ar_turn, 2),
                               refit_deviation_m=round(ar_dev, 3))
                print('一次拟合(含起点位姿)：起点切向偏差 %.2f°，最大偏差 %.3f m'
                      % (ar_turn, ar_dev), flush=True)
                skip_extra = True
            else:
                skip_extra = False
        else:
            skip_extra = False'''
assert old in s
s = s.replace(old, new)

# 后续旧的 anchor/smooth 步骤在 skip_extra 时跳过
s = s.replace('''        if cfg['planner'].get('anchor_start', True):''',
              '''        if cfg['planner'].get('anchor_start', True) and not skip_extra:''')
s = s.replace('''        refit_rms = float(cfg['planner'].get('refit_rms_m', 0) or 0)''',
              '''        refit_rms = 0.0 if skip_extra else float(cfg['planner'].get('refit_rms_m', 0) or 0)''')
s = s.replace('''        re_xy, re_turn = anchor_start(path.xy, ego_xy, np.deg2rad(ego_wp.transform.rotation.yaw))''',
              '''        re_xy, re_turn = (path.xy, 0.0) if skip_extra else anchor_start(
            path.xy, ego_xy, np.deg2rad(ego_wp.transform.rotation.yaw))''')
s = s.replace('''        smooth_iters = int(cfg['planner'].get('smooth_iters', 0) or 0)''',
              '''        smooth_iters = 0 if skip_extra else int(cfg['planner'].get('smooth_iters', 0) or 0)''')
g.write_text(s, encoding='utf8')

cfg = root/'configs/scenarios/S1_lane_block.yaml'
t = cfg.read_text(encoding='utf8')
if 'anchored_refit_rms' not in t:
    t = t.replace('  refit_rms_m: 0.18      # 平滑样条重拟合的 RMS 容差（米）',
                  '  anchored_refit_rms: 0.18   # 一次拟合（含起点位姿硬约束）的 RMS 容差\n  refit_rms_m: 0')
cfg.write_text(t, encoding='utf8')
print('一次拟合（起点硬约束）已接入')
