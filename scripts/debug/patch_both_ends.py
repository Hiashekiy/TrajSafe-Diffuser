"""给 path_smooth.py 追加 pin_both_ends（两端都钉扎）。"""
from pathlib import Path

p = Path('src/carla_bridge/path_smooth.py')
s = p.read_text(encoding='utf8')
if 'def pin_both_ends' in s:
    print('already there')
else:
    s += '''

def pin_both_ends(path_xy, start_xy, start_yaw, end_xy, end_yaw,
                  sample_m=12.0, end_gain=0.6):
    """两端都钉扎的三次样条：起点位置+切向、终点位置+切向全部硬约束。

    与 pin_and_spline 同一套做法（沿弧长每 sample_m 取一个节点，用
    make_interp_spline 的 bc_type 把两端一阶导钉死，曲线内部 C2 连续）。
    end_gain 缩小终点切向的手柄长度，让末段收敛更平缓（配合终点速度归零）。
    """
    from scipy.interpolate import make_interp_spline
    from src.carla_bridge.path_profile import Path as _Path
    p = np.asarray(path_xy, float)
    if len(p) < 10:
        return p, 0.0, 0.0
    start = np.asarray(start_xy, float)
    end = np.asarray(end_xy, float)
    geom = _Path.build(p)
    total = float(geom.s[-1])
    ss = np.arange(0.0, total, sample_m)
    if ss[-1] < total-0.5:
        ss = np.append(ss, total)
    knots = np.array([geom.pose(float(s))[0] for s in ss])
    knots[0] = start
    knots[-1] = end
    m0 = np.array([np.cos(start_yaw), np.sin(start_yaw)])
    m1 = np.array([np.cos(end_yaw), np.sin(end_yaw)])
    d = np.linalg.norm(np.diff(knots, axis=0), axis=1)
    tt = np.concatenate([[0.0], np.cumsum(d)])
    tt = tt/tt[-1]
    bc = ([(1, m0*total)], [(1, m1*total*end_gain)])
    spl = make_interp_spline(tt, knots, k=3, bc_type=bc)
    uu = np.linspace(0.0, 1.0, max(600, int(total/0.2)))
    dense = spl(uu)
    dense[0] = start
    dense[-1] = end
    yaw0 = float(np.arctan2(*(dense[min(4, len(dense)-1)]-dense[0])[::-1]))
    turn = float(np.rad2deg(abs((yaw0-start_yaw+np.pi) % (2*np.pi)-np.pi)))
    return dense, turn, 0.0
'''
    p.write_text(s, encoding='utf8')
    print('appended')
