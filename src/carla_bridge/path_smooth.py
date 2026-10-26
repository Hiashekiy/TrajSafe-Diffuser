"""轨迹平滑与起点对齐（供 runner 调用）。"""
from __future__ import annotations

import numpy as np


def refit_spline(path_xy, rms=0.18):
    """平滑三次样条重拟合；返回 (新点列, 相对原轨迹的最大偏差 m)。"""
    from scipy.interpolate import splprep, splev
    p = np.asarray(path_xy, float)
    if len(p) < 12:
        return p, 0.0
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(d)])
    u = u/u[-1]
    tck, _ = splprep([p[:, 0], p[:, 1]], u=u, s=len(p)*float(rms)**2, k=3)
    x, y = splev(np.linspace(0.0, 1.0, len(p)), tck)
    new = np.column_stack([x, y])
    return new, float(np.max(np.linalg.norm(new-p, axis=1)))


def align_start_rigid(path_xy, start_xy, start_yaw, k=8):
    """刚体旋转+平移，使起点落在自车中心、起始切向等于车身朝向（保曲率）。"""
    p = np.asarray(path_xy, float).copy()
    if len(p) < k+2:
        return p, 0.0
    v = p[k]-p[0]
    if np.linalg.norm(v) < 1e-6:
        return p, 0.0
    ang = float(start_yaw)-float(np.arctan2(v[1], v[0]))
    ang = (ang+np.pi) % (2*np.pi)-np.pi
    c, s = np.cos(ang), np.sin(ang)
    out = (p-p[0]) @ np.array([[c, -s], [s, c]])+np.asarray(start_xy, float)
    return out, float(np.rad2deg(abs(ang)))


def lead_in_quintic(path_xy, start_xy, start_yaw, length=6.0, step=0.25):
    """五次 Hermite 前导段：位置/切向/曲率同时匹配，接缝处 C² 连续。

    起点曲率取 0（车身笔直），接缝处取原轨迹在 ``length`` 处的曲率。
    """
    from src.carla_bridge.path_profile import Path as _Path
    p = np.asarray(path_xy, float)
    geom = _Path.build(p)
    L = float(min(length, geom.s[-1]*0.3))
    if L <= 1.0:
        return p
    p1, yaw1 = geom.pose(L)

    s_a = max(0.0, L-1.0)
    s_b = min(geom.s[-1], L+1.0)
    pa, _ = geom.pose(s_a)
    pb, _ = geom.pose(s_b)
    yaw_a = float(np.arctan2(*(pa-p1)[::-1])) if s_a < L-1e-6 else yaw1
    yaw_b = float(np.arctan2(*(pb-p1)[::-1])) if s_b > L+1e-6 else yaw1
    d1 = (yaw1-yaw_a+np.pi) % (2*np.pi)-np.pi
    d2 = (yaw_b-yaw1+np.pi) % (2*np.pi)-np.pi
    k1 = float((d1+d2)/max(s_b-s_a, 1e-6))

    p0 = np.asarray(start_xy, float)
    m0 = np.array([np.cos(start_yaw), np.sin(start_yaw)])
    n0 = np.array([-np.sin(start_yaw), np.cos(start_yaw)])
    m1 = np.array([np.cos(yaw1), np.sin(yaw1)])
    n1 = np.array([-np.sin(yaw1), np.cos(yaw1)])
    k0 = 0.0

    pts = []
    for t in np.arange(0.0, 1.0, step/L):
        h00 = 1-10*t**3+15*t**4-6*t**5
        h10 = t-6*t**3+8*t**4-3*t**5
        h20 = 0.5*t**2-1.5*t**3+1.5*t**4-0.5*t**5
        h01 = 10*t**3-15*t**4+6*t**5
        h11 = -4*t**3+7*t**4-3*t**5
        h21 = 0.5*t**3-t**4+0.5*t**5
        pts.append(h00*p0+h10*L*m0+h20*L*L*k0*n0+h01*p1+h11*L*m1+h21*L*L*k1*n1)
    keep = geom.xy[geom.s > L+0.35]      # 跳过重复点，避免退化线段
    out = np.vstack([np.array(pts), keep]) if len(keep) else np.array(pts)
    # 去掉间距过小的相邻点
    d = np.linalg.norm(np.diff(out, axis=0), axis=1)
    good = np.concatenate([[True], d > 0.05])
    return out[good]


def bend_start(path_xy, start_xy, start_yaw, length=10.0):
    """把轨迹头部平滑地"掰"到车身朝向上：旋转角在 length 内按余弦权重衰减到 0。

    权重函数 C1 连续，因此不会产生折角；远离起点处完全不变。
    """
    p = np.asarray(path_xy, float).copy()
    if len(p) < 8:
        return p, 0.0
    d1 = p[1]-p[0]
    if np.linalg.norm(d1) < 1e-9:
        return p, 0.0
    ang = float(start_yaw)-float(np.arctan2(d1[1], d1[0]))
    ang = (ang+np.pi) % (2*np.pi)-np.pi
    if abs(ang) < np.deg2rad(0.5):
        return p, 0.0
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    w = np.where(s < length, 0.5*(1.0+np.cos(np.pi*np.clip(s/length, 0.0, 1.0))), 0.0)
    rel = p-np.asarray(start_xy, float)
    out = rel.copy()
    for i, wi in enumerate(w):
        if wi <= 0:
            continue
        a = ang*wi
        c, sn = np.cos(a), np.sin(a)
        out[i] = np.array([[c, -sn], [sn, c]]) @ rel[i]
    return out+np.asarray(start_xy, float), float(np.rad2deg(abs(ang)))


def pin_and_refit(path_xy, start_xy, start_yaw, pin_m=4.0, rms=0.10, step=0.2):
    """端点钉扎 + 平滑三次样条重拟合。

    1) 把前 pin_m 米重建为从自车位姿出发、切向一致的三次 Hermite 段（端点钉扎）；
    2) 再对整条曲线做一次平滑样条重拟合，抹掉接缝处的一阶不连续，
       得到处处 C2 连续的曲线。
    返回 (新点列, 起点切向偏差 deg, 相对原轨迹最大偏差 m)。
    """
    from src.carla_bridge.path_profile import Path as _Path
    p = np.asarray(path_xy, float)
    if len(p) < 12:
        return p, 0.0, 0.0
    start = np.asarray(start_xy, float)
    geom = _Path.build(p)
    L = float(min(pin_m, geom.s[-1]*0.3))
    if L > 1.0:
        p1, yaw1 = geom.pose(L)
        m0 = np.array([np.cos(start_yaw), np.sin(start_yaw)])
        m1 = np.array([np.cos(yaw1), np.sin(yaw1)])
        seg = []
        for t in np.arange(0.0, 1.0, step/L):
            h00 = 2*t**3-3*t**2+1
            h10 = t**3-2*t**2+t
            h01 = -2*t**3+3*t**2
            h11 = t**3-t**2
            seg.append(h00*start+h10*L*m0+h01*p1+h11*L*m1)
        keep = geom.xy[geom.s > L+0.3]
        pinned = np.vstack([np.array(seg), keep]) if len(keep) else np.array(seg)
    else:
        pinned = p.copy()
    pinned = pinned+(start-pinned[0])
    ref, dev = refit_spline(pinned, rms=rms)
    ref = ref+(start-ref[0])                     # 重拟合后再平移回自车中心
    yaw0 = float(np.arctan2(*(ref[min(3, len(ref)-1)]-ref[0])[::-1]))
    turn = float(np.rad2deg(abs((yaw0-start_yaw+np.pi) % (2*np.pi)-np.pi)))
    return ref, turn, float(dev)


def pin_and_refit_hard(path_xy, start_xy, start_yaw, pin_m=4.0, knot_m=8.0,
                       start_weight=300.0, lead_n=3):
    """硬约束版本：带权最小二乘 B 样条拟合。

    前 pin_m 米（含起点）与起点后方的 lead 虚拟点都给极大权重，
    其余点权重 1。这样起点位置与切向被"钉住"，同时整条曲线仍是
    单一 B 样条，处处 C2 连续，没有接缝。
    """
    from scipy.interpolate import LSQUnivariateSpline
    from src.carla_bridge.path_profile import Path as _Path
    p = np.asarray(path_xy, float)
    if len(p) < 16:
        return p, 0.0, 0.0
    start = np.asarray(start_xy, float)
    geom = _Path.build(p)
    L = float(min(pin_m, geom.s[-1]*0.3))

    pts, wts = [], []
    m0 = np.array([np.cos(start_yaw), np.sin(start_yaw)])
    for k in range(lead_n, 0, -1):          # 起点后方虚拟点，保证起始切向
        pts.append(start-m0*(0.6*k))
        wts.append(start_weight)
    pts.append(start); wts.append(start_weight)
    for s in np.arange(0.5, L, 0.5):        # 钉扎段
        q, _ = geom.pose(float(s))
        pts.append(q); wts.append(start_weight*0.5)
    for s in np.arange(L, geom.s[-1], 0.5):  # 其余部分
        q, _ = geom.pose(float(s))
        pts.append(q); wts.append(1.0)
    pts = np.array(pts)
    wts = np.array(wts)

    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(d)])
    t = t/t[-1]
    n_int = max(3, int(round(t[-1]*0)+int(np.ceil(t.size/12))))
    inner = np.linspace(0.0, 1.0, n_int+2)[1:-1]
    inner = inner[(inner > 4*1e-3) & (inner < 1-4*1e-3)]
    splx = LSQUnivariateSpline(t, pts[:, 0], inner, k=3, w=wts)
    sply = LSQUnivariateSpline(t, pts[:, 1], inner, k=3, w=wts)
    tt = np.linspace(0.0, 1.0, 900)
    dense = np.column_stack([splx(tt), sply(tt)])
    dense = dense+(start-dense[0])
    yaw0 = float(np.arctan2(*(dense[min(3, len(dense)-1)]-dense[0])[::-1]))
    turn = float(np.rad2deg(abs((yaw0-start_yaw+np.pi) % (2*np.pi)-np.pi)))
    step = max(1, len(dense)//len(p))
    dev = float(np.max(np.linalg.norm(dense[::step][:len(p)]-p[:len(dense[::step])], axis=1)))
    return dense, turn, dev


def pin_and_spline(path_xy, start_xy, start_yaw, sample_m=3.0):
    """边界切向硬约束的三次样条：起点导数 = 车身朝向（精确 0°）。

    做法：沿弧长每 sample_m 取一个点，用 make_interp_spline 的 bc_type
    把两端一阶导钉死；曲线内部 C2 连续，天然无折角。
    """
    from scipy.interpolate import make_interp_spline
    from src.carla_bridge.path_profile import Path as _Path
    p = np.asarray(path_xy, float)
    if len(p) < 10:
        return p, 0.0, 0.0
    start = np.asarray(start_xy, float)
    geom = _Path.build(p)
    total = float(geom.s[-1])
    ss = np.arange(0.0, total, sample_m)
    if ss[-1] < total-0.5:
        ss = np.append(ss, total)
    knots = np.array([geom.pose(float(s))[0] for s in ss])
    knots[0] = start
    m0 = np.array([np.cos(start_yaw), np.sin(start_yaw)])
    _, yaw_end = geom.pose(total)
    m1 = np.array([np.cos(yaw_end), np.sin(yaw_end)])
    d = np.linalg.norm(np.diff(knots, axis=0), axis=1)
    tt = np.concatenate([[0.0], np.cumsum(d)])
    tt = tt/tt[-1]
    scale = total                              # 导数按弧长归一
    bc = ([(1, m0*scale)], [(1, m1*scale)])     # 两端一阶导：起点沿车身朝向，终点沿原轨迹
    spl = make_interp_spline(tt, knots, k=3, bc_type=bc)
    uu = np.linspace(0.0, 1.0, max(600, int(total/0.2)))
    dense = spl(uu)
    dense[0] = start
    yaw0 = float(np.arctan2(*(dense[min(4, len(dense)-1)]-dense[0])[::-1]))
    turn = float(np.rad2deg(abs((yaw0-start_yaw+np.pi) % (2*np.pi)-np.pi)))
    return dense, turn, 0.0


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
