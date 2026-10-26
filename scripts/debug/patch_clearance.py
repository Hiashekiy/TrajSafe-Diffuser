"""给 dashboard 加“拟合后净空修复”：保证最终轨迹不穿过障碍。"""
from pathlib import Path

p = Path('scripts/dashboard_app.py')
s = p.read_text(encoding='utf8')

anchor = "def plan(world, seed):"
code = '''def path_min_clearance(path, bodies, half_len, half_wid, step=1.0, margin=0.0):
    """轨迹上自车矩形到各障碍多边形的最小距离（米）。"""
    from shapely.geometry import Polygon
    polys = [Polygon(b.corners(0.0)) for b in bodies]
    if not polys:
        return 99.0
    worst = 99.0
    for s in np.arange(0.0, path.s[-1], step):
        p, yaw = path.pose(float(s))
        c, sn = np.cos(yaw), np.sin(yaw)
        d = np.array([c, sn])
        n = np.array([-sn, c])
        rect = Polygon([p+d*(half_len+margin)+n*(half_wid+margin),
                        p+d*(half_len+margin)-n*(half_wid+margin),
                        p-d*(half_len+margin)-n*(half_wid+margin),
                        p-d*(half_len+margin)+n*(half_wid+margin)])
        for P in polys:
            worst = min(worst, float(rect.distance(P)))
    return worst


def push_clear(xy, bodies, radius=0.6, iters=3):
    """把穿过/贴着障碍的点沿远离障碍中心的方向推出去，再轻微平滑。"""
    from shapely.geometry import Point, Polygon
    polys = [Polygon(b.corners(0.0)) for b in bodies]
    pts = np.asarray(xy, float).copy()
    if not polys:
        return pts
    for _ in range(iters):
        moved = 0
        for i in range(len(pts)):
            p = pts[i]
            for P in polys:
                if Point(p[0], p[1]).distance(P) >= radius:
                    continue
                c = np.array([P.centroid.x, P.centroid.y])
                v = p-c
                nv = float(np.linalg.norm(v))
                dirv = v/nv if nv > 1e-6 else np.array([0.0, 1.0])
                for t in np.linspace(0.5, 10.0, 40):
                    q = p+dirv*t
                    if Point(q[0], q[1]).distance(P) >= radius:
                        pts[i] = q
                        moved += 1
                        break
        if not moved:
            break
        # 平滑被推开的点，避免折角（端点不动）
        sm = pts.copy()
        sm[1:-1] = 0.5*pts[1:-1]+0.25*pts[:-2]+0.25*pts[2:]
        pts = sm
    return pts


'''
s = s.replace(anchor, code+anchor, 1)

old = """    pin, turn, dev = pin_both_ends(raw, ego_xy, np.deg2rad(ego_yaw),
                                   goal_xy, goal_yaw,
                                   sample_m=float(c['planner'].get('pin_sample_m', 12.0)),
                                   end_gain=float(c['planner'].get('pin_end_gain', 0.6)))
    path = GeomPath.build(pin)"""
new = """    sample_m = float(c['planner'].get('pin_sample_m', 12.0))
    end_gain = float(c['planner'].get('pin_end_gain', 0.6))
    pin, turn, dev = pin_both_ends(raw, ego_xy, np.deg2rad(ego_yaw), goal_xy, goal_yaw,
                                   sample_m=sample_m, end_gain=end_gain)
    path = GeomPath.build(pin)
    # 拟合会把原始路径的绕行弯“切掉”，所以拟合后再查一次净空，不够就推出去重拟合
    hl, hw = BODY['car']
    if STATE['ego_actor'] is not None:
        bb = STATE['ego_actor'].bounding_box
        hl, hw = float(bb.extent.x), float(bb.extent.y)
    want = 0.30
    for _ in range(3):
        got = path_min_clearance(path, bodies, hl, hw)
        if got >= want:
            break
        fixed = push_clear(pin, bodies, radius=0.65)
        pin, turn, dev = pin_both_ends(fixed, ego_xy, np.deg2rad(ego_yaw),
                                       goal_xy, goal_yaw,
                                       sample_m=max(4.0, sample_m*0.6), end_gain=end_gain)
        path = GeomPath.build(pin)
    clearance = path_min_clearance(path, bodies, hl, hw)"""
s = s.replace(old, new, 1)

# 提示栏用自己算的净空
old = """    if check is not None:
        note += ' ｜ 越界 %s m ｜ 最小间隙 %s m ｜ %s' % ("""
new = """    note += ' ｜ 最小间隙 %.2f m%s' % (clearance, '' if clearance >= 0.30 else '（不足）')
    if check is not None:
        note += ' ｜ 越界 %s m ｜ 校验 %s' % ("""
s = s.replace(old, new, 1)
s = s.replace("""            check['max_road_excursion_m'],
            '—' if check['min_clearance_m'] is None else check['min_clearance_m'],
            '通过' if check['passed'] else '不通过')""",
"""            check['max_road_excursion_m'],
            '通过' if check['passed'] else '不通过')""", 1)

p.write_text(s, encoding='utf8')
print('patched ok')
