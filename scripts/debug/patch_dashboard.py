"""dashboard 修复：全部车道计入自由空间 + 相机沿轨迹 + 右图可点击 + 障碍吸附车道。"""
from pathlib import Path

f = Path(__file__).resolve().parents[2] / 'scripts/dashboard_app.py'
t = f.read_text(encoding='utf8')

# ---------------- 1) 全部车道（不分方向）都算自由空间 ----------------
old = t[t.index('def road_quads_at('):t.index('def obstacle_bodies()')]
new = '''def road_quads_at(world, ego_xy, back_m=40.0, fwd_m=120.0, lanes_each_side=4):
    """自车所在道路的【全部】可行驶车道都算自由空间（不分方向）。"""
    from src.carla_bridge.occupancy import road_quads
    import carla
    cmap = world.get_map()
    w0 = cmap.get_waypoint(carla.Location(x=float(ego_xy[0]), y=float(ego_xy[1]), z=1.0),
                           project_to_road=True)
    back, probe = [], w0
    while len(back)*2.0 < back_m:
        prv = probe.previous(2.0)
        if len(prv) != 1:
            break
        probe = prv[0]
        back.append(probe)
    chain = list(reversed(back))+[w0]
    i_ego = len(chain)-1
    while (len(chain)-i_ego-1)*2.0 < fwd_m:
        nxt = chain[-1].next(2.0)
        if not nxt:
            break
        same = [w for w in nxt if w.lane_id == chain[-1].lane_id]
        chain.append((same or nxt)[0])

    quads = road_quads(chain)
    lanes = 1
    for side in (+1, -1):
        for k in range(1, lanes_each_side+1):
            pts = []
            for wp in chain:
                x = wp
                ok = True
                for _ in range(k):
                    x = x.get_left_lane() if side > 0 else x.get_right_lane()
                    if x is None or x.lane_type.name != 'Driving':
                        ok = False
                        break
                if ok:
                    pts.append(x)
            if len(pts) < 2:
                break
            runs, cur = [], [pts[0]]
            for a, b in zip(pts[:-1], pts[1:]):
                d = np.hypot(a.transform.location.x-b.transform.location.x,
                             a.transform.location.y-b.transform.location.y)
                if d < 6.0:
                    cur.append(b)
                else:
                    if len(cur) >= 2:
                        runs.append(cur)
                    cur = [b]
            if len(cur) >= 2:
                runs.append(cur)
            for run in runs:
                quads = quads+road_quads(run)
                lanes += 1
    return chain, quads, i_ego


'''
t = t.replace(old, new)

# ---------------- 2) 相机沿轨迹中段方向 ----------------
old = '''    bp = world.get_blueprint_library().find('sensor.camera.rgb')'''
new = '''    cam_yaw = float(STATE.get('lane_yaw') or 0.0)
    if path_xy is not None and len(path_xy) > 6:
        k = len(path_xy)//2
        cam_yaw = float(np.rad2deg(np.arctan2(*(path_xy[k+2]-path_xy[k])[::-1])))
    bp = world.get_blueprint_library().find('sensor.camera.rgb')'''
assert old in t
t = t.replace(old, new, 1)
t = t.replace("carla.Rotation(pitch=-90.0, yaw=float(STATE['lane_yaw'] or 0.0))))",
              "carla.Rotation(pitch=-90.0, yaw=cam_yaw)))")

# ---------------- 3) 保存像素→世界 的单应矩阵，供右图点击 ----------------
old = '''    img = rgb.astype(np.float32)'''
new = '''    # 记录 世界→像素 单应矩阵，供右图点击反算
    try:
        probe_w = np.array([[centre[0], centre[1]], [centre[0]+15, centre[1]],
                            [centre[0], centre[1]+15], [centre[0]+15, centre[1]+15]], np.float32)
        probe_p = px(probe_w).astype(np.float32)
        H = cv2.getPerspectiveTransform(probe_w, probe_p)
        STATE['view_H'] = H.tolist()
        STATE['view_size'] = [int(w), int(h)]
    except Exception:
        STATE['view_H'] = None

    img = rgb.astype(np.float32)'''
assert old in t
t = t.replace(old, new, 1)

# ---------------- 4) 右图点击 + 障碍吸附车道 ----------------
old = '''                if self.path == '/api/click':
                    import carla
                    mode = p.get('mode', 'ego')
                    if p.get('side') == 'view':
                        return self._json(dict(ok=False, msg='请在左图点击设置'))
                    frame, occ, road, chain, i_ego = build_occupancy(w)
                    res = occ.shape[0]
                    u, v = float(p['u']), float(p['v'])
                    sx = (u*res)/(res-1)*2-1
                    sy = (v*res)/(res-1)*2-1
                    wx, wy = frame.world(np.array([[sx, sy]], float))[0]'''
new = '''                if self.path == '/api/click':
                    import carla
                    mode = p.get('mode', 'ego')
                    u, v = float(p['u']), float(p['v'])
                    if p.get('side') == 'view':
                        H = STATE.get('view_H')
                        if not H:
                            return self._json(dict(ok=False, msg='先生成一次轨迹再点右图'))
                        Hm = np.array(H, float)
                        W, Hh = STATE.get('view_size', [1600, 900])
                        inv = np.linalg.inv(Hm)
                        vec = inv @ np.array([u*W, v*Hh, 1.0])
                        wx, wy = float(vec[0]/vec[2]), float(vec[1]/vec[2])
                    else:
                        frame, occ, road, chain, i_ego = build_occupancy(w)
                        res = occ.shape[0]
                        sx = (u*res)/(res-1)*2-1
                        sy = (v*res)/(res-1)*2-1
                        wx, wy = frame.world(np.array([[sx, sy]], float))[0]'''
assert old in t
t = t.replace(old, new, 1)

old = '''                    wp = w.get_map().get_waypoint(carla.Location(x=float(wx), y=float(wy), z=1.0))
                    if mode == 'ego':'''
new = '''                    wp = w.get_map().get_waypoint(carla.Location(x=float(wx), y=float(wy), z=1.0),
                                                  project_to_road=True)
                    wx, wy = float(wp.transform.location.x), float(wp.transform.location.y)
                    if mode == 'ego':'''
assert old in t
t = t.replace(old, new, 1)

f.write_text(t, encoding='utf8')
print('dashboard: 全部车道 / 相机沿轨迹 / 右图可点 / 吸附车道 均已修复')
