"""加 /api/set_pose（世界坐标直接设起点/终点/障碍），并把批量脚本改成用它。"""
from pathlib import Path

p = Path('scripts/dashboard_app.py')
s = p.read_text(encoding='utf8')

if '/api/set_pose' not in s:
    s = s.replace("""                if self.path == '/api/maps':""",
"""                if self.path == '/api/set_pose':
                    import carla
                    if p.get('ego'):
                        e = p['ego']
                        wp = w.get_map().get_waypoint(
                            carla.Location(x=float(e[0]), y=float(e[1]), z=1.0),
                            project_to_road=True)
                        STATE['ego'] = [float(e[0]), float(e[1]),
                                         float(wp.transform.rotation.yaw),
                                         float(wp.transform.location.z)]
                    if p.get('goal'):
                        g = p['goal']
                        STATE['goal'] = [float(g[0]), float(g[1]), 0.0]
                    if p.get('obstacles') is not None:
                        out = []
                        for o in p['obstacles']:
                            wp = w.get_map().get_waypoint(
                                carla.Location(x=float(o[0]), y=float(o[1]), z=1.0),
                                project_to_road=True)
                            out.append(dict(x=float(o[0]), y=float(o[1]),
                                            yaw=float(wp.transform.rotation.yaw),
                                            z=float(wp.transform.location.z),
                                            kind=o[2] if len(o) > 2 else 'car'))
                        STATE['obstacles'] = out
                    STATE['plan'] = None
                    respawn(w)
                    return self._json(dict(ok=True, occ=occ_png(w), view=view_png(w),
                                           ego=STATE['ego'], goal=STATE['goal'],
                                           region=STATE['region'], msg=status()))
                if self.path == '/api/maps':""", 1)
    p.write_text(s, encoding='utf8')
    print('server patched')

q = Path('scripts/batch_dashboard_figs.py')
t = q.read_text(encoding='utf8')
t = t.replace("""        ego = post(args.port, '/api/click', dict(mode='ego', u=0.5, v=0.5)).get('ego')
        if not ego:
            continue
        try:
            pg, wp = lane_point(ego[0], ego[1], int(dist/2))
            po, _ = lane_point(ego[0], ego[1], int((dist*0.45)/2))
        except Exception:
            continue
        post(args.port, '/api/click', dict(mode='goal', **dict(zip(('u', 'v'), uv_of(
            reg, pg.transform.location.x, pg.transform.location.y)))))
        if mode:
            if mode == 2:
                left = po.get_left_lane()
                tgt = left if (left is not None and left.lane_type.name == 'Driving'
                               and np.sign(left.lane_id) == np.sign(po.lane_id)) else po
            else:
                tgt = po
            post(args.port, '/api/click', dict(mode='car', **dict(zip(('u', 'v'), uv_of(
                reg, tgt.transform.location.x, tgt.transform.location.y)))))""",
"""        ex, ey = reg['cx'], reg['cy']          # 区域中心 = 该 spawn 点，直接用世界坐标
        try:
            pg, wp = lane_point(ex, ey, int(dist/2))
            po, _ = lane_point(ex, ey, int((dist*0.45)/2))
        except Exception:
            continue
        obs = []
        if mode:
            if mode == 2:
                left = po.get_left_lane()
                tgt = left if (left is not None and left.lane_type.name == 'Driving'
                               and np.sign(left.lane_id) == np.sign(po.lane_id)) else po
            else:
                tgt = po
            obs = [[tgt.transform.location.x, tgt.transform.location.y, 'car']]
        post(args.port, '/api/set_pose', dict(
            ego=[ex, ey],
            goal=[pg.transform.location.x, pg.transform.location.y],
            obstacles=obs))""", 1)
q.write_text(t, encoding='utf8')
print('batch patched')
