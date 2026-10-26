"""重做 overlay 配色与图层：低饱和填充 + 清晰描边 + 只留必要元素。"""
from pathlib import Path

g = Path(__file__).resolve().parents[2]/'scripts/carla_scenario_sim.py'
s = g.read_text(encoding='utf8')

start = s.index('        def overlay(rgb, inverse, text):')
end = s.index('        text = ("%s   plan %.0f ms')

new = '''        # ---------------- 论文级配色（BGR） ----------------
        C_CORR_FILL = (196, 172, 132)    # 淡蓝灰：安全走廊
        C_CORR_EDGE = (168, 120, 70)     # 中蓝：走廊轮廓
        C_BAND_FILL = (150, 208, 160)    # 淡绿：车身扫掠带
        C_BAND_EDGE = (86, 146, 60)      # 深绿：带边界
        C_WIDTH     = (150, 208, 160)
        C_TRAJ      = (235, 235, 235)    # 近白：规划轨迹
        C_OBST      = (72, 72, 214)      # 红：障碍
        C_START     = (36, 150, 240)     # 橙：起点
        C_GOAL      = (72, 72, 214)      # 红：终点
        C_TEXT      = (238, 238, 238)
        C_SUB       = (170, 175, 180)
        C_BAR       = (32, 32, 36)

        def overlay(rgb, inverse, text):
            import cv2
            from shapely.geometry import Polygon as _Poly, MultiPolygon as _Multi, LineString as _Line
            from shapely.ops import unary_union as _union
            h, w = rgb.shape[:2]
            ground = z

            def project(points):
                points = np.asarray(points, float).reshape(-1, 2)
                world = np.column_stack([points, np.full(len(points), ground), np.ones(len(points))])
                cam_pts = world @ inverse.T
                if np.any(cam_pts[:, 0] <= 0.05):
                    return None
                f = w/2.0
                px = np.column_stack([w/2 + f*cam_pts[:, 1]/cam_pts[:, 0],
                                      h/2 - f*cam_pts[:, 2]/cam_pts[:, 0]])
                return np.rint(px).astype(np.int32)

            img = rgb.copy()

            def fill(part, colour, alpha):
                pts = project(np.asarray(part.exterior.coords)[:-1])
                if pts is None or len(pts) < 3:
                    return
                layer = img.copy()
                cv2.fillPoly(layer, [pts], colour)
                cv2.addWeighted(layer, alpha, img, 1-alpha, 0, img)

            def stroke(part, colour, thick):
                pts = project(np.asarray(part.exterior.coords)[:-1])
                if pts is not None and len(pts) >= 3:
                    cv2.polylines(img, [pts], True, colour, thick, cv2.LINE_AA)

            # 1) 安全走廊：所有格子求并 → 一条轮廓（不再画 128 条线）
            band_clip = _Line(path.xy).buffer(7.5, cap_style=2, join_style=2)
            parts = []
            for poly in corridor:
                wp = _Poly(frame.world(np.asarray(poly)))
                if not wp.is_valid:
                    wp = wp.buffer(0)
                inter = wp.intersection(band_clip)
                if not inter.is_empty:
                    parts.extend(inter.geoms if isinstance(inter, _Multi) else [inter])
            if parts:
                sil = _union(parts)
                for part in (sil.geoms if isinstance(sil, _Multi) else [sil]):
                    fill(part, C_CORR_FILL, 0.30)
                for part in (sil.geoms if isinstance(sil, _Multi) else [sil]):
                    stroke(part, C_CORR_EDGE, 2)

            # 2) 车身扫掠带：填充 + 两侧边界 + 每 5 m 一道截面
            sweep = np.vstack([band_l, band_r[::-1]])
            pts = project(sweep)
            if pts is not None:
                layer = img.copy()
                cv2.fillPoly(layer, [pts], C_BAND_FILL)
                cv2.addWeighted(layer, 0.32, img, 0.68, 0, img)
            for line in (band_l, band_r):
                pl = project(line)
                if pl is not None:
                    cv2.polylines(img, [pl], False, C_BAND_EDGE, 2, cv2.LINE_AA)
            for s_i in np.arange(0.0, path.s[-1], 5.0):
                p_i, yaw_i = path.pose(float(s_i))
                n = np.array([-np.sin(yaw_i), np.cos(yaw_i)])
                seg = project(np.array([p_i+n*half, p_i-n*half]))
                if seg is not None:
                    cv2.line(img, tuple(seg[0]), tuple(seg[1]), C_BAND_EDGE, 1, cv2.LINE_AA)

            # 3) 规划轨迹（中心线，近白细线压在带上）
            cl = project(path.xy)
            if cl is not None:
                cv2.polylines(img, [cl], False, C_TRAJ, 2, cv2.LINE_AA)

            # 4) 障碍物：只描边
            for o in obstacle_bodies:
                pts_o = project(np.vstack([o.corners(0.0)]))
                if pts_o is not None:
                    cv2.polylines(img, [pts_o], True, C_OBST, 2, cv2.LINE_AA)

            # 5) 起点 / 终点
            for point, colour, label, off in ((ego_xy, C_START, 'START', (12, -10)),
                                              (goal_xy, C_GOAL, 'GOAL', (12, -10))):
                pt = project(point[None])
                if pt is None:
                    continue
                cv2.circle(img, tuple(pt[0]), 9, colour, 2, cv2.LINE_AA)
                cv2.putText(img, label, (int(pt[0][0])+off[0], int(pt[0][1])+off[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, .58, colour, 2, cv2.LINE_AA)

            # 6) 顶部信息条 + 图例
            bar = np.zeros((86, w, 3), np.uint8)
            bar[:] = C_BAR
            img = np.vstack([bar, img])
            cv2.putText(img, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, .62, C_TEXT, 1, cv2.LINE_AA)
            legend = [('safety corridor', C_CORR_FILL), ('body sweep', C_BAND_FILL),
                      ('planned path', C_TRAJ), ('static obstacle', C_OBST)]
            xoff = 16
            for label, colour in legend:
                cv2.rectangle(img, (xoff, 52), (xoff+18, 66), colour, -1)
                cv2.rectangle(img, (xoff, 52), (xoff+18, 66), (110, 110, 110), 1)
                cv2.putText(img, label, (xoff+24, 64), cv2.FONT_HERSHEY_SIMPLEX, .46, C_SUB, 1, cv2.LINE_AA)
                xoff += 24 + 9*len(label) + 26
            return img

'''
g.write_text(s[:start] + new + s[end:], encoding='utf8')
print('overlay 重画完成（低饱和 + 合并轮廓 + 图例）')
