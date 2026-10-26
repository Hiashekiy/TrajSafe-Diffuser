"""把可视化从 debug 画线改成「俯视 RGB + Python 投影半透明带」。"""
from pathlib import Path

g = Path(__file__).resolve().parents[2]/'scripts/carla_scenario_sim.py'
s = g.read_text(encoding='utf8')
start = s.index('        life = 900.0')
end = s.index('        np.savez_compressed(out/')

new = '''        # ============ 俯视 RGB + Python 侧投影半透明叠加 ============
        z = ego.get_transform().location.z + 0.06
        corridor = (result['viz'] or {}).get('corridor', {}).get('polygons', [])
        lane_yaw = ego_wp.transform.rotation.yaw

        half = ego_body.half_width + 0.15
        band_l, band_r = [], []
        for s_i in np.arange(0.0, path.s[-1], 1.0):
            p_i, yaw_i = path.pose(float(s_i))
            n = np.array([-np.sin(yaw_i), np.cos(yaw_i)])
            band_l.append(p_i + n*half)
            band_r.append(p_i - n*half)
        band_l, band_r = np.array(band_l), np.array(band_r)
        centre_world = path.pose(path.s[-1]*0.5)[0]

        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '1280')
        cam_bp.set_attribute('image_size_y', '720')
        cam_bp.set_attribute('fov', '90')
        cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(z=60)))
        actors.append(cam)
        captured = []
        cam.listen(lambda img: captured.append(
            (np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, :3].copy(),
             np.asarray(img.transform.get_inverse_matrix(), dtype=float))))

        def topdown(centre, height):
            return carla.Transform(carla.Location(x=float(centre[0]), y=float(centre[1]), z=z+height),
                                   carla.Rotation(pitch=-90.0, yaw=float(lane_yaw)))

        n_frames = max(2, args.frames)
        for s_i in np.linspace(0.0, path.s[-1], n_frames):      # 视频：俯视沿轨迹平移
            cam.set_transform(topdown(path.pose(float(s_i))[0], 34.0))
            world.tick()
        cam.set_transform(topdown(centre_world, 52.0))         # 截图：整条轨迹一屏
        for _ in range(4):
            world.tick()

        def overlay(rgb, inverse, text):
            import cv2
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
            for poly in corridor:                               # 走廊：淡青绿
                pts = project(frame.world(np.asarray(poly)))
                if pts is None:
                    continue
                layer = img.copy()
                cv2.fillPoly(layer, [pts], (170, 225, 190))
                cv2.addWeighted(layer, 0.16, img, 0.84, 0, img)
            pts = project(np.vstack([band_l, band_r[::-1]]))     # 车身扫掠带
            if pts is not None:
                layer = img.copy()
                cv2.fillPoly(layer, [pts], (70, 215, 95))
                cv2.addWeighted(layer, 0.34, img, 0.66, 0, img)
            for s_i in np.arange(0.0, path.s[-1], 3.0):          # 每 3 m 一道截面
                p_i, yaw_i = path.pose(float(s_i))
                n = np.array([-np.sin(yaw_i), np.cos(yaw_i)])
                seg = project(np.array([p_i+n*half, p_i-n*half]))
                if seg is not None:
                    cv2.line(img, tuple(seg[0]), tuple(seg[1]), (35, 150, 60), 2, cv2.LINE_AA)
            cl = project(path.xy)                                # 中心线
            if cl is not None:
                cv2.polylines(img, [cl], False, (20, 110, 35), 2, cv2.LINE_AA)
            for o in obstacle_bodies:                            # 障碍框
                pts_o = project(np.vstack([o.corners(0.0)]))
                if pts_o is not None:
                    cv2.polylines(img, [pts_o], True, (60, 60, 235), 2, cv2.LINE_AA)
            for point, colour, label in ((ego_xy, (255, 170, 60), 'START'),
                                         (goal_xy, (60, 60, 235), 'GOAL')):
                pt = project(point[None])
                if pt is not None:
                    cv2.circle(img, tuple(pt[0]), 8, colour, 2, cv2.LINE_AA)
                    cv2.putText(img, label, (int(pt[0][0])+10, int(pt[0][1])-8),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, colour, 2, cv2.LINE_AA)
            bar = np.zeros((74, w, 3), np.uint8)
            bar[:] = (26, 26, 30)
            img = np.vstack([bar, img])
            cv2.putText(img, text, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, .64, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(img, 'GREEN band: body sweep   PALE: safety corridor   RED: static obstacle   ORANGE: start',
                        (16, 58), cv2.FONT_HERSHEY_SIMPLEX, .5, (190, 195, 200), 1, cv2.LINE_AA)
            return img

        text = ("%s %s   plan %.0f ms   corridor %s cells   min clearance %s m   footprint %s"
                % (cfg['id'], cfg['name'], plan_ms, summary['corridor_cells'],
                   summary['footprint']['min_clearance_m'],
                   'PASS' if summary['footprint']['passed'] else 'FAIL'))
        if captured:
            import cv2
            cv2.imwrite(str(out/'screenshot.png'), overlay(*captured[-1], text))
            hh, ww = overlay(*captured[0], text).shape[:2]
            vw = cv2.VideoWriter(str(out/'scenario.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, (ww, hh))
            for rgb_i, inv_i in captured:
                vw.write(overlay(rgb_i, inv_i, text))
            vw.release()
            summary['frames'] = len(captured)

'''
g.write_text(s[:start] + new + s[end:], encoding='utf8')
print('重写可视化段完成')
