"""确认右图朝向：前进方向是否朝上、自车右侧是否朝右。"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, 'E:/CARLA_0.9.16/PythonAPI/carla')
import cv2
import carla


def main():
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    sp = world.get_map().get_spawn_points()[45]
    ex, ey = sp.location.x, sp.location.y
    yaw = float(np.deg2rad(sp.rotation.yaw))
    span = 110.0
    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', '900')
    bp.set_attribute('image_size_y', '900')
    bp.set_attribute('fov', '90')
    cam = world.spawn_actor(bp, carla.Transform(
        carla.Location(x=ex, y=ey, z=span/2.0),
        carla.Rotation(pitch=-90.0, yaw=float(np.rad2deg(yaw)))))
    frames = []
    cam.listen(lambda img: frames.append((
        np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, :3].copy(),
        np.asarray(img.transform.get_inverse_matrix(), float))))
    st = world.get_settings()
    was = st.synchronous_mode
    st.synchronous_mode = True
    st.fixed_delta_seconds = 0.05
    world.apply_settings(st)
    for _ in range(6):
        world.tick()
    st.synchronous_mode = was
    world.apply_settings(st)
    cam.destroy()
    rgb, inv = frames[-1]
    h, w = rgb.shape[:2]

    def px(pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        hom = np.column_stack([pts, np.full(len(pts), 0.06), np.ones(len(pts))])
        c3 = hom @ inv.T
        c3[:, 0] = np.maximum(c3[:, 0], 0.05)
        f = w/2.0
        return np.column_stack([w/2+f*c3[:, 1]/c3[:, 0], h/2-f*c3[:, 2]/c3[:, 0]])

    d = np.array([np.cos(yaw), np.sin(yaw)])
    n = np.array([-np.sin(yaw), np.cos(yaw)])
    ego = np.array([ex, ey])
    ahead, right = ego+d*20, ego+n*20
    print('ego', px(ego).round(1))
    print('ahead+20m', px(ahead).round(1))
    print('right+20m', px(right).round(1))
    img = rgb.copy()
    for p, col in ((ego, (240, 240, 240)), (ahead, (80, 220, 90)), (right, (60, 60, 235))):
        u, v = px(p)[0]
        cv2.circle(img, (int(u), int(v)), 8, col, -1)
    cv2.line(img, tuple(px(ego)[0].astype(int)), tuple(px(ahead)[0].astype(int)), (80, 220, 90), 2)
    cv2.line(img, tuple(px(ego)[0].astype(int)), tuple(px(right)[0].astype(int)), (60, 60, 235), 2)
    ok, buf = cv2.imencode('.png', img)
    out = ROOT/'outputs'/'probe_orient.png'
    out.write_bytes(buf.tobytes())
    print('wrote', out)


if __name__ == '__main__':
    main()
