"""单测：俯视相机能否真的取到图（不依赖 dashboard）。"""
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
    cmap = world.get_map()
    sps = cmap.get_spawn_points()
    sp = sps[45]
    cx, cy = sp.location.x, sp.location.y
    yaw = sp.rotation.yaw
    print('map', cmap.name, 'spawn45', round(cx, 1), round(cy, 1), round(yaw, 1))

    span = 110.0
    z = span/2.0
    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', '900')
    bp.set_attribute('image_size_y', '900')
    bp.set_attribute('fov', '90')
    cam = world.spawn_actor(bp, carla.Transform(
        carla.Location(x=cx, y=cy, z=z), carla.Rotation(pitch=-90.0, yaw=yaw)))
    frames = []
    cam.listen(lambda img: frames.append(np.frombuffer(img.raw_data, np.uint8).reshape(
        img.height, img.width, 4)[:, :, :3].copy()))
    st = world.get_settings()
    was = st.synchronous_mode
    st.synchronous_mode = True
    st.fixed_delta_seconds = 0.05
    world.apply_settings(st)
    for _ in range(6):
        world.tick()
    print('frames', len(frames))
    st.synchronous_mode = was
    world.apply_settings(st)
    cam.destroy()
    if not frames:
        print('NO FRAME')
        return
    img = frames[-1]
    print('mean', img.mean(), 'max', img.max(), 'shape', img.shape)
    out = ROOT/'outputs'/'probe_view.png'
    ok, buf = cv2.imencode('.png', img)
    out.write_bytes(buf.tobytes())
    print('wrote', out)


if __name__ == '__main__':
    main()
