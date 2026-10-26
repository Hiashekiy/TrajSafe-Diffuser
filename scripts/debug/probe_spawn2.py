"""复现 dashboard 的 spawn 序列，确认车到底落在哪、能不能拍到。"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, 'E:/CARLA_0.9.16/PythonAPI/carla')
import cv2
import carla

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(20.0)
world = client.get_world()
cmap = world.get_map()
sp = cmap.get_spawn_points()[45]
wp = cmap.get_waypoint(sp.location)
x, y = wp.transform.location.x, wp.transform.location.y
yaw = wp.transform.rotation.yaw
print('target', round(x, 1), round(y, 1), round(yaw, 1))

lib = world.get_blueprint_library()
for name in ('vehicle.audi.tt', 'vehicle.audi.a2'):
    try:
        lib.find(name)
        print('blueprint ok', name)
    except Exception as exc:
        print('blueprint missing', name, exc)

bp = lib.find('vehicle.audi.tt')
tf = carla.Transform(carla.Location(x=x, y=y, z=wp.transform.location.z+0.25),
                     carla.Rotation(yaw=yaw))
act = world.try_spawn_actor(bp, tf)
print('spawn ->', act)
if act is not None:
    print('  tf after spawn', act.get_transform().location)
    act.set_simulate_physics(False)
    print('  tf after physics off', act.get_transform().location)
    act.set_transform(tf)
    print('  tf after set_transform', act.get_transform().location)
    print('  bbox extent', act.bounding_box.extent)

cam_bp = lib.find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '700')
cam_bp.set_attribute('image_size_y', '700')
cam_bp.set_attribute('fov', '90')
cam = world.spawn_actor(cam_bp, carla.Transform(
    carla.Location(x=x, y=y, z=90.0), carla.Rotation(pitch=-90.0, yaw=0.0)))
frames = []
cam.listen(lambda img: frames.append(
    np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, :3].copy()))
st = world.get_settings()
was = st.synchronous_mode
st.synchronous_mode = True
st.fixed_delta_seconds = 0.05
world.apply_settings(st)
for _ in range(5):
    world.tick()
st.synchronous_mode = was
world.apply_settings(st)
cam.stop()
cam.destroy()
print('frames', len(frames))
img = frames[-1] if frames else np.zeros((700, 700, 3), np.uint8)
crop = img[300:400, 300:400]
print('centre crop mean', crop.mean().round(1))
Path(ROOT/'outputs'/'probe_spawn2.png').write_bytes(cv2.imencode('.png', img)[1].tobytes())
if act is not None:
    act.destroy()
print('wrote outputs/probe_spawn2.png')
