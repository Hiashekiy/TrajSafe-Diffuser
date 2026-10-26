"""在 dashboard 同一点位复现 spawn + 取图，确认车是否能被拍到。"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, 'E:/CARLA_0.9.16/PythonAPI/carla')
import cv2
import carla

X, Y, Z = -41.7, 48.7, 0.0
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(20.0)
world = client.get_world()
lib = world.get_blueprint_library()

st = world.get_settings()
st.synchronous_mode = True
st.fixed_delta_seconds = 0.05
world.apply_settings(st)
world.tick()

bp = lib.find('vehicle.audi.tt')
bp.set_attribute('role_name', 'probe_car')
tf = carla.Transform(carla.Location(x=X, y=Y, z=Z+0.25), carla.Rotation(yaw=-90.0))
act = world.try_spawn_actor(bp, tf)
print('spawn', act)
world.tick()
print('tf1', act.get_transform().location)
act.set_simulate_physics(False)
act.set_transform(tf)
world.tick()
print('tf2', act.get_transform().location)

cam_bp = lib.find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '900')
cam_bp.set_attribute('image_size_y', '900')
cam_bp.set_attribute('fov', '90')
cam = world.spawn_actor(cam_bp, carla.Transform(
    carla.Location(x=X, y=Y, z=90.0), carla.Rotation(pitch=-90.0, yaw=0.0)))
frames = []
cam.listen(lambda img: frames.append(
    np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, :3].copy()))
for _ in range(5):
    world.tick()
print('frames', len(frames), 'tf3', act.get_transform().location)
st.synchronous_mode = False
world.apply_settings(st)
cam.stop()
cam.destroy()
img = frames[-1] if frames else np.zeros((900, 900, 3), np.uint8)
Path(ROOT/'outputs'/'probe_spawn4.png').write_bytes(cv2.imencode('.png', img)[1].tobytes())
print('crop mean', img[420:480, 420:480].mean().round(1))
act.destroy()
print('wrote outputs/probe_spawn4.png')
