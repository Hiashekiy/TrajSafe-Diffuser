"""确认 try_spawn_actor 的车能出现在俯视图里。"""
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
x, y, yaw = sp.location.x, sp.location.y, sp.rotation.yaw
wp = cmap.get_waypoint(sp.location)
print('road z', wp.transform.location.z, 'spawn z', sp.location.z)

bp = world.get_blueprint_library().find('vehicle.tesla.model3')
tf = carla.Transform(carla.Location(x=x, y=y, z=wp.transform.location.z+0.25),
                     carla.Rotation(yaw=yaw))
act = world.try_spawn_actor(bp, tf)
print('spawned', act)
if act is not None:
    act.set_simulate_physics(False)
    print('after physics off', act.get_transform().location)

cbp = world.get_blueprint_library().find('static.prop.constructioncone')
cact = world.try_spawn_actor(cbp, carla.Transform(
    carla.Location(x=x+8, y=y, z=wp.transform.location.z+0.15)))
print('cone', cact)

cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '700')
cam_bp.set_attribute('image_size_y', '700')
cam_bp.set_attribute('fov', '90')
cam = world.spawn_actor(cam_bp, carla.Transform(
    carla.Location(x=x, y=y, z=60.0), carla.Rotation(pitch=-90.0, yaw=0.0)))
frames = []
cam.listen(lambda img: frames.append(
    np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, :3].copy()))
st = world.get_settings()
was = st.synchronous_mode
st.synchronous_mode = True
st.fixed_delta_seconds = 0.05
world.apply_settings(st)
for _ in range(6):
    world.tick()
st.synchronous_mode = was
world.apply_settings(st)
cam.stop()
cam.destroy()
print('frames', len(frames))
img = frames[-1] if frames else np.zeros((700, 700, 3), np.uint8)
Path(ROOT/'outputs'/'probe_spawn.png').write_bytes(cv2.imencode('.png', img)[1].tobytes())
print('wrote outputs/probe_spawn.png')
for a in (act, cact):
    if a is not None:
        a.destroy()
