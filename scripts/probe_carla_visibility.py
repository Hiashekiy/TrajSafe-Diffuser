"""Compare native camera vehicle visibility at two altitudes, restoring the world."""
from pathlib import Path
import queue
import carla
import cv2
import numpy as np


def main():
    out = Path('outputs/closed_loop/visibility_probe')
    out.mkdir(parents=True, exist_ok=True)
    client = carla.Client('localhost', 2000)
    client.set_timeout(20)
    world = client.get_world()
    saved = world.get_settings()
    actors = []
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = .05
        world.apply_settings(settings)
        point = carla.Location(x=193.0344, y=61.7508, z=.5)
        print('vertical_ray', world.cast_ray(carla.Location(point.x, point.y, 150), carla.Location(point.x, point.y, -5)))
        bp = world.get_blueprint_library().find('vehicle.audi.tt')
        bp.set_attribute('color', '255,0,0')
        car = world.spawn_actor(bp, carla.Transform(point, carla.Rotation(yaw=90)))
        actors.append(car)
        cameras = []
        for height in (35, 100):
            bp = world.get_blueprint_library().find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', '1280')
            bp.set_attribute('image_size_y', '720')
            bp.set_attribute('fov', '100')
            camera = world.spawn_actor(bp, carla.Transform(carla.Location(point.x, 24.8, height), carla.Rotation(pitch=-90)))
            actors.append(camera)
            frames = queue.Queue()
            camera.listen(frames.put)
            cameras.append((height, camera, frames))
        for _ in range(35):
            world.tick()
        for height, camera, frames in cameras:
            image = frames.get(timeout=20)
            while not frames.empty():
                image = frames.get()
            arr = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)[:, :, :3]
            cv2.imwrite(str(out / f'height_{height}.png'), arr)
            print('saved', height, 'car', car.get_location())
    finally:
        for actor in reversed(actors):
            if actor.type_id.startswith('sensor.'):
                actor.stop()
            actor.destroy()
        world.apply_settings(saved)


if __name__ == '__main__':
    main()
