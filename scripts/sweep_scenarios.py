"""批量生成场景方案图：14 个无遮挡路段 × 4 组起终点/障碍位置 = 56 个方案。

    python scripts/sweep_scenarios.py --out outputs/scenarios/sweep

每个方案输出 vXX/{plan.npz, view.npz, summary.json, fig.png}，最后拼 contact_sheet.png。
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, 'E:/CARLA_0.9.16/PythonAPI/carla')

# find_open_spot.py 扫出来的"头顶零遮挡 + 笔直"路段
SPAWNS = [12, 26, 44, 45, 50, 60, 61, 98, 99, 101, 108, 118, 15, 47]

# (障碍距离 m, 障碍横移 m, 终点距离 m)
VARIANTS = [(12.0, 2.2, 26.0), (18.0, 2.2, 34.0), (12.0, 1.8, 26.0), (22.0, 2.4, 38.0)]

LANES_EACH_SIDE = 1
CORRIDOR_MARGIN_M = 1.0
CORRIDOR_AXIS_MAX = 0.035
STEPS = 4


def body_of(actor):
    from src.carla_bridge.dynamic_objects import Body
    tf, box = actor.get_transform(), actor.bounding_box
    c, s = np.cos(np.deg2rad(tf.rotation.yaw)), np.sin(np.deg2rad(tf.rotation.yaw))
    centre = np.array([tf.location.x, tf.location.y]) + np.array([[c, -s], [s, c]]) @ np.array([box.location.x, box.location.y])
    return Body(actor.id, centre, np.deg2rad(tf.rotation.yaw), np.zeros(2),
                float(box.extent.x), float(box.extent.y))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='outputs/scenarios/sweep')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args(argv)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    import carla
    from src.carla_bridge.frame import SceneFrame
    from src.carla_bridge.occupancy import raster, road_quads
    from src.carla_bridge.planner_adapter import PlannerAdapter
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    cmap = world.get_map()
    settings = world.get_settings()
    original = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    adapter = PlannerAdapter(dict(
        model_id='K4P_c48_oneshot:best_task',
        config='configs/config_160k4p_c48_oneshot.yaml',
        dataset='160m-k4p-c48', device='cuda', steps=STEPS,
        region_stride=1, corridor_margin_m=CORRIDOR_MARGIN_M,
        corridor_axis_max_scene=CORRIDOR_AXIS_MAX, visualize=True))

    bp_lib = world.get_blueprint_library()
    ego_bp = bp_lib.find('vehicle.audi.tt')
    obs_bp = bp_lib.find('vehicle.mini.cooper_s')
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '1280')
    cam_bp.set_attribute('image_size_y', '720')
    cam_bp.set_attribute('fov', '90')
    cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(z=200)))
    frames = []
    cam.listen(lambda img: frames.append(
        (np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, :3].copy(),
         np.asarray(img.transform.get_inverse_matrix(), dtype=float))))

    plans = [(sp, v) for sp in SPAWNS for v in VARIANTS]
    if args.limit:
        plans = plans[:args.limit]
    print('方案数', len(plans), flush=True)

    idx = 0
    for spawn_index, (obs_d, obs_lat, goal_d) in plans:
        idx += 1
        tag = 'v%02d' % idx
        out = out_root/tag
        out.mkdir(parents=True, exist_ok=True)
        actors = []
        try:
            sp = cmap.get_spawn_points()[spawn_index]
            w0 = cmap.get_waypoint(sp.location)
            back, probe = [], w0
            while len(back)*2.0 < 40.0:
                prv = probe.previous(2.0)
                if len(prv) != 1:
                    break
                probe = prv[0]
                back.append(probe)
            chain = list(reversed(back))+[w0]
            i_ego = len(chain)-1
            while (len(chain)-i_ego-1)*2.0 < 90.0:
                nxt = chain[-1].next(2.0)
                if len(nxt) != 1:
                    break
                chain.append(nxt[0])
            if len(chain)-i_ego-1 < int(goal_d/2.0)+2:
                print(tag, 'skip: 路段太短', flush=True)
                continue

            tp = sp
            tp.location.z += 0.3
            tp.rotation.yaw = w0.transform.rotation.yaw
            ego = world.try_spawn_actor(ego_bp, tp)
            if ego is None:
                print(tag, 'skip: ego spawn 失败', flush=True)
                continue
            actors.append(ego)

            j = i_ego+int(round(obs_d/2.0))
            w = chain[j]
            right = w.transform.get_right_vector()
            otf = w.transform
            otf.location.x += right.x*obs_lat
            otf.location.y += right.y*obs_lat
            otf.location.z += 0.3
            obst = world.try_spawn_actor(obs_bp, otf)
            if obst is None:
                print(tag, 'skip: 障碍 spawn 失败', flush=True)
                continue
            actors.append(obst)
            world.tick()

            ego_body = body_of(ego)
            ego_xy = ego_body.xy
            frame = SceneFrame(ego_xy, np.deg2rad(w0.transform.rotation.yaw))
            goal_xy = np.array([chain[i_ego+int(round(goal_d/2.0))].transform.location.x,
                                chain[i_ego+int(round(goal_d/2.0))].transform.location.y])
            obstacle_bodies = [body_of(obst)]

            quads = road_quads(chain)

            def side(side_sign, k):
                res = []
                for wp in chain:
                    x = wp
                    for _ in range(k):
                        x = x.get_left_lane() if side_sign > 0 else x.get_right_lane()
                        if x is None or x.lane_type.name != 'Driving':
                            x = None
                            break
                    if x is not None:
                        res.append(x)
                return res if len(res) >= 2 else None

            for side_sign in (+1, -1):
                for k in range(1, LANES_EACH_SIDE+1):
                    lane = side(side_sign, k)
                    if lane is None:
                        break
                    runs, cur = [], [lane[0]]
                    for a, b in zip(lane[:-1], lane[1:]):
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

            occupancy, road_mask = raster(frame, quads, obstacle_bodies, 0)
            condition = frame.scene(np.array([ego_xy, goal_xy])).astype(np.float32)
            t0 = time.perf_counter()
            try:
                result = adapter(dict(id=0, time=0.0, frame=frame, occupancy=occupancy,
                                      road=road_mask, condition=condition, goal=goal_xy, seed=43))
            except ValueError as exc:
                print(tag, 'skip:', exc, flush=True)
                continue
            plan_ms = 1000*(time.perf_counter()-t0)
            path = result['path']

            road_area = unary_union([Polygon(q) for q in quads]).buffer(1e-4)
            obst_poly = Polygon(obstacle_bodies[0].corners(0.0))
            clear, out_of_road = np.inf, 0
            for s in np.arange(0.0, path.s[-1], 2.0):
                p, yaw = path.pose(float(s))
                rb = body_of(ego)
                rect = Polygon(body_of(ego).__class__(-1, p, yaw, np.zeros(2), rb.half_length, rb.half_width).corners(0.0))
                if not road_area.covers(rect):
                    outside = rect.difference(road_area)
                    exc = float(outside.buffer(0).exterior.distance(road_area)) if not outside.is_empty else 0.0
                    if exc > 0.20:          # 车道四边形接缝造成的假出界不计
                        out_of_road += 1
                clear = min(clear, rect.distance(obst_poly))

            # 俯视取景：让整条轨迹进画面
            centre = path.pose(path.s[-1]*0.5)[0]
            length = float(path.s[-1])
            height = max(18.0, (length+16.0)/1.125)
            frames.clear()
            cam.set_transform(carla.Transform(
                carla.Location(x=float(centre[0]), y=float(centre[1]), z=ego.get_transform().location.z+0.06+height),
                carla.Rotation(pitch=-90.0, yaw=float(w0.transform.rotation.yaw))))
            for _ in range(4):
                world.tick()
            if frames:
                rgb_last, inv_last = frames[-1]
                np.savez_compressed(out/'view.npz', rgb=rgb_last, inverse=inv_last,
                                    ground_z=float(ego.get_transform().location.z+0.06),
                                    half_width=float(ego_body.half_width),
                                    obstacle_corners=np.array([obstacle_bodies[0].corners(0.0)], dtype=object))
            np.savez_compressed(
                out/'plan.npz', curve_world=path.xy, controls=result['controls'],
                occupancy=occupancy, road=road_mask, condition=condition,
                anchor=frame.origin, yaw=frame.yaw, goal=goal_xy,
                cand_xy=np.zeros((0, 128, 2)), cand_mask=np.zeros(0, bool), selected=int(result['topology']),
                cand_geom=np.zeros((0, 1, 2)), cand_len=np.zeros(0),
                corridors=np.array([np.asarray(p) for p in (result['viz'] or {}).get('corridor', {}).get('polygons', [])],
                                   dtype=object),
                ellipse_center=np.zeros((0, 2)), ellipse_ab=np.zeros((0, 2)))
            summary = dict(scenario='v%02d' % idx, name='spawn%d 障碍%.0fm 横移%.1f 终点%.0fm'
                                                     % (spawn_index, obs_d, obs_lat, goal_d),
                           spawn_index=spawn_index, obs_d=obs_d, obs_lat=obs_lat, goal_d=goal_d,
                           plan_ms=round(plan_ms, 1), alm_status=result['alm_status'],
                           corridor_cells=(result['corridor'] or {}).get('num_cells'),
                           footprint=dict(min_clearance_m=None if not np.isfinite(clear) else round(float(clear), 3),
                                          out_of_road=int(out_of_road),
                                          passed=bool(out_of_road == 0 and np.isfinite(clear) and clear > 0.3)))
            (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf8')
            print(tag, 'ok  alm=%s  clear=%.2f  oor=%d  %.0fms'
                  % (result['alm_status'], -1 if not np.isfinite(clear) else clear, out_of_road, plan_ms), flush=True)
        except Exception as exc:
            print(tag, 'fail:', type(exc).__name__, exc, flush=True)
        finally:
            for a in reversed(actors):
                try:
                    a.destroy()
                except Exception:
                    pass

    try:
        cam.destroy()
    except Exception:
        pass
    world.apply_settings(original)

    # ---- 出图 + 总览
    dirs = sorted([d for d in out_root.iterdir() if d.is_dir() and (d/'plan.npz').exists()])
    print('出图', len(dirs), '张', flush=True)
    for d in dirs:
        subprocess.run([sys.executable, str(ROOT/'scripts/render_schematic.py'),
                        '--run', str(d), '--out', str(d/'fig.png'), '--dpi', '110'],
                       cwd=str(ROOT), capture_output=True)
    build_sheet(dirs, out_root/'contact_sheet.png')
    print('done', flush=True)


def build_sheet(dirs, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    dirs = [d for d in dirs if (d/'fig.png').exists()]
    if not dirs:
        return
    cols = 6
    rows = (len(dirs)+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.4, rows*2.1), dpi=110)
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, d in zip(axes.ravel(), dirs):
        try:
            img = mpimg.imread(d/'fig.png')
        except Exception:
            continue
        ax.imshow(img)
        ax.set_title(d.name, fontsize=8)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_png, facecolor='white', bbox_inches='tight')
    print('saved', out_png, flush=True)


if __name__ == '__main__':
    main()
