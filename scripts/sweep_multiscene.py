"""多场景批量生成：8 种障碍布局 × 12 个路段（+4 个补充）= 100 张正方形插图。

    python scripts/sweep_multiscene.py --out outputs/scenarios/batch100

每个场景输出 <id>/{plan.npz, view.npz, summary.json, fig.png(900x900)}，
最后拼 contact_sheet.png 供挑选。
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

# find_open_spot.py 扫出来的「头顶零遮挡 + 笔直」路段
SPAWNS = [12, 15, 26, 44, 45, 47, 50, 60, 61, 98, 99, 101, 108, 118]

MINI = 'vehicle.mini.cooper_s'
AUDI = 'vehicle.audi.tt'
TESLA = 'vehicle.tesla.model3'
CONE = 'static.prop.constructioncone'
BARRIER = 'static.prop.streetbarrier'

# 每种布局: (名字, 目标距离 m, [(blueprint, 纵向距离 m, 横向偏移 m), ...])
LAYOUTS = [
    ('L1_roadside_parked', 65.0, [(MINI, 35.0, 0.5)]),
    ('L2_blocking_centred', 65.0, [(MINI, 30.0, 0.9)]),
    ('L3_two_staggered', 65.0, [(MINI, 25.0, -0.8), (TESLA, 48.0, 1.0)]),
    ('L4_three_s_curve', 70.0, [(MINI, 20.0, -0.9), (AUDI, 38.0, 1.0), (MINI, 56.0, -0.9)]),
    ('L5_cone_corridor', 65.0, [(CONE, 22.0, -2.0), (CONE, 22.0, 2.0),
                                (CONE, 32.0, -2.0), (CONE, 32.0, 2.0),
                                (CONE, 42.0, -2.0), (CONE, 42.0, 2.0)]),
    ('L6_parked_row', 65.0, [(MINI, 20.0, 2.4), (AUDI, 32.0, 2.4), (MINI, 44.0, 2.4)]),
    ('L7_barrier_plus_car', 68.0, [(BARRIER, 28.0, 2.2), (MINI, 46.0, 1.4)]),
    ('L8_two_side_squeeze', 65.0, [(MINI, 26.0, -1.6), (AUDI, 26.0, 1.9)]),
    ('L9_double_cone_gate', 62.0, [(CONE, 24.0, -2.2), (CONE, 24.0, 2.2), (MINI, 40.0, 1.2)]),
    ('L10_far_blocker', 70.0, [(TESLA, 45.0, 0.8)]),
]

EXTRA = [
    ('X1_long_range', 101, 80.0, [(MINI, 40.0, 0.6)]),
    ('X2_short_range', 44, 45.0, [(MINI, 26.0, 0.6)]),
    ('X3_cone_plus_car', 60, 68.0, [(CONE, 24.0, -2.0), (CONE, 30.0, -2.0), (MINI, 42.0, 1.4)]),
    ('X4_three_staggered', 50, 72.0, [(AUDI, 22.0, 1.2), (MINI, 40.0, -1.0), (TESLA, 58.0, 1.2)]),
]

TOPUP = [
    ('T1_goal45', 47, 45.0, [(MINI, 24.0, 0.7)]),
    ('T2_goal55', 108, 55.0, [(MINI, 30.0, 0.8)]),
    ('T3_goal75', 118, 75.0, [(TESLA, 42.0, 0.7)]),
    ('T4_goal80', 26, 80.0, [(MINI, 44.0, 0.8)]),
    ('T5_cones', 12, 60.0, [(CONE, 20.0, -2.1), (CONE, 20.0, 2.1), (CONE, 34.0, -2.1), (CONE, 34.0, 2.1)]),
    ('T6_parked', 15, 66.0, [(MINI, 22.0, 2.5), (AUDI, 36.0, 2.5)]),
    ('T7_barrier', 45, 62.0, [(BARRIER, 26.0, 2.3)]),
    ('T8_twoCars', 60, 68.0, [(MINI, 28.0, -0.9), (MINI, 46.0, 0.9)]),
    ('T9_single', 99, 70.0, [(TESLA, 38.0, 0.6)]),
    ('T10_three', 50, 72.0, [(MINI, 24.0, 1.0), (AUDI, 42.0, -1.0), (MINI, 58.0, 1.0)]),
    ('T11_gentle', 61, 66.0, [(MINI, 34.0, 0.6)]),
    ('T12_parked2', 98, 66.0, [(AUDI, 24.0, 2.5), (MINI, 40.0, 2.5)]),
    ('T13_barrier2', 44, 64.0, [(BARRIER, 30.0, 2.3)]),
    ('T14_single2', 45, 68.0, [(MINI, 36.0, 0.7)]),
    ('T15_cones2', 47, 62.0, [(CONE, 22.0, -2.2), (CONE, 22.0, 2.2), (CONE, 36.0, -2.2), (CONE, 36.0, 2.2)]),
    ('T16_pair2', 15, 70.0, [(MINI, 30.0, -1.0), (TESLA, 50.0, 1.0)]),
]

STEPS = 16
LANES_EACH_SIDE = 2
CORRIDOR_MARGIN_M = 1.0
CORRIDOR_AXIS_MAX = 0.035
PIN_SAMPLE_M = 12.0


def body_of(actor):
    from src.carla_bridge.dynamic_objects import Body
    tf, box = actor.get_transform(), actor.bounding_box
    c, s = np.cos(np.deg2rad(tf.rotation.yaw)), np.sin(np.deg2rad(tf.rotation.yaw))
    centre = np.array([tf.location.x, tf.location.y]) + np.array([[c, -s], [s, c]]) @ np.array([box.location.x, box.location.y])
    return Body(actor.id, centre, np.deg2rad(tf.rotation.yaw), np.zeros(2),
                float(box.extent.x), float(box.extent.y))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='outputs/scenarios/batch100')
    ap.add_argument('--size', type=int, default=900)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--topup', action='store_true')
    args = ap.parse_args(argv)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    import carla
    from src.carla_bridge.frame import SceneFrame
    from src.carla_bridge.occupancy import raster, road_quads
    from src.carla_bridge.planner_adapter import PlannerAdapter
    from src.carla_bridge.path_smooth import pin_and_spline
    from src.carla_bridge.path_profile import Path as GeomPath
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    cmap = world.get_map()
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    adapter = PlannerAdapter(dict(
        model_id='K4P_c48_oneshot:best_task',
        config='configs/config_160k4p_c48_oneshot.yaml',
        dataset='160m-k4p-c48', device='cuda', steps=STEPS,
        region_stride=1, corridor_margin_m=CORRIDOR_MARGIN_M,
        corridor_axis_max_scene=CORRIDOR_AXIS_MAX, visualize=True))

    bl = world.get_blueprint_library()
    ego_bp = bl.find(AUDI)
    ego_bp.set_attribute('role_name', 'ego')
    cam_bp = bl.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '1600')
    cam_bp.set_attribute('image_size_y', '900')
    cam_bp.set_attribute('fov', '90')
    cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(z=300)))
    frames = []
    cam.listen(lambda img: frames.append(
        (np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, :3].copy(),
         np.asarray(img.transform.get_inverse_matrix(), dtype=float))))

    jobs = []
    for name, spawn, goal_m, obstacles in (TOPUP if args.topup else []):
        jobs.append((name, spawn, goal_m, obstacles, name))
    for si, (name, goal_m, obstacles) in enumerate([] if args.topup else LAYOUTS):
        for k, spawn in enumerate(SPAWNS):
            jobs.append((f'{name}_{k+1:02d}', spawn, goal_m, obstacles, name))
    for name, spawn, goal_m, obstacles in EXTRA:
        jobs.append((f'X_{name}', spawn, goal_m, obstacles, name))
    if args.limit:
        jobs = jobs[:args.limit]
    print('场景数', len(jobs), flush=True)

    ok = 0
    for tag, spawn_index, goal_m, obstacles, name in jobs:
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
            while (len(chain)-i_ego-1)*2.0 < max(goal_m+30.0, 110.0):
                nxt = chain[-1].next(2.0)
                if not nxt:
                    break
                same = [w for w in nxt if w.lane_id == chain[-1].lane_id]
                chain.append((same or nxt)[0])
            if (len(chain)-i_ego-1)*2.0 < goal_m+8.0:
                print(tag, 'skip: 路段不足', flush=True)
                continue

            tp = sp
            tp.location.z += 0.3
            tp.rotation.yaw = w0.transform.rotation.yaw
            ego = world.try_spawn_actor(ego_bp, tp)
            if ego is None:
                order = [s for s in SPAWNS if s != spawn_index]
                for alt in order:
                    sp_alt = cmap.get_spawn_points()[alt]
                    wp_alt = cmap.get_waypoint(sp_alt.location)
                    tp_alt = sp_alt
                    tp_alt.location.z += 0.3
                    tp_alt.rotation.yaw = wp_alt.transform.rotation.yaw
                    ego = world.try_spawn_actor(ego_bp, tp_alt)
                    if ego is not None:
                        sp, w0 = sp_alt, wp_alt
                        back, probe = [], w0
                        while len(back)*2.0 < 40.0:
                            prv = probe.previous(2.0)
                            if len(prv) != 1:
                                break
                            probe = prv[0]
                            back.append(probe)
                        chain = list(reversed(back))+[w0]
                        i_ego = len(chain)-1
                        while (len(chain)-i_ego-1)*2.0 < max(goal_m+30.0, 110.0):
                            nxt = chain[-1].next(2.0)
                            if not nxt:
                                break
                            same = [w for w in nxt if w.lane_id == chain[-1].lane_id]
                            chain.append((same or nxt)[0])
                        break
            if ego is None:
                print(tag, 'skip: ego 占用', flush=True)
                continue
            actors.append(ego)

            placed = []
            for bp_name, d_m, lat in obstacles:
                j = i_ego+int(round(d_m/2.0))
                if j >= len(chain):
                    continue
                w = chain[j]
                right = w.transform.get_right_vector()
                tf = w.transform
                tf.location.x += right.x*lat
                tf.location.y += right.y*lat
                tf.location.z += 0.35
                bp = bl.find(bp_name)
                bp.set_attribute('role_name', 'obs_%d' % len(placed))
                a = world.try_spawn_actor(bp, tf)
                if a is not None:
                    actors.append(a)
                    placed.append(a)
            if not placed:
                print(tag, 'skip: 障碍未放置', flush=True)
                continue
            world.tick()

            ego_body = body_of(ego)
            ego_xy = ego_body.xy
            frame = SceneFrame(ego_xy, np.deg2rad(w0.transform.rotation.yaw))
            i_goal = min(len(chain)-1, i_ego+int(round(goal_m/2.0)))
            goal_xy = np.array([chain[i_goal].transform.location.x, chain[i_goal].transform.location.y])
            obstacle_bodies = [body_of(a) for a in placed]

            quads = road_quads(chain)
            ego_sign = np.sign(w0.lane_id) or 1.0

            def side_chain(side, k):
                res = []
                for wp in chain:
                    x = wp
                    for _ in range(k):
                        x = x.get_left_lane() if side > 0 else x.get_right_lane()
                        if x is None or x.lane_type.name != 'Driving' or np.sign(x.lane_id) != ego_sign:
                            x = None
                            break
                    if x is not None:
                        res.append(x)
                if len(res) < 2:
                    return None
                runs, cur = [], [res[0]]
                for a, b in zip(res[:-1], res[1:]):
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
                return runs

            for side in (+1, -1):
                for k in range(1, LANES_EACH_SIDE+1):
                    runs = side_chain(side, k)
                    if runs is None:
                        break
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
            path_xy = result['path'].xy
            pin_xy, pin_turn, _ = pin_and_spline(path_xy, ego_xy,
                                                 np.deg2rad(w0.transform.rotation.yaw),
                                                 sample_m=PIN_SAMPLE_M)
            path = GeomPath.build(pin_xy)

            road_area = unary_union([Polygon(q) for q in quads]).buffer(1e-4)
            obst_polys = [Polygon(o.corners(0.0)) for o in obstacle_bodies]
            clear, out_road = np.inf, 0
            for s in np.arange(0.0, path.s[-1], 2.0):
                p, yaw = path.pose(float(s))
                rect = Polygon(body_of(ego).__class__(-1, p, yaw, np.zeros(2),
                                                    ego_body.half_length, ego_body.half_width).corners(0.0))
                if not road_area.covers(rect):
                    outside = rect.difference(road_area)
                    exc = float(outside.buffer(0).exterior.distance(road_area)) if not outside.is_empty else 0.0
                    if exc > 0.20:
                        out_road += 1
                for poly in obst_polys:
                    clear = min(clear, rect.distance(poly))

            centre = path.pose(path.s[-1]*0.45)[0]
            height = (path.s[-1]+18.0)/1.125
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
                                    half_length=float(ego_body.half_length),
                                    ego_corners=ego_body.corners(0.0),
                                    obstacle_corners=np.array([o.corners(0.0) for o in obstacle_bodies],
                                                              dtype=object))
            np.savez_compressed(
                out/'plan.npz', curve_world=path.xy, occupancy=occupancy, road=road_mask,
                condition=condition, anchor=frame.origin, yaw=frame.yaw, goal=goal_xy,
                corridors=np.array([np.asarray(p) for p in
                                    (result['viz'] or {}).get('corridor', {}).get('polygons', [])], dtype=object))
            summary = dict(scenario=tag, name=name, spawn_index=spawn_index, goal_m=goal_m,
                           n_obstacles=len(obstacle_bodies), plan_ms=round(plan_ms, 1),
                           alm_status=result['alm_status'],
                           corridor_cells=(result['corridor'] or {}).get('num_cells'),
                           pin_tangent_deg=round(pin_turn, 2),
                           min_clearance_m=None if not np.isfinite(clear) else round(float(clear), 3),
                           out_of_road=int(out_road),
                           passed=bool(out_road == 0 and np.isfinite(clear) and clear > 0.2))
            (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf8')
            ok += 1
            print('%s ok  %s  间距 %.2f  越界 %d  %.0fms' %
                  (tag, result['alm_status'], -1 if not np.isfinite(clear) else clear, out_road, plan_ms), flush=True)
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
    print('成功场景', ok, flush=True)

    dirs = sorted([d for d in out_root.iterdir() if d.is_dir() and (d/'plan.npz').exists()])
    print('出图', len(dirs), '张', flush=True)
    for d in dirs:
        subprocess.run([sys.executable, str(ROOT/'scripts/render_style.py'),
                        '--run', str(d), '--out', str(d/'fig.png'),
                        '--square', '1', '--size', str(args.size)], cwd=str(ROOT), capture_output=True)
    build_sheet(dirs, out_root/'contact_sheet.png')
    print('done', flush=True)


def build_sheet(dirs, out_png, cols=8):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    dirs = [d for d in dirs if (d/'fig.png').exists()]
    if not dirs:
        return
    rows = (len(dirs)+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.2, rows*2.4), dpi=100)
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, d in zip(axes.ravel(), dirs):
        try:
            ax.imshow(mpimg.imread(d/'fig.png'))
        except Exception:
            continue
        ax.set_title(d.name[:26], fontsize=6)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_png, facecolor='white', bbox_inches='tight')
    print('saved', out_png, flush=True)


if __name__ == '__main__':
    main()
