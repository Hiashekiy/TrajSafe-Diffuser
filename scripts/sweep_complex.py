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

# 弯道 / 路口 / 分叉 路段（find_curve_junction.py 扫出）
CURVE_SPAWNS = [24, 25, 141, 37, 2, 36, 81, 139]      # 含 90° 以上急弯（障碍放直段）
JUNCTION_SPAWNS = [26, 143, 54, 118, 97, 43]
FORK_SPAWNS = [58, 86, 103, 135]
MERGELC_SPAWNS = [61, 15, 44, 45, 50, 60, 12, 26]

MINI = 'vehicle.mini.cooper_s'
AUDI = 'vehicle.audi.tt'
TESLA = 'vehicle.tesla.model3'
CONE = 'static.prop.constructioncone'
BARRIER = 'static.prop.streetbarrier'

# 每个场景: (tag, spawn, goal_m, [(blueprint, 纵向 m, 横向 m, 偏航 deg), ...])
SCENES = []
for k, sp in enumerate(CURVE_SPAWNS):
    SCENES.append(('C%02d_curve_obstacle' % (k+1), sp, 62.0, [(CONE, 16.0, 2.0, 0.0)]))   # 弯道用锥桶（体积小，不堵转弯带）
for k, sp in enumerate(JUNCTION_SPAWNS):
    SCENES.append(('J%02d_junction_block' % (k+1), sp, 66.0, [(MINI, 18.0, 0.8, 0.0)]))   # 路口前直段
for k, sp in enumerate(FORK_SPAWNS):
    SCENES.append(('F%02d_fork_two_obstacles' % (k+1), sp, 70.0,
                   [(MINI, 16.0, 1.0, 0.0), (CONE, 36.0, -1.8, 0.0)]))
for k, sp in enumerate(MERGELC_SPAWNS):
    # 前车正在变道：车头偏 20°、压在两车道之间
    SCENES.append(('M%02d_lead_lane_change' % (k+1), sp, 64.0, [(TESLA, 24.0, 1.1, 20.0)]))
for k, sp in enumerate(CURVE_SPAWNS[:4]):
    SCENES.append(('C%02d_curve_pair' % (k+9), sp, 68.0,
                   [(CONE, 12.0, -2.0, 0.0), (CONE, 12.0, 2.0, 0.0)]))
for k, sp in enumerate(CURVE_SPAWNS):
    SCENES.append(('C%02d_curve_pure' % (k+13), sp, 60.0, []))
for k, sp in enumerate(JUNCTION_SPAWNS[:4]):
    SCENES.append(('J%02d_junction_barrier' % (k+9), sp, 68.0, [(BARRIER, 20.0, 2.2, 0.0)]))

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

    jobs = [(tag, sp, goal, obs, tag) for tag, sp, goal, obs in SCENES]
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
                order = [s for s in (CURVE_SPAWNS + JUNCTION_SPAWNS + FORK_SPAWNS + MERGELC_SPAWNS) if s != spawn_index]
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
            for bp_name, d_m, lat, dyaw in obstacles:
                j = i_ego+int(round(d_m/2.0))
                if j >= len(chain):
                    continue
                w = chain[j]
                right = w.transform.get_right_vector()
                tf = w.transform
                tf.location.x += right.x*lat
                tf.location.y += right.y*lat
                tf.location.z += 0.35
                tf.rotation.yaw += dyaw
                bp = bl.find(bp_name)
                bp.set_attribute('role_name', 'obs_%d' % len(placed))
                a = world.try_spawn_actor(bp, tf)
                if a is not None:
                    actors.append(a)
                    placed.append(a)
            if not placed and obstacles:
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
            result = None
            for goal_try in [goal_m, goal_m+5, goal_m-5, goal_m+10, goal_m-10, goal_m+15]:
                gi = i_ego+int(round(goal_try/2.0))
                if gi >= len(chain):
                    continue
                g_xy = np.array([chain[gi].transform.location.x, chain[gi].transform.location.y])
                cond_try = frame.scene(np.array([ego_xy, g_xy])).astype(np.float32)
                try:
                    result = adapter(dict(id=0, time=0.0, frame=frame, occupancy=occupancy,
                                          road=road_mask, condition=cond_try, goal=g_xy, seed=43))
                    goal_xy, condition, goal_m = g_xy, cond_try, goal_try
                    break
                except ValueError:
                    continue
            if result is None:
                print(tag, 'skip: NO_GEOMETRIC_CANDIDATE(6 个目标点都失败)', flush=True)
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
                    if outside.is_empty:
                        exc = 0.0
                    else:
                        parts = outside.geoms if hasattr(outside, 'geoms') else [outside]
                        exc = max(float(g.distance(road_area)) for g in parts)
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
                           passed=bool(out_road == 0 and ((not obstacle_bodies) or
                                                          (np.isfinite(clear) and clear > 0.2))))
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
