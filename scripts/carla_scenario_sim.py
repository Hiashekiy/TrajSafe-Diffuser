"""CARLA 静态障碍场景：占据图 → TrajSafe(加速版) 轨迹 → 车身扫掠带可视化。

不执行车辆控制：规划只调用一次，然后把「规划轨迹 + 车身扫掠带 + 走廊 + 障碍物」
画在 CARLA 三维场景里，用斜俯视相机出截图与视频（ghost 车辆只做回放，不参与物理）。

    python scripts/carla_scenario_sim.py --scenario configs/scenarios/S1_lane_block.yaml
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GREEN = (40, 220, 90)
AMBER = (255, 180, 40)
RED = (255, 70, 70)


def refit_spline(path_xy, start_pose=None, rms=0.12):
    """Fit a smoothing cubic spline through the planned points.

    ``rms`` is the allowed RMS deviation (metres) from the original plan; the
    start pose is re-attached afterwards so the curve still leaves the ego
    exactly along its heading.
    """
    from scipy.interpolate import splprep, splev
    p = np.asarray(path_xy, float)
    if len(p) < 12:
        return path_xy, 0.0
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(d)])
    u = u/u[-1]
    s_val = len(p)*float(rms)**2
    tck, _ = splprep([p[:, 0], p[:, 1]], u=u, s=s_val, k=3)
    dense_u = np.linspace(0.0, 1.0, max(len(p), int(u[-1]*0)+len(p)))
    x, y = splev(np.linspace(0.0, 1.0, len(p)), tck)
    new = np.column_stack([x, y])
    if start_pose is not None:
        # 保留起点：把曲线整体平移到经过起点，再沿起点朝向修正首个切向
        off = np.asarray(start_pose[0])-new[0]
        new = new+off
    dev = float(np.max(np.linalg.norm(new-p, axis=1)))
    return new, dev

def smooth_path(path_xy, iters=60, lam=0.45, pin_points=12):
    """Curvature smoothing: pin the first ``pin_points`` (start pose) and the last point."""
    p = np.asarray(path_xy, float).copy()
    n = len(p)
    if n < pin_points+3 or iters <= 0:
        return path_xy, 0.0
    anchor = p[:pin_points].copy()
    for _ in range(iters):
        q = p.copy()
        q[1:-1] = (1.0-lam)*p[1:-1] + lam*0.5*(p[:-2]+p[2:])
        q[:pin_points] = anchor
        q[-1] = p[-1]
        p = q
    dev = float(np.max(np.linalg.norm(p-np.asarray(path_xy, float), axis=1)))
    return p, dev

def refit_anchored(path_xy, start_xy, start_yaw, rms=0.18, lead_m=6.0, lead_n=6):
    """One-shot C2 spline fit that leaves the ego exactly along its heading.

    A synthetic lead-in (points behind the ego along -heading) is appended before
    fitting; afterwards the curve is trimmed at the ego centre.  Because the fit
    is global there is no joint and therefore no curvature discontinuity.
    """
    from scipy.interpolate import splprep, splev
    p = np.asarray(path_xy, float)
    if len(p) < 12:
        return path_xy, 0.0, 0.0
    start = np.asarray(start_xy, float)
    m0 = np.array([np.cos(start_yaw), np.sin(start_yaw)])
    lead = np.array([start - m0*(lead_m*(i+1)/lead_n) for i in range(lead_n)])[::-1]
    # 原轨迹从 3 m 之后开始接入，避免与前置点在起点附近打架
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s_arc = np.concatenate([[0.0], np.cumsum(d)])
    body = p[s_arc > 3.0]
    pts = np.vstack([lead, start[None], body])
    dd = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(dd)])
    u = u/u[-1]
    tck, _ = splprep([pts[:, 0], pts[:, 1]], u=u, s=len(pts)*float(rms)**2, k=3)
    u_start = float(u[len(lead)])          # 自车中心在拟合点列中的参数位置
    uu = np.linspace(u_start, 1.0, 900)
    x, y = splev(uu, tck)
    trimmed = np.column_stack([x, y])
    if len(trimmed) < 8:
        return path_xy, 0.0, 0.0
    # 刚体旋转：让起始切向严格等于车身朝向，同时保持曲率连续（不产生折角）
    v0 = trimmed[min(6, len(trimmed)-1)]-trimmed[0]
    if np.linalg.norm(v0) > 1e-6:
        ang = start_yaw-float(np.arctan2(v0[1], v0[0]))
        ang = (ang+np.pi) % (2*np.pi)-np.pi
        c, s_ = np.cos(ang), np.sin(ang)
        rel = trimmed-trimmed[0]
        trimmed = rel @ np.array([[c, s_], [-s_, c]]) + start
    trimmed[0] = start
    tangent = start_yaw
    dev = float(np.max(np.linalg.norm(trimmed[::5]-p[np.minimum(
        np.arange(0, len(trimmed), 5)*len(p)//len(trimmed), len(p)-1)], axis=1))) if len(p) else 0.0
    return trimmed, float(np.rad2deg(abs((tangent-start_yaw+np.pi) % (2*np.pi)-np.pi))), dev

def anchor_start(path_xy, start_xy, start_yaw, length=4.0, step=0.2):
    """Rebuild the first ``length`` metres as a cubic Hermite that matches the ego pose."""
    from src.carla_bridge.path_profile import Path as _Path
    nm = 3
    L = min(length, float(_Path.build(path_xy).s[-1])*0.25)
    if L <= 0.5:
        return path_xy, 0.0
    geom = _Path.build(path_xy)
    p1, yaw1 = geom.pose(L)
    m0 = np.array([np.cos(start_yaw), np.sin(start_yaw)])
    m1 = np.array([np.cos(yaw1), np.sin(yaw1)])
    yaw0 = float(np.arctan2(*(geom.xy[1]-geom.xy[0])[::-1]))
    turn = abs((yaw0-start_yaw+np.pi) % (2*np.pi) - np.pi)   # 比较【起始切向】与车身朝向
    offset = float(np.linalg.norm(geom.xy[0]-np.asarray(start_xy, float)))
    if turn < np.deg2rad(6.0) and offset < 0.10:
        return path_xy, 0.0
    turn = float(np.rad2deg(max(turn, 0.0))) if offset < 0.10 else max(float(np.rad2deg(turn)), 0.1)
    pts = []
    for t in np.arange(0.0, 1.0, step/L):
        h00 = 2*t**3-3*t**2+1
        h10 = t**3-2*t**2+t
        h01 = -2*t**3+3*t**2
        h11 = t**3-t**2
        pts.append(h00*np.asarray(start_xy) + h10*L*m0 + h01*p1 + h11*L*m1)
    pts.append(p1)
    keep = geom.xy[geom.s > L]
    return np.vstack([np.array(pts), keep]), float(turn)

def extend_forward(chain, i_ego, need_m):
    """Extend along the lane; at a junction keep the same lane_id instead of failing."""
    while (len(chain)-i_ego-1)*2.0 < need_m:
        nxt = chain[-1].next(2.0)
        if not nxt:
            break
        lane = chain[-1].lane_id
        same = [w for w in nxt if w.lane_id == lane]
        chain.append((same or nxt)[0])
    return chain

def xy_wp(wp):
    return np.array([wp.transform.location.x, wp.transform.location.y])


def body_of(actor):
    """(centre, yaw, half_length, half_width) 取自真实 CARLA BoundingBox。"""
    from src.carla_bridge.dynamic_objects import Body
    tf, box = actor.get_transform(), actor.bounding_box
    c, s = np.cos(np.deg2rad(tf.rotation.yaw)), np.sin(np.deg2rad(tf.rotation.yaw))
    centre = np.array([tf.location.x, tf.location.y]) + np.array([[c, -s], [s, c]]) @ np.array([box.location.x, box.location.y])
    return Body(actor.id, centre, np.deg2rad(tf.rotation.yaw), np.zeros(2),
                float(box.extent.x), float(box.extent.y))


def corners(centre, yaw, hl, hw, margin=0.0):
    from src.carla_bridge.dynamic_objects import Body
    return Body(-1, np.asarray(centre), float(yaw), np.zeros(2), hl, hw).corners(margin)


def footprint_check(path, ego_body, obstacles, road_area, step_m=2.0, margin=0.15):
    """车身级检查：沿轨迹放真实车身矩形，查道路包含与障碍重叠。"""
    from shapely.geometry import Polygon
    from shapely.prepared import prep
    allowed = prep(road_area.buffer(1e-6)) if road_area is not None else None
    polys = [Polygon(o.corners(0.0)) for o in obstacles]
    out_road, hit, clear, excursion = 0, set(), np.inf, 0.0
    n = 0
    for s in np.arange(0.0, path.s[-1], step_m):
        p, yaw = path.pose(float(s))
        rect = Polygon(corners(p, yaw, ego_body.half_length, ego_body.half_width, margin))
        n += 1
        if allowed is not None and not allowed.covers(rect):
            # 按越界距离判定：车道四边形之间的接缝也会让 covers 失败，但越界距离接近 0
            outside = rect.difference(road_area)
            exc = float(outside.buffer(0).exterior.distance(road_area)) if not outside.is_empty else 0.0
            excursion = max(excursion, exc)
            if exc > 0.20:
                out_road += 1
        for k, poly in enumerate(polys):
            if rect.intersects(poly):
                hit.add(k)
            clear = min(clear, rect.distance(poly))
    return dict(samples=n, out_of_road=out_road, overlapping_obstacles=sorted(hit),
                min_clearance_m=None if not np.isfinite(clear) else round(float(clear), 3),
                max_road_excursion_m=round(excursion, 3),
                passed=bool(out_road == 0 and not hit))


def draw_scene(carla, world, path, ego_body, corridor_polys, frame, obstacles, goal_xy, life, z):
    d = world.debug
    half = ego_body.half_width + 0.15
    left, right = [], []
    for s in np.arange(0.0, path.s[-1], 1.0):
        p, yaw = path.pose(float(s))
        n = np.array([-np.sin(yaw), np.cos(yaw)])
        left.append(p+n*half)
        right.append(p-n*half)
    left, right = np.array(left), np.array(right)
    # 扫掠带：左右边界 + 每 3 m 一个横向截面（CARLA 无半透明多边形，用线族表达宽度）
    for line in (left, right):
        for a, b in zip(line[:-1], line[1:]):
            d.draw_line(carla.Location(x=float(a[0]), y=float(a[1]), z=z),
                        carla.Location(x=float(b[0]), y=float(b[1]), z=z),
                        thickness=0.022, color=carla.Color(*GREEN), life_time=life)
    for s in np.arange(0.0, path.s[-1], 6.0):
        p, yaw = path.pose(float(s))
        n = np.array([-np.sin(yaw), np.cos(yaw)])
        a, b = p+n*half, p-n*half
        d.draw_line(carla.Location(x=float(a[0]), y=float(a[1]), z=z),
                    carla.Location(x=float(b[0]), y=float(b[1]), z=z),
                    thickness=0.018, color=carla.Color(*GREEN), life_time=life)
    for poly in corridor_polys:
        w = frame.world(np.asarray(poly))
        for a, b in zip(w, np.roll(w, -1, axis=0)):
            d.draw_line(carla.Location(x=float(a[0]), y=float(a[1]), z=z+0.02),
                        carla.Location(x=float(b[0]), y=float(b[1]), z=z+0.02),
                        thickness=0.012, color=carla.Color(*AMBER), life_time=life)
    for o in obstacles:
        c = o.corners(0.0)
        for a, b in zip(c, np.roll(c, -1, axis=0)):
            d.draw_line(carla.Location(x=float(a[0]), y=float(a[1]), z=z+0.05),
                        carla.Location(x=float(b[0]), y=float(b[1]), z=z+0.05),
                        thickness=0.035, color=carla.Color(*RED), life_time=life)
    d.draw_point(carla.Location(x=float(goal_xy[0]), y=float(goal_xy[1]), z=z+0.3),
                 size=0.45, color=carla.Color(*RED), life_time=life)
    d.draw_string(carla.Location(x=float(goal_xy[0]), y=float(goal_xy[1]), z=z+1.2),
                  'GOAL', draw_shadow=False, color=carla.Color(*RED), life_time=life)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenario', required=True)
    ap.add_argument('--out')
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--frames', type=int, default=120, help='视频帧数（ghost 沿轨迹回放）')
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.scenario).read_text(encoding='utf8'))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = Path(args.out or (Path(cfg['record']['dir'])/(f"{cfg['id']}_{stamp}")))
    out.mkdir(parents=True, exist_ok=True)
    (out/'scenario.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf8')

    import carla
    from src.carla_bridge.frame import SceneFrame
    from src.carla_bridge.occupancy import raster, road_quads
    from src.carla_bridge.planner_adapter import PlannerAdapter
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    # CARLA 在本机偶发 RPC 挂起：短超时重试，避免一次 180 s 卡死
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(20.0)
    deadline = time.time()+240
    world = None
    while time.time() < deadline:
        try:
            world = client.get_world()
            break
        except RuntimeError:
            time.sleep(5)
    if world is None:
        raise RuntimeError('CARLA_UNRESPONSIVE')
    if cfg.get('map') and world.get_map().name.rsplit('/', 1)[-1] != cfg['map']:
        try:
            world = client.load_world(cfg['map'])
        except RuntimeError:
            print('load_world 失败，沿用当前地图', world.get_map().name, flush=True)
    cmap = world.get_map()
    print('MAP', cmap.name, flush=True)

    summary = dict(scenario=cfg['id'], name=cfg['name'], status='INITIALIZING', plan_ok=False)
    actors = []
    settings = world.get_settings()
    original = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = cfg['record']['fixed_dt']
    world.apply_settings(settings)
    try:
        bp_lib = world.get_blueprint_library()
        ego_wp = None
        ego_tf = None
        if cfg['ego'].get('spawn_index') is not None:
            sp = cmap.get_spawn_points()[int(cfg['ego']['spawn_index'])]
            ego_wp = cmap.get_waypoint(sp.location)
            ego_tf = sp
        if ego_wp is None and cfg['ego'].get('road_id') is not None:
            ego_wp = cmap.get_waypoint_xodr(cfg['ego']['road_id'], cfg['ego']['lane_id'], cfg['ego']['s'])
        if ego_wp is None:
            # 自动选路段：从 spawn 点里挑一段有左邻车道、且能连续延伸 140 m 的
            for sp in cmap.get_spawn_points():
                w0 = cmap.get_waypoint(sp.location)
                if w0.get_left_lane() is None or w0.get_left_lane().lane_type.name != 'Driving':
                    continue
                probe, ok = w0, True
                for _ in range(int(cfg['grid']['route_length_m']/2.0)):
                    if probe.get_left_lane() is None:
                        ok = False
                        break
                    nxt = probe.next(2.0)
                    if len(nxt) != 1:
                        ok = False
                        break
                    probe = nxt[0]
                if ok:
                    ego_wp = w0
                    break
            if ego_wp is None:
                raise RuntimeError('EGO_ROAD_INVALID')
            print('AUTO_ROAD', ego_wp.road_id, ego_wp.lane_id, round(ego_wp.s, 1), flush=True)
        back_m = float(cfg['grid'].get('route_back_m', 40.0))
        back = []
        probe = ego_wp
        while len(back)*2.0 < back_m:
            prv = probe.previous(2.0)
            if len(prv) != 1:
                break
            probe = prv[0]
            back.append(probe)
        chain = list(reversed(back))+[ego_wp]
        i_ego = len(chain)-1
        chain = extend_forward(chain, i_ego, cfg['grid']['route_length_m'])

        ego_bp = bp_lib.find(cfg['ego']['blueprint'])
        ego_bp.set_attribute('role_name', 'ego')
        tp = ego_tf if ego_tf is not None else ego_wp.transform
        tp.location.z += 0.3
        tp.rotation.yaw = ego_wp.transform.rotation.yaw      # 车身朝向必须与车道方向一致
        ego = world.try_spawn_actor(ego_bp, tp)
        if ego is None:
            # spawn 点被 NPC 占用：换下一个候选点，并【重建链条与坐标系】（顺序不能颠倒）
            for alt in (cfg['ego'].get('spawn_candidates') or []):
                sp_alt = cmap.get_spawn_points()[int(alt)]
                wp_alt = cmap.get_waypoint(sp_alt.location)
                tf_alt = sp_alt
                tf_alt.location.z += 0.3
                tf_alt.rotation.yaw = wp_alt.transform.rotation.yaw
                ego = world.try_spawn_actor(ego_bp, tf_alt)
                if ego is not None:
                    ego_wp, tp = wp_alt, tf_alt
                    print('spawn 被占用，改用 spawn_index', alt, flush=True)
                    back = []
                    probe = ego_wp
                    while len(back)*2.0 < back_m:
                        prv = probe.previous(2.0)
                        if len(prv) != 1:
                            break
                        probe = prv[0]
                        back.append(probe)
                    chain = list(reversed(back))+[ego_wp]
                    i_ego = len(chain)-1
                    chain = extend_forward(chain, i_ego, cfg['grid']['route_length_m'])
                    break
        if ego is None:
            raise RuntimeError('EGO_SPAWN_FAILED')
        actors.append(ego)

        for i, spec in enumerate(cfg.get('obstacles') or []):
            j = min(len(chain)-1, i_ego+int(round(spec['distance_m']/2.0)))
            w = chain[j]
            right = w.transform.get_right_vector()
            tf = w.transform
            tf.location.x += right.x*spec.get('lateral_m', 0.0)
            tf.location.y += right.y*spec.get('lateral_m', 0.0)
            tf.location.z += 0.3
            tf.rotation.yaw += spec.get('yaw_offset_deg', 0.0)
            bp = bp_lib.find(spec['blueprint'])
            bp.set_attribute('role_name', f'obstacle_{i}')
            a = world.try_spawn_actor(bp, tf)
            if a is None:
                raise RuntimeError(f'OBSTACLE_SPAWN_FAILED_{i}')
            actors.append(a)

        world.tick()

        ego_body = body_of(ego)
        ego_xy = ego_body.xy
        # 帧朝向取车道方向（spawn 点的 yaw 可能是反的）
        frame = SceneFrame(ego_xy, np.deg2rad(ego_wp.transform.rotation.yaw))
        avail_m = (len(chain)-i_ego-1)*2.0
        goal_m = min(cfg['goal']['distance_m'], max(12.0, avail_m-6.0))
        if goal_m < cfg['goal']['distance_m']:
            print('路段只有 %.0f m，目标点缩短到 %.0f m' % (avail_m, goal_m), flush=True)
        i_goal = min(len(chain)-1, i_ego+int(round(goal_m/2.0)))
        goal_xy = xy_wp(chain[i_goal])
        obstacle_bodies = [body_of(a) for a in actors[1:]]

        quads = road_quads(chain)
        lanes_included = 1

        ego_lane_sign = np.sign(ego_wp.lane_id) or 1.0

        def same_direction(wp):
            # 只纳入与自车同向的车道（车道 id 同号）；对向车道视为不可行驶
            return wp is not None and wp.lane_type.name == 'Driving' and np.sign(wp.lane_id) == ego_lane_sign

        def side_chain(side, k):
            """向同侧数第 k 条可行驶车道；逐点纳入，缺的点跳过（不再整条丢弃）。"""
            out = []
            for w in chain:
                x = w
                for _ in range(k):
                    x = x.get_left_lane() if side > 0 else x.get_right_lane()
                    if not same_direction(x):
                        x = None
                        break
                if x is not None:
                    out.append(x)
            return out if len(out) >= 2 else None

        each_side = int(cfg['grid'].get('lanes_each_side', 1))
        if cfg['grid'].get('all_lanes') or each_side > 0:
            for side in (+1, -1):
                for k in range(1, (4 if cfg['grid'].get('all_lanes') else each_side+1)):
                    lane = side_chain(side, k)
                    if lane is None:
                        break
                    runs, cur = [], [lane[0]]
                    for a, b in zip(lane[:-1], lane[1:]):
                        if np.linalg.norm(np.array([a.transform.location.x - b.transform.location.x,
                                                    a.transform.location.y - b.transform.location.y])) < 6.0:
                            cur.append(b)
                        else:
                            if len(cur) >= 2:
                                runs.append(cur)
                            cur = [b]
                    if len(cur) >= 2:
                        runs.append(cur)
                    for run in runs:
                        quads = quads + road_quads(run)
                        lanes_included += 1
        elif cfg['grid'].get('left_lane'):
            lane = side_chain(+1, 1)
            if lane is not None:
                quads = quads + road_quads(lane)
                lanes_included += 1
        summary['lanes_in_grid'] = lanes_included
        road_area = unary_union([Polygon(q) for q in quads]).buffer(1e-4)
        occupancy, road_mask = raster(frame, quads, obstacle_bodies, cfg['grid']['inflation_cells'])
        condition = frame.scene(np.array([ego_xy, goal_xy])).astype(np.float32)

        adapter = PlannerAdapter(cfg['planner'])
        request = dict(id=0, time=0.0, frame=frame, occupancy=occupancy, road=road_mask,
                       condition=condition, goal=goal_xy, seed=43)
        t0 = time.perf_counter()
        try:
            result = adapter(request)
        except ValueError as exc:
            summary.update(status='NO_GEOMETRIC_CANDIDATE', error=str(exc),
                           occupancy_free_ratio=float((occupancy == 0).mean()))
            (out/'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf8')
            print(json.dumps(summary, ensure_ascii=False, indent=1))
            return 3
        plan_ms = 1000*(time.perf_counter()-t0)
        path = result['path']
        pin_m = float(cfg['planner'].get('pin_start_m', 0) or 0)
        ar_rms = float(cfg['planner'].get('anchored_refit_rms', 0) or 0)
        if ar_rms > 0:
            from src.carla_bridge.path_profile import Path as _PathAR
            ar_xy, ar_turn, ar_dev = refit_anchored(
                path.xy, ego_xy, np.deg2rad(ego_wp.transform.rotation.yaw), rms=ar_rms)
            if ar_turn < 90.0 and len(ar_xy) > 10:
                path = _PathAR.build(ar_xy)
                summary.update(refit_tangent_deg=round(ar_turn, 2),
                               refit_deviation_m=round(ar_dev, 3))
                print('一次拟合(含起点位姿)：起点切向偏差 %.2f°，最大偏差 %.3f m'
                      % (ar_turn, ar_dev), flush=True)
                skip_extra = True
            else:
                skip_extra = False
        else:
            skip_extra = False
        if cfg['planner'].get('anchor_start', True) and not skip_extra:
            from src.carla_bridge.path_profile import Path as _Path
            ego_yaw = np.deg2rad(ego_wp.transform.rotation.yaw)
            new_xy, turned = anchor_start(path.xy, ego_xy, ego_yaw)
            if turned:
                path = _Path.build(new_xy)
                summary['start_anchor_deg'] = round(turned, 1)
                print('起点锚定：原切向与车身相差 %.1f°，已重建前 4 m' % turned, flush=True)
        refit_rms = 0.0 if (skip_extra or pin_m > 0) else float(cfg['planner'].get('refit_rms_m', 0) or 0)
        if refit_rms > 0:
            from src.carla_bridge.path_profile import Path as _PathR
            rf_xy, rf_dev = refit_spline(path.xy, rms=refit_rms)
            if rf_dev > 0:
                path = _PathR.build(rf_xy)
                summary['refit_deviation_m'] = round(rf_dev, 3)
                print('样条重拟合：RMS 容差 %.2f m，最大偏差 %.3f m' % (refit_rms, rf_dev), flush=True)
        if pin_m > 0:
            from src.carla_bridge.path_profile import Path as _PathP
            from src.carla_bridge.path_smooth import pin_and_spline as pin_and_refit
            pn_xy, pn_turn, pn_dev = pin_and_refit(
                path.xy, ego_xy, np.deg2rad(ego_wp.transform.rotation.yaw),
                sample_m=float(cfg['planner'].get('pin_sample_m', 12.0)))
            path = _PathP.build(pn_xy)
            summary.update(pin_start_m=pin_m, pin_tangent_deg=round(pn_turn, 2),
                           pin_deviation_m=round(pn_dev, 3))
            print('端点钉扎 %.1f m + 平滑重拟合：起点切向 %.2f°，最大偏差 %.3f m'
                  % (pin_m, pn_turn, pn_dev), flush=True)
        bend_m = 0.0 if pin_m > 0 else float(cfg['planner'].get('bend_start_m', 0) or 0)
        if bend_m > 0:
            from src.carla_bridge.path_profile import Path as _PathB
            from src.carla_bridge.path_smooth import bend_start
            bd_xy, bd_ang = bend_start(path.xy, ego_xy,
                                       np.deg2rad(ego_wp.transform.rotation.yaw), length=bend_m)
            bd_xy = bd_xy+(np.asarray(ego_xy, float)-bd_xy[0])   # 平移到自车中心（刚体，不改形状）
            path = _PathB.build(bd_xy)
            summary['bend_start_deg'] = round(bd_ang, 2)
            print('起点平滑弯折 %.1f m：旋转 %.2f°，并平移到自车中心' % (bend_m, bd_ang), flush=True)
        lead_m = float(cfg['planner'].get('lead_in_m', 0) or 0)
        if lead_m > 0:
            from src.carla_bridge.path_profile import Path as _PathL
            from src.carla_bridge.path_smooth import lead_in_quintic
            ld_xy = lead_in_quintic(path.xy, ego_xy, np.deg2rad(ego_wp.transform.rotation.yaw),
                                    length=lead_m)
            path = _PathL.build(ld_xy)
            summary['lead_in_m'] = lead_m
            print('五次 Hermite 前导段 %.1f m（C2 接缝）' % lead_m, flush=True)
        # 刚体对齐起点：平移+旋转，不改变曲率，因此不会产生折角
        if cfg['planner'].get('rigid_align_start', True):
            from src.carla_bridge.path_profile import Path as _PathRG
            from src.carla_bridge.path_smooth import align_start_rigid
            rg_xy, rg_ang = align_start_rigid(path.xy, ego_xy,
                                              np.deg2rad(ego_wp.transform.rotation.yaw))
            if rg_ang > 0.05:
                path = _PathRG.build(rg_xy)
                summary['rigid_align_deg'] = round(rg_ang, 2)
                print('起点刚体对齐：旋转 %.2f°' % rg_ang, flush=True)
        smooth_iters = 0 if skip_extra else int(cfg['planner'].get('smooth_iters', 0) or 0)
        if smooth_iters:
            from src.carla_bridge.path_profile import Path as _Path2
            sm_xy, dev = smooth_path(path.xy, iters=smooth_iters,
                                     lam=float(cfg['planner'].get('smooth_lambda', 0.45)))
            if dev > 0:
                path = _Path2.build(sm_xy)
                summary['smooth_deviation_m'] = round(dev, 3)
                print('曲率平滑：%d 次迭代，与原始规划最大偏差 %.3f m' % (smooth_iters, dev), flush=True)
        # 仅平移，使曲线起点落在自车中心（刚体平移，不改切向、不产生折角）
        if cfg['planner'].get('translate_to_ego', True) and pin_m <= 0:
            from src.carla_bridge.path_profile import Path as _PathT
            shift = np.asarray(ego_xy, float)-path.xy[0]
            if np.linalg.norm(shift) > 0.02:
                path = _PathT.build(path.xy+shift)
                print('平移 %.3f m 使起点对齐自车中心' % np.linalg.norm(shift), flush=True)
        summary.update(plan_ok=True, plan_ms=round(plan_ms, 1),
                       corridor_cells=(result['corridor'] or {}).get('num_cells'),
                       corridor_min_overlap=(result['corridor'] or {}).get('min_overlap'),
                       alm_status=result['alm_status'], topology=int(result['topology']),
                       occupancy_free_ratio=float((occupancy == 0).mean()),
                       goal_distance_m=float(np.linalg.norm(goal_xy-ego_xy)))
        summary['footprint'] = footprint_check(path, ego_body, obstacle_bodies, road_area,
                                               margin=cfg['safety']['road_margin_m'])
        summary['status'] = 'PLANNED' if summary['footprint']['passed'] else 'FOOTPRINT_CHECK_FAILED'

        # ============ 俯视 RGB + Python 侧投影半透明叠加 ============
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
        # 截图取景：镜头压在自车前方 14 m，高度 30 m -> 纵向约 34 m，自车在下缘、前方 30 m 可见
        _s_shot = path.s[-1]*0.45
        cam.set_transform(topdown(path.pose(_s_shot)[0], (path.s[-1]+18.0)/1.125))
        for _ in range(4):
            world.tick()

        # ---------------- 论文级配色（BGR） ----------------
        C_CORR_FILL = (196, 172, 132)    # 淡蓝灰：安全走廊
        C_CORR_EDGE = (168, 120, 70)     # 中蓝：走廊轮廓
        C_BAND_FILL = (150, 208, 160)    # 淡绿：车身扫掠带
        C_BAND_EDGE = (86, 146, 60)      # 深绿：带边界
        C_WIDTH     = (150, 208, 160)
        C_TRAJ      = (235, 235, 235)    # 近白：规划轨迹
        C_OBST      = (72, 72, 214)      # 红：障碍
        C_START     = (36, 150, 240)     # 橙：起点
        C_GOAL      = (72, 72, 214)      # 红：终点
        C_TEXT      = (238, 238, 238)
        C_SUB       = (170, 175, 180)
        C_BAR       = (32, 32, 36)

        def overlay(rgb, inverse, text):
            import cv2
            from shapely.geometry import Polygon as _Poly, MultiPolygon as _Multi, LineString as _Line
            from shapely.ops import unary_union as _union
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

            def fill(part, colour, alpha):
                pts = project(np.asarray(part.exterior.coords)[:-1])
                if pts is None or len(pts) < 3:
                    return
                layer = img.copy()
                cv2.fillPoly(layer, [pts], colour)
                cv2.addWeighted(layer, alpha, img, 1-alpha, 0, img)

            def stroke(part, colour, thick):
                pts = project(np.asarray(part.exterior.coords)[:-1])
                if pts is not None and len(pts) >= 3:
                    cv2.polylines(img, [pts], True, colour, thick, cv2.LINE_AA)

            # 1) 安全走廊：所有格子求并 → 一条轮廓（不再画 128 条线）
            band_clip = _Line(path.xy).buffer(7.5, cap_style=2, join_style=2)
            parts = []
            for poly in corridor:
                wp = _Poly(frame.world(np.asarray(poly)))
                if not wp.is_valid:
                    wp = wp.buffer(0)
                inter = wp.intersection(band_clip)
                if not inter.is_empty:
                    parts.extend(inter.geoms if isinstance(inter, _Multi) else [inter])
            if parts:
                sil = _union(parts)
                for part in (sil.geoms if isinstance(sil, _Multi) else [sil]):
                    fill(part, C_CORR_FILL, 0.30)
                for part in (sil.geoms if isinstance(sil, _Multi) else [sil]):
                    stroke(part, C_CORR_EDGE, 2)

            # 2) 车身扫掠带：填充 + 两侧边界 + 每 5 m 一道截面
            sweep = np.vstack([band_l, band_r[::-1]])
            pts = project(sweep)
            if pts is not None:
                layer = img.copy()
                cv2.fillPoly(layer, [pts], C_BAND_FILL)
                cv2.addWeighted(layer, 0.32, img, 0.68, 0, img)
            for line in (band_l, band_r):
                pl = project(line)
                if pl is not None:
                    cv2.polylines(img, [pl], False, C_BAND_EDGE, 2, cv2.LINE_AA)
            for s_i in np.arange(0.0, path.s[-1], 5.0):
                p_i, yaw_i = path.pose(float(s_i))
                n = np.array([-np.sin(yaw_i), np.cos(yaw_i)])
                seg = project(np.array([p_i+n*half, p_i-n*half]))
                if seg is not None:
                    cv2.line(img, tuple(seg[0]), tuple(seg[1]), C_BAND_EDGE, 1, cv2.LINE_AA)

            # 3) 规划轨迹（中心线，近白细线压在带上）
            cl = project(path.xy)
            if cl is not None:
                cv2.polylines(img, [cl], False, C_TRAJ, 2, cv2.LINE_AA)

            # 4) 障碍物：只描边
            for o in obstacle_bodies:
                pts_o = project(np.vstack([o.corners(0.0)]))
                if pts_o is not None:
                    cv2.polylines(img, [pts_o], True, C_OBST, 2, cv2.LINE_AA)

            # 5) 起点 / 终点
            for point, colour, label, off in ((ego_xy, C_START, 'START', (12, -10)),
                                              (goal_xy, C_GOAL, 'GOAL', (12, -10))):
                pt = project(point[None])
                if pt is None:
                    continue
                cv2.circle(img, tuple(pt[0]), 9, colour, 2, cv2.LINE_AA)
                cv2.putText(img, label, (int(pt[0][0])+off[0], int(pt[0][1])+off[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, .58, colour, 2, cv2.LINE_AA)

            # 6) 顶部信息条 + 图例
            bar = np.zeros((86, w, 3), np.uint8)
            bar[:] = C_BAR
            img = np.vstack([bar, img])
            cv2.putText(img, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, .62, C_TEXT, 1, cv2.LINE_AA)
            legend = [('safety corridor', C_CORR_FILL), ('body sweep', C_BAND_FILL),
                      ('planned path', C_TRAJ), ('static obstacle', C_OBST)]
            xoff = 16
            for label, colour in legend:
                cv2.rectangle(img, (xoff, 52), (xoff+18, 66), colour, -1)
                cv2.rectangle(img, (xoff, 52), (xoff+18, 66), (110, 110, 110), 1)
                cv2.putText(img, label, (xoff+24, 64), cv2.FONT_HERSHEY_SIMPLEX, .46, C_SUB, 1, cv2.LINE_AA)
                xoff += 24 + 9*len(label) + 26
            return img

        text = ("%s   plan %.0f ms   corridor %s cells   clearance %s m   footprint %s"
                % (cfg['id'], plan_ms, summary['corridor_cells'],
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

        viz_d = result.get('viz') or {}
        cand = viz_d.get('candidates') or {}
        # 俯视原图 + 相机矩阵留给 render_figure.py 出图（CARLA 只负责"拍照"）
        if captured:
            rgb_last, inv_last = captured[-1]
            np.savez_compressed(out/'view.npz', rgb=rgb_last, inverse=inv_last,
                                ground_z=float(z), half_width=float(ego_body.half_width),
                                half_length=float(ego_body.half_length),
                                ego_corners=ego_body.corners(0.0),
                                obstacle_corners=np.array([o.corners(0.0) for o in obstacle_bodies],
                                                          dtype=object))
        np.savez_compressed(
            out/'plan.npz', curve_world=path.xy, controls=result['controls'],
            occupancy=occupancy, road=road_mask, condition=condition,
            anchor=frame.origin, yaw=frame.yaw, goal=goal_xy,
            cand_xy=(cand.get('xy') if cand else np.zeros((0, 2))),
            cand_mask=(cand.get('mask') if cand else np.zeros(0, bool)),
            selected=int(viz_d.get('selected', -1)),
            cand_geom=(cand.get('geometry') if cand else np.zeros((0, 2))),
            cand_len=(cand.get('lengths') if cand else np.zeros(0)),
            corridors=np.array([np.asarray(p) for p in (viz_d.get('corridor') or {}).get('polygons', [])],
                               dtype=object),
            ellipse_center=(np.zeros(0) if result.get('ellipse_center') is None else np.asarray(result['ellipse_center'], dtype=float)),
            ellipse_ab=(np.zeros((0, 2)) if result.get('ellipse_ab') is None else np.asarray(result['ellipse_ab'], dtype=float)))
        summary['outputs'] = dict(dir=str(out), video=str(out/'scenario.mp4'),
                                  screenshot=str(out/'screenshot.png'))
        (out/'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf8')
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0 if summary['status'] == 'PLANNED' else 2
    finally:
        try:
            world.apply_settings(original)      # 先退同步模式，再销毁 actor，避免服务端卡死
        except Exception:
            pass
        for a in reversed(actors):
            try:
                a.destroy()
            except Exception:
                pass


if __name__ == '__main__':
    raise SystemExit(main())
