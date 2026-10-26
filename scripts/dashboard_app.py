"""交互式场景工具：随机地块 → 点起点/终点/放障碍 → 生成轨迹（双视图）。

    python scripts/dashboard_app.py --port 8815
    http://127.0.0.1:8815

左 = 占据栅格（整块地，全部车道为自由空间 + 障碍）
右 = CARLA 实景俯视（自车/障碍车都是真实 spawn 的 actor）+ 论文风格轨迹

两块图看向同一块地（同一中心、同一朝向、同一跨度），点击任一张都能放东西。
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, 'E:/CARLA_0.9.16/PythonAPI/carla')

SPAN = 160.0        # 地块边长 (m)：与模型输入完全一致
RES = 256           # 地块栅格边长：与模型输入完全一致
CELL = SPAN/RES

VIEW_PX = 1400      # 右侧实景边长
OCC_PX = 3          # 左侧上采样倍数

BODY = {            # 半长, 半宽 (m)
    'car': (2.10, 1.00),
    'truck': (4.00, 1.30),
    'cone': (0.25, 0.25),
    'barrier': (1.20, 0.30),
}
BP = {
    'ego': 'vehicle.tesla.model3',
    'car': 'vehicle.audi.a2',
    'truck': 'vehicle.carlamotors.carlacola',
    'cone': 'static.prop.constructioncone',
    'barrier': 'static.prop.streetbarrier',
}

SWEEP_RGB = (80, 220, 90)
CORE_RGB = (150, 255, 170)

STATE = {
    'map': None,
    'region': None,     # dict(cx, cy)
    'ego': None,        # [x, y, yaw_deg, z]
    'goal': None,       # [x, y, z]
    'obstacles': [],    # dict(x, y, yaw, z, kind)
    'plan': None,       # dict(xy, corridor, check, seed, msg)
    'actors': [],
    'ego_actor': None,
    'obs_actors': [],
    'version': 0,
    'seed': 43,
    'side': 'auto',
    'cands': [],
    'paint': [],          # [{mode:'block'|'free', r:米, pts:[[x,y],...]}]
}
CFG_PATH = ROOT/'configs'/'scenarios'/'S1_lane_block.yaml'
CFG = {'planner': {}, 'grid': {}, 'safety': {}}
BUILD = time.strftime('%Y%m%d_%H%M%S', time.localtime(Path(__file__).stat().st_mtime))
SCENE_DIR = ROOT/'outputs'/'dashboard_scenes'


def invalidate_caches():
    TOPO['map'], TOPO['segments'] = None, None
    CAM['key'], CAM['rgb'], CAM['inv'] = None, None, None
    MEDIAN['map'], MEDIAN['sign'], MEDIAN['lines'] = None, None, None
    CYCLE['order'], CYCLE['i'] = None, 0


def snapshot():
    """当前场景的可保存内容（地块 + 起点 + 终点 + 障碍 + 种子）。"""
    return dict(map=STATE['map'],
                region=STATE['region'],
                ego=STATE['ego'],
                goal=STATE['goal'],
                obstacles=STATE['obstacles'],
                seed=STATE['seed'],
                saved=time.strftime('%Y-%m-%d %H:%M:%S'))


def load_index():
    p = SCENE_DIR/'index.json'
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf8'))
    except Exception:
        return {}


def write_index(data):
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    p = SCENE_DIR/'index.json'
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf8')
    tmp.replace(p)
LOCK = threading.Lock()
CAM = {'key': None, 'rgb': None, 'inv': None}
TOPO = {'map': None, 'segments': None}


# --------------------------------------------------------------------- CARLA
def carla_world():
    import carla
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(20.0)
    return client.get_world()


def topology_segments(world):
    """整张地图所有 Driving 车道：缓存的是【纯 numpy 折线】，不是 CARLA 对象。

    服务端每个请求都会新建 client，旧 client 的 waypoint 句柄会失效，
    缓存 waypoint 会导致车道线画到错误位置（这正是候选通道时有时无的原因）。
    """
    cmap = world.get_map()
    if TOPO['map'] == cmap.name and TOPO['segments'] is not None:
        return TOPO['segments']
    segs = []
    for a, b in cmap.get_topology():
        if a is None or b is None or a.lane_type.name != 'Driving':
            continue
        pts, probe = [], a
        for _ in range(400):
            pts.append([probe.transform.location.x, probe.transform.location.y])
            nxt = [w for w in probe.next(2.0) if w.lane_id == probe.lane_id]
            if not nxt:
                break
            probe = nxt[0]
            if probe.transform.location.distance(b.transform.location) < 2.0:
                pts.append([probe.transform.location.x, probe.transform.location.y])
                break
        if len(pts) >= 2:
            segs.append((np.asarray(pts, float), float(a.lane_width)/2.0,
                         float(np.sign(a.lane_id) or 1.0)))
    TOPO['map'], TOPO['segments'] = cmap.name, segs
    return segs


def wp_xy(wps):
    return np.array([[w.transform.location.x, w.transform.location.y] for w in wps], float)


def polyline_quads(pts, hw):
    """折线 + 半宽 → 四边形（与 occupancy.road_quads 同一几何）。"""
    pts = np.asarray(pts, float)
    quads = []
    for a, b in zip(pts[:-1], pts[1:]):
        d = b-a
        n = np.hypot(*d)
        if n < 1e-6:
            continue
        r = np.array([-d[1], d[0]])/n*hw
        quads.append(np.array([a+r, a-r, b-r, b+r]))
    return quads


def lane_polylines(world, centre, half_span, direction=None):
    """direction=None 取全部车道；否则只取同向车道（lane_id 同号）。"""
    out = []
    for pts, hw, sign in topology_segments(world):
        if direction is not None and sign != direction:
            continue
        near = pts[(np.abs(pts[:, 0]-centre[0]) < half_span)
                   & (np.abs(pts[:, 1]-centre[1]) < half_span)]
        if len(near) >= 2:
            out.append((near, hw))
    return out


def split_runs(pts, gap=6.0):
    """按间距断开成连续段（numpy 版本）。"""
    runs, cur = [], [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        if np.hypot(*(b-a)) < gap:
            cur.append(b)
        else:
            if len(cur) >= 2:
                runs.append(np.array(cur))
            cur = [b]
    if len(cur) >= 2:
        runs.append(np.array(cur))
    return runs


def build_runs(wps, gap=6.0):
    """把 waypoint 序列按 6 m 断点切成连续段。"""
    runs, cur = [], [wps[0]]
    for a, b in zip(wps[:-1], wps[1:]):
        if np.hypot(a.transform.location.x-b.transform.location.x,
                    a.transform.location.y-b.transform.location.y) < gap:
            cur.append(b)
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = [b]
    if len(cur) >= 2:
        runs.append(cur)
    return runs


def frame_quads(world, frame, half_span, direction=None):
    """把坐标系范围内的车道做成四边形（弯道处按 6 m 断开）；direction 限定同向。"""
    quads = []
    c = frame.origin
    for pts, hw, sign in topology_segments(world):
        if direction is not None and sign != direction:
            continue
        near = pts[(np.abs(pts[:, 0]-c[0]) < half_span)
                   & (np.abs(pts[:, 1]-c[1]) < half_span)]
        if len(near) < 2:
            continue
        for run in split_runs(near):
            quads += polyline_quads(run, hw)
    return quads


# ----------------------------------------------------------------- 几何/像素
def region_frame():
    """模型输入帧 = 左图 = 160 m / 256²，世界对齐（+x 向右、+y 向下）。"""
    from src.carla_bridge.frame import SceneFrame
    r = STATE['region']
    return SceneFrame(np.array([r['cx'], r['cy']], float), 0.0, size=SPAN, resolution=RES)


def region_px(pts):
    """世界 → 模型栅格像素（就是左图的像素坐标，不再旋转）。"""
    frame = region_frame()
    return frame.grid(np.asarray(pts, float).reshape(-1, 2))


def region_world(u, v):
    """左图像素（归一化 0..1）→ 世界。"""
    frame = region_frame()
    col, row = u*(RES-1), v*(RES-1)
    scene = np.array([[col*2.0/RES-1.0, row*2.0/RES-1.0]])
    return tuple(float(x) for x in frame.world(scene)[0])


def snap(world, x, y):
    import carla
    wp = world.get_map().get_waypoint(
        carla.Location(x=float(x), y=float(y), z=1.0), project_to_road=True)
    return float(wp.transform.location.x), float(wp.transform.location.y), \
        float(wp.transform.rotation.yaw), float(wp.transform.location.z)


GOOD_SPAWNS = [12, 15, 26, 44, 45, 47, 50, 60, 61, 98, 99, 101, 108, 118]  # 已验证的直路段
CYCLE = {'order': None, 'i': 0}


def pick_region(world):
    """随机地块：Town10 用已验证的直路段；其它地图随机挑（按路面占比择优）。"""
    sps = world.get_map().get_spawn_points()
    if 'Town10' not in world.get_map().name:
        rng = np.random.default_rng()
        best = None
        for _ in range(10):
            sp = sps[int(rng.integers(len(sps)))]
            STATE['region'] = dict(cx=float(sp.location.x), cy=float(sp.location.y))
            ratio = float(region_road(world).mean())
            if best is None or ratio > best[0]:
                best = (ratio, dict(STATE['region']))
            if ratio >= 0.08:
                break
        STATE['region'] = best[1]
        return -1
    if CYCLE['order'] is None:
        CYCLE['order'] = list(GOOD_SPAWNS)
        np.random.shuffle(CYCLE['order'])
    best = None
    for _ in range(len(CYCLE['order'])):
        idx = CYCLE['order'][CYCLE['i'] % len(CYCLE['order'])]
        CYCLE['i'] += 1
        sp = sps[int(idx)]
        STATE['region'] = dict(cx=float(sp.location.x), cy=float(sp.location.y))
        ratio = float(region_road(world).mean())
        if best is None or ratio > best[0]:
            best = (ratio, dict(STATE['region']))
        if ratio >= 0.08:
            return idx
    STATE['region'] = best[1]
    return -1


def region_road(world):
    """地块内可行驶区域（1=可行驶），全部车道不管方向。"""
    import cv2
    r = STATE['region']
    road = np.zeros((RES, RES), np.uint8)
    SHIFT = 4
    for pts, hw in lane_polylines(world, (r['cx'], r['cy']), SPAN*0.78):
        g = np.rint(region_px(pts)*(1 << SHIFT)).astype(np.int32)
        th = max(1, int(math.ceil(2.0*hw/CELL)))
        cv2.polylines(road, [g], False, 1, th, cv2.LINE_8, shift=SHIFT)
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return road


def obstacle_polys():
    """障碍四角：优先用 CARLA 真实 bounding box，没有 actor 时退回标称尺寸。"""
    out = []
    import carla_scenario_sim as css
    for i, o in enumerate(STATE['obstacles']):
        if i < len(STATE['obs_actors']) and STATE['obs_actors'][i] is not None:
            out.append(css.body_of(STATE['obs_actors'][i]).corners(0.0))
        else:
            hl, hw = BODY[o['kind']]
            out.append(body_poly((o['x'], o['y']), o['yaw'], hl, hw))
    return out


def ego_poly():
    if STATE['ego'] is None:
        return None
    import carla_scenario_sim as css
    if STATE['ego_actor'] is not None:
        return css.body_of(STATE['ego_actor']).corners(0.0)
    e = STATE['ego']
    return body_poly(e[:2], e[2], *BODY['car'])


def body_poly(centre, yaw_deg, half_len, half_wid):
    c = np.asarray(centre, float)
    y = np.deg2rad(yaw_deg)
    d = np.array([np.cos(y), np.sin(y)])
    n = np.array([-np.sin(y), np.cos(y)])
    return np.array([c+d*half_len+n*half_wid, c+d*half_len-n*half_wid,
                     c-d*half_len-n*half_wid, c-d*half_len+n*half_wid])


# ------------------------------------------------------------------ 真实车辆
def clear_actors(world):
    for a in STATE['actors']:
        try:
            a.destroy()
        except Exception:
            pass
    STATE['actors'] = []


def cfg():
    """论文出图用的同一份配置（S1_lane_block.yaml）。"""
    if CFG.get('planner'):
        return CFG
    import yaml
    CFG.clear()
    CFG.update(yaml.safe_load(CFG_PATH.read_text(encoding='utf8')))
    return CFG


def spawn_actor(world, kind, x, y, yaw, z, blueprint=None):
    """只负责 spawn 并返回 (actor, transform)；落地由 respawn 的 tick 序列完成。"""
    import carla
    lib = world.get_blueprint_library()
    try:
        bp = lib.find(blueprint or BP[kind])
    except Exception:
        return None, None
    try:
        bp.set_attribute('role_name', 'dash_'+kind)     # 便于重启后清理残留
    except Exception:
        pass
    tf = carla.Transform(carla.Location(x=float(x), y=float(y), z=float(z)+0.25),
                         carla.Rotation(yaw=float(yaw)))
    act = None
    nyaw = float(yaw)
    dx, dy = np.cos(np.deg2rad(nyaw)), np.sin(np.deg2rad(nyaw))
    nx, ny = -dy, dx                                        # 横向单位向量
    for dz, off, lat in ((0.0, 0.0, 0.0), (0.0, 1.6, 0.0), (0.0, -1.6, 0.0),
                         (0.0, 0.0, 1.4), (0.0, 0.0, -1.4), (0.5, 0.0, 0.0)):
        tf.location.z = float(z)+0.25+dz
        tf.location.x = float(x)+off*dx+lat*nx
        tf.location.y = float(y)+off*dy+lat*ny
        try:
            act = world.try_spawn_actor(bp, tf)
        except Exception:
            act = None
        if act is not None:
            break
        act = None
    if act is None:
        print('spawn 失败', kind, blueprint or BP[kind], round(x, 1), round(y, 1), flush=True)
    return act, tf


def respawn(world):
    """按当前 ego/障碍 重新 spawn 真实车辆。

    CARLA 必须先 tick 一次 actor 才“落地”，否则 set_simulate_physics(False)
    会把车弹回世界原点、set_transform 也无效。所以整段都在同步模式里做。
    """
    with LOCK:
        st = world.get_settings()
        was = st.synchronous_mode
        st.synchronous_mode = True
        st.fixed_delta_seconds = 0.05
        world.apply_settings(st)
        try:
            clear_actors(world)
            STATE['ego_actor'], STATE['obs_actors'] = None, []
            world.tick()
            made = []
            if STATE['ego']:
                e = STATE['ego']
                act, tf = spawn_actor(world, 'ego', e[0], e[1], e[2], e[3],
                                      blueprint=cfg()['ego'].get('blueprint'))
                if act is not None:
                    STATE['ego_actor'] = act
                    STATE['actors'].append(act)
                    made.append((act, tf))
            for o in STATE['obstacles']:
                act, tf = spawn_actor(world, o['kind'], o['x'], o['y'], o['yaw'], o['z'])
                if act is not None:
                    STATE['obs_actors'].append(act)
                    STATE['actors'].append(act)
                    made.append((act, tf))
            world.tick()                     # 让 actor 注册
            for act, tf in made:
                try:
                    act.set_simulate_physics(False)
                except Exception:
                    pass
                try:
                    act.set_transform(tf)     # 必须有，否则停在原点
                except Exception:
                    pass
            world.tick()
        finally:
            st.synchronous_mode = False        # 本工具绝不把世界留在同步模式
            world.apply_settings(st)
    STATE['version'] += 1


# -------------------------------------------------------------------- 实景图
def grab(world):
    import carla
    r = STATE['region']
    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', str(VIEW_PX))
    bp.set_attribute('image_size_y', str(VIEW_PX))
    bp.set_attribute('fov', '90')
    cam = world.spawn_actor(bp, carla.Transform(
        carla.Location(x=r['cx'], y=r['cy'], z=SPAN/2.0),
        carla.Rotation(pitch=-90.0, yaw=-90.0)))   # 与左图一致：+x 向右、+y 向下
    frames = []
    cam.listen(lambda img: frames.append((
        np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, :3].copy(),
        np.asarray(img.transform.get_inverse_matrix(), float))))
    st = world.get_settings()
    was = st.synchronous_mode
    st.synchronous_mode = True
    st.fixed_delta_seconds = 0.05
    world.apply_settings(st)
    try:
        for _ in range(5):
            world.tick()
    finally:
        st.synchronous_mode = False            # 本工具绝不把世界留在同步模式
        world.apply_settings(st)
        try:
            cam.stop()
            cam.destroy()
        except Exception:
            pass
    if not frames:
        return np.zeros((VIEW_PX, VIEW_PX, 3), np.uint8), np.eye(4)
    return frames[-1]


def capture(world):
    key = (round(STATE['region']['cx'], 2), round(STATE['region']['cy'], 2),
           STATE['version'])
    if CAM['key'] == key and CAM['rgb'] is not None:
        return CAM['rgb'], CAM['inv']
    with LOCK:
        if CAM['key'] == key and CAM['rgb'] is not None:
            return CAM['rgb'], CAM['inv']
        rgb, inv = grab(world)
        CAM.update(key=key, rgb=rgb, inv=inv)
    return rgb, inv


def projector(inv, w, h, ground=0.06):
    def px(points):
        pts = np.asarray(points, float).reshape(-1, 2)
        hom = np.column_stack([pts, np.full(len(pts), ground), np.ones(len(pts))])
        c3 = hom @ inv.T
        c3[:, 0] = np.maximum(c3[:, 0], 0.05)
        f = w/2.0
        return np.column_stack([w/2+f*c3[:, 1]/c3[:, 0], h/2-f*c3[:, 2]/c3[:, 0]])
    return px


def paper_overlay(img, px, path_xy, half_w, hidden, sweep_extra=0.30, layers=14,
                  core_ratio=0.62):
    """论文风格：柔和边扫掠带 + 主体 + 中心高亮 + 中心线（与 render_style.py 一致）。"""
    import cv2
    from src.carla_bridge.path_profile import Path as GeomPath
    geom = GeomPath.build(np.asarray(path_xy, float))
    h, w = img.shape[:2]
    SHIFT = 4
    SCALE = 1 << SHIFT

    def px_sub(points):
        return np.rint(px(points)*SCALE).astype(np.int32)

    keep = (1.0-hidden)[:, :, None]

    def band(half, steps=0.4):
        left, right = [], []
        for s in np.arange(0.0, geom.s[-1], steps):
            p, yaw = geom.pose(float(s))
            n = np.array([-np.sin(yaw), np.cos(yaw)])
            left.append(p+n*half)
            right.append(p-n*half)
        return np.vstack([np.array(left), np.array(right)[::-1]])

    def blend(poly, colour, alpha):
        nonlocal img
        mask = np.zeros((h, w), np.float32)
        cv2.fillPoly(mask, [px_sub(poly)], 1.0, shift=SHIFT, lineType=cv2.LINE_AA)
        mask = cv2.GaussianBlur(mask, (0, 0), 3.0)[:, :, None]*keep
        img = img*(1-alpha*mask)+np.array(colour, np.float32)*alpha*mask

    sweep = half_w+sweep_extra
    for k in range(layers):
        t = k/(layers-1) if layers > 1 else 1.0
        blend(band(sweep*(1.0+0.45*(1.0-t))), SWEEP_RGB, 0.042)
    blend(band(sweep), SWEEP_RGB, 0.32)
    blend(band(sweep*0.92), SWEEP_RGB, 0.10)
    blend(band(sweep*core_ratio), CORE_RGB, 0.24)
    blend(band(sweep*core_ratio*0.6), CORE_RGB, 0.18)

    out = np.clip(img, 0, 255).astype(np.uint8)
    line_mask = np.zeros((h, w), np.uint8)
    cv2.polylines(line_mask, [px_sub(path_xy)], False, 255, 2, cv2.LINE_AA, shift=SHIFT)
    line_mask = cv2.GaussianBlur(line_mask, (0, 0), 0.8).astype(np.float32)/255.0
    line_mask = (line_mask*keep[:, :, 0])[:, :, None]
    return (out.astype(np.float32)*(1-0.85*line_mask)
            + np.array((240, 252, 246), np.float32)*0.80*line_mask).astype(np.uint8)


def hidden_mask(px, w, h):
    """车身/障碍占位（绿色不上车身）。"""
    import cv2
    SHIFT = 4
    mask = np.zeros((h, w), np.float32)
    polys = []
    if STATE['ego']:
        polys.append(ego_poly())
    polys += obstacle_polys()
    for poly in polys:
        pts = np.rint(px(poly)*(1 << SHIFT)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1.0, shift=SHIFT, lineType=cv2.LINE_AA)
    if mask.any():
        mask = cv2.erode(mask, np.ones((13, 13), np.uint8))
        mask = cv2.GaussianBlur(mask, (0, 0), 1.2)
        mask = np.clip(mask*1.2, 0.0, 1.0)
    return mask


def view_png(world):
    """右图：真实实景 + （有轨迹时）论文风格叠加。"""
    import cv2
    rgb, inv = capture(world)
    h, w = rgb.shape[:2]
    px = projector(inv, w, h)
    img = rgb.astype(np.float32)
    plan = STATE['plan']
    if plan and len(plan['xy']) > 5:
        img = paper_overlay(img, px, plan['xy'], BODY['car'][1], hidden_mask(px, w, h))
    out = np.clip(img, 0, 255).astype(np.uint8)
    if not (plan and len(plan['xy']) > 5) and STATE['goal']:
        g = px(np.array([STATE['goal'][:2]]))[0]
        cv2.drawMarker(out, (int(g[0]), int(g[1])), (60, 60, 235),
                       cv2.MARKER_TILTED_CROSS, 22, 3, cv2.LINE_AA)
    ok, buf = cv2.imencode('.png', out)
    return base64.b64encode(buf.tobytes()).decode()


# ------------------------------------------------------------------- 占据栅格
def occ_png(world):
    """左图 = 模型输入（同一张 256² 数组），只额外叠自车/终点/走廊/轨迹。"""
    import cv2
    SHIFT = 4
    frame = region_frame()
    road = lane_raster(world, frame, None)            # 车道（白）
    occ = apply_paint(road.copy(), frame)             # 叠加涂抹后的模型输入
    img = np.full((RES, RES, 3), 96, np.uint8)        # 非道路（深灰）
    img[road == 0] = (232, 232, 232)
    img[(road == 0) & (occ == 1)] = (200, 140, 60)    # 你涂的不可行驶（蓝）
    img[(road == 1) & (occ == 0)] = (90, 190, 90)     # 你涂的可行驶（绿）
    for poly in obstacle_polys():
        g = np.rint(region_px(poly)*(1 << SHIFT)).astype(np.int32)
        cv2.fillPoly(img, [g], (70, 70, 225), shift=SHIFT, lineType=cv2.LINE_AA)
    if STATE.get('cands') and not STATE['plan']:
        cols = [(60, 200, 255), (255, 120, 60), (60, 255, 160), (200, 60, 255),
                (60, 120, 255), (255, 200, 60)]
        for i, poly in enumerate(STATE['cands']):
            g = np.rint(region_px(poly)*(1 << SHIFT)).astype(np.int32)
            cv2.polylines(img, [g], False, cols[i % len(cols)], 3, cv2.LINE_AA, shift=SHIFT)
            mid = poly[len(poly)//2]
            gp = region_px(np.array([mid]))[0]
            cv2.putText(img, str(i+1), (int(gp[0])+4, int(gp[1])-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, cols[i % len(cols)], 2, cv2.LINE_AA)
    plan = STATE['plan']
    if plan and plan['corridor']:
        cells = [c for c in plan['corridor'] if len(c) >= 3]
        if cells:
            areas = np.array([abs(cv2.contourArea(np.asarray(c, np.float32))) for c in cells])
            med = float(np.median(areas)) or 1.0
            mask = np.zeros((RES, RES), np.uint8)
            for c, a in zip(cells, areas):
                if a > 5.0*med:
                    continue
                g = cv2.convexHull(np.rint(region_px(c)*(1 << SHIFT)).astype(np.int32))
                cv2.fillPoly(mask, [g], 255, shift=SHIFT, lineType=cv2.LINE_AA)
            m = cv2.GaussianBlur(mask.astype(np.float32)/255.0, (0, 0), 1.0)[:, :, None]
            img = (img.astype(np.float32)*(1-0.18*m)
                   + np.array((205, 175, 130), np.float32)*0.18*m).astype(np.uint8)
    if plan and len(plan['xy']) > 2:
        g = np.rint(region_px(plan['xy'])*(1 << SHIFT)).astype(np.int32)
        cv2.polylines(img, [g], False, (235, 235, 45), 2, cv2.LINE_AA, shift=SHIFT)
    if STATE['ego']:
        g = np.rint(region_px(ego_poly())*(1 << SHIFT)).astype(np.int32)
        cv2.fillPoly(img, [g], (40, 140, 240), shift=SHIFT, lineType=cv2.LINE_AA)
    if STATE['goal']:
        g = region_px(np.array([STATE['goal'][:2]]))[0]      # drawMarker 不支持 shift
        cv2.drawMarker(img, (int(round(g[0])), int(round(g[1]))), (60, 60, 235),
                       cv2.MARKER_TILTED_CROSS, 13, 3, cv2.LINE_AA)
    img = cv2.resize(img, (RES*OCC_PX, RES*OCC_PX), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf.tobytes()).decode()


# ---------------------------------------------------------------------- 规划
def ego_lane_sign(world, xy=None):
    """某点所在车道的方向（lane_id 符号）。取不到返回 None（=不做方向过滤）。"""
    pts = STATE['ego'] if xy is None else xy
    if pts is None:
        return None
    import carla
    wp = world.get_map().get_waypoint(
        carla.Location(x=float(pts[0]), y=float(pts[1]), z=1.0), project_to_road=True)
    return float(np.sign(wp.lane_id) or 1.0)


JUNCTION_KEEP_M = 20.0     # 短车道（路口内转弯/connecting lane）永远保留，否则路口断掉


def seg_len(pts):
    p = np.asarray(pts, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def lane_raster(world, frame, direction=None):
    """车道中心线按真实车道宽画粗线。

    direction 给定时：同向车道全保留；反向车道只保留路口内的短车道
    （否则十字路口的转弯 lane 会被过滤掉，走廊在路口断开；掉头也靠这些短车道）。
    """
    import cv2
    res = frame.resolution
    cell = frame.size/res
    free = np.zeros((res, res), np.uint8)
    SHIFT = 4
    for pts, hw, sign in topology_segments(world):
        if direction is not None and sign != direction and seg_len(pts) > JUNCTION_KEEP_M:
            continue
        sel = pts[(np.abs(pts[:, 0]-frame.origin[0]) < frame.size*0.72)
                  & (np.abs(pts[:, 1]-frame.origin[1]) < frame.size*0.72)]
        if len(sel) < 2:
            continue
        g = np.rint(frame.grid(sel)*(1 << SHIFT)).astype(np.int32)
        th = max(1, int(math.ceil(2.0*hw/cell)))
        cv2.polylines(free, [g], False, 1, th, cv2.LINE_8, shift=SHIFT)
    free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return 1-free


def scene_direction(world):
    """不再按方向过滤车道：所有车道都算自由空间。

    逆行由 median_lines() 画出的中线墙来阻止（只挡相邻反向车道之间，
    所以路口左拐/掉头仍然可以通过），而不是把整条对向车道判成障碍。
    """
    return None


MEDIAN = {'map': None, 'sign': None, 'lines': None}


def median_lines(world, sign):
    """相邻反向车道之间的中线折线（世界坐标），按地图+方向缓存。"""
    if MEDIAN['map'] == world.get_map().name and MEDIAN['sign'] == sign \
            and MEDIAN['lines'] is not None:
        return MEDIAN['lines']
    from scipy.spatial import cKDTree
    segs = topology_segments(world)
    opp = [q for q, hw, sg in segs if sg != sign]
    lines = []
    if opp:
        tree = cKDTree(np.vstack(opp))
        for q, hw, sg in segs:
            if sg != sign or len(q) < 4:
                continue
            d = np.gradient(q, axis=0)
            nr = np.linalg.norm(d, axis=1)
            nr[nr < 1e-6] = 1.0
            t = d/nr[:, None]
            n = np.stack([-t[:, 1], t[:, 0]], axis=1)
            mid = []
            for i in range(len(q)):
                for off in (hw+1.75, -(hw+1.75)):
                    probe = q[i]+n[i]*off
                    if tree.query(probe)[0] < 1.8:
                        mid.append(q[i]+n[i]*off*0.5)
                        break
            if len(mid) >= max(4, int(0.5*len(q))):
                lines.append(np.asarray(mid, float))
    MEDIAN.update(map=world.get_map().name, sign=sign, lines=lines)
    return lines


def fill_small_gaps(occ, max_cells=260):
    """把被路面包围、面积很小的空洞填成可行驶。

    车道中线栅格在路口拐角会漏掉几格空地（CARLA 实景是路面），
    这里按连通块面积区分：小于 max_cells 格且不接触画面边界的空洞 → 填平；
    建筑街区那种大块区域（几千格）不会被误填。
    """
    from scipy.ndimage import label
    blocked = (occ == 1).astype(np.uint8)
    lab, n = label(blocked, structure=np.ones((3, 3)))
    if n == 0:
        return occ
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    sizes = np.bincount(lab.ravel())
    out = occ.copy()
    for k in range(1, n+1):
        if k in border:
            continue
        if sizes[k] <= max_cells:
            out[lab == k] = 0
    return out


def apply_paint(mask, frame):
    """把用户涂抹按顺序盖到栅格上（block=1 不可行驶，free=0 可行驶）。"""
    import cv2
    cell = frame.size/frame.resolution
    for stroke in STATE['paint']:
        rr = max(1, int(round(float(stroke['r'])/cell)))
        val = 1 if stroke['mode'] == 'block' else 0
        g = frame.grid(np.asarray(stroke['pts'], float))
        for cx, cy in g:
            cv2.circle(mask, (int(round(cx)), int(round(cy))), rr, val, -1)
    return mask


def scene_raster(world, frame):
    """模型的占据图：全部车道都算自由空间（不分方向、不加中线）。

    之前的"中线墙"实现会把路口画乱、还误伤路缘，已撤回。
    要限制逆行应该在车道级建图上做，不能用散点拼。
    """
    return apply_paint(fill_small_gaps(lane_raster(world, frame, None)), frame)


def build_scene(world):
    """模型输入 = 左边那张图：160 m / 256²，车道按真实宽度画，0=自由。

    所有车道都算自由空间（不分方向）；障碍用 CARLA 真实 bounding box 盖上去，
    不膨胀；不挪起点、不挪终点。
    """
    import cv2
    import carla_scenario_sim as css
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    frame = region_frame()
    direction = scene_direction(world)
    road = scene_raster(world, frame)                 # 0=可行驶（与左图逐像素一致）
    occupancy = road.copy()
    # 障碍一律进栅格：有真实 actor 用它的 bounding box，spawn 失败则用标称尺寸兜底
    from src.carla_bridge.dynamic_objects import Body
    bodies = []
    for i, o in enumerate(STATE['obstacles']):
        if i < len(STATE['obs_actors']) and STATE['obs_actors'][i] is not None:
            bodies.append(css.body_of(STATE['obs_actors'][i]))
        else:
            hl, hw = BODY.get(o['kind'], BODY['car'])
            bodies.append(Body(-1, np.array([o['x'], o['y']], float),
                               np.deg2rad(o['yaw']), np.zeros(2), hl, hw))
    for body in bodies:
        cv2.fillPoly(occupancy, [np.rint(frame.grid(body.corners())).astype(np.int32)], 1)
    quads = frame_quads(world, frame, frame.size*0.72, direction)
    road_area = unary_union([Polygon(q) for q in quads]).buffer(1e-4) if quads else None
    return frame, occupancy, road, road_area, bodies


SIDE = {'mode': 'auto', 'vec': None, 'orig': None, 'kept': None, 'total': None}
FORCE = {'idx': None}          # 手动指定候选序号（None=交给模型选）


def install_candidate_bias():
    """绕行侧过滤：候选生成后，按“候选相对车头的平均横向偏移符号”筛掉另一侧。

    只影响交��模型挑哪条候选，不改模型、不改走廊、不改栅格。
    指定的一侧一条候选都没有时，退回不过滤（不会把功能搞死）。
    """
    import src.geometry.skeleton_paths as sp
    if SIDE['orig'] is not None:
        return
    SIDE['orig'] = sp.generate_candidates

    def wrapped(graph, start, goal, cfg):
        cands = SIDE['orig'](graph, start, goal, cfg)
        vec = SIDE['vec']
        if vec is None or cands.num_valid == 0:
            return cands
        start = np.asarray(start, float)
        vec = np.asarray(vec, float)
        lat = np.zeros(cands.num_slots)
        for i in range(cands.num_slots):
            if not cands.mask[i]:
                continue
            poly = cands.geometry[i] if i < len(cands.geometry) else cands.coords[i]
            poly = np.asarray(poly, float)
            prof = (poly-start) @ vec
            lat[i] = float(prof[int(np.argmax(np.abs(prof)))])  # 取摆幅最大处的符号
        keep = (lat > 0.4) if SIDE['mode'] == 'right' else (lat < -0.4)
        keep = keep & cands.mask
        if FORCE['idx'] is not None:
            vi = cands.valid_index()
            if len(vi):
                keep = np.zeros_like(cands.mask)
                keep[vi[int(FORCE['idx']) % len(vi)]] = True
        SIDE['total'] = int(cands.num_valid)
        if not keep.any():
            SIDE['kept'] = None
            return cands
        cands.mask = keep
        SIDE['kept'] = int(keep.sum())
        return cands

    sp.generate_candidates = wrapped


ANCHOR = {'relaxed': False}


def relax_anchors():
    """骨架锚点：直接取最近的骨架节点（不分方向）。

    项目自带 src/carla_bridge/skeleton_query.py::QueryGraph.visible_anchors 要求锚点
    严格落在起点正前方 / 终点正后方 3-18 m、车头 +-35 度锥内，骨架一旦分叉或横向就
    一个锚点都取不到（实测 spawn 12/50/60 锚点 0 个 -> 算不出候选通道）。

    车后方那段无用骨架在拿到路径后由 trim_to_ego() 截掉，所以这里不需要方向门控。
    """
    if ANCHOR['relaxed']:
        return
    import src.carla_bridge.skeleton_query as sq
    from src.geometry.skeleton_graph import SkeletonGraph
    sq.QueryGraph.visible_anchors = (
        lambda self, point_scene, max_candidates=16:
        SkeletonGraph.visible_anchors(self, point_scene, max_candidates))
    ANCHOR['relaxed'] = True
    print('骨架锚点：取最近节点（车后段由 trim_to_ego 截掉）', flush=True)


def trim_to_ego(path_xy, ego_xy, step=0.5):
    """把车投影到规划出的骨架上：截掉车后方那段，再把车本身接成起点。"""
    from src.carla_bridge.path_profile import Path as GeomPath
    geom = GeomPath.build(np.asarray(path_xy, float))
    ego = np.asarray(ego_xy, float)
    if geom.s[-1] < 2.0:
        return np.asarray(path_xy, float)
    grid = np.arange(0.0, geom.s[-1], step)
    pts = np.array([geom.pose(float(s))[0] for s in grid])
    s0 = float(grid[int(np.argmin(np.linalg.norm(pts-ego, axis=1)))])
    keep = np.array([geom.pose(float(s))[0] for s in np.arange(s0, geom.s[-1], step)])
    if len(keep) < 3:
        return np.asarray(path_xy, float)
    return np.vstack([ego[None, :], keep])


def snap_free(frame, occ, xy, radius_cells=20):
    """把点吸附到模型栅格里最近的自由格（用户点的终点可能压在不可行驶区）。"""
    g = frame.grid(np.asarray(xy, float).reshape(1, 2))[0]
    res = occ.shape[0]
    c0, r0 = int(round(g[0])), int(round(g[1]))
    best = None
    for r in range(max(0, r0-radius_cells), min(res, r0+radius_cells+1)):
        for c in range(max(0, c0-radius_cells), min(res, c0+radius_cells+1)):
            if occ[r, c] == 0:
                d2 = (c-c0)**2+(r-r0)**2
                if best is None or d2 < best[0]:
                    best = (d2, c, r)
    if best is None:
        return None
    scene = np.array([[best[1]*2.0/res-1.0, best[2]*2.0/res-1.0]])
    return frame.world(scene)[0]


def goal_lane_yaw(world, goal_xy, raw_path):
    """终点切向：取终点所在车道方向；若与原始轨迹末端方向相反则翻转 180°。"""
    import carla
    wp = world.get_map().get_waypoint(
        carla.Location(x=float(goal_xy[0]), y=float(goal_xy[1]), z=1.0), project_to_road=True)
    yaw = float(np.deg2rad(wp.transform.rotation.yaw))
    try:
        from src.carla_bridge.path_profile import Path as GeomPath
        _, yaw_raw = GeomPath.build(np.asarray(raw_path, float)).pose(
            float(GeomPath.build(np.asarray(raw_path, float)).s[-1]))
        if np.cos(yaw-yaw_raw) < 0:
            yaw += np.pi
    except Exception:
        pass
    return yaw


def candidate_paths(world):
    """只算骨架候选路线（不跑扩散），返回世界坐标折线列表。"""
    from src.carla_bridge.planner_adapter import PlannerAdapter
    from src.carla_bridge.skeleton_query import query_graph
    from src.geometry.skeleton_paths import generate_candidates
    d = install_candidate_bias() and None
    adapter = PlannerAdapter(cfg()['planner'])
    frame, occ, road, road_area, bodies = build_scene(world)
    cond = frame.scene(np.array([STATE['ego'][:2], STATE['goal'][:2]], float)).astype(np.float32)
    old = FORCE['idx']
    FORCE['idx'] = None
    graph = query_graph(adapter.engine.get_graph(adapter.engine.occupancy_key(occ), occ),
                        *cond)
    cands = generate_candidates(graph, cond[0], cond[1], adapter.online_cand_cfg)
    FORCE['idx'] = old
    out = []
    if cands.num_valid:
        for i in cands.valid_index():
            poly = cands.geometry[i] if i < len(cands.geometry) else cands.coords[i]
            out.append(np.asarray(frame.world(np.asarray(poly, float)), float))
    STATE['cands'] = out
    return out


def path_min_clearance(path, bodies, half_len, half_wid, step=1.0, margin=0.0):
    """轨迹上自车矩形到各障碍多边形的最小距离（米）。"""
    from shapely.geometry import Polygon
    polys = [Polygon(b.corners(0.0)) for b in bodies]
    if not polys:
        return 99.0
    worst = 99.0
    info = (None, None)
    for s in np.arange(0.0, path.s[-1], step):
        p, yaw = path.pose(float(s))
        c, sn = np.cos(yaw), np.sin(yaw)
        d = np.array([c, sn])
        n = np.array([-sn, c])
        rect = Polygon([p+d*(half_len+margin)+n*(half_wid+margin),
                        p+d*(half_len+margin)-n*(half_wid+margin),
                        p-d*(half_len+margin)-n*(half_wid+margin),
                        p-d*(half_len+margin)+n*(half_wid+margin)])
        for k, P in enumerate(polys):
            dist = float(rect.distance(P))
            if dist < worst:
                worst = dist
                info = (k, float(s))
    path_min_clearance.info = info
    return worst


def push_clear(xy, bodies, radius=0.6, iters=3):
    """把穿过/贴着障碍的点沿远离障碍中心的方向推出去，再轻微平滑。"""
    from shapely.geometry import Point, Polygon
    polys = [Polygon(b.corners(0.0)) for b in bodies]
    pts = np.asarray(xy, float).copy()
    if not polys:
        return pts
    for _ in range(iters):
        moved = 0
        for i in range(len(pts)):
            p = pts[i]
            for P in polys:
                if Point(p[0], p[1]).distance(P) >= radius:
                    continue
                c = np.array([P.centroid.x, P.centroid.y])
                v = p-c
                nv = float(np.linalg.norm(v))
                dirv = v/nv if nv > 1e-6 else np.array([0.0, 1.0])
                for t in np.linspace(0.5, 10.0, 40):
                    q = p+dirv*t
                    if Point(q[0], q[1]).distance(P) >= radius:
                        pts[i] = q
                        moved += 1
                        break
        if not moved:
            break
        # 平滑被推开的点，避免折角（端点不动）
        sm = pts.copy()
        sm[1:-1] = 0.5*pts[1:-1]+0.25*pts[:-2]+0.25*pts[2:]
        pts = sm
    return pts


def free_tol(frame, occ, xy, k=1):
    """点所在格（含 ±k 邻域）是否有自由格。只放宽判定，不移动点。"""
    g = frame.grid(np.asarray(xy, float).reshape(1, 2))[0]
    res = occ.shape[0]
    c0, r0 = int(np.floor(g[0])), int(np.floor(g[1]))
    for r in range(max(0, r0-k), min(res, r0+k+1)):
        for c in range(max(0, c0-k), min(res, c0+k+1)):
            if occ[r, c] == 0:
                return True
    return False


def free_components(frame, occ, ego_xy, goal_xy):
    """自由空间连通块诊断：(块数, 起点块, 起点块格数, 终点块, 终点块格数, 是否同块)。"""
    from scipy.ndimage import label
    lab, n = label((occ == 0).astype(np.uint8), structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    def cell(pt):
        g = frame.grid(np.asarray(pt, float).reshape(1, 2))[0]
        r = int(np.floor(g[1])); c = int(np.floor(g[0]))
        r = min(max(r, 0), lab.shape[0]-1); c = min(max(c, 0), lab.shape[1]-1)
        return int(lab[r, c])
    a, b = cell(ego_xy), cell(goal_xy)
    sa = int(sizes[a]) if 0 < a < len(sizes) else 0
    sb = int(sizes[b]) if 0 < b < len(sizes) else 0
    return n, a, sa, b, sb, (a == b and a > 0)


def plan(world, seed):
    """起点、终点直接喂给模型：没有距离上限，只要点在图上。"""
    import carla_scenario_sim as css
    from src.carla_bridge.path_profile import Path as GeomPath
    from src.carla_bridge.path_smooth import pin_and_spline, pin_both_ends
    from src.carla_bridge.planner_adapter import PlannerAdapter

    c = cfg()
    e, g = STATE['ego'], STATE['goal']
    ego_xy = np.array(e[:2], float)
    ego_yaw = float(e[2])
    goal_xy = np.array(g[:2], float)
    frame, occ, road, road_area, bodies = build_scene(world)

    # 起点、终点都用用户点的原坐标：不吸附、不挪动
    for name, pt in (('起点', ego_xy), ('终点', goal_xy)):
        if not free_tol(frame, occ, pt, 1):
            raise ValueError('%s不在可行驶区域（点在路面上再试）' % name)

    install_candidate_bias()
    SIDE['mode'] = STATE.get('side', 'auto')
    if SIDE['mode'] == 'auto':
        SIDE['vec'] = None
    else:
        h = np.deg2rad(ego_yaw)
        SIDE['vec'] = np.array([-np.sin(h), np.cos(h)])   # 车头右侧单位向量
    SIDE['kept'] = SIDE['total'] = None

    ncomp, ca, sa, cb, sb, same = free_components(frame, occ, ego_xy, goal_xy)
    if not same:
        raise ValueError(
            '起点和终点不在同一块可行驶区域（自由空间共 %d 块：起点在第 %d 块 %d 格，'
            '终点在第 %d 块 %d 格）。说明涂抹/障碍把通道封死了，用「涂可行驶」打通一段再试。'
            % (ncomp, ca, sa, cb, sb))

    adapter = PlannerAdapter(c['planner'])
    t0 = time.perf_counter()

    def attempt(occ_x, road_x, goal_x):
        cond = frame.scene(np.array([ego_xy, goal_x])).astype(np.float32)
        return adapter(dict(id=0, time=0.0, frame=frame, occupancy=occ_x, road=road_x,
                            condition=cond, goal=goal_x, seed=int(seed)))

    try:
        res = attempt(occ, road, goal_xy)
    except ValueError:
        # 锚点门控太严（要求起终点沿车头方向 3–18 m 内有骨架节点）→ 放宽为最近节点
        relax_anchors()
        res = attempt(occ, road, goal_xy)
    ms = 1000*(time.perf_counter()-t0)

    # 截掉车后方那段骨架，把车本身接成起点，再做两端钉扎拟合
    raw = trim_to_ego(res['path'].xy, ego_xy)
    goal_yaw = goal_lane_yaw(world, goal_xy, raw)
    sample_m = float(c['planner'].get('pin_sample_m', 12.0))
    pin, turn, dev = pin_and_spline(raw, ego_xy, np.deg2rad(ego_yaw), sample_m=sample_m)
    path = GeomPath.build(pin)
    # 拟合会把原始路径的绕行弯“切掉”，所以拟合后再查一次净空，不够就推出去重拟合
    hl, hw = BODY['car']
    if STATE['ego_actor'] is not None:
        bb = STATE['ego_actor'].bounding_box
        hl, hw = float(bb.extent.x), float(bb.extent.y)
    want = 0.30
    for _ in range(3):
        got = path_min_clearance(path, bodies, hl, hw)
        if got >= want:
            break
        fixed = push_clear(pin, bodies, radius=0.65)
        pin, turn, dev = pin_and_spline(fixed, ego_xy, np.deg2rad(ego_yaw),
                                        sample_m=max(4.0, sample_m*0.6))
        path = GeomPath.build(pin)
    clearance = path_min_clearance(path, bodies, hl, hw)
    if clearance < 0.30:
        k, ss = getattr(path_min_clearance, 'info', (None, None))
        who = '未知'
        if k is not None and k < len(STATE['obstacles']):
            o = STATE['obstacles'][k]
            who = '第 %d 个障碍（%s，位置 %.1f,%.1f）在轨迹 s=%.0f m 处' % (
                k+1, o.get('kind'), o['x'], o['y'], ss or 0.0)
        raise ValueError('绕不开：净空 %.2f m，最近的障碍是%s。\n'
                         '可选：删掉该障碍、把终点挪到障碍前方、换个种子重抽、'
                         '或用「涂可行驶」把通道刷宽' % (clearance, who))
    if STATE['ego_actor'] is not None:
        ego_body = css.body_of(STATE['ego_actor'])
        try:
            check = css.footprint_check(path, ego_body, bodies, road_area,
                                        margin=float(c['safety'].get('road_margin_m', 0.0)))
        except Exception as exc:                     # MultiPolygon 越界碎片会让它抛错
            check = None
            print('footprint_check 异常：%s' % exc, flush=True)
    else:
        check = None

    corridor = (res['viz'] or {}).get('corridor', {}).get('polygons', [])
    STATE['seed'] = int(seed)
    STATE['plan'] = dict(xy=np.asarray(pin, float),
                         corridor=[frame.world(np.asarray(p, float)) for p in corridor],
                         check=check, seed=int(seed))
    note = '种子 %d ｜ 规划 %.0f ms ｜ 走廊 %s 格 ｜ %s ｜ 轨迹 %.0f m ｜ 终点 %.0f m' % (
        seed, ms, (res['corridor'] or {}).get('num_cells'), res['alm_status'],
        path.s[-1], np.linalg.norm(goal_xy-ego_xy))
    if SIDE['mode'] != 'auto':
        note += ' ｜ 绕行侧 %s（候选 %s/%s）' % (
            '右' if SIDE['mode'] == 'right' else '左',
            SIDE['kept'] if SIDE['kept'] is not None else '同侧为空',
            SIDE['total'])
    note += ' ｜ 最小间隙 %.2f m%s' % (clearance, '' if clearance >= 0.30 else '（不足）')
    if check is not None:
        note += ' ｜ 越界 %s m ｜ 校验 %s' % (
            check['max_road_excursion_m'],
            '通过' if check['passed'] else '不通过')
    return note


def status():
    msg = []
    if STATE['ego'] is None:
        msg.append('请点“起点”后在地图上点一下')
    if STATE['goal'] is None:
        msg.append('再点“终点”设目标')
    if STATE['ego'] and STATE['goal']:
        msg.append('终点距离 %.0f m' % np.linalg.norm(
            np.array(STATE['goal'][:2])-np.array(STATE['ego'][:2])))
    if STATE['obstacles']:
        msg.append('障碍 %d（已渲染 %d）' % (len(STATE['obstacles']), len(STATE['obs_actors'])))
    if STATE['ego'] and STATE['ego_actor'] is None:
        msg.append('⚠ 起点处车 spawn 失败（位置被占用），换个点')
    if STATE['plan']:
        msg.append('已生成轨迹')
    return ' ｜ '.join(msg)


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>TrajSafe 场景工具</title>
<style>
body{font-family:system-ui,Segoe UI,sans-serif;margin:0;background:#0f1013;color:#eee}
#bar{padding:8px 12px;background:#1b1c21;display:flex;gap:8px;align-items:center;flex-wrap:wrap;
     position:sticky;top:0;z-index:9;border-bottom:1px solid #2a2c33}
button{background:#33363f;border:1px solid #444a55;color:#e8e8e8;padding:7px 13px;border-radius:6px;
       cursor:pointer;font-size:13px}
button:hover{background:#3d424d}
button.active{background:#2d7ef7;border-color:#2d7ef7;color:#fff}
button.primary{background:#1f9d55;border-color:#1f9d55;color:#fff}
button:disabled{opacity:.45;cursor:default}
#msg{color:#8ecdf7;font-size:12.5px;margin-left:6px}
#wrap{display:flex;gap:10px;padding:10px;align-items:flex-start}
.pane{background:#15171c;border:1px solid #23262d;border-radius:8px;padding:6px;flex:1}
.pane h4{margin:2px 0 6px;font-size:12.5px;color:#8b93a1;font-weight:600}
canvas{width:100%;display:block;border-radius:4px;background:#000;cursor:crosshair}
</style></head><body>
<div id="bar">
  <b style="margin-right:6px">TrajSafe 场景工具</b><span style="color:#6b7280;font-size:11px;margin-right:8px">v__BUILD__</span>
  <select id="maps" style="background:#33363f;color:#e8e8e8;border:1px solid #444a55;
          border-radius:6px;padding:7px 8px;font-size:13px;max-width:150px"></select>
  <button onclick="loadMap(this)">加载地图</button>
  <button onclick="newMap(this)">换个地块</button>
  <button class="tool active" data-mode="ego" onclick="setMode(this)">起点</button>
  <button class="tool" data-mode="goal" onclick="setMode(this)">终点</button>
  <button class="tool" data-mode="car" onclick="setMode(this)">放车辆</button>
  <button class="tool" data-mode="truck" onclick="setMode(this)">放大车</button>
  <button class="tool" data-mode="cone" onclick="setMode(this)">放锥桶</button>
  <button class="tool" data-mode="barrier" onclick="setMode(this)">放施工栏</button>
  <button class="tool" data-mode="del" onclick="setMode(this)">删障碍</button>
  <button onclick="undo()">撤销障碍</button>
  <button onclick="clearObs()">清空障碍</button>
  <button class="primary" id="gen" onclick="run()" disabled>生成轨迹</button>
  <button id="reroll" onclick="reroll()" disabled>换种子重抽</button>
  <select id="side" onchange="STATE_side=this.value" title="绕行侧"
          style="background:#33363f;color:#e8e8e8;border:1px solid #444a55;
          border-radius:6px;padding:7px 8px;font-size:13px">
    <option value="auto">绕行：自动</option>
    <option value="left">绕行：左</option>
    <option value="right">绕行：右</option>
  </select>
  <button id="paintBlock" onclick="setPaint('block',this)">涂不可行驶</button>
  <button id="paintFree" onclick="setPaint('free',this)">涂可行驶</button>
  <select id="brush" style="background:#33363f;color:#e8e8e8;border:1px solid #444a55;
          border-radius:6px;padding:7px 8px;font-size:13px">
    <option value="1.5">笔刷 1.5 m</option>
    <option value="2.5" selected>笔刷 2.5 m</option>
    <option value="4">笔刷 4 m</option>
    <option value="6">笔刷 6 m</option>
  </select>
  <button onclick="paintUndo()">撤销涂抹</button>
  <button onclick="paintClear()">清空涂抹</button>
  <button id="cropbtn" onclick="toggleCrop()">截图</button>
  <button onclick="showCands()">看候选路线</button>
  <select id="cand" style="background:#33363f;color:#e8e8e8;border:1px solid #444a55;
          border-radius:6px;padding:7px 8px;font-size:13px">
    <option value="auto">候选：自动</option>
  </select>
  <button onclick="saveScene()">保存场景</button>
  <select id="scenes" style="background:#33363f;color:#e8e8e8;border:1px solid #444a55;
          border-radius:6px;padding:7px 8px;font-size:13px;max-width:180px"></select>
  <button onclick="loadScene()">载入</button>
  <button onclick="delScene()">删除场景</button>
  <span id="msg">加载中…</span>
</div>
<div id="wrap">
  <div class="pane"><h4>占据栅格（整块地，全部车道自由）</h4><canvas id="occ"></canvas></div>
  <div class="pane"><h4>CARLA 实景（同一块地，同视角）+ 轨迹</h4><canvas id="view"></canvas></div>
</div>
<script>
let addMode='ego', occImg=null, viewImg=null, st={};
const cOcc=document.getElementById('occ'), cView=document.getElementById('view');
const gOcc=cOcc.getContext('2d'), gView=cView.getContext('2d');
function draw(){
  if(occImg){cOcc.width=occImg.width;cOcc.height=occImg.height;gOcc.drawImage(occImg,0,0);}
  if(viewImg){cView.width=viewImg.width;cView.height=viewImg.height;gView.drawImage(viewImg,0,0);}
  if(cropRect){
    const g=(cropPane===cOcc)?gOcc:gView;
    g.save();
    g.strokeStyle='#ff2d55'; g.lineWidth=2;
    g.strokeRect(cropRect.x, cropRect.y, cropRect.side, cropRect.side);
    g.fillStyle='rgba(0,0,0,0.35)';
    g.fillRect(cropRect.x, cropRect.y, cropRect.side, 18);
    g.fillStyle='#fff'; g.font='12px system-ui';
    g.fillText(Math.round(cropRect.side)+'×'+Math.round(cropRect.side),
               cropRect.x+4, cropRect.y+13);
    g.restore();
  }
}
function setImg(id,b64){const im=new Image();im.onload=()=>{if(id==='occ')occImg=im;else viewImg=im;draw();};im.src='data:image/png;base64,'+b64;}
const BUILD='__BUILD__';
function checkBuild(s){ if(s&&s.build&&s.build!==BUILD){
  msg('检测到服务端已更新，正在刷新…'); setTimeout(()=>location.reload(true),400); } }
async function post(u,b){
  const ctl=new AbortController();
  const timer=setTimeout(()=>ctl.abort(), 60000);
  try{
    const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},
                           body:JSON.stringify(b||{}),signal:ctl.signal});
    const txt=await r.text();
    try{ const j=JSON.parse(txt); checkBuild(j); return j; }
    catch(e){ msg('服务器返回异常 HTTP '+r.status+'：'+txt.slice(0,140)); return {}; }
  }catch(e){
    msg('请求失败：'+(e.name==='AbortError'?'服务器 60 s 无响应（可能卡住，请重启服务）':e.message));
    return {};
  }finally{ clearTimeout(timer); }
}
function msg(s){document.getElementById('msg').textContent=s;}
function setMode(btn){
  addMode=btn.dataset.mode;
  if(typeof paintMode!=='undefined' && paintMode){ paintMode=null;
    document.getElementById('paintBlock').style.background='#33363f';
    document.getElementById('paintFree').style.background='#33363f'; }
  document.querySelectorAll('.tool').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  msg('当前：'+btn.textContent+'　点任一张图放置');
}
function apply(s){
  st=Object.assign({},st,s);
  if(s.occ) setImg('occ',s.occ);
  if(s.view) setImg('view',s.view);
  document.getElementById('gen').disabled=!(s.ego&&s.goal);
  document.getElementById('reroll').disabled=!(s.ego&&s.goal);
  msg(s.msg||'');
}
async function reload(){
  const s=await post('/api/state'); st=s; apply(s);
}
async function refreshMaps(){
  const s=await post('/api/maps');
  const sel=document.getElementById('maps');
  if(!sel||!s.maps) return;
  sel.innerHTML=s.maps.map(m=>'<option'+(m===s.cur?' selected':'')+'>'+m+'</option>').join('');
}
async function loadMap(btn){
  const n=document.getElementById('maps').value;
  if(!n) return;
  btn.disabled=true; msg('正在加载 '+n+'（10~40 秒）…');
  apply(await post('/api/load_map',{name:n}));
  btn.disabled=false;
}
async function newMap(btn){
  btn.disabled=true; msg('切换地块中…');
  apply(await post('/api/map'));
  btn.disabled=false;
}
async function undo(){ apply(await post('/api/undo')); }
async function clearObs(){
  if(!confirm('清空所有障碍？')) return;
  apply(await post('/api/clear_obstacles'));
}
async function run(seed){
  const g=document.getElementById('gen'); g.disabled=true; msg('规划中…');
  const body={side:document.getElementById('side').value,
              cand:document.getElementById('cand').value};
  if(seed!==undefined) body.seed=seed;
  apply(await post('/api/plan',body));
  g.disabled=false;
}
async function reroll(){
  // 换个随机种子再抽一次（卡池不同，结果不同）
  await run(Math.floor(Math.random()*100000));
}
function fillScenes(s){
  const sel=document.getElementById('scenes');
  sel.innerHTML=(s.names||[]).map(n=>{
    const i=(s.info&&s.info[n])||{}; const tag=i.n_obs?(' ·'+i.n_obs+'障碍'):'';
    return '<option value="'+n+'">'+n+tag+'</option>';
  }).join('');
}
async function refreshScenes(){ fillScenes(await post('/api/scenes')); }
async function showCands(){
  if(!(st.ego&&st.goal)){ msg('先设起点和终点'); return; }
  msg('算候选路线中…');
  const s=await post('/api/candidates');
  if(s.occ) setImg('occ',s.occ);
  if(s.view) setImg('view',s.view);
  const sel=document.getElementById('cand');
  sel.innerHTML='<option value="auto">候选：自动</option>'+
    Array.from({length:s.n||0},(_,i)=>'<option value="'+(i+1)+'">候选 '+(i+1)+'</option>').join('');
  msg(s.msg||'');
}
async function saveScene(){
  const cur=document.getElementById('scenes').value||'';
  const name=(prompt('场景名字（同名会覆盖）：', cur||('场景'+new Date().toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'})))) ;
  if(!name) return;
  const s=await post('/api/save',{name:name});
  fillScenes(await post('/api/scenes'));
  document.getElementById('scenes').value=name;
  msg(s.msg||'');
}
async function loadScene(){
  const n=document.getElementById('scenes').value;
  if(!n){ msg('先选一个场景'); return; }
  apply(await post('/api/load',{name:n}));
}
async function delScene(){
  const n=document.getElementById('scenes').value;
  if(!n) return;
  if(!confirm('删除场景「'+n+'」？')) return;
  const s=await post('/api/delete_scene',{name:n});
  fillScenes(await post('/api/scenes'));
  msg(s.msg||'');
}
function clickPos(c,ev){const r=c.getBoundingClientRect();
  return {u:(ev.clientX-r.left)/r.width, v:(ev.clientY-r.top)/r.height};}
let paintMode=null, painting=false, strokePts=[];
function setPaint(m,btn){
  paintMode=(paintMode===m)?null:m;
  cropMode=false;
  document.getElementById('paintBlock').style.background = paintMode==='block'?'#c02d2d':'#33363f';
  document.getElementById('paintFree').style.background  = paintMode==='free'?'#1f7a3f':'#33363f';
  msg(paintMode? '涂抹模式：在左图上拖动（'+document.getElementById('brush').value+' m 笔刷）' : '已退出涂抹');
}
function paintUV(c,ev){
  const p=canvasPos(c,ev);
  return [p.x/c.width, p.y/c.height, p];
}
function paintDown(c,ev){
  if(!paintMode) return false;
  painting=true; strokePts=[paintUV(c,ev)]; c.__paintPane=c; return true;
}
function paintMove(c,ev){
  if(!painting||!paintMode) return;
  const t=paintUV(c,ev); strokePts.push(t);
  const g=(c===cOcc)?gOcc:gView;
  const rp=(+document.getElementById('brush').value)/(160/256)*(c.width/256);
  g.save(); g.fillStyle=(paintMode==='block')?'rgba(230,60,60,0.35)':'rgba(120,255,150,0.30)';
  g.beginPath(); g.arc(t[2].x,t[2].y,rp,0,6.2832); g.fill(); g.restore();
}
async function paintUp(){
  if(!painting||!paintMode){ painting=false; return; }
  painting=false;
  const pts=strokePts.map(t=>[t[0],t[1]]);
  strokePts=[];
  if(pts.length<2) return;
  apply(await post('/api/paint',{mode:paintMode, r:+document.getElementById('brush').value, pts:pts}));
}
async function paintUndo(){ apply(await post('/api/paint_undo')); }
async function paintClear(){ if(!confirm('清空所有涂抹？')) return; apply(await post('/api/paint_clear')); }

let cropMode=false, cropPane=null, cropStart=null, cropRect=null;
function toggleCrop(){
  cropMode=!cropMode; cropRect=null; cropStart=null; cropPane=null;
  const b=document.getElementById('cropbtn');
  b.style.background=cropMode?'#c02d2d':'#33363f';
  b.style.borderColor=cropMode?'#c02d2d':'#444a55';
  msg(cropMode?'截图模式：在任意一张图上拖出选框（自动裁成正方形）':'已退出截图模式');
  draw();
}
function canvasPos(c,ev){
  const r=c.getBoundingClientRect();
  return {x:(ev.clientX-r.left)*c.width/r.width, y:(ev.clientY-r.top)*c.height/r.height};
}
function squareRect(a,b,W,H){
  const side=Math.max(Math.abs(b.x-a.x),Math.abs(b.y-a.y),48);
  let cx=(a.x+b.x)/2, cy=(a.y+b.y)/2;
  let x=cx-side/2, y=cy-side/2;
  x=Math.min(Math.max(x,0),Math.max(0,W-side));
  y=Math.min(Math.max(y,0),Math.max(0,H-side));
  return {x:x, y:y, side:Math.min(side,Math.min(W,H))};
}
function saveCrop(pane,rect){
  const src=(pane==='occ')?occImg:viewImg;      // 从原图裁，不带任何界面叠加
  const out=document.createElement('canvas');
  out.width=Math.round(rect.side); out.height=Math.round(rect.side);
  out.getContext('2d').drawImage(src, rect.x, rect.y, rect.side, rect.side,
                                 0, 0, out.width, out.height);
  const a=document.createElement('a');
  const tag=(pane==='occ')?'occupancy':'carla';
  a.download='crop_'+tag+'_'+Date.now()+'.png';
  a.href=out.toDataURL('image/png');
  a.click();
  msg('已下载正方形截图 '+out.width+'×'+out.height+'（'+tag+'）');
}
function cordSel(c,ev){
  if(!cropMode) return;
  cropPane=c; cropStart=canvasPos(c,ev); cropRect=null; draw();
}
function cordMove(ev){
  if(!cropMode||!cropStart||!cropPane) return;
  const p=canvasPos(cropPane,ev);
  cropRect=squareRect(cropStart,p,cropPane.width,cropPane.height);
  draw();
}
function cordUp(){
  if(!cropMode||!cropStart||!cropPane) return;
  const pane=(cropPane===cOcc)?'occ':'view', r=cropRect;
  cropStart=null; cropPane=null; cropRect=null;
  draw();                                        // 先抹掉选框，再按原图裁
  if(r&&r.side>=48) saveCrop(pane, r);
}
window.addEventListener('mousemove',cordMove);
window.addEventListener('mouseup',cordUp);
window.addEventListener('mouseup',paintUp);

async function clicked(canvas,ev){
  if(cropMode||paintMode) return;   // 截图/涂抹模式下不放置任何东西

  const p=clickPos(canvas,ev);
  apply(await post('/api/click',{mode:addMode,u:p.u,v:p.v}));
}
cOcc.addEventListener('mousedown',e=>{ if(cropMode){e.preventDefault(); cordSel(cOcc,e); return;}
  if(paintDown(cOcc,e)){e.preventDefault();} });
cOcc.addEventListener('mousemove',e=>paintMove(cOcc,e));
cView.addEventListener('mousedown',e=>{ if(cropMode){e.preventDefault(); cordSel(cView,e); return;}
  if(paintDown(cView,e)){e.preventDefault();} });
cView.addEventListener('mousemove',e=>paintMove(cView,e));
cOcc.onclick=e=>clicked(cOcc,e);
cView.onclick=e=>clicked(cView,e);
reload();
refreshScenes();
refreshMaps();
</script></body></html>"""


# ---------------------------------------------------------------------- HTTP
def serve(port):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    holder = {'world': carla_world()}
    st = holder['world'].get_settings()
    if st.synchronous_mode:                     # 上次被强杀可能把世界冻住
        st.synchronous_mode = False
        holder['world'].apply_settings(st)
        print('世界被留在同步模式，已解除', flush=True)
    left = 0
    for a in holder['world'].get_actors():
        if str(a.attributes.get('role_name', '')).startswith('dash_'):
            try:
                a.destroy()
                left += 1
            except Exception:
                pass
    if left:
        print('清理上次残留车辆 %d 台' % left, flush=True)
    STATE['map'] = holder['world'].get_map().name
    r = pick_region(holder['world'])
    STATE['region_road'] = r

    def world():
        try:
            return carla_world()
        except Exception:
            return holder['world']

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj):
            obj = dict(obj)
            obj.setdefault('build', BUILD)
            self._send(200, json.dumps(obj).encode('utf8'), 'application/json; charset=utf-8')

        def _body(self):
            n = int(self.headers.get('Content-Length') or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n).decode('utf8'))
            except Exception:
                return {}

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                self._send(200, HTML.replace('__BUILD__', BUILD).encode('utf8'),
                           'text/html; charset=utf-8')
            else:
                self._send(404, b'not found', 'text/plain')

        def do_POST(self):
            w = world()
            try:
                st = w.get_settings()
                if st.synchronous_mode and not LOCK.locked():
                    st.synchronous_mode = False
                    w.apply_settings(st)
                    print('检测到世界卡在同步模式，已解除', flush=True)
            except Exception:
                pass
            p = self._body()
            try:
                if self.path == '/api/set_pose':
                    import carla
                    if p.get('ego'):
                        e = p['ego']
                        wp = w.get_map().get_waypoint(
                            carla.Location(x=float(e[0]), y=float(e[1]), z=1.0),
                            project_to_road=True)
                        STATE['ego'] = [float(e[0]), float(e[1]),
                                         float(wp.transform.rotation.yaw),
                                         float(wp.transform.location.z)]
                    if p.get('goal'):
                        g = p['goal']
                        STATE['goal'] = [float(g[0]), float(g[1]), 0.0]
                    if p.get('obstacles') is not None:
                        out = []
                        for o in p['obstacles']:
                            wp = w.get_map().get_waypoint(
                                carla.Location(x=float(o[0]), y=float(o[1]), z=1.0),
                                project_to_road=True)
                            out.append(dict(x=float(o[0]), y=float(o[1]),
                                            yaw=float(wp.transform.rotation.yaw),
                                            z=float(wp.transform.location.z),
                                            kind=o[2] if len(o) > 2 else 'car'))
                        STATE['obstacles'] = out
                    STATE['plan'] = None
                    respawn(w)
                    return self._json(dict(ok=True, occ=occ_png(w), view=view_png(w),
                                           ego=STATE['ego'], goal=STATE['goal'],
                                           region=STATE['region'], msg=status()))
                if self.path == '/api/maps':
                    import carla
                    cl = carla.Client('127.0.0.1', 2000)
                    cl.set_timeout(20.0)
                    try:
                        names = sorted(cl.get_available_maps())
                    except Exception:
                        names = [STATE['map']]
                    return self._json(dict(ok=True, maps=[n.rsplit('/', 1)[-1] for n in names],
                                           cur=STATE['map'].rsplit('/', 1)[-1]))
                if self.path == '/api/load_map':
                    import carla
                    name = str(p.get('name') or '').strip()
                    if not name:
                        return self._json(dict(ok=False, msg='未指定地图'))
                    cl = carla.Client('127.0.0.1', 2000)
                    cl.set_timeout(60.0)
                    try:
                        cl.load_world(name)
                    except Exception as exc:
                        return self._json(dict(ok=False, msg='加载地图失败：%s' % exc))
                    w2 = cl.get_world()
                    holder['world'] = w2
                    STATE.update(ego=None, goal=None, obstacles=[], plan=None, paint=[], cands=[])
                    STATE['map'] = w2.get_map().name
                    invalidate_caches()
                    pick_region(w2)
                    respawn(w2)
                    return self._json(dict(ok=True, region=STATE['region'],
                                           occ=occ_png(w2), view=view_png(w2),
                                           ego=None, goal=None, map=STATE['map'],
                                           msg='已加载 %s ｜ 点起点开始' % STATE['map'].rsplit('/', 1)[-1]))
                if self.path == '/api/map':
                    STATE.update(ego=None, goal=None, obstacles=[], plan=None, cands=[])
                    pick_region(w)
                    respawn(w)
                    return self._json(dict(ok=True, region=STATE['region'],
                                           occ=occ_png(w), view=view_png(w),
                                           ego=None, goal=None,
                                           msg='新地块 %s ｜ %s' % (
                                               STATE['map'], status())))
                if self.path == '/api/clear_obstacles':
                    STATE['obstacles'] = []
                    STATE['plan'] = None
                    respawn(w)
                    return self._json(dict(ok=True, occ=occ_png(w), view=view_png(w),
                                           ego=STATE['ego'], goal=STATE['goal'],
                                           region=STATE['region'],
                                           msg='已清空障碍 ｜ '+status()))
                if self.path == '/api/undo':
                    if STATE['obstacles']:
                        STATE['obstacles'].pop()
                        STATE['plan'] = None
                        respawn(w)
                    return self._json(dict(ok=True, occ=occ_png(w), view=view_png(w),
                                           ego=STATE['ego'], goal=STATE['goal'],
                                           msg='撤销一个障碍 ｜ '+status()))
                if self.path == '/api/click':
                    import carla
                    mode = p.get('mode', 'ego')
                    x, y = region_world(float(p['u']), float(p['v']))
                    # 位置一律用点的原坐标（不吸附）；只有车头朝向取最近车道方向
                    wp = w.get_map().get_waypoint(
                        carla.Location(x=float(x), y=float(y), z=1.0), project_to_road=True)
                    yaw = float(wp.transform.rotation.yaw)
                    z = float(wp.transform.location.z)
                    if mode == 'goal' and STATE['ego'] is None:
                        return self._json(dict(ok=False, msg='先设起点'))
                    if mode == 'del':
                        if not STATE['obstacles']:
                            return self._json(dict(ok=False, msg='当前没有障碍'))
                        d2 = [np.hypot(o['x']-x, o['y']-y) for o in STATE['obstacles']]
                        k = int(np.argmin(d2))
                        if d2[k] > 4.0:
                            return self._json(dict(ok=False,
                                                   msg='离最近的障碍 %.1f m，点近一点' % d2[k]))
                        STATE['obstacles'].pop(k)
                        STATE['plan'] = None
                        respawn(w)
                        return self._json(dict(ok=True, occ=occ_png(w), view=view_png(w),
                                               ego=STATE['ego'], goal=STATE['goal'],
                                               region=STATE['region'],
                                               msg='已删除第 %d 个障碍 ｜ %s' % (k+1, status())))
                    if mode == 'ego':
                        STATE['ego'] = [float(x), float(y), yaw, z]
                    elif mode == 'goal':
                        STATE['goal'] = [float(x), float(y), z]
                    else:
                        STATE['obstacles'].append(dict(x=float(x), y=float(y),
                                                       yaw=yaw, z=z, kind=mode))
                    STATE['plan'] = None
                    respawn(w)
                    return self._json(dict(ok=True, occ=occ_png(w), view=view_png(w),
                                           ego=STATE['ego'], goal=STATE['goal'],
                                           region=STATE['region'], msg=status()))
                if self.path == '/api/save':
                    name = str(p.get('name') or '').strip()
                    if not name:
                        return self._json(dict(ok=False, msg='场景名为空'))
                    idx = load_index()
                    idx[name] = snapshot()
                    write_index(idx)
                    return self._json(dict(ok=True, names=sorted(idx),
                                           msg='已保存场景「%s」（共 %d 个）' % (name, len(idx))))
                if self.path == '/api/scenes':
                    idx = load_index()
                    return self._json(dict(ok=True, names=sorted(idx),
                                           info={k: dict(n_obs=len(v.get('obstacles') or []),
                                                         saved=v.get('saved'),
                                                         map=v.get('map'))
                                                 for k, v in idx.items()}))
                if self.path == '/api/load':
                    name = str(p.get('name') or '').strip()
                    idx = load_index()
                    if name not in idx:
                        return self._json(dict(ok=False, msg='没有这个场景：%s' % name))
                    snap = idx[name]
                    STATE['region'] = snap.get('region')
                    STATE['ego'] = snap.get('ego')
                    STATE['goal'] = snap.get('goal')
                    STATE['obstacles'] = snap.get('obstacles') or []
                    STATE['seed'] = snap.get('seed', 43)
                    STATE['plan'] = None
                    respawn(w)
                    return self._json(dict(ok=True, occ=occ_png(w), view=view_png(w),
                                           ego=STATE['ego'], goal=STATE['goal'],
                                           region=STATE['region'],
                                           msg='已还原场景「%s」｜ %s' % (name, status())))
                if self.path == '/api/delete_scene':
                    name = str(p.get('name') or '').strip()
                    idx = load_index()
                    if name in idx:
                        idx.pop(name)
                        write_index(idx)
                    return self._json(dict(ok=True, names=sorted(idx),
                                           msg='已删除场景「%s」' % name))
                if self.path == '/api/paint':
                    mode = 'block' if p.get('mode') == 'block' else 'free'
                    r = float(p.get('r', 2.0))
                    pts = []
                    for uv in (p.get('pts') or []):
                        x, y = region_world(float(uv[0]), float(uv[1]))
                        pts.append([x, y])
                    if pts:
                        STATE['paint'].append(dict(mode=mode, r=r, pts=pts))
                        STATE['plan'] = None
                    return self._json(dict(ok=True, occ=occ_png(w), view=view_png(w),
                                           ego=STATE['ego'], goal=STATE['goal'],
                                           region=STATE['region'],
                                           msg='涂抹 %d 笔（%s）' % (
                                               len(STATE['paint']),
                                               '不可行驶' if mode == 'block' else '可行驶')))
                if self.path == '/api/paint_undo':
                    if STATE['paint']:
                        STATE['paint'].pop()
                        STATE['plan'] = None
                    return self._json(dict(ok=True, occ=occ_png(w), view=view_png(w),
                                           ego=STATE['ego'], goal=STATE['goal'],
                                           region=STATE['region'],
                                           msg='撤销一笔，剩 %d 笔' % len(STATE['paint'])))
                if self.path == '/api/paint_clear':
                    STATE['paint'] = []
                    STATE['plan'] = None
                    return self._json(dict(ok=True, occ=occ_png(w), view=view_png(w),
                                           ego=STATE['ego'], goal=STATE['goal'],
                                           region=STATE['region'], msg='已清空涂抹'))
                if self.path == '/api/candidates':
                    STATE['plan'] = None
                    if not (STATE['ego'] and STATE['goal']):
                        return self._json(dict(ok=False, msg='需要先设起点和终点'))
                    paths = candidate_paths(w)
                    return self._json(dict(ok=True, n=len(paths), occ=occ_png(w),
                                           view=view_png(w),
                                           msg='候选路线 %d 条（左图彩色线，数字为序号）' % len(paths)))
                if self.path == '/api/state':
                    return self._json(dict(ok=True, occ=occ_png(w), view=view_png(w),
                                           ego=STATE['ego'], goal=STATE['goal'],
                                           region=STATE['region'], msg=status()))
                if self.path == '/api/plan':
                    STATE['plan'] = None
                    if not (STATE['ego'] and STATE['goal']):
                        return self._json(dict(ok=False, occ=occ_png(w), view=view_png(w),
                                               ego=STATE['ego'], goal=STATE['goal'],
                                               msg='需要先设起点和终点'))
                    if p.get('side') in ('auto', 'left', 'right'):
                        STATE['side'] = p['side']
                    if 'cand' in p:
                        v = p['cand']
                        FORCE['idx'] = None if v in (None, '', -1, 'auto') else int(v)
                    seed = p.get('seed')
                    if seed is None:
                        seed = STATE['seed']
                    try:
                        note = plan(w, int(seed))
                    except Exception as exc:
                        STATE['plan'] = None
                        return self._json(dict(ok=False, occ=occ_png(w), view=view_png(w),
                                               ego=STATE['ego'], goal=STATE['goal'],
                                               msg='规划失败：%s' % exc))
                    return self._json(dict(ok=True, occ=occ_png(w), view=view_png(w),
                                           ego=STATE['ego'], goal=STATE['goal'],
                                           region=STATE['region'], msg=note))
                self._send(404, b'not found', 'text/plain')
            except Exception as exc:
                self._json(dict(ok=False, msg='%s: %s' % (type(exc).__name__, exc)))

    print('打开 http://127.0.0.1:%d' % port, flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8815)
    serve(ap.parse_args().port)
