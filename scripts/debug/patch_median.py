"""中线隔离带：只在相邻反向车道之间画墙，对向车道本身仍可通行（可左拐）。"""
from pathlib import Path

p = Path('scripts/dashboard_app.py')
s = p.read_text(encoding='utf8')

# 1) scene_direction 恢复为不过滤
a = s.index('def scene_direction(world):')
b = s.index('def build_scene(world):')
s = s[:a] + '''def scene_direction(world):
    """不再按方向过滤车道：所有车道都算自由空间。

    逆行由 median_lines() 画出的中线墙来阻止（只挡相邻反向车道之间，
    所以路口左拐/掉头仍然可以通过），而不是把整条对向车道判成障碍。
    """
    return None


''' + s[b:]

# 2) 中线隔离带 + 统一的场景栅格
if 'def median_lines' not in s:
    s = s.replace("def build_scene(world):",
'''MEDIAN = {'map': None, 'sign': None, 'lines': None}


def median_lines(world, sign):
    """相邻反向车道之间的中线折线（世界坐标），按地图+方向缓存。"""
    if MEDIAN['map'] == world.get_map().name and MEDIAN['sign'] == sign \\
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


def scene_raster(world, frame):
    """模型的占据图 = 全部车道 + 中线隔离墙（0=可行驶）。"""
    import cv2
    occ = lane_raster(world, frame, None)
    sign = ego_lane_sign(world)
    if sign is not None:
        cell = frame.size/frame.resolution
        for line in median_lines(world, sign):
            g = np.rint(frame.grid(line)*(1 << 4)).astype(np.int32)
            cv2.polylines(occ, [g], False, 1, max(1, int(round(0.7/cell))),
                          cv2.LINE_8, shift=4)
    return occ


def build_scene(world):''', 1)

# 3) build_scene / occ_png 都用 scene_raster
s = s.replace("""    road = lane_raster(world, frame, direction)       # 0=可行驶（与左图逐像素一致）""",
              """    road = scene_raster(world, frame)                 # 0=可行驶（与左图逐像素一致）""", 1)
s = s.replace("""    occ = lane_raster(world, frame, scene_direction(world))""",
              """    occ = scene_raster(world, frame)""", 1)

p.write_text(s, encoding='utf8')
print('patched ok')
