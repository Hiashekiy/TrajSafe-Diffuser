"""左图按来源上色：路=白、你涂的不可行驶=蓝、你涂的可行驶=绿、障碍=红、其它=深灰。"""
from pathlib import Path

p = Path('scripts/dashboard_app.py')
s = p.read_text(encoding='utf8')

old = """    frame = region_frame()
    occ = scene_raster(world, frame)
    img = np.full((RES, RES, 3), 96, np.uint8)
    img[occ == 0] = (232, 232, 232)"""
new = """    frame = region_frame()
    road = lane_raster(world, frame, None)            # 车道（白）
    occ = apply_paint(road.copy(), frame)             # 叠加涂抹后的模型输入
    img = np.full((RES, RES, 3), 96, np.uint8)        # 非道路（深灰）
    img[road == 0] = (232, 232, 232)
    img[(road == 0) & (occ == 1)] = (200, 140, 60)    # 你涂的不可行驶（蓝）
    img[(road == 1) & (occ == 0)] = (90, 190, 90)     # 你涂的可行驶（绿）"""
assert old in s
s = s.replace(old, new, 1)

p.write_text(s, encoding='utf8')
print('patched ok')
