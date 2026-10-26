"""S1 可视化调优：换地点 / 缩短视距 / 走廊逐格描边 / HUD 改英文。"""
from pathlib import Path

root = Path(__file__).resolve().parents[2]

# ---------- 1. 场景：更近的终点、更细的走廊格子、换 spawn 66 ----------
cfg = root/'configs/scenarios/S1_lane_block.yaml'
t = cfg.read_text(encoding='utf8')
t = t.replace('  spawn_index: 69', '  spawn_index: 66')
t = t.replace('  distance_m: 55.0          # 沿车道的目标点距离', '  distance_m: 38.0          # 沿车道的目标点距离（缩短：视野 50 m 内看清车身）')
t = t.replace('  region_stride: 8', '  region_stride: 1          # 128 个小区，走廊才画得清（演示用，牺牲约 0.3 s）')
cfg.write_text(t, encoding='utf8')

# ---------- 2. runner：俯视高度 26 m + 走廊逐格描边 + HUD 英文 ----------
g = root/'scripts/carla_scenario_sim.py'
s = g.read_text(encoding='utf8')
s = s.replace('cam.set_transform(topdown(path.pose(float(s_i))[0], 34.0))',
              'cam.set_transform(topdown(path.pose(float(s_i))[0], 26.0))')
s = s.replace('cam.set_transform(topdown(centre_world, 52.0))',
              'cam.set_transform(topdown(centre_world, 26.0))')

# 走廊：交替色 + 逐格描边（原来只有单色 0.16 填充，重叠后看不见）
old_corr = '''            for poly in corridor:                               # 走廊：淡青绿
                pts = project(frame.world(np.asarray(poly)))
                if pts is None:
                    continue
                layer = img.copy()
                cv2.fillPoly(layer, [pts], (170, 225, 190))
                cv2.addWeighted(layer, 0.16, img, 0.84, 0, img)'''
new_corr = '''            for i, poly in enumerate(corridor):                 # 走廊：逐格交替色 + 描边
                pts = project(frame.world(np.asarray(poly)))
                if pts is None:
                    continue
                fill = (150, 210, 255) if i % 2 else (120, 235, 215)
                layer = img.copy()
                cv2.fillPoly(layer, [pts], fill)
                cv2.addWeighted(layer, 0.22, img, 0.78, 0, img)
                cv2.polylines(img, [pts], True, (60, 150, 220), 1, cv2.LINE_AA)'''
assert old_corr in s
s = s.replace(old_corr, new_corr)

s = s.replace('''        text = ("%s %s   plan %.0f ms   corridor %s cells   min clearance %s m   footprint %s"
                % (cfg['id'], cfg['name'], plan_ms, summary['corridor_cells'],''',
'''        text = ("%s   plan %.0f ms   corridor %s cells   clearance %s m   footprint %s"
                % (cfg['id'], plan_ms, summary['corridor_cells'],''')
g.write_text(s, encoding='utf8')
print('runner: 26 m 俯视 + 走廊逐格描边 + HUD 英文')
