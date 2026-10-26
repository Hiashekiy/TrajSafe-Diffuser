"""论文级可视化：走廊收窄成管、3 车道、障碍贴路缘、干净构图。"""
from pathlib import Path

root = Path(__file__).resolve().parents[2]

# ---------- adapter：支持 corridor 椭圆最大半轴 ----------
a = root/'src/carla_bridge/planner_adapter.py'
t = a.read_text(encoding='utf8')
old = '''        if self.cfg.get('region_stride') is not None:'''
new = '''        if self.cfg.get('corridor_axis_max_scene') is not None:
            region = dict(corridor.get('region') or {})
            region['ellipse_axis_max'] = float(self.cfg['corridor_axis_max_scene'])
            corridor['region'] = region
        if self.cfg.get('region_stride') is not None:'''
assert old in t
a.write_text(t.replace(old, new, 1), encoding='utf8')

# ---------- runner：车道数改为“每侧 N 条” ----------
g = root/'scripts/carla_scenario_sim.py'
s = g.read_text(encoding='utf8')
old_lanes = '''        if cfg['grid'].get('all_lanes'):
            for side in (+1, -1):
                for k in (1, 2, 3):'''
new_lanes = '''        each_side = int(cfg['grid'].get('lanes_each_side', 1))
        if cfg['grid'].get('all_lanes') or each_side > 0:
            for side in (+1, -1):
                for k in range(1, (4 if cfg['grid'].get('all_lanes') else each_side+1)):'''
assert old_lanes in s
s = s.replace(old_lanes, new_lanes)
g.write_text(s, encoding='utf8')

# ---------- 配置 ----------
c = root/'configs/scenarios/S1_lane_block.yaml'
c = c.read_text(encoding='utf8')
c = c.replace('  all_lanes: true      # 占据图纳入同向所有可行驶车道（不只是左邻一条）',
              '  lanes_each_side: 1   # 自车车道 + 左右各 1 条（与训练分布一致，走廊才有管子形状）')
c = c.replace('  corridor_margin_m: 1.0   # 走廊构建时每侧留 1.0 m（替代障碍膨胀）',
              '  corridor_margin_m: 1.0    # 走廊构建时每侧留 1.0 m（替代障碍膨胀）\n'
              '  corridor_axis_max_scene: 0.06   # 椭圆最大半轴 0.06×80 = 4.8 m，走廊收成管子')
c = c.replace('    lateral_m: 1.3', '    lateral_m: 1.9')
c = c.replace('  distance_m: 38.0', '  distance_m: 38.0')
c.write_text(c, encoding='utf8')
print('adapter/runner/config 已改')
