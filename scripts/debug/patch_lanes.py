"""把占据图的车道范围从「自车+左邻一条」改成「同向所有可行驶车道」。"""
from pathlib import Path

g = Path(__file__).resolve().parents[2]/'scripts/carla_scenario_sim.py'
s = g.read_text(encoding='utf8')

old = '''        quads = road_quads(chain)
        if cfg['grid'].get('left_lane'):
            left = [w.get_left_lane() for w in chain]
            left = [w for w in left if w is not None]
            if len(left) >= 2:
                quads += road_quads(left)'''

new = '''        quads = road_quads(chain)
        lanes_included = 1

        def side_chain(side, k):
            """链上每个 waypoint 向同一侧数第 k 条可行驶车道；任一点取不到就整条放弃。"""
            out = []
            for w in chain:
                x = w
                for _ in range(k):
                    x = x.get_left_lane() if side > 0 else x.get_right_lane()
                    if x is None or x.lane_type.name != 'Driving':
                        return None
                out.append(x)
            return out if len(out) >= 2 else None

        if cfg['grid'].get('all_lanes'):
            for side in (+1, -1):
                for k in (1, 2, 3):
                    lane = side_chain(side, k)
                    if lane is None:
                        break
                    quads = quads + road_quads(lane)
                    lanes_included += 1
        elif cfg['grid'].get('left_lane'):
            lane = side_chain(+1, 1)
            if lane is not None:
                quads = quads + road_quads(lane)
                lanes_included += 1
        summary['lanes_in_grid'] = lanes_included'''

assert old in s, 'lane block not found'
s = s.replace(old, new)
g.write_text(s, encoding='utf8')
print('占据图改为可配置的多车道')
