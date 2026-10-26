"""逐点纳入相邻车道（不再要求整条链都连续）。"""
from pathlib import Path

g = Path(__file__).resolve().parents[2]/'scripts/carla_scenario_sim.py'
s = g.read_text(encoding='utf8')

old = '''        def side_chain(side, k):
            """链上每个 waypoint 向同一侧数第 k 条可行驶车道；任一点取不到就整条放弃。"""
            out = []
            for w in chain:
                x = w
                for _ in range(k):
                    x = x.get_left_lane() if side > 0 else x.get_right_lane()
                    if x is None or x.lane_type.name != 'Driving':
                        return None
                out.append(x)
            return out if len(out) >= 2 else None'''

new = '''        def side_chain(side, k):
            """向同侧数第 k 条可行驶车道；逐点纳入，缺的点跳过（不再整条丢弃）。"""
            out = []
            for w in chain:
                x = w
                for _ in range(k):
                    x = x.get_left_lane() if side > 0 else x.get_right_lane()
                    if x is None or x.lane_type.name != 'Driving':
                        x = None
                        break
                if x is not None:
                    out.append(x)
            return out if len(out) >= 2 else None'''

assert old in s
s = s.replace(old, new)

# 逐点纳入后相邻点可能不相邻，road_quads 会画长斜条；限制相邻点距离
old2 = '''                    lane = side_chain(side, k)
                    if lane is None:
                        break
                    quads = quads + road_quads(lane)
                    lanes_included += 1'''
new2 = '''                    lane = side_chain(side, k)
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
                        lanes_included += 1'''
assert old2 in s
s = s.replace(old2, new2)
g.write_text(s, encoding='utf8')
print('车道改为逐点纳入 + 分段（相邻点超过 6 m 就断开）')
