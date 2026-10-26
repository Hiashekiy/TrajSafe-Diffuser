"""无条件按自车方向分组（长对向车道永远不可行驶）。"""
from pathlib import Path

p = Path('scripts/dashboard_app.py')
s = p.read_text(encoding='utf8')

a = s.index('def scene_direction(world):')
b = s.index('def build_scene(world):')
new = '''def scene_direction(world):
    """无条件按自车所在车道的方向分组：对向的长车道一律不可行驶。

    不再依据终点放松（那样会让整条对向车道重新变自由，车就一路开过去）。
    路口内不足 20 m 的短车道仍然保留，十字路口左转/掉头不受影响。
    """
    return ego_lane_sign(world)


'''
s = s[:a] + new + s[b:]
p.write_text(s, encoding='utf8')
print('patched ok')
