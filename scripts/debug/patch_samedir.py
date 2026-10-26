"""重新启用同向车道分组（中线变墙，不窜到对面车道）。"""
from pathlib import Path

p = Path('scripts/dashboard_app.py')
s = p.read_text(encoding='utf8')

old = """def scene_direction(world):
    \"\"\"暂时不做方向过滤：所有车道（含对向）都算自由空间。

    保留函数是为了以后要按起终点方向收窄时改这里即可。
    \"\"\"
    return None"""
new = """def scene_direction(world):
    \"\"\"车道方向分组：起终点同向时，只保留同向车道 + 路口内短车道。

    反向的长车道判为不可行驶，等于把中线变成墙，轨迹不会窜到对面车道。
    终点落在对向车道（掉头）时不过滤，保证能掉过去。
    \"\"\"
    se = ego_lane_sign(world)
    sg = ego_lane_sign(world, STATE['goal']) if STATE['goal'] else None
    if se is None or sg is None or se != sg:
        return None
    return se"""
assert old in s, 'anchor missing'
s = s.replace(old, new, 1)
p.write_text(s, encoding='utf8')
print('patched ok')
