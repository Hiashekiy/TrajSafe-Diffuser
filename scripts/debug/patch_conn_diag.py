"""失败时给出连通性诊断：自由区分几块、起终点各在哪块。"""
from pathlib import Path

p = Path('scripts/dashboard_app.py')
s = p.read_text(encoding='utf8')

if 'def free_components' not in s:
    s = s.replace("def plan(world, seed):",
'''def free_components(frame, occ, ego_xy, goal_xy):
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


def plan(world, seed):''', 1)

old = """    adapter = PlannerAdapter(c['planner'])
    t0 = time.perf_counter()"""
new = """    ncomp, ca, sa, cb, sb, same = free_components(frame, occ, ego_xy, goal_xy)
    if not same:
        raise ValueError(
            '起点和终点不在同一块可行驶区域（自由空间共 %d 块：起点在第 %d 块 %d 格，'
            '终点在第 %d 块 %d 格）。说明涂抹/障碍把通道封死了，用「涂可行驶」打通一段再试。'
            % (ncomp, ca, sa, cb, sb))

    adapter = PlannerAdapter(c['planner'])
    t0 = time.perf_counter()"""
assert old in s
s = s.replace(old, new, 1)

p.write_text(s, encoding='utf8')
print('patched ok')
