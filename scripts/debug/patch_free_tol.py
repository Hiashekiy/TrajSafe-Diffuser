"""起终点自由区判定加 1 格容差（点仍不移动）。"""
from pathlib import Path

p = Path('scripts/dashboard_app.py')
s = p.read_text(encoding='utf8')

if 'def free_tol' not in s:
    s = s.replace("def plan(world, seed):",
"""def free_tol(frame, occ, xy, k=1):
    \"\"\"点所在格（含 ±k 邻域）是否有自由格。只放宽判定，不移动点。\"\"\"
    g = frame.grid(np.asarray(xy, float).reshape(1, 2))[0]
    res = occ.shape[0]
    c0, r0 = int(np.floor(g[0])), int(np.floor(g[1]))
    for r in range(max(0, r0-k), min(res, r0+k+1)):
        for c in range(max(0, c0-k), min(res, c0+k+1)):
            if occ[r, c] == 0:
                return True
    return False


def plan(world, seed):""", 1)

s = s.replace("""    for name, pt in (('起点', ego_xy), ('终点', goal_xy)):
        if not frame.free(occ, pt.reshape(1, 2))[0]:
            raise ValueError('%s不在可行驶区域（点在路面上再试）' % name)""",
"""    for name, pt in (('起点', ego_xy), ('终点', goal_xy)):
        if not free_tol(frame, occ, pt, 1):
            raise ValueError('%s不在可行驶区域（点在路面上再试）' % name)""", 1)

p.write_text(s, encoding='utf8')
print('patched ok')
