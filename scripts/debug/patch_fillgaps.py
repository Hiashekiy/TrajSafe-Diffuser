"""自动填补路口里的小空洞（被路面包围、面积小的非自由块）。"""
from pathlib import Path

p = Path('scripts/dashboard_app.py')
s = p.read_text(encoding='utf8')

if 'def fill_small_gaps' not in s:
    s = s.replace("def apply_paint(mask, frame):",
'''def fill_small_gaps(occ, max_cells=260):
    """把被路面包围、面积很小的空洞填成可行驶。

    车道中线栅格在路口拐角会漏掉几格空地（CARLA 实景是路面），
    这里按连通块面积区分：小于 max_cells 格且不接触画面边界的空洞 → 填平；
    建筑街区那种大块区域（几千格）不会被误填。
    """
    from scipy.ndimage import label
    blocked = (occ == 1).astype(np.uint8)
    lab, n = label(blocked, structure=np.ones((3, 3)))
    if n == 0:
        return occ
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    sizes = np.bincount(lab.ravel())
    out = occ.copy()
    for k in range(1, n+1):
        if k in border:
            continue
        if sizes[k] <= max_cells:
            out[lab == k] = 0
    return out


def apply_paint(mask, frame):''', 1)

s = s.replace("""    return apply_paint(lane_raster(world, frame, None), frame)""",
              """    return apply_paint(fill_small_gaps(lane_raster(world, frame, None)), frame)""", 1)

p.write_text(s, encoding='utf8')
print('patched ok')
