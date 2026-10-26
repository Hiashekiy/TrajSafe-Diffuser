"""把 relax_anchors 换成“纯最近锚点”，并追加 trim_to_ego。"""
from pathlib import Path

p = Path('scripts/dashboard_app.py')
s = p.read_text(encoding='utf8')
a = s.index('def relax_anchors():')
b = s.index('def snap_free(')
new = '''def relax_anchors():
    """骨架锚点：直接取最近的骨架节点（不分方向）。

    项目自带 src/carla_bridge/skeleton_query.py::QueryGraph.visible_anchors 要求锚点
    严格落在起点正前方 / 终点正后方 3-18 m、车头 +-35 度锥内，骨架一旦分叉或横向就
    一个锚点都取不到（实测 spawn 12/50/60 锚点 0 个 -> 算不出候选通道）。

    车后方那段无用骨架在拿到路径后由 trim_to_ego() 截掉，所以这里不需要方向门控。
    """
    if ANCHOR['relaxed']:
        return
    import src.carla_bridge.skeleton_query as sq
    from src.geometry.skeleton_graph import SkeletonGraph
    sq.QueryGraph.visible_anchors = (
        lambda self, point_scene, max_candidates=16:
        SkeletonGraph.visible_anchors(self, point_scene, max_candidates))
    ANCHOR['relaxed'] = True
    print('骨架锚点：取最近节点（车后段由 trim_to_ego 截掉）', flush=True)


def trim_to_ego(path_xy, ego_xy, step=0.5):
    """把车投影到规划出的骨架上：截掉车后方那段，再把车本身接成起点。"""
    from src.carla_bridge.path_profile import Path as GeomPath
    geom = GeomPath.build(np.asarray(path_xy, float))
    ego = np.asarray(ego_xy, float)
    if geom.s[-1] < 2.0:
        return np.asarray(path_xy, float)
    grid = np.arange(0.0, geom.s[-1], step)
    pts = np.array([geom.pose(float(s))[0] for s in grid])
    s0 = float(grid[int(np.argmin(np.linalg.norm(pts-ego, axis=1)))])
    keep = np.array([geom.pose(float(s))[0] for s in np.arange(s0, geom.s[-1], step)])
    if len(keep) < 3:
        return np.asarray(path_xy, float)
    return np.vstack([ego[None, :], keep])


'''
p.write_text(s[:a]+new+s[b:], encoding='utf8')
print('patched', len(new))
