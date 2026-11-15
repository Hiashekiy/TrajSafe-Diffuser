"""Hard synthetic scenes: mazes, room grids, spirals, combs, decoys.

The CARLA crop is a road network: a handful of 2.5-4 m wide corridors, so the
grid search is almost 1-D and A* expands a few hundred cells.  These scenes are
built to stress the SEARCH instead:

    maze_*      braided random maze (loops -> several routes of similar length)
    rooms_*     room grid with 2-cell doors (many alternative routes)
    spiral      one long corridor: the goal is behind a forced detour
    comb        a trunk with dead-end branches (deceptive local progress)
    long_wall   a long wall with a gap at the far end (octile misleads)

For every scene four KINDS of reference are generated, so the guidance is tested
against the quality of the diffusion prior:

    optimal   A* path resampled            ("diffusion got it right")
    detour    start -> far dead cell -> goal ("diffusion wandered")
    straight  straight line (always crosses walls, "diffusion ignored the map")
    decoy     smooth Bezier through the wrong channel
"""

from __future__ import annotations

import numpy as np

from .grid_map import GridSpec, build_safe_grid
from .scenes import Scene

__all__ = ["hard_scenes", "reference_for", "REFERENCE_KINDS"]

REFERENCE_KINDS = ("optimal", "detour", "straight", "decoy")


# ---------------------------------------------------------------------------
# map generators
# ---------------------------------------------------------------------------
def maze_grid(seed=0, cells=24, corridor=3, braid=0.25):
    """Recursive-backtracker maze; ``braid`` knocks extra walls out (loops)."""
    rng = np.random.default_rng(seed)
    step = corridor + 1
    W = cells * step + 1
    occ = np.ones((W, W), dtype=np.uint8)

    def blk(i, j):
        r = 1 + i * step
        c = 1 + j * step
        return slice(r, r + corridor), slice(c, c + corridor)

    def wall(i, j, di, dj):
        """One-cell-thick wall between block (i,j) and block (i+di,j+dj)."""
        if di != 0:
            r0 = 1 + min(i, i + di) * step + corridor
            c0 = 1 + j * step
            return slice(r0, r0 + 1), slice(c0, c0 + corridor)
        r0 = 1 + i * step
        c0 = 1 + min(j, j + dj) * step + corridor
        return slice(r0, r0 + corridor), slice(c0, c0 + 1)

    seen = np.zeros((cells, cells), dtype=bool)
    stack = [(0, 0)]
    seen[0, 0] = True
    while stack:
        i, j = stack[-1]
        occ[blk(i, j)] = 0
        nbrs = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
        nbrs = [(a, b) for a, b in nbrs
                if 0 <= a < cells and 0 <= b < cells and not seen[a, b]]
        if not nbrs:
            stack.pop()
            continue
        a, b = nbrs[rng.integers(len(nbrs))]
        occ[wall(i, j, a - i, b - j)] = 0
        seen[a, b] = True
        stack.append((a, b))

    if braid > 0:                          # open extra walls -> loops
        for i in range(cells):
            for j in range(cells):
                for di, dj in ((1, 0), (0, 1)):
                    a, b = i + di, j + dj
                    if a >= cells or b >= cells:
                        continue
                    if rng.random() < braid:
                        occ[wall(i, j, di, dj)] = 0
    start = (1 + 0 * step + corridor // 2, 1 + 0 * step + corridor // 2)
    goal = (1 + (cells - 1) * step + corridor // 2,
            1 + (cells - 1) * step + corridor // 2)
    return occ, start, goal


def rooms_grid(seed=0, rooms=3, size=30, corridor=3, doors=2):
    """Grid of rooms separated by walls with ``doors`` openings per shared wall."""
    rng = np.random.default_rng(seed)
    W = rooms * size + 1
    occ = np.zeros((W, W), dtype=np.uint8)
    t = corridor                                 # wall thickness
    for k in range(rooms + 1):
        pos = min(k * size, W - t)
        occ[pos:pos + t, :] = 1
        occ[:, pos:pos + t] = 1
    for i in range(rooms):
        for j in range(rooms):
            # right wall
            if j + 1 < rooms:
                x = min((j + 1) * size, W - t)
                for _ in range(doors):
                    r = int(rng.integers(i * size + 2, (i + 1) * size - 2))
                    occ[r:r + corridor, x:x + t] = 0
            if i + 1 < rooms:
                y = min((i + 1) * size, W - t)
                for _ in range(doors):
                    c = int(rng.integers(j * size + 2, (j + 1) * size - 2))
                    occ[y:y + t, c:c + corridor] = 0
    r0, c0 = 3, 3
    r1, c1 = W - 4, W - 4
    occ[r0:r0 + corridor, c0:c0 + corridor] = 0
    occ[r1 - corridor:r1, c1 - corridor:c1] = 0
    return occ, (r0 + corridor // 2, c0 + corridor // 2), \
        (r1 - corridor // 2, c1 - corridor // 2)


def spiral_grid(size=81, corridor=3, turns=4):
    """One long spiral corridor: the goal sits at the centre of a forced detour.

    The corridor is rasterised along an explicit polyline, so connectivity is
    guaranteed by construction (an earlier ring-by-ring version produced
    disconnected rings -- solved here, not patched with a gap).
    """
    occ = np.ones((size, size), dtype=np.uint8)
    lo, hi = 2, size - 3
    pts = []
    while lo < hi:
        pts.append(((lo, lo), (lo, hi)))
        pts.append(((lo, hi), (hi, hi)))
        pts.append(((hi, hi), (hi, lo)))
        pts.append(((hi, lo), (lo + corridor + 1, lo)))
        lo += corridor + 1
        hi -= corridor + 1
    half = max(1, corridor // 2)
    for (r0, c0), (r1, c1) in pts:
        r_lo, r_hi = min(r0, r1) - half, max(r0, r1) + half + 1
        c_lo, c_hi = min(c0, c1) - half, max(c0, c1) + half + 1
        occ[max(r_lo, 0):min(r_hi, size), max(c_lo, 0):min(c_hi, size)] = 0
    occ[:, 0] = 1
    occ[:, -1] = 1
    occ[0, :] = 1
    occ[-1, :] = 1
    start = (2 + half, 2 + half)
    occ[start] = 0
    goal = _farthest_reachable(occ, start)
    return occ, start, goal


def comb_grid(rows=6, length=30, corridor=3, gap=5):
    """A trunk corridor with ``rows`` dead-end branches (deceptive progress)."""
    H = (rows + 1) * (corridor + gap) + 1
    W = length + 1
    occ = np.ones((H, W), dtype=np.uint8)
    trunk = H // 2
    occ[trunk:trunk + corridor, :] = 0
    for i in range(rows):
        if i % 2 == 0:                      # above the trunk
            r = trunk - (i // 2 + 1) * (corridor + gap)
        else:                               # below the trunk
            r = trunk + (i // 2 + 1) * (corridor + gap)
        if r < 1 or r + corridor >= H:
            continue
        occ[r:r + corridor, 2:length - 4] = 0
        occ[min(r, trunk):max(r + corridor, trunk + corridor),
            2:2 + corridor] = 0             # link to the trunk
    # start/goal at the tips of two DIFFERENT dead-end branches
    start = (max(1, trunk - (0 // 2 + 1) * (corridor + gap)) + corridor // 2,
             4)
    goal = (min(H - 2, trunk + (1 // 2 + 1) * (corridor + gap)) + corridor // 2,
            length - 6)
    occ[start] = 0
    occ[goal] = 0
    return occ, start, goal


def long_wall_grid(size=81, gap=6):
    """A long wall with one gap at the far end (the octile heuristic misleads)."""
    occ = np.zeros((size, size), dtype=np.uint8)
    occ[:, size // 2] = 1                     # full-height wall
    occ[size - gap:, size // 2] = 0           # gap at the bottom
    occ[:, size // 2 + 1] = 1
    occ[size - gap:, size // 2 + 1] = 0
    start = (6, 4)
    goal = (6, size - 5)
    return occ, start, goal


def _farthest_reachable(occ, start_rc):
    """4-connected BFS: the reachable free cell with the largest step count.

    Used to place the goal so that every hard scene is solvable BY CONSTRUCTION
    (an isolated centre cell made the first spiral version return NO_PATH).
    """
    from collections import deque

    h, w = occ.shape
    dist = -np.ones((h, w), dtype=np.int32)
    sr, sc = int(start_rc[0]), int(start_rc[1])
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    best, best_d = (sr, sc), 0
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if occ[nr, nc] != 0 or dist[nr, nc] >= 0:
                continue
            dist[nr, nc] = dist[r, c] + 1
            if dist[nr, nc] > best_d:
                best_d, best = dist[nr, nc], (nr, nc)
            q.append((nr, nc))
    return best


# ---------------------------------------------------------------------------
# scene assembly
# ---------------------------------------------------------------------------
def _scene(ident, name, occ, start_rc, goal_rc, cell_m=1.0) -> Scene:
    h, w = occ.shape
    spec = GridSpec(h, w, cell_m, cell_m * max(h, w) / 2.0)
    return Scene(ident, name, occ, spec, tuple(int(v) for v in start_rc),
                 tuple(int(v) for v in goal_rc), None, "hard scene")


def _fix_goal(occ, start, goal):
    if occ[goal] != 0 or _farthest_reachable(occ, start) == goal:
        return goal
    return _farthest_reachable(occ, start)


def hard_scenes(cell_m=1.0):
    """The hard suite (6 scenes)."""
    out = []
    out.append(_scene("H1", "maze_a", *maze_grid(seed=1, cells=24, corridor=3,
                                                 braid=0.22), cell_m=cell_m))
    out.append(_scene("H2", "maze_b", *maze_grid(seed=7, cells=26, corridor=3,
                                                 braid=0.10), cell_m=cell_m))
    out.append(_scene("H3", "rooms", *rooms_grid(seed=3, rooms=3, size=26,
                                                 corridor=3, doors=2),
                      cell_m=cell_m))
    out.append(_scene("H4", "spiral", *spiral_grid(size=81, corridor=3, turns=4),
                      cell_m=cell_m))
    out.append(_scene("H5", "comb", *comb_grid(rows=6, length=30, corridor=3),
                      cell_m=cell_m))
    out.append(_scene("H6", "long_wall", *long_wall_grid(size=81, gap=8),
                      cell_m=cell_m))
    return out


# ---------------------------------------------------------------------------
# references of different quality
# ---------------------------------------------------------------------------
def reference_for(scene: Scene, kind: str, rng=None) -> np.ndarray:
    """Build a reference polyline (scene coords) of the requested quality."""
    rng = rng if rng is not None else np.random.default_rng(0)
    spec = scene.spec
    safe = build_safe_grid(scene.occupancy, spec, robot_radius_m=0.0,
                           safety_margin_m=0.0)
    from .baselines import astar
    from .grid_neighbors import build_neighbor_table
    nbr = build_neighbor_table(safe.free, spec.cell_m)
    s, g = scene.start_idx, scene.goal_idx

    def rc(idx):
        return spec.index_to_rc(np.asarray(idx, dtype=np.int64))

    if kind == "optimal":
        res = astar(safe, s, g, nbr)
        cells = rc(res.path)
    elif kind == "detour":
        free = np.flatnonzero(safe.free.reshape(-1))
        best = None
        for _ in range(40):
            cand = int(free[rng.integers(len(free))])
            r1 = astar(safe, s, cand, nbr)
            r2 = astar(safe, cand, g, nbr)
            if r1.path is None or r2.path is None:
                continue
            total = r1.cost_m + r2.cost_m
            if best is None or total > best[0]:
                best = (total, r1.path, r2.path)
        if best is None:
            raise RuntimeError("no detour reference for %s" % scene.ident)
        cells = np.concatenate([rc(best[1])[:-1], rc(best[2])])
    elif kind == "straight":
        a, b = np.array(scene.start_rc, float), np.array(scene.goal_rc, float)
        t = np.linspace(0.0, 1.0, 40)[:, None]
        cells = a[None] + t * (b - a)[None]
    elif kind == "decoy":
        a, b = np.array(scene.start_rc, float), np.array(scene.goal_rc, float)
        # a smooth curve that swings into a DIFFERENT corridor, but whose control
        # point stays inside the map (otherwise the whole reference degenerates)
        h, w = spec.height, spec.width
        ctrl = np.array([
            np.clip((a[0] + b[0]) / 2 - 0.35 * abs(b[1] - a[1]), 2, h - 3),
            np.clip((a[1] + b[1]) / 2, 2, w - 3)])
        t = np.linspace(0.0, 1.0, 60)[:, None]
        cells = ((1 - t) ** 2) * a[None] + 2 * (1 - t) * t * ctrl[None] \
            + (t ** 2) * b[None]
    else:
        raise ValueError("unknown reference kind %r" % (kind,))

    world = spec.cell_to_world(np.asarray(cells, dtype=np.float64)[:, ::-1])
    world[0] = scene.start_world
    world[-1] = scene.goal_world
    return world
