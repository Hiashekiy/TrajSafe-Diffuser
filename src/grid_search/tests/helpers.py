"""Shared helpers for the grid-search unit tests."""

from __future__ import annotations

import numpy as np

from ..benchmark import Case, evaluate_case, prepare_case
from ..grid_map import GridSpec, build_safe_grid
from ..grid_neighbors import build_neighbor_table
from ..scenes import make_scene


def case_from_scene(scene, start_world=None, goal_world=None,
                    ref_world=None) -> Case:
    return Case(case_id=scene.ident, occupancy=scene.occupancy, spec=scene.spec,
                start_world=scene.start_world if start_world is None else start_world,
                goal_world=scene.goal_world if goal_world is None else goal_world,
                ref_world=scene.ref_world if ref_world is None else ref_world,
                meta={"scene": scene.name})


def prep_scene(ident, robot_radius_m=0.0, safety_margin_m=0.0, J=3, **kw):
    """Prepare a synthetic acceptance scene with NO extra inflation."""
    scene = make_scene(ident)
    case = case_from_scene(scene, **kw)
    return prepare_case(case, robot_radius_m=robot_radius_m,
                        safety_margin_m=safety_margin_m, J=J)


def run(prep, method="pdg", w=1.2, alpha_d=1.0, alpha_p=1.0, **kw):
    return evaluate_case(prep, method, method, w, alpha_d, alpha_p, **kw)


def naive_focal_search(safe, start_idx, goal_idx, w=1.2, secondary="f0_index",
                       guidance=None, alpha_d=1.0, alpha_p=1.0, sigma_m=None,
                       L_ref_m=None):
    """Deliberately dumb FOCAL: rescan the whole OPEN dict every iteration.

    Used by the tests to prove the heap/lazy-deletion maintenance in
    ``focal_astar.focal_search`` implements the same rule (design section 5.4).
    """
    import heapq

    from ..focal_astar import octile_heuristic
    from ..grid_neighbors import build_neighbor_table

    spec = safe.spec
    nbr = build_neighbor_table(safe.free, spec.cell_m)
    goal_rc = spec.index_to_rc(np.array([goal_idx]))[0]
    cell_m = spec.cell_m
    cleared = np.asarray(safe.clearance_m).reshape(-1)

    g = {start_idx: 0.0}
    parent = {start_idx: -1}
    open_set = {start_idx}
    closed = set()

    def h(node):
        return float(octile_heuristic(node // spec.width, node % spec.width,
                                      goal_rc, cell_m))

    def f0(node):
        return g[node] + h(node)

    expanded = 0
    while open_set:
        f_min = min(f0(v) for v in open_set)
        focal = [v for v in open_set if f0(v) <= w * f_min + 1e-12]
        if guidance is not None and getattr(guidance, "valid", False) \
                and secondary == "guidance":
            cand = np.asarray(focal, dtype=np.int64)
            hd = guidance.score(cand, np.array([g[v] for v in focal]),
                                alpha_d, alpha_p, sigma_m, L_ref_m)
            order = sorted(range(len(focal)),
                           key=lambda i: (hd[i], f0(focal[i]), -cleared[focal[i]],
                                          focal[i]))
        elif secondary == "clearance":
            order = sorted(range(len(focal)),
                           key=lambda i: (-cleared[focal[i]], f0(focal[i]),
                                          focal[i]))
        elif secondary == "index":
            order = sorted(range(len(focal)), key=lambda i: focal[i])
        else:
            order = sorted(range(len(focal)),
                           key=lambda i: (f0(focal[i]), focal[i]))
        current = focal[order[0]]
        open_set.discard(current)
        expanded += 1
        if current == goal_idx:
            path = [current]
            while parent[path[-1]] >= 0:
                path.append(parent[path[-1]])
            return {"status": "SUCCESS", "cost": g[current],
                    "expanded": expanded, "path": path[::-1]}
        closed.add(current)
        for k in range(8):
            nb = int(nbr.idx[current, k])
            if nb < 0:
                continue
            tentative = g[current] + float(nbr.cost[k])
            if tentative + 1e-12 < g.get(nb, np.inf):
                g[nb] = tentative
                parent[nb] = current
                open_set.add(nb)
                closed.discard(nb)
    return {"status": "NO_PATH", "cost": np.inf, "expanded": expanded,
            "path": None}
