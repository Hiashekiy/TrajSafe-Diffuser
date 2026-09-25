#!/usr/bin/env python
"""_diag_c48_failures.py - why did the C=48 run still collide on its 9 failures?

For every failing test index of ``outputs/oneshot_k4p_c48`` (ALM on, 16 steps,
seed 0 -- the eval protocol) this asks the ONE question that separates the two
possible causes:

    the curve is inside the corridor  AND  the corridor covers obstacle cells
        -> the CONSTRAINT SET is unsound (ALM has nothing to fix)

    the curve is outside the corridor (violation > 0)
        -> the OPTIMISER ran out of budget

Method: re-run the failure through the dashboard backend, take the corridor it
actually froze (``corridor.cells[*].polygon``, 128 convex cells in scene
coordinates) and the final curve (``alm.frames[-1].safe_p``), then

  * rebuild the SAME free-space test the sampler uses (``_free_mask``: bilinear
    ``grid_sample`` of the occupancy, free iff value <= 0.5) to locate the
    colliding curve points;
  * rasterise every corridor cell on the 256^2 grid and count the OBSTACLE
    cells strictly inside it;
  * for each colliding point, look at the corridor cell responsible for its
    progress and report how many obstacle cells that cell covers and how far
    the point is from the nearest obstacle cell centre.

    python scripts/_diag_c48_failures.py [idx ...]
"""
from __future__ import annotations

import json
import sys
import urllib.request

import numpy as np
from matplotlib.path import Path

ROOT = "D:/ProjectDirectory/Neural-IRISDiffuser"
API = "http://127.0.0.1:8765"
DATASET = "160k4p_c48"
MODEL = "K4P_c48_oneshot:best_task"
RES = 256
CELL_M = 2.0 / RES * 80.0          # metres per cell (160 m crop)
FAILURES = [167, 291, 325, 333, 342, 343, 348, 366, 368]


def _get(url):
    return json.load(urllib.request.urlopen(url, timeout=600))


def generate(idx, steps=16, seed=0, alm=True, model=MODEL, dataset=DATASET):
    s = _get("%s/sample?dataset=%s&split=test&index=%d" % (API, dataset, idx))
    body = dict(sample_key=s["key"], split="test", dataset=dataset, model_id=model,
                seed=seed, condition=s["condition"], obstacles=[],
                alm_enabled=alm, steps=steps)
    req = urllib.request.Request("%s/generate" % API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def bilinear_occupancy(occ: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Exactly ``F.grid_sample(occ, points, bilinear, border, align_corners=False)``.

    ``occ [R,R]`` with row j centred at scene y = -1 + (j+0.5)/R.
    """
    gx = (points[:, 0] + 1.0) * (RES / 2.0) - 0.5
    gy = (points[:, 1] + 1.0) * (RES / 2.0) - 0.5
    x0 = np.floor(gx).astype(int)
    y0 = np.floor(gy).astype(int)
    fx = gx - x0
    fy = gy - y0
    x1, y1 = x0 + 1, y0 + 1
    cx0, cx1 = np.clip(x0, 0, RES - 1), np.clip(x1, 0, RES - 1)
    cy0, cy1 = np.clip(y0, 0, RES - 1), np.clip(y1, 0, RES - 1)
    v00 = occ[cy0, cx0]
    v01 = occ[cy0, cx1]
    v10 = occ[cy1, cx0]
    v11 = occ[cy1, cx1]
    return (v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy)
            + v10 * (1 - fx) * fy + v11 * fx * fy)


def grid_centres():
    c = (np.arange(RES) + 0.5) * (2.0 / RES) - 1.0
    X, Y = np.meshgrid(c, c)                      # X = scene x, Y = scene y
    return np.stack([X.ravel(), Y.ravel()], axis=-1)


CENTRES = grid_centres()


def cell_obstacle_cells(polygon, wall: np.ndarray) -> tuple[int, int]:
    """(#obstacle cells inside the polygon, #cells inside the polygon)."""
    inside = Path(np.asarray(polygon)).contains_points(CENTRES).reshape(RES, RES)
    n = int(inside.sum())
    return int((inside & wall).sum()), n


def analyse(idx: int, occ: np.ndarray, wall: np.ndarray):
    out = generate(idx)
    alm = out["alm"]
    cells = (out.get("corridor") or {}).get("cells") or []
    curve = np.asarray(alm["frames"][-1]["safe_p"], dtype=np.float64)   # [128,2]
    fv = out.get("final_validation") or {}
    tm = (out.get("topology") or {}).get("metrics") or {}

    val = bilinear_occupancy(occ.astype(np.float64), curve)
    hit = (val > 0.5) | (np.abs(curve) > 1.0).any(axis=-1)
    hit_idx = np.where(hit)[0]

    # corridor coverage of obstacles
    per_cell = []
    for c in cells:
        if not c.get("polygon"):
            per_cell.append((0, 0))
            continue
        per_cell.append(cell_obstacle_cells(c["polygon"], wall))
    obst_in_cells = np.array([p[0] for p in per_cell])
    anchors = np.array([float(c.get("anchor_s", i / max(len(cells) - 1, 1)))
                        for i, c in enumerate(cells)])

    rows = []
    for j in hit_idx:
        s = j / (len(curve) - 1)
        k = int(np.argmin(np.abs(anchors - s)))
        # distance from the colliding point to the nearest obstacle cell centre
        wy, wx = np.where(wall)
        if len(wx):
            sx = (wx + 0.5) * (2.0 / RES) - 1.0
            sy = (wy + 0.5) * (2.0 / RES) - 1.0
            d = np.hypot(sx - curve[j, 0], sy - curve[j, 1]).min() * 80.0
        else:
            d = float("nan")
        rows.append((int(j), float(val[j]), k, int(obst_in_cells[k]) if len(obst_in_cells) else -1,
                     float(d)))

    return dict(
        idx=idx, collision=bool(fv.get("final_collision")),
        own_violation=fv.get("final_max_constraint_violation"),
        membership=fv.get("final_corridor_membership_rate"),
        n_hit=int(hit.sum()), free_rate=float(1.0 - hit.mean()),
        cells_obstacle_total=int(obst_in_cells.sum()) if len(obst_in_cells) else 0,
        cells_obstacle_max=int(obst_in_cells.max()) if len(obst_in_cells) else 0,
        cells_with_obstacle=int((obst_in_cells > 0).sum()),
        ellipse_collision_rate=tm.get("ellipse_collision_rate"),
        center_clearance_cells=tm.get("min_center_clearance_cells"),
        rows=rows,
    )


def main(argv):
    occ_all = np.load("%s/data/carla_processed_160k4p_c48/test/occupancy.npy" % ROOT,
                      mmap_mode="r")
    idxs = [int(a) for a in argv[1:]] or FAILURES
    print("cell size = %.3f m   (1 cell = 0.625 m)" % CELL_M)
    print()
    hdr = ("idx   | cl | own_viol | hit/128 | corridor covers obstacle cells | "
           "ellipse_hit | clearance(cells)")
    print(hdr)
    print("-" * len(hdr))
    details = []
    for i in idxs:
        occ = np.asarray(occ_all[i], dtype=np.float64)
        wall = occ > 0.5
        r = analyse(i, occ, wall)
        details.append(r)
        print("%-5d | %-2s | %+8.5f | %3d/128 | total %4d  max/cell %3d  cells %3d/128 | "
              "%6.2f%% | %.2f"
              % (r["idx"], "Y" if r["collision"] else "n",
                 r["own_violation"] if r["own_violation"] is not None else float("nan"),
                 r["n_hit"], r["cells_obstacle_total"], r["cells_obstacle_max"],
                 r["cells_with_obstacle"],
                 100.0 * (r["ellipse_collision_rate"] or 0.0),
                 r["center_clearance_cells"] or float("nan")), flush=True)
    print()
    for r in details:
        if not r["rows"]:
            continue
        print("sample %d: colliding curve points (param j, bilinear occ, responsible cell k, "
              "obstacle cells in that cell, distance to nearest obstacle centre)"
              % r["idx"])
        for j, v, k, n, d in r["rows"][:12]:
            print("    j=%-3d occ=%.3f  cell#%-3d obstacle_cells_in_cell=%-3d  nearest_obstacle=%.2f m"
                  % (j, v, k, n, d))
        print()
    with open("%s/outputs/diag_c48_failures.json" % ROOT, "w", encoding="utf-8") as fh:
        json.dump(details, fh, indent=2, ensure_ascii=False)
    print("written outputs/diag_c48_failures.json")


if __name__ == "__main__":
    main(sys.argv)
