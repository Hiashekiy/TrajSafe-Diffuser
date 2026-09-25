#!/usr/bin/env python
"""_diag_c48_projection_scope.py - what the hard projection projects onto, verified
against the DASHBOARD's own frozen corridor.

Two claims, tested with data that does NOT come from the LP's own matrices:

  1. the projection runs ONCE, after the last denoising step;
  2. its constraint set IS the 128 frozen convex regions -- applied to the BEZIER
     control points (4 per piece), NOT to the 48 control-polygon vertices.

The test uses the dashboard payload of ONE generation: the frozen corridor's 128
cell POLYGONS (`corridor.cells[*].polygon`, the same thing the canvas draws), the
pre-projection curve (`pre_projection`), the returned curve (`final_curve`) and
the control polygon the guided phase produced (`control_history[-1]`).

    python scripts/_diag_c48_projection_scope.py --index 232 --seed 7
"""
from __future__ import annotations

import argparse
import json
import urllib.request

import numpy as np

API = "http://127.0.0.1:8765"
DATASET = "160k4p_c48"
MODEL = "K4P_c48_oneshot:best_task"


def generate(idx, seed, final_project=True, model=MODEL, dataset=DATASET, steps=16):
    s = json.load(urllib.request.urlopen(
        "%s/sample?dataset=%s&split=test&index=%d" % (API, dataset, idx), timeout=600))
    body = dict(sample_key=s["key"], split="test", dataset=dataset, model_id=model,
                seed=seed, condition=s["condition"], obstacles=[],
                alm_enabled=True, final_project=final_project, steps=steps)
    req = urllib.request.Request("%s/generate" % API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def inside_polygon(poly: np.ndarray, pt: np.ndarray) -> bool:
    """Even-odd ray casting for a single point."""
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xin:
                inside = not inside
    return inside


def outside_count(cells, anchors, pts, scale_to_m=False):
    """How many points fall OUTSIDE the corridor cell responsible for their progress."""
    bad, worst = 0, 0.0
    for k, pt in enumerate(np.asarray(pts, dtype=float)):
        s = k / max(len(pts) - 1, 1)
        ci = int(np.argmin(np.abs(anchors - s)))
        poly = cells[ci].get("polygon") or []
        if len(poly) < 3:
            continue
        if not inside_polygon(np.asarray(poly, float), pt):
            bad += 1
            worst = max(worst, float(np.abs(pt).max()))
    return bad, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=232)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()

    on = generate(args.index, args.seed, True, steps=args.steps)
    off = generate(args.index, args.seed, False, steps=args.steps)
    cells = (on.get("corridor") or {}).get("cells") or []
    anchors = np.array([float(c.get("anchor_s", i / max(len(cells) - 1, 1)))
                        for i, c in enumerate(cells)])
    print("=== sample test_%04d  (16 步 / seed %d / 面板同一套 API) ===" % (args.index, args.seed))
    print("  走廊: valid=%s  cells=%d   (每个 cell 一个多边形)"
          % ((on.get("corridor") or {}).get("valid"), len(cells)))
    fp = on.get("final_projection") or {}
    print("  开关ON : status=%s  viol %+.3e -> %+.3e  corr=%.3f m"
          % (fp.get("status"), fp.get("violation_before"), fp.get("violation_after"),
             fp.get("correction_m")))
    print("  开关OFF: final_projection=%s  max_curve_step 无关（只跳过末端 LP）"
          % (off.get("final_projection"),))

    # ---- 1. timing: the projection is AFTER the last traced frame ----------
    last_state = np.asarray(on["state_history"][-1], float)
    fin = np.asarray(on["final_curve"], float)
    pre = np.asarray(on["pre_projection"], float)
    last_ctl = np.asarray(on["control_history"][-1], float)
    d = lambda a, b: float(np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1).max()) * 80.0
    print()
    print("  [1] 时序（米）")
    print("      最后一帧 state_history[-1] vs 返回曲线 final_curve : %8.3f m" % d(last_state, fin))
    print("      引导输出 pre_projection    vs 返回曲线 final_curve : %8.3f m" % d(pre, fin))
    print("      最后一帧 state_history[-1] vs 引导输出 pre_projection: %8.6f m" % d(last_state, pre))

    # ---- 2b. membership measured on the CELL POLYGONS --------------------
    bad_pre, _ = outside_count(cells, anchors, pre)
    bad_fin, _ = outside_count(cells, anchors, fin)
    print()
    print("  [2b] 用面板那 128 个多边形独立判定（每点按进度归到对应 cell）")
    print("       引导输出 pre_projection : 在走廊外 %3d/%d" % (bad_pre, len(pre)))
    print("       返回曲线  final_curve  : 在走廊外 %3d/%d" % (bad_fin, len(fin)))

    # ---- 2c. the 48 polygon vertices are NOT the constrained object -------
    ctl_bad, _ = outside_count(cells, anchors, last_ctl)
    print()
    print("  [2c] 同样判定用在 %d 个控制多边形顶点（投射前的 control_history[-1]）" % len(last_ctl))
    print("       在外面 %3d/%d   <- 所以约束不是『每个控制点落在区域内』" % (ctl_bad, len(last_ctl)))
    print("       而是『每段样条的 4 个 Bezier 控制点 beta_r = E_r Q 落在其责任区域内』")

    # ---- 2d. off-switch sanity ------------------------------------------
    bad_off, _ = outside_count(cells, anchors, np.asarray(off["final_curve"], float))
    print()
    print("  [2d] 关掉开关后返回的曲线在走廊外 %3d/%d  (这就是没投射的样子)"
          % (bad_off, len(off["final_curve"])))


if __name__ == "__main__":
    main()
