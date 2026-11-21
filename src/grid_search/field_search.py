"""Bridge the bundled field-guided-search package to V5 scene coordinates.

All fields and the A* fallback use the same inflated occupancy. The original
network occupancy is never passed to or mutated by this adapter.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time

import numpy as np


def guidance_class():
    # Load the supplied source explicitly; an older pip installation cannot win.
    name = "src._bundled_field_guided_search"
    if name not in sys.modules:
        root = Path(__file__).resolve().parents[1] / "field-guided-search/src/field_guided_search"
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name].GuidanceField


def search_path(safe, spec, start, goal, curve, config):
    """Return a 128-point scene path and existing sampler-compatible statistics.

No free endpoint snapping or straight-line fallback: a blocked endpoint or
disconnected map is an explicit failure, not a usable search result.
"""
    t0 = time.perf_counter()
    options = dict(config.get("field_guided") or {})
    defaults = dict(lam=0.01, weight="quadratic", distance="manhattan",
                    source_method="drop", num_samples=64, method="auto")
    gf = guidance_class()((~safe.free).astype(np.uint8), **{
        k: options.get(k, v) for k, v in defaults.items()})
    start, goal = np.asarray(start), np.asarray(goal)
    start_rc = tuple(np.rint(spec.world_to_cell(start)[::-1]).astype(int))
    goal_rc = tuple(np.rint(spec.world_to_cell(goal)[::-1]).astype(int))
    for label, rc in (("start", start_rc), ("goal", goal_rc)):
        if not (0 <= rc[0] < spec.height and 0 <= rc[1] < spec.width
                and safe.free[rc]):
            raise RuntimeError(f"field-guided search: {label} blocked/outside inflated map")
    stats = dict(backend="field_guided", guide_valid=False, fallback=None,
                 expanded=0, guide_ms=0.0, search_ms=0.0, status="NA")
    tg = time.perf_counter()
    try:
        gf.build(spec.world_to_cell(np.asarray(curve))[..., ::-1])
        stats["guide_valid"] = True
    except RuntimeError as exc:
        # The package reports no surviving source explicitly. A* stays on the
        # same inflated map and does not pretend a guidance field was built.
        stats["guide_error"] = str(exc)
    stats["guide_ms"] = (time.perf_counter() - tg) * 1000
    search_defaults = dict(alpha=0.15, gamma=1.0, adaptive=False,
                           clearance_beta=0.10, clearance_mode="quadratic_hinge",
                           clearance_threshold=3.0)
    result = None
    if stats["guide_valid"]:
        result = gf.search(start_rc, goal_rc, **{
            k: options.get(k, v) for k, v in search_defaults.items()})
    if result is None or result["status"] != "success":
        result = gf.astar(start_rc, goal_rc)
        stats["fallback"] = "astar"
    stats.update(status=result["status"].upper(), expanded=result["expanded"])
    if result["status"] != "success":
        raise RuntimeError(f"field-guided search failed: {result['status']}")
    pts = spec.cell_to_world(np.asarray(result["path"], dtype=float)[:, ::-1])
    pts[0], pts[-1] = start, goal
    lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.r_[0., np.cumsum(lengths)]
    sample = np.linspace(0, cum[-1], 128)
    path = np.stack([np.interp(sample, cum, pts[:, k]) for k in range(2)], -1)
    # Resampling a polyline can cut obstacle corners. Check each resulting
    # segment at <= 0.1-cell spacing before it can become a fallback output.
    cells = spec.world_to_cell(path)
    for a, b in zip(cells[:-1], cells[1:]):
        n = max(2, int(np.ceil(np.linalg.norm(b - a) * 10)) + 1)
        rc = np.rint(np.linspace(a, b, n)[:, ::-1]).astype(int)
        if (np.any(rc < 0) or np.any(rc[:, 0] >= spec.height)
                or np.any(rc[:, 1] >= spec.width)
                or not safe.free[rc[:, 0], rc[:, 1]].all()):
            raise RuntimeError("search path crosses inflated obstacles after resampling")
    stats["path_len_m"] = float(cum[-1] * spec.scene_to_meter)
    stats["search_ms"] = (time.perf_counter() - t0) * 1000
    stats["preprocess_ms"] = result.get("preprocess_ms", 0.0)
    stats["queue_search_ms"] = result.get("search_ms", 0.0)
    return path.astype(np.float32), stats
