"""Unified metric definitions for 实验 1/2/3 (see 实验/docs/实验1.md §4).

One implementation, used by every experiment, so the three tables cannot drift
apart.  Everything is computed from the FINAL B-spline control polygon plus the
reference curve, which is the smallest common denominator of all runs.

Definitions (units: position m, curvature 1/m, squared curvature 1/m^2):

* arc length ``L``            -- dense polyline length of the exact B-spline
* detour ratio ``R_L``        -- ``L / ||G - S||_2``
* ``RMSE``/``NRMSE``          -- 128-point position error vs ``curve_gt``,
                                 normalised by the REFERENCE length
* ``kappa_max``               -- ``max_u |x'y'' - y'x''| / (x'^2 + y'^2)^{3/2}``
* ``kappa2_mean``             -- ``(1/L) * integral kappa^2 ds``

Curvature uses the EXACT B-spline derivative basis (Cox-de Boor: the derivative
of a degree-p B-spline is the degree-(p-1) B-spline of the divided differences),
never a finite-difference surrogate of the old "smoothness" column.
"""

from __future__ import annotations

import functools

import numpy as np

#: dense sampling density used for length / curvature quadrature.  Matches the
#: 512-point discrete collision protocol so every metric sees the same curve.
DENSE_POINTS = 512
#: grid used to invert arc length before curvature evaluation (parameter space)
DENSE_U = 4096
#: curvature is evaluated on an ARC-LENGTH-uniform grid with this spacing (m).
#:
#: WHY not on a uniform parameter grid: ``kappa(u)`` is the geometric curvature
#: and is parameterisation invariant analytically, but the B-spline parameter
#: ``u`` is NOT arc length.  The SDF/ALM corrections can move two neighbouring
#: control points to within ~1e-4 scene of each other, which makes the spline
#: trace a sub-millimetre loop; ``sup_u kappa`` then diverges (measured: 3.3e4
#: 1/m, i.e. a 30 um radius, on a 100 m trajectory) and the sample mean becomes
#: meaningless.  Evaluating the SAME analytic formula (exact N'(u), N''(u)) at
#: arc-length-uniform positions removes that artefact while keeping the
#: definition -- this is the standard way curvature along a path is reported.
GEOMETRY_STEP_M = 0.5
#: ``|dr/du|`` below this (scene units per unit parameter) is a degenerate
#: curve: curvature is undefined there and the sample is flagged, not dropped.
SPEED_EPS = 1e-9
#: ``min|r'| / max|r'|`` below this marks the trajectory as curvature-degenerate
#: (parametric stall / micro-cusp, ``kappa`` unbounded).  The guide requires such
#: curves to be flagged and counted, never silently dropped.
STALL_RATIO = 1e-2
#: ``||G - S||`` below this (m) makes the detour ratio meaningless.
ENDPOINT_EPS_M = 1e-6


# ---------------------------------------------------------------------------
# exact B-spline derivative bases
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def derivative_bases(knots_bytes: bytes, num_controls: int, degree: int,
                     num_points: int):
    """``[P,C]`` basis matrices for the curve and its 1st/2nd derivative.

    ``C(u)  = B0 @ Q``, ``C'(u) = B1 @ Q``, ``C''(u) = B2 @ Q`` with the same
    control polygon ``Q``; all three are exact (no finite differences).
    """
    from src.geometry.bspline import (bspline_basis_matrix,
                                      bspline_basis_derivative_matrix,
                                      bspline_derivative_operator)

    knots = np.frombuffer(knots_bytes, dtype=np.float64).copy()
    params = np.linspace(0.0, 1.0, int(num_points))
    b0 = bspline_basis_matrix(knots, int(num_controls), int(degree), params)
    d1 = bspline_derivative_operator(knots, int(num_controls), int(degree))
    b1 = bspline_basis_derivative_matrix(knots, int(num_controls), int(degree),
                                         params)
    # second derivative = derivative of the degree-(p-1) curve on U[1:-1]
    reduced = knots[1:-1]
    b2 = bspline_basis_derivative_matrix(
        reduced, int(num_controls) - 1, int(degree) - 1, params) @ d1
    return b0, b1, b2


def _bases_for(codec, num_points: int = DENSE_POINTS):
    knots = np.asarray(codec._knots64, dtype=np.float64).reshape(-1)
    return derivative_bases(knots.tobytes(), int(codec.num_controls),
                           int(codec.degree), int(num_points))


def curve_geometry(controls, codec, scene_to_meter: float,
                   step_m: float = GEOMETRY_STEP_M) -> dict:
    """Arc length and curvature of one or many control polygons.

    ``controls``: array ``[C,2]`` or ``[B,C,2]`` in scene units.  Every returned
    array has length ``B`` (``B=1`` for a single polygon).

    * ``length_m``             dense polyline length of the exact B-spline
    * ``kappa_max_inv_m``      ``max_k |T_k - T_{k-1}| / ds`` on an
                               arc-length-uniform grid of spacing ``step_m``
    * ``kappa2_mean_inv_m2``   ``mean_k kappa_k^2`` (uniform arc length, i.e.
                               ``(1/L) * integral kappa^2 ds``)
    * ``kappa_max_analytic``   ``sup_u |x'y''-y'x''|/(x'^2+y'^2)^{3/2}`` from the
                               exact 1st/2nd derivative bases (reported for
                               validation only, see the note below)

    Both curvature numbers come from the EXACT B-spline derivative bases,
    ``T_k = C'(u_k)/|C'(u_k)|`` and ``C''(u_k)`` respectively; neither is a
    second-difference surrogate of the old "smoothness" column.

    WHY the tangent-difference estimate is the reported one: ``sup_u kappa`` is
    taken over the *parameterisation*, and the SDF/ALM corrections can place two
    neighbouring control points within ~1e-4 scene of each other, making the
    spline trace a sub-millimetre loop.  There ``kappa`` is genuinely unbounded
    (measured 3.3e4 1/m on a 100 m trajectory) and a handful of such samples
    dominates any mean.  ``|T_k - T_{k-1}| / ds`` is the same quantity evaluated
    over one arc-length step, so it is bounded by ``2/ds``, exact for constant
    curvature, and converges to the analytic value; ``micro_loop`` reports the
    affected trajectories instead of hiding them.
    """
    from src.geometry.bspline import (bspline_basis_derivative_matrix,
                                      bspline_basis_matrix,
                                      bspline_derivative_operator)

    q = np.asarray(controls, dtype=np.float64)
    if q.ndim == 2:
        q = q[None]
    if q.ndim != 3 or q.shape[-1] != 2:
        raise ValueError("controls must be [C,2] or [B,C,2], got %s"
                         % (q.shape,))
    knots = np.asarray(codec._knots64, dtype=np.float64).reshape(-1)
    num_controls, degree = int(codec.num_controls), int(codec.degree)
    bases = derivative_bases(knots.tobytes(), num_controls, degree, DENSE_U)
    b0, b1_dense = bases[0], bases[1]
    d1_op = bspline_derivative_operator(knots, num_controls, degree)
    reduced = knots[1:-1]
    s_scale = float(scene_to_meter)
    n = len(q)

    dense = np.einsum("pk,bkd->bpd", b0, q) * s_scale
    seg = np.linalg.norm(np.diff(dense, axis=1), axis=-1)
    cum = np.concatenate([np.zeros((n, 1)), np.cumsum(seg, axis=1)], axis=1)
    length = cum[:, -1]
    speed_dense = np.linalg.norm(np.einsum("pk,bkd->bpd", b1_dense, q),
                                 axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        peak = np.maximum(speed_dense.max(axis=1), SPEED_EPS)
        speed_ratio = np.where(speed_dense.max(axis=1) > SPEED_EPS,
                               speed_dense.min(axis=1) / peak, 0.0)
    u_dense = np.linspace(0.0, 1.0, DENSE_U)
    kappa_max = np.full(n, np.nan)
    kappa2 = np.full(n, np.nan)
    kappa_max_analytic = np.full(n, np.nan)
    kappa2_analytic = np.full(n, np.nan)
    degenerate = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(length[i]) or length[i] <= 0.0:
            degenerate[i] = True
            continue
        count = int(length[i] // float(step_m))
        if count < 4:
            # too short for a meaningful sup; fall back to a finer grid
            count = 4
        s_k = np.linspace(0.0, length[i], count + 1)
        u_k = np.interp(s_k, cum[i], u_dense)
        ds = float(s_k[1] - s_k[0])
        b_1 = bspline_basis_derivative_matrix(knots, num_controls, degree, u_k)
        b_2 = bspline_basis_derivative_matrix(
            reduced, num_controls - 1, degree - 1, u_k) @ d1_op
        vel = (b_1 @ q[i]) * s_scale
        acc = (b_2 @ q[i]) * s_scale
        speed = np.linalg.norm(vel, axis=-1)
        if float(speed.max()) <= SPEED_EPS:
            degenerate[i] = True
            continue
        tangent = vel / np.maximum(speed, SPEED_EPS)[:, None]
        kappa = np.linalg.norm(np.diff(tangent, axis=0), axis=-1) / ds
        finite = np.isfinite(kappa)
        if not finite.any():
            degenerate[i] = True
            continue
        kappa_max[i] = float(kappa[finite].max())
        kappa2[i] = float((kappa[finite] ** 2).mean())
        cross = vel[:, 0] * acc[:, 1] - vel[:, 1] * acc[:, 0]
        analytic = np.abs(cross) / np.maximum(speed, SPEED_EPS) ** 3
        ok = np.isfinite(analytic)
        if ok.any():
            kappa_max_analytic[i] = float(analytic[ok].max())
            kappa2_analytic[i] = float((analytic[ok] ** 2).mean())
    return {
        "length_m": length,
        "kappa_max_inv_m": kappa_max,
        "kappa2_mean_inv_m2": kappa2,
        "kappa_max_analytic_inv_m": kappa_max_analytic,
        "kappa2_analytic_inv_m2": kappa2_analytic,
        "degenerate": degenerate,
        "speed_ratio": speed_ratio,
        "micro_loop": speed_ratio < STALL_RATIO,
        "step_m": float(step_m),
        "num_points": int(DENSE_U),
    }


def polyline_length(points, scene_to_meter: float) -> np.ndarray:
    """Polyline length of ``[H,2]``/``[B,H,2]`` scene-space points, in m."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim == 2:
        p = p[None]
    return np.linalg.norm(np.diff(p, axis=1), axis=-1).sum(axis=1) * \
        float(scene_to_meter)


def detour_ratio(length_m, start, goal, scene_to_meter: float) -> np.ndarray:
    """``L / ||G - S||``; NaN where the straight-line distance is degenerate."""
    s = np.asarray(start, dtype=np.float64).reshape(-1, 2)
    g = np.asarray(goal, dtype=np.float64).reshape(-1, 2)
    chord = np.linalg.norm(g - s, axis=-1) * float(scene_to_meter)
    length_m = np.asarray(length_m, dtype=np.float64).reshape(-1)
    out = np.full(length_m.shape, np.nan)
    ok = chord > ENDPOINT_EPS_M
    out[ok] = length_m[ok] / chord[ok]
    return out


def nrmse_pct(rmse_m, ref_length_m) -> np.ndarray:
    """``100 * RMSE / L_ref``; NaN where the reference length is degenerate."""
    rmse = np.asarray(rmse_m, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref_length_m, dtype=np.float64).reshape(-1)
    out = np.full(rmse.shape, np.nan)
    ok = ref > ENDPOINT_EPS_M
    out[ok] = 100.0 * rmse[ok] / ref[ok]
    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _mean_or_none(values):
    vals = np.asarray([v for v in values if v is not None], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else None


def enrich(rows, controls, reference, endpoints, codec, scene_to_meter,
           step_m: float = GEOMETRY_STEP_M) -> list:
    """Attach the 实验 1 metric set to per-task ``rows`` (in place, returned).

    ``clearance`` is defined ONLY on collision-free trajectories (实验 1 §4.1
    metric 2): ``None`` for a colliding sample, so that later aggregation cannot
    silently average a negative signed distance into the mean clearance.
    """
    geo = curve_geometry(controls, codec, scene_to_meter, step_m)
    ref_len = polyline_length(reference, scene_to_meter)
    length = geo["length_m"]
    start = endpoints[:, 0, :] if np.asarray(endpoints).ndim == 3 else endpoints
    goal = endpoints[:, 1, :] if np.asarray(endpoints).ndim == 3 else endpoints
    detour = detour_ratio(length, start, goal, scene_to_meter)
    rmse = np.asarray([r["rmse_m"] for r in rows], dtype=np.float64)
    nrmse = nrmse_pct(rmse, ref_len)
    for i, row in enumerate(rows):
        row["length_m"] = float(length[i])
        row["detour_ratio"] = None if not np.isfinite(detour[i]) else \
            float(detour[i])
        row["ref_length_m"] = float(ref_len[i])
        row["nrmse_pct"] = None if not np.isfinite(nrmse[i]) else float(nrmse[i])
        row["kappa_max_inv_m"] = (None if not np.isfinite(geo["kappa_max_inv_m"][i])
                                  else float(geo["kappa_max_inv_m"][i]))
        row["kappa2_mean_inv_m2"] = (
            None if not np.isfinite(geo["kappa2_mean_inv_m2"][i])
            else float(geo["kappa2_mean_inv_m2"][i]))
        row["degenerate_curve"] = bool(geo["degenerate"][i])
        row["micro_loop_curve"] = bool(geo["micro_loop"][i])
        row["curvature_step_m"] = geo["step_m"]
        row["clearance_m"] = (None if row["collision"]
                              else max(0.0, float(row["sdf_min_distance_m"])))
    return rows


#: the six safety/quality columns of 实验 1 表 1 (+ bookkeeping)
TABLE1_COLUMNS = ("collision_pct", "clearance_m", "detour_ratio", "nrmse_pct",
                  "kappa_max_inv_m", "kappa2_mean_inv_m2")


def summarise(rows, columns, extra: dict | None = None) -> dict:
    """Mean/std/count per numeric column over per-task ``rows``.

    ``None``/NaN are excluded per column and the surviving count is reported, so
    a metric that only applies to a subset (e.g. clearance on collision-free
    trajectories) can never be mistaken for a whole-dataset mean.
    """
    out = {"n": len(rows)}
    for name in columns:
        vals = np.asarray([r.get(name) for r in rows], dtype=object)
        num = np.asarray([np.nan if v is None else float(v) for v in vals],
                         dtype=np.float64)
        finite = np.isfinite(num)
        out[name] = {
            "mean": float(num[finite].mean()) if finite.any() else None,
            "std": float(num[finite].std(ddof=0)) if finite.any() else None,
            "valid_n": int(finite.sum()),
            "missing_n": int((~finite).sum()),
        }
    if extra:
        out.update(extra)
    return out
