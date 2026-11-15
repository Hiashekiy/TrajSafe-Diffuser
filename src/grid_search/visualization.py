"""Visualisation of REAL search data (design section 9).

All figures are drawn from the actual occupancy, the actual reference polyline,
the actual guidance field and the actual expansion order of a run -- nothing is
schematic.  Cell coordinates are (row, col) with row 0 at scene_y = -1, so
``imshow(origin='lower')`` reproduces the project's map orientation.
"""

from __future__ import annotations

import os

import numpy as np

__all__ = ["render_case_figures", "FIGURES"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


FIGURES = ("01_occupancy", "02_safe_grid", "03_reference_overlay",
           "04_distance_field", "05_progress_field", "06_expansions",
           "07_path_compare", "08_failure_search")


def _mask_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def _ref_rc(ref):
    rc = np.asarray(ref.points, dtype=np.float64)
    return rc[:, 1], rc[:, 0]           # row, col for imshow-style plotting


def render_case_figures(prep, result, outdir, prefix="case", dpi=110,
                        expanded_limit=5000) -> dict:
    """Write the eight required figures for one prepared case + one result."""
    plt = _plt()
    os.makedirs(outdir, exist_ok=True)
    case = prep.case
    occ = np.asarray(case.occupancy)
    safe = prep.safe
    spec = case.spec
    paths = {}

    def save(fig, name):
        path = os.path.join(outdir, "%s_%s.png" % (prefix, name))
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths[name] = path

    # 1 -- raw occupancy ---------------------------------------------------
    sr = _w2c(case.start_world, spec)
    gr = _w2c(case.goal_world, spec)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest")
    ax.scatter([sr[1]], [sr[0]], marker="*", s=140, c="#1f77b4",
               label="start", zorder=6)
    ax.scatter([gr[1]], [gr[0]], marker="*", s=140, c="#d62728", label="goal",
               zorder=6)
    ax.set_title("%s raw occupancy" % case.case_id, fontsize=9)
    ax.legend(loc="upper right", fontsize=7)
    _mask_axes(ax)
    save(fig, "01_occupancy")

    # 2 -- inflated safe grid ---------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest",
              alpha=0.45)
    blocked = np.ma.masked_where(safe.free, np.ones_like(occ, dtype=float))
    ax.imshow(blocked, origin="lower", cmap="Reds", alpha=0.55,
              interpolation="nearest", vmin=0, vmax=1.4)
    ax.set_title("%s safe grid (inflation %.2f cells, %.2f m)"
                 % (case.case_id, safe.inflation_cells,
                    safe.inflation_cells * spec.cell_m), fontsize=9)
    _mask_axes(ax)
    save(fig, "02_safe_grid")

    # 3 -- reference overlay (colliding part marked) -----------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest")
    if prep.ref.valid:
        rr, cc = _ref_rc(prep.ref)
        ax.plot(cc, rr, color="#1f77b4", lw=1.4, label="diffusion x0 reference")
        hit = _collide_mask(prep.ref, case)
        if hit is not None and hit.any():
            ax.scatter(cc[hit], rr[hit], s=14, c="#d62728", marker="x",
                       label="reference on obstacle", zorder=6)
    else:
        ax.text(0.03, 0.97, "reference invalid: %s" % (prep.ref.reason or "-"),
                transform=ax.transAxes, va="top", fontsize=8, color="#d62728")
    sr = _w2c(case.start_world, spec)
    gr = _w2c(case.goal_world, spec)
    ax.scatter([sr[1]], [sr[0]], marker="*", s=150, c="#2ca02c", zorder=7,
               label="start")
    ax.scatter([gr[1]], [gr[0]], marker="*", s=150, c="#d62728", zorder=7,
               label="goal")
    ax.set_title("%s reference (length %.1f m)"
                 % (case.case_id, prep.ref.length_m or 0.0), fontsize=9)
    ax.legend(loc="upper right", fontsize=6)
    _mask_axes(ax)
    save(fig, "03_reference_overlay")

    # 4 -- distance field --------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    d0 = np.asarray(prep.guidance.dist_m[:, 0]).reshape(spec.height, spec.width)
    finite = np.isfinite(d0)
    vmax = np.percentile(d0[finite], 98) if finite.any() else 1.0
    im = ax.imshow(d0, origin="lower", cmap="viridis", vmax=max(vmax, 1e-6))
    fig.colorbar(im, ax=ax, fraction=0.046, label="distance to reference (m)")
    ax.set_title("%s distance field (nearest of %d candidates)"
                 % (case.case_id, prep.guidance.dist_m.shape[1]), fontsize=9)
    _mask_axes(ax)
    save(fig, "04_distance_field")

    # 5 -- progress field --------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    u0 = np.asarray(prep.guidance.prog[:, 0]).reshape(spec.height, spec.width)
    im = ax.imshow(np.ma.masked_invalid(u0), origin="lower", cmap="twilight",
                   vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, label="projected progress u_j")
    ax.set_title("%s progress projection (nearest segment)" % case.case_id,
                 fontsize=9)
    _mask_axes(ax)
    save(fig, "05_progress_field")

    # 6 -- expansion order / snapshots -------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest")
    exp = np.asarray(result.expanded if result.expanded is not None else [],
                     dtype=np.int64)
    if len(exp):
        rc = spec.index_to_rc(exp[:expanded_limit])
        sc = ax.scatter(rc[:, 1], rc[:, 0], c=np.arange(len(rc)), s=1.2,
                        cmap="plasma", alpha=0.75)
        fig.colorbar(sc, ax=ax, fraction=0.046, label="expansion order")
        for n, snap in sorted((result.snapshots or {}).items()):
            sn = spec.index_to_rc(np.asarray(snap, dtype=np.int64))
            ax.scatter(sn[:, 1], sn[:, 0], s=0.8, facecolors="none",
                       edgecolors="#00d0ff", linewidths=0.35,
                       label="first %d" % n)
    ax.set_title("%s expansions: %d nodes (peak OPEN %d)"
                 % (case.case_id, len(exp), result.stats.get("peak_open_size", 0)),
                 fontsize=9)
    ax.legend(loc="upper right", fontsize=6)
    _mask_axes(ax)
    save(fig, "06_expansions")

    # 7 -- final path vs reference ----------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest")
    if prep.ref.valid:
        rr, cc = _ref_rc(prep.ref)
        ax.plot(cc, rr, color="#1f77b4", lw=1.0, ls="--", alpha=0.8,
                label="diffusion x0")
    if result.path is not None:
        prc = spec.index_to_rc(np.asarray(result.path, dtype=np.int64))
        ax.plot(prc[:, 1], prc[:, 0], color="#ff7f0e", lw=1.8,
                label="grid path %.1f m" % result.cost_m)
    ax.scatter([sr[1]], [sr[0]], marker="*", s=140, c="#2ca02c", zorder=7)
    ax.scatter([gr[1]], [gr[0]], marker="*", s=140, c="#d62728", zorder=7)
    ax.set_title("%s path vs reference (%s)" % (case.case_id, result.status),
                 fontsize=9)
    ax.legend(loc="upper right", fontsize=7)
    _mask_axes(ax)
    save(fig, "07_path_compare")

    # 8 -- failure / search boundary --------------------------------------
    if result.status != "SUCCESS":
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest")
        if result.expanded is not None and len(result.expanded):
            rc = spec.index_to_rc(np.asarray(result.expanded, dtype=np.int64))
            ax.scatter(rc[:, 1], rc[:, 0], s=1.2, c="#2ca02c",
                       label="expanded (%d)" % len(rc))
        if result.open_at_end is not None and len(result.open_at_end):
            rc = spec.index_to_rc(np.asarray(result.open_at_end, dtype=np.int64))
            ax.scatter(rc[:, 1], rc[:, 0], s=1.2, c="#ff7f0e", alpha=0.6,
                       label="OPEN at exit (%d)" % len(rc))
        ax.scatter([sr[1]], [sr[0]], marker="*", s=140, c="#1f77b4", zorder=7)
        ax.scatter([gr[1]], [gr[0]], marker="*", s=140, c="#d62728", zorder=7)
        ax.set_title("%s FAILED (%s)" % (case.case_id, result.status), fontsize=9)
        ax.legend(loc="upper right", fontsize=6)
        _mask_axes(ax)
        save(fig, "08_failure_search")

    return paths


def _w2c(world, spec):
    c = spec.world_to_cell(np.asarray(world, dtype=np.float64))
    return float(c[1]), float(c[0])       # row, col


def _collide_mask(ref, case):
    spec = case.spec
    rc = spec.index_to_rc(spec.cell_to_index(ref.points))
    ok = ((rc[:, 0] >= 0) & (rc[:, 0] < spec.height)
          & (rc[:, 1] >= 0) & (rc[:, 1] < spec.width))
    hit = np.zeros(len(rc), dtype=bool)
    if ok.any():
        occ = np.asarray(case.occupancy)
        hit[ok] = occ[rc[ok, 0], rc[ok, 1]].astype(bool)
    return hit
