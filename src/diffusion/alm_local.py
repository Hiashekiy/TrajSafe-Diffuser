"""Shared closed loop for train + inference, reusing the LEGACY corridor code.

No local packing, no custom cell selection: the sequence below is copied from
``src/diffusion/sampler.py`` (the pipeline that already shipped):

    EllipseRegionBuilder.build_from_ellipse      -> per-query convex cells
    build_safety_corridor(overlap + bridge)      -> a closed convex corridor
    build_constraint_pack                        -> Bezier-extracted constraints
    bspline_alm_correct(warm-started dual)       -> projected clean prediction
    dense_validation                             -> verification only

Everything is called with the legacy argument names so the behaviour stays the
one that was already validated; the only v5-specific part is that ``gamma`` (the
bridge skeleton) is the searched path instead of a candidate skeleton, and the
anchors are ``i/(Q-1)`` because v5 uses Q=64 queries.
"""

from __future__ import annotations

import numpy as np
import torch

from ..geometry.bspline_constraints import build_constraint_pack
from ..geometry.convex_region import EllipseRegionBuilder
from ..geometry.ellipse_densify import densify_ellipses
from ..geometry.safety_corridor import build_safety_corridor
from .bspline_alm import bspline_alm_correct
from .sampler import dense_validation

__all__ = ["collision_mask", "collision_intervals", "alm_closed_loop"]

_REGION_KEYS = ("safety_margin", "obstacle_window_half", "obstacle_dilation",
                "corridor_chunk_size", "ellipse_axis_min", "ellipse_axis_max")


def collision_mask(occupancy, curve):
    """Discrete collision per dense curve point (verification statistic)."""
    occ = (occupancy.detach().cpu().numpy() if torch.is_tensor(occupancy)
           else np.asarray(occupancy))
    occ = np.squeeze(occ)
    if occ.ndim != 2:
        raise ValueError("occupancy must reduce to [H,W], got %s" % (occ.shape,))
    curve = (curve.detach().cpu().numpy() if torch.is_tensor(curve)
             else np.asarray(curve))
    curve = curve.reshape(-1, 2)
    res = occ.shape[-1]
    px = np.rint((curve[:, 0] + 1.0) / 2.0 * res - 0.5).astype(int)
    py = np.rint((curve[:, 1] + 1.0) / 2.0 * res - 0.5).astype(int)
    ok = (px >= 0) & (px < res) & (py >= 0) & (py < res)
    hit = np.ones(len(curve), dtype=bool)
    hit[ok] = occ[py[ok], px[ok]].astype(bool)
    return hit, (~ok)


def collision_intervals(hit, pad=4, min_q=8, n_query=64):
    """Contiguous collision index ranges mapped to ellipse-query indices."""
    idx = np.flatnonzero(hit)
    if len(idx) == 0:
        return []
    L = len(hit)
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    q = max(int(n_query), 2)
    spans = []
    for a, b in out:
        qa = int(np.floor(a / max(L - 1, 1) * (q - 1))) - pad
        qb = int(np.ceil(b / max(L - 1, 1) * (q - 1))) + pad
        spans.append((max(0, qa), min(q - 1, qb)))
    return spans


def alm_closed_loop(model, occ, cond, q0_raw, ell, cfg, gamma=None, lam=None):
    """Corridor -> pack -> ALM -> dense validation, exactly as the legacy sampler.

    ``gamma [B,K,2]`` is the bridge skeleton (the searched path); ``lam`` is the
    dual of the previous reverse step for warm starting.  Returns
    ``(q0_safe, lam, info)``; ``info["alm_ok"]`` is True only when the legacy
    dense verification reports no collision *and* every corridor was valid.
    """
    B = int(q0_raw.shape[0])
    dev = q0_raw.device
    alm_cfg = dict(cfg.get("alm") or {})
    builder_cfg = {k: alm_cfg[k] for k in _REGION_KEYS if k in alm_cfg}
    corridor_cfg = dict(cfg.get("corridor") or {})
    corridor_cfg.update(builder_cfg)

    centers = ell["center"]
    shape4 = ell["shape4"]
    Q = int(centers.shape[1])
    anchors = torch.linspace(0.0, 1.0, Q, device=dev, dtype=centers.dtype)

    dens_cfg = dict(corridor_cfg.get("densify") or {})
    dens_on = bool(dens_cfg.get("enabled", True))
    inserted = 0
    corridors = [None] * B
    for b in range(B):
        builder = EllipseRegionBuilder(occ[b:b + 1].float(), builder_cfg)
        c_b, s_b, a_b = centers[b], shape4[b], anchors
        if dens_on:
            # fill corridor gaps by inserting midpoints that REUSE the previous
            # ellipse (the shape head is untouched, no geometry is invented)
            c_b, s_b, a_b, dst = densify_ellipses(
                builder, centers[b].detach().cpu().numpy(),
                shape4[b].detach().cpu().numpy(),
                anchors.detach().cpu().numpy(),
                min_overlap=float(dens_cfg.get("min_overlap",
                                               corridor_cfg.get("min_overlap_ratio",
                                                                0.10))),
                max_insert_per_gap=int(dens_cfg.get("max_insert_per_gap", 2)),
                max_total=int(dens_cfg.get("max_total", 256)))
            inserted += int(dst["inserted"])
            c_b = torch.as_tensor(c_b, dtype=centers.dtype, device=dev)
            s_b = torch.as_tensor(s_b, dtype=shape4.dtype, device=dev)
            a_b = torch.as_tensor(a_b, dtype=anchors.dtype, device=dev)
        g = None if gamma is None else gamma[b]
        glen = 1 if g is None else int(g.shape[0])
        corridors[b] = build_safety_corridor(
            builder, c_b, s_b, a_b, gamma=g,
            gamma_lengths=max(glen, 1), config=corridor_cfg)

    valid = [bool(getattr(c, "valid", False)) for c in corridors]
    reasons = [getattr(c, "failure_reason", "") or "" for c in corridors]
    info = {"alm": False, "alm_ok": False, "corridor_valid": all(valid),
            "valid": valid, "reason": "", "per_sample": []}
    if not any(valid):
        info["reason"] = "corridor_invalid:" + ";".join(
            r[:40] for r in reasons if r)
        return q0_raw, lam, info

    try:
        pack = build_constraint_pack(
            model.bspline, corridors, device=dev, dtype=torch.float32,
            margin=float(alm_cfg.get("constraint_margin", 0.0)))
    except Exception as exc:      # legacy degradation: no pack -> no constraint
        info["reason"] = "pack_failed:%s" % repr(exc)[:80]
        return q0_raw, lam, info
    info["num_pieces"] = int(pack.num_pieces.sum())
    overlaps = [min(c.overlap_ratio) for c in corridors
                if getattr(c, "overlap_ratio", None)]
    info["overlap_min"] = float(min(overlaps)) if overlaps else None
    info["bridge_cells"] = int(sum(getattr(c, "bridge_cell_count", 0)
                                   for c in corridors))
    info["inserted_ellipses"] = int(inserted)
    # expose what was built (polygons only; nothing geometric is changed here)
    info["cells"] = [[np.asarray(c.polygon, dtype=float) for c in cor.cells
                      if getattr(c, "valid", False) and c.polygon is not None]
                     for cor in corridors]

    # The dual is [B,P,4,F]; F depends on the occupancy window of each sample,
    # so it can change between reverse steps.  Legacy inherits it blindly; when
    # the shape no longer matches we must drop it instead of broadcasting wrong.
    if lam is not None and (int(lam.shape[1]) != int(pack.piece_A.shape[1])
                            or int(lam.shape[-1]) != int(pack.piece_A.shape[2])):
        lam = None
        info["lam_reset"] = True

    q0_safe, lam_new, st = bspline_alm_correct(
        q0_raw, pack, model.bspline, lam, alm_cfg,
        inner_steps=int(alm_cfg.get("inner_steps", 6)))
    info["alm"] = True
    info["alm_stats"] = {k: (float(v) if isinstance(v, (int, float)) else v)
                         for k, v in (st or {}).items()}

    val = dense_validation(model, model.bspline, q0_safe, pack, occ, cond,
                           num_points=int(alm_cfg.get("dense_verify_points",
                                                      512)), cfg=alm_cfg)
    per = []
    for b in range(B):
        v = val[b]
        free = bool(not v["final_collision"])
        per.append({"ok": bool(free and valid[b]),
                    "collision_frac": 1.0 - float(v["final_free_rate"]),
                    "corridor_valid": valid[b],
                    "violation": v.get("final_max_constraint_violation"),
                    "membership": v.get("final_corridor_membership_rate")})
    info["per_sample"] = per
    info["alm_ok"] = all(p["ok"] for p in per)
    return q0_safe, lam_new, info
