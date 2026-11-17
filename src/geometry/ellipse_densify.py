"""Insert ellipse centres into corridor gaps, reusing the previous ellipse.

The corridor of section 5.3 fails to close when two consecutive convex cells do
not overlap enough (``bridge_failed`` / ``bridge_gamma_failed``): the ellipses
predicted along the searched path simply leave a hole between them.

This module fills those holes the cheapest possible way — it inserts one (or a
few) points at the midpoint of the gap and REUSES THE PREVIOUS ELLIPSE's shape
there, then lets ``build_safety_corridor`` do its normal overlap/bridge work on
the densified sequence.  Nothing else changes: the shape head is untouched and
no geometry is invented, the extra regions are copies of a region the network
already predicted.

Overlap is measured with the helpers the corridor builder itself uses
(``halfspaces_to_vertices`` / ``overlap_ratio``), so "needs an insert" means
exactly what the builder means by "needs a bridge".
"""

from __future__ import annotations

import numpy as np

from .safety_corridor import halfspaces_to_vertices, overlap_ratio

__all__ = ["densify_ellipses"]


def _cell_polygons(builder, centers, shape4):
    import torch
    dev = None
    groups = getattr(builder, '_groups', None)
    if groups:
        idx = groups[0][0]
        dev = idx.device if torch.is_tensor(idx) else None
    c_t = torch.as_tensor(np.asarray(centers), dtype=torch.float32)[None]
    s_t = torch.as_tensor(np.asarray(shape4), dtype=torch.float32)[None]
    if dev is not None:
        c_t, s_t = c_t.to(dev), s_t.to(dev)
    A, b, mask, valid = builder.build_from_ellipse(c_t, s_t)
    A = A[0].detach().cpu().numpy()
    b = b[0].detach().cpu().numpy()
    mask = mask[0].detach().cpu().numpy().astype(bool)
    out = []
    for i in range(len(centers)):
        if not bool(valid[0, i]):
            out.append(None)
            continue
        keep = mask[i]
        if keep.sum() < 3:
            out.append(None)
            continue
        out.append(halfspaces_to_vertices(A[i][keep], b[i][keep]))
    return out


def densify_ellipses(builder, centers, shape4, anchors, *, min_overlap: float = 0.10,
                     max_insert_per_gap: int = 2, max_total: int = 256):
    """Return ``(centers, shape4, anchors, stats)`` with gap-filling inserts.

    A gap is a consecutive pair whose cell overlap is below ``min_overlap``.  For
    each such gap we insert midpoints between the two centres and give them the
    LEFT ellipse's ``shape4`` (the instruction: reuse the previous ellipse), then
    re-evaluate the two new adjacent pairs, up to ``max_insert_per_gap``.
    """
    centers = np.asarray(centers, dtype=np.float64)
    shape4 = np.asarray(shape4, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64).reshape(-1)
    stats = {"inserted": 0, "gaps": 0}

    polygons = _cell_polygons(builder, centers, shape4)
    polys = list(polygons)
    cs = [centers[i].copy() for i in range(len(centers))]
    sh = [shape4[i].copy() for i in range(len(shape4))]
    an = [float(anchors[i]) for i in range(len(anchors))]

    def ratio(i, j):
        pa, pb = polys[i], polys[j]
        if pa is None or pb is None:
            return 0.0
        return float(overlap_ratio(pa, pb))

    i = 0
    while i < len(cs) - 1 and len(cs) < max_total:
        r = ratio(i, i + 1)
        if r >= min_overlap:
            i += 1
            continue
        stats["gaps"] += 1
        # insert between i and i+1, reusing the LEFT ellipse (shape copied)
        added = 0
        for _ in range(int(max_insert_per_gap)):
            if len(cs) >= max_total:
                break
            mid_c = 0.5 * (cs[i] + cs[i + 1])
            mid_s = sh[i].copy()
            mid_a = 0.5 * (an[i] + an[i + 1])
            poly = _cell_polygons(builder, mid_c[None], mid_s[None])[0]
            cs.insert(i + 1, mid_c)
            sh.insert(i + 1, mid_s)
            an.insert(i + 1, mid_a)
            polys.insert(i + 1, poly)
            added += 1
            if ratio(i, i + 1) >= min_overlap:
                break
        stats["inserted"] += added
        i += 1 + added

    if stats["inserted"] == 0:
        return centers, shape4, anchors, stats
    order = np.argsort(np.asarray(an))
    return (np.stack(cs)[order], np.stack(sh)[order],
            np.asarray(an)[order], stats)
