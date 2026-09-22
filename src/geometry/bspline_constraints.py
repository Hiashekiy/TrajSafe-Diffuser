"""Exact B-spline safety constraints for a frozen safety corridor.

    SafetyCorridor
        -> responsibility intervals on the curve parameter u
        -> exact subdivision (responsibility boundaries  U  original knots)
        -> exact cubic Bezier extraction per piece
        -> linear inequality pack  A_j E_{l,r} Q <= b_j

ALM never sees an ellipse, a Skeleton or a polygon: it only ever receives the
padded :class:`BSplineConstraintPack` built here.

Why the subdivision is mandatory
--------------------------------
A cubic B-spline piece is a cubic polynomial ONLY between two consecutive
knots.  On ``[u_a, u_b]`` that does not cross a knot, the exact Bezier form is

    b0 = C(u_a)
    b1 = C(u_a) + du/3 * C'(u_a)
    b2 = C(u_b) - du/3 * C'(u_b)
    b3 = C(u_b)

with ``du = u_b - u_a``.  Because ``C(u) = N(u) Q`` and ``C'(u) = N'(u) Q``,
every Bezier control is linear in the 32 controls:

    b_{l,r} = E_{l,r} Q,        E_l0 = N(u_a)
                                E_l1 = N(u_a) + du/3 N'(u_a)
                                E_l2 = N(u_b) - du/3 N'(u_b)
                                E_l3 = N(u_b)

The Bezier convex-hull property then gives the CONTINUOUS constraint

    b_{l,0..3} in R_j   =>   C(u) in R_j  for every u in [u_a, u_b]

i.e. the curve is safe everywhere, not merely on a 128-point sample grid.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch

from .bspline import (BSplineCodec, bspline_basis_derivative_matrix,
                      bspline_basis_matrix)
from .safety_corridor import SafetyCorridor

__all__ = [
    "BSplineConstraintPack",
    "responsibility_intervals",
    "exact_subdivision",
    "bezier_extraction",
    "piece_region_assignment",
    "region_table",
    "build_constraint_pack",
    "build_constraint_pack_from_regions",
    "bezier_eval",
]

_MIN_PIECE_WIDTH = 1e-9

# Content-keyed cache of the sample-INDEPENDENT part of a constraint pack, i.e.
# the exact Bezier extraction and the region assignment.  Keyed by the codec's
# knot vector, the anchor progresses and the knot boundaries, so a different
# codec / corridor anchors can never hit a stale entry.  See
# :func:`_piece_structure`.
_PIECE_CACHE: "OrderedDict[tuple, tuple | None]" = OrderedDict()
_PIECE_CACHE_MAX = 8


def _piece_structure(codec: BSplineCodec, anchors, knot_boundaries):
    """``(extraction [P,4,C], region [P], intervals [P,2])`` or ``None``.

    This is the SAMPLE-INDEPENDENT half of a constraint pack: it only depends on
    the codec (knots) and on the corridor anchors, so every sample of a batch -
    and every micro-batch of a training run - shares it.  Building it costs
    ~15 ms per sample (two numpy basis matrices per piece, ~156 pieces), which
    dominated the training rollout, hence the cache.
    """
    key = (np.asarray(codec._knots64, dtype=np.float64).tobytes(),
           int(codec.num_controls), int(codec.degree),
           np.asarray(anchors, dtype=np.float64).tobytes(),
           np.asarray(knot_boundaries, dtype=np.float64).tobytes())
    if key in _PIECE_CACHE:
        _PIECE_CACHE.move_to_end(key)
        return _PIECE_CACHE[key]

    tau, _ = responsibility_intervals(anchors)
    u = exact_subdivision(tau, knot_boundaries)
    a, bb = u[:-1], u[1:]
    keep = (bb - a) > _MIN_PIECE_WIDTH
    a, bb = a[keep], bb[keep]
    if len(a) == 0:
        value = None
    else:
        region = piece_region_assignment(u, tau, len(anchors))[keep]
        pieces = [bezier_extraction(codec, float(ai), float(bi))
                  for ai, bi in zip(a, bb)]
        value = (torch.stack(pieces, dim=0),            # [P,4,C] float64 CPU
                 region,                                # [P]
                 np.stack([a, bb], axis=-1))            # [P,2]
    _PIECE_CACHE[key] = value
    while len(_PIECE_CACHE) > _PIECE_CACHE_MAX:
        _PIECE_CACHE.popitem(last=False)
    return value


# ---------------------------------------------------------------------------
# parameter bookkeeping
# ---------------------------------------------------------------------------


def responsibility_intervals(anchors, eps: float = 1e-9):
    """Anchor progresses -> region responsibility boundaries.

        tau_0 = 0, tau_M = 1, tau_j = (s_{j-1} + s_j)/2

    Region ``j`` is responsible for ``u in [tau_j, tau_{j+1}]``.  Returns
    ``(tau [M+1], order [M])`` where ``order`` permutes the input anchors into
    ascending order.
    """
    s = np.asarray(anchors, dtype=np.float64).reshape(-1)
    order = np.argsort(s, kind="stable")
    s = np.clip(s[order], 0.0, 1.0).astype(np.float64, copy=True)
    for i in range(1, len(s)):
        if s[i] <= s[i - 1]:
            s[i] = s[i - 1] + eps
    tau = np.empty(len(s) + 1, dtype=np.float64)
    tau[0] = 0.0
    tau[-1] = 1.0
    if len(s) > 1:
        tau[1:-1] = 0.5 * (s[:-1] + s[1:])
    tau = np.clip(tau, 0.0, 1.0)
    return tau, order


def exact_subdivision(tau, knot_boundaries=None) -> np.ndarray:
    """``U = {tau_j} u {distinct original knot values}``, sorted and unique."""
    values = [float(v) for v in np.asarray(tau, dtype=np.float64).reshape(-1)]
    if knot_boundaries is not None:
        values.extend(float(v) for v in
                      np.asarray(knot_boundaries, dtype=np.float64).reshape(-1))
    values = [v for v in values if -1e-12 <= v <= 1.0 + 1e-12]
    values.extend([0.0, 1.0])
    u = np.unique(np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0))
    return u


def piece_region_assignment(u: np.ndarray, tau: np.ndarray,
                            num_regions: int) -> np.ndarray:
    """Region id of every ``[u_l, u_{l+1}]`` piece, by its midpoint."""
    if len(u) < 2:
        return np.zeros(0, dtype=np.int64)
    mid = 0.5 * (u[:-1] + u[1:])
    region = np.searchsorted(tau, mid, side="right") - 1
    return np.clip(region, 0, max(0, num_regions - 1)).astype(np.int64)


# ---------------------------------------------------------------------------
# exact Bezier extraction
# ---------------------------------------------------------------------------


def bezier_extraction(codec: BSplineCodec, u_a: float, u_b: float,
                      dtype=torch.float64) -> torch.Tensor:
    """``E [4, C]`` with ``beta_r = E_r Q`` for the piece ``[u_a, u_b]``."""
    knots = codec._knots64
    nc, p = codec.num_controls, codec.degree
    params = np.asarray([u_a, u_b], dtype=np.float64)
    n = bspline_basis_matrix(knots, nc, p, params)                # [2, C]
    dn = bspline_basis_derivative_matrix(knots, nc, p, params)    # [2, C]
    du = float(u_b) - float(u_a)
    rows = np.stack([
        n[0],
        n[0] + (du / 3.0) * dn[0],
        n[1] - (du / 3.0) * dn[1],
        n[1],
    ], axis=0)
    return torch.as_tensor(rows, dtype=dtype)


def bezier_eval(controls: torch.Tensor, t) -> torch.Tensor:
    """Evaluate a cubic Bezier ``[...,4,2]`` at ``t`` -> ``[...,K,2]``."""
    t = torch.as_tensor(t, dtype=controls.dtype).reshape(-1)
    omt = 1.0 - t
    b = torch.stack([omt ** 3, 3.0 * omt ** 2 * t, 3.0 * omt * t ** 2,
                     t ** 3], dim=-1)                  # [K,4]
    return torch.einsum("kr,...rd->...kd", b, controls)


# ---------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------


@dataclass
class BSplineConstraintPack:
    """Padded, frozen constraint pack consumed by the control-space ALM."""

    extraction: torch.Tensor        # [B,P,4,C]
    piece_A: torch.Tensor           # [B,P,F,2]
    piece_b: torch.Tensor           # [B,P,F]
    face_mask: torch.Tensor         # [B,P,F] bool
    piece_mask: torch.Tensor        # [B,P]   bool
    piece_region_id: torch.Tensor   # [B,P]   long
    intervals: torch.Tensor         # [B,P,2]
    num_pieces: torch.Tensor        # [B]     long

    @property
    def num_controls(self) -> int:
        return int(self.extraction.shape[-1])

    @property
    def num_constraints(self) -> int:
        """Number of ACTIVE scalar inequalities ``4 * faces`` per piece."""
        return int((self.piece_mask[:, :, None, None]
                    & self.face_mask[:, :, None, :]).sum())

    def to(self, device=None, dtype=None) -> "BSplineConstraintPack":
        def cast(x):
            if x.dtype.is_floating_point:
                return x.to(device=device, dtype=dtype)
            return x.to(device=device)
        return BSplineConstraintPack(
            extraction=cast(self.extraction), piece_A=cast(self.piece_A),
            piece_b=cast(self.piece_b), face_mask=cast(self.face_mask),
            piece_mask=cast(self.piece_mask),
            piece_region_id=cast(self.piece_region_id),
            intervals=cast(self.intervals), num_pieces=cast(self.num_pieces))

    def index_select(self, rows: torch.Tensor) -> "BSplineConstraintPack":
        return BSplineConstraintPack(
            extraction=self.extraction.index_select(0, rows),
            piece_A=self.piece_A.index_select(0, rows),
            piece_b=self.piece_b.index_select(0, rows),
            face_mask=self.face_mask.index_select(0, rows),
            piece_mask=self.piece_mask.index_select(0, rows),
            piece_region_id=self.piece_region_id.index_select(0, rows),
            intervals=self.intervals.index_select(0, rows),
            num_pieces=self.num_pieces.index_select(0, rows))

    def summary(self) -> dict:
        return {
            "num_pieces": [int(v) for v in self.num_pieces.tolist()],
            "num_pieces_max": int(self.extraction.shape[1]),
            "num_faces_max": int(self.piece_A.shape[2]),
            "num_controls": self.num_controls,
            "num_active_constraints": self.num_constraints,
        }


def region_table(corridor: SafetyCorridor, device=None, dtype=torch.float32,
                 margin: float = 0.0):
    """``(A [M,Fmax,2], b [M,Fmax], mask [M,Fmax])`` for a corridor."""
    cells = corridor.cells
    fmax = max((len(c.A) for c in cells), default=0)
    m = len(cells)
    A = np.zeros((m, fmax, 2), dtype=np.float64)
    b = np.zeros((m, fmax), dtype=np.float64)
    mask = np.zeros((m, fmax), dtype=bool)
    for i, cell in enumerate(cells):
        f = len(cell.A)
        if f:
            A[i, :f] = cell.A
            # the normals are unit vectors, so subtracting a scalar from b
            # shrinks the cell INWARD by exactly that distance.  A
            # projection lands ON the constraint boundary, which eats the
            # whole margin; a positive margin is what keeps the corrected
            # curve off the corridor edge instead of hugging it.
            b[i, :f] = cell.b - float(margin)
            mask[i, :f] = True
    return (torch.as_tensor(A, dtype=dtype, device=device),
            torch.as_tensor(b, dtype=dtype, device=device),
            torch.as_tensor(mask, device=device))


def _sample_pieces(codec: BSplineCodec, A_reg, b_reg, mask_reg, anchors,
                   knot_boundaries):
    """Exact Bezier extraction + region assignment of ONE region table.

    ``A_reg [M,F,2]``, ``b_reg [M,F]``, ``mask_reg [M,F]`` describe ``M`` convex
    cells (the same layout :func:`region_table` produces), ``anchors [M]`` their
    progresses.  Returns ``None`` when no usable piece exists.

    The extraction / region / interval part depends ONLY on the codec knots and
    the anchors - never on the sample - so it comes from
    :func:`_piece_structure`, which caches it.  That matters a lot in training:
    the two-step rollout rebuilds the pack on every micro-batch and the numpy
    Bezier extraction was its dominant cost (measured ~15 ms/sample).
    """
    structure = _piece_structure(codec, anchors, knot_boundaries)
    if structure is None:
        return None
    extraction, region, intervals = structure
    return {
        "extraction": extraction,                            # [P,4,C] (shared)
        "region": region,
        "intervals": intervals,                              # [P,2] (shared)
        "A": A_reg, "b": b_reg, "mask": mask_reg,
    }


def _assemble_pack(per_sample, num_controls, device, dtype):
    """Pad a list of per-sample piece dicts (``None`` = fully masked row)."""
    B = len(per_sample)
    C = int(num_controls)
    pmax = max((p["extraction"].shape[0] for p in per_sample if p), default=0)
    fmax = max((p["A"].shape[1] for p in per_sample if p), default=0)

    # Fast path: every sample shares the SAME extraction tensor (the normal
    # case - one anchor set for the whole batch), so a single expand + device
    # copy replaces B per-sample copies.  This runs on every training
    # micro-batch.
    shared = {}
    for pack in per_sample:
        if pack is not None:
            shared[id(pack["extraction"])] = pack["extraction"]
    if len(shared) == 1 and next(iter(shared.values())).shape[0] == pmax:
        one = next(iter(shared.values()))
        extraction = one.to(dtype=dtype, device=device)[None].expand(
            B, -1, -1, -1).contiguous()
    else:
        extraction = torch.zeros(B, pmax, 4, C, dtype=dtype, device=device)
        for i, pack in enumerate(per_sample):
            if pack is None:
                continue
            p = pack["extraction"].shape[0]
            extraction[i, :p] = pack["extraction"].to(dtype=dtype, device=device)

    piece_A = torch.zeros(B, pmax, fmax, 2, dtype=dtype, device=device)
    piece_b = torch.zeros(B, pmax, fmax, dtype=dtype, device=device)
    face_mask = torch.zeros(B, pmax, fmax, dtype=torch.bool, device=device)
    piece_mask = torch.zeros(B, pmax, dtype=torch.bool, device=device)
    piece_region_id = torch.full((B, pmax), -1, dtype=torch.long, device=device)
    intervals = torch.zeros(B, pmax, 2, dtype=dtype, device=device)
    num_pieces = torch.zeros(B, dtype=torch.long, device=device)

    for i, pack in enumerate(per_sample):
        if pack is None:
            continue
        p = pack["extraction"].shape[0]
        faces = pack["A"].shape[1]
        row = pack["region"]
        # ``A`` / ``b`` / ``mask`` are torch tensors on the corridor path
        # (region_table) and numpy arrays on the offline-cache path
        # (build_constraint_pack_from_regions), so they are normalised here.
        A_reg = torch.as_tensor(np.asarray(pack["A"]), dtype=dtype,
                                device=device)[row]
        b_reg = torch.as_tensor(np.asarray(pack["b"]), dtype=dtype,
                                device=device)[row]
        m_reg = torch.as_tensor(np.asarray(pack["mask"]), device=device)[row]
        piece_A[i, :p, :faces] = A_reg
        piece_b[i, :p, :faces] = b_reg
        face_mask[i, :p, :faces] = m_reg
        piece_mask[i, :p] = True
        piece_region_id[i, :p] = torch.as_tensor(row, device=device).long()
        intervals[i, :p] = torch.as_tensor(np.asarray(pack["intervals"]),
                                           dtype=dtype, device=device)
        num_pieces[i] = p

    return BSplineConstraintPack(
        extraction=extraction, piece_A=piece_A, piece_b=piece_b,
        face_mask=face_mask, piece_mask=piece_mask,
        piece_region_id=piece_region_id, intervals=intervals,
        num_pieces=num_pieces)


def build_constraint_pack(codec: BSplineCodec,
                          corridors: list,
                          knot_boundaries=None,
                          device=None,
                          dtype=torch.float32,
                          margin: float = 0.0) -> BSplineConstraintPack:
    """Build the padded pack for a batch of frozen corridors.

    Entries that are ``None`` or ``corridor.valid == False`` become fully masked
    rows (no constraint, ALM is a no-op), so an activation failure degrades to
    raw diffusion instead of raising.
    """
    if knot_boundaries is None:
        knot_boundaries = codec.knot_span_boundaries()
    knot_boundaries = np.asarray(knot_boundaries, dtype=np.float64).reshape(-1)

    per_sample = []
    for corridor in corridors:
        if corridor is None or not getattr(corridor, "valid", False) \
                or corridor.num_cells < 2:
            per_sample.append(None)
            continue
        A_reg, b_reg, mask_reg = region_table(corridor, margin=margin)
        anchors = corridor.anchors()
        per_sample.append(_sample_pieces(codec, A_reg, b_reg, mask_reg,
                                         anchors, knot_boundaries))

    return _assemble_pack(per_sample, codec.num_controls, device, dtype)


def build_constraint_pack_from_regions(
        codec: BSplineCodec,
        cell_A: torch.Tensor,
        cell_b: torch.Tensor,
        cell_mask: torch.Tensor,
        anchors=None,
        knot_boundaries=None,
        device=None,
        dtype=torch.float32,
        margin: float = 0.0,
        sample_valid: torch.Tensor | None = None) -> BSplineConstraintPack:
    """The SAME pack as :func:`build_constraint_pack`, from an explicit table.

    Training cannot afford to build a :class:`SafetyCorridor` per sample
    (~240 ms), so ``scripts/data/carla_full/04_build_alm_constraints.py`` stores
    the corridor of every sample as a padded half-space table::

        cell_A [B,R,F,2]  cell_b [B,R,F]  cell_mask [B,R] bool

    ``cell_mask`` flags the CELLS (regions), not the faces: the script pads
    unused faces with ``A = 0`` / ``b = +inf`` inside a valid cell, so the face
    mask is derived here as ``cell_mask & isfinite(b) & |A| > 0``.  ``anchors
    [R]`` are the cell progresses (the offline cache uses
    ``linspace(0, 1, R)``, the same anchors the inference sampler passes to the
    corridor builder), and ``sample_valid [B]`` masks the samples whose corridor
    never closed.

    The returned pack is the CONTINUOUS constraint set of the inference ALM:
    ``A_j E_{l,r} Q <= b_j`` for every piece and its four exact Bezier controls,
    so a loss written on it is the training-time twin of
    :func:`src.diffusion.bspline_alm.bspline_alm_correct`.
    """
    if knot_boundaries is None:
        knot_boundaries = codec.knot_span_boundaries()
    knot_boundaries = np.asarray(knot_boundaries, dtype=np.float64).reshape(-1)

    A_all = torch.as_tensor(cell_A).detach().cpu()
    b_all = torch.as_tensor(cell_b).detach().cpu().to(torch.float64)
    m_all = torch.as_tensor(cell_mask).detach().cpu().to(torch.bool)
    B, R = int(m_all.shape[0]), int(m_all.shape[1])
    if anchors is None:
        anchors = np.linspace(0.0, 1.0, R)
    if torch.is_tensor(anchors):
        anchors = anchors.detach().cpu().numpy()
    anchors = np.asarray(anchors, dtype=np.float64).reshape(-1)

    per_sample = []
    for b in range(B):
        if sample_valid is not None and not bool(sample_valid[b]):
            per_sample.append(None)
            continue
        cell_ok = m_all[b].numpy()
        if not cell_ok.any():
            per_sample.append(None)
            continue
        A_reg = A_all[b].numpy().astype(np.float64)
        b_reg = b_all[b].numpy()
        # padded faces carry A = 0 / b = +inf: they are never allowed to bind,
        # so the per-cell flag is expanded into a per-face mask here
        face_ok = (cell_ok[:, None] & np.isfinite(b_reg)
                   & (np.abs(A_reg).sum(axis=-1) > 0.0))
        if not face_ok.any():
            per_sample.append(None)
            continue
        if float(margin) != 0.0:
            b_reg = b_reg - float(margin)
        per_sample.append(_sample_pieces(codec, A_reg, b_reg, face_ok, anchors,
                                         knot_boundaries))

    return _assemble_pack(per_sample, codec.num_controls, device, dtype)
