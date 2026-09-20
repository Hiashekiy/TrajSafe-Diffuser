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
    "bezier_eval",
]

_MIN_PIECE_WIDTH = 1e-9


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


def region_table(corridor: SafetyCorridor, device=None, dtype=torch.float32):
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
            b[i, :f] = cell.b
            mask[i, :f] = True
    return (torch.as_tensor(A, dtype=dtype, device=device),
            torch.as_tensor(b, dtype=dtype, device=device),
            torch.as_tensor(mask, device=device))


def build_constraint_pack(codec: BSplineCodec,
                          corridors: list,
                          knot_boundaries=None,
                          device=None,
                          dtype=torch.float32) -> BSplineConstraintPack:
    """Build the padded pack for a batch of frozen corridors.

    Entries that are ``None`` or ``corridor.valid == False`` become fully masked
    rows (no constraint, ALM is a no-op), so an activation failure degrades to
    raw diffusion instead of raising.
    """
    if knot_boundaries is None:
        knot_boundaries = codec.knot_span_boundaries()
    knot_boundaries = np.asarray(knot_boundaries, dtype=np.float64).reshape(-1)
    B = len(corridors)
    C = codec.num_controls

    per_sample = []
    for corridor in corridors:
        if corridor is None or not getattr(corridor, "valid", False) \
                or corridor.num_cells < 2:
            per_sample.append(None)
            continue
        A_reg, b_reg, mask_reg = region_table(corridor)
        tau, _ = responsibility_intervals(corridor.anchors())
        u = exact_subdivision(tau, knot_boundaries)
        a, bb = u[:-1], u[1:]
        keep = (bb - a) > _MIN_PIECE_WIDTH
        a, bb = a[keep], bb[keep]
        if len(a) == 0:
            per_sample.append(None)
            continue
        region = piece_region_assignment(u, tau, corridor.num_cells)[keep]
        pieces = [bezier_extraction(codec, float(ai), float(bi))
                  for ai, bi in zip(a, bb)]
        per_sample.append({
            "extraction": torch.stack(pieces, dim=0),        # [P,4,C]
            "region": region,
            "intervals": np.stack([a, bb], axis=-1),         # [P,2]
            "A": A_reg, "b": b_reg, "mask": mask_reg,
        })

    pmax = max((p["extraction"].shape[0] for p in per_sample if p), default=0)
    fmax = max((p["A"].shape[1] for p in per_sample if p), default=0)

    extraction = torch.zeros(B, pmax, 4, C, dtype=dtype, device=device)
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
        extraction[i, :p] = pack["extraction"].to(dtype=dtype, device=device)
        row = pack["region"]
        piece_A[i, :p, :faces] = pack["A"].to(dtype=dtype,
                                             device=device)[row]
        piece_b[i, :p, :faces] = pack["b"].to(dtype=dtype,
                                             device=device)[row]
        face_mask[i, :p, :faces] = pack["mask"].to(device=device)[row]
        piece_mask[i, :p] = True
        piece_region_id[i, :p] = torch.as_tensor(row, device=device).long()
        intervals[i, :p] = torch.as_tensor(pack["intervals"], dtype=dtype,
                                           device=device)
        num_pieces[i] = p

    return BSplineConstraintPack(
        extraction=extraction, piece_A=piece_A, piece_b=piece_b,
        face_mask=face_mask, piece_mask=piece_mask,
        piece_region_id=piece_region_id, intervals=intervals,
        num_pieces=num_pieces)
