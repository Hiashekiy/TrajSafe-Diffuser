"""Fixed cubic-B-spline codec for the control-space diffusion state.

The diffusion state of this version is the 32-control polygon

    Q_t in R^{B x 32 x 2}          (cubic, clamped, degree 3, 36 knots)

and the network still reasons on the decoded 128-point curve

    P_t = B_128 Q_t in R^{B x 128 x 2}.

Everything here is FIXED geometry (no learned parameters):

  * ``BSplineCodec.decode_controls(q)``   : q [B,32,2] -> p [B,128,2]
  * ``BSplineCodec.fit_curve_to_controls(p, start, goal)``
        endpoint-constrained least-squares projection
        Q_0 = start, Q_31 = goal, only the 30 interior controls are fitted.

The clamped knot vector guarantees C(0) = Q_0 and C(1) = Q_31, therefore hard
endpoint conditioning on the CONTROL polygon is equivalent to hard endpoints on
the decoded curve.

The knot vector is loaded from the dataset root (``data/carla_v1/bspline_knots.npy``,
[36] for 32 controls / degree 3) so that the offline labels, the offline
control-GT refit and the network use exactly one basis.
"""

from __future__ import annotations

import functools
import os

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "bspline_basis_matrix",
    "bspline_derivative_operator",
    "bspline_basis_derivative_matrix",
    "default_knots_path",
    "numpy_fit_curve_to_controls",
    "BSplineCodec",
    "TrajectoryToControlHead",
]

_DEFAULT_KNOTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "carla_v1", "bspline_knots.npy")


def _as_param_array(params) -> np.ndarray:
    """Accept lists, numpy arrays or tensors on ANY device."""
    if torch.is_tensor(params):
        params = params.detach().cpu()
    return np.asarray(params, dtype=np.float64).reshape(-1)


def default_knots_path() -> str:
    return _DEFAULT_KNOTS


# ---------------------------------------------------------------------------
# basis (Cox-de Boor, NURBS-book style, robust at the clamped ends)
# ---------------------------------------------------------------------------


def _basis_row(u: float, num_controls: int, degree: int,
               knots: np.ndarray) -> np.ndarray:
    """One row of the [curve_points, num_controls] basis matrix."""
    p = int(degree)
    nc = int(num_controls)
    row = np.zeros(nc, dtype=np.float64)
    u = float(min(max(u, knots[p]), knots[nc]))
    span = int(np.searchsorted(knots, u, side="right")) - 1
    span = min(max(span, p), nc - 1)
    left = np.zeros(p + 1, dtype=np.float64)
    right = np.zeros(p + 1, dtype=np.float64)
    N = np.zeros(p + 1, dtype=np.float64)
    N[0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u
        saved = 0.0
        for r in range(j):
            den = right[r + 1] + left[j - r]
            temp = 0.0 if den == 0.0 else N[r] / den
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved
    row[span - p:span + 1] = N
    return row


def bspline_basis_matrix(knots, num_controls: int, degree: int,
                         params) -> np.ndarray:
    """[n_params, num_controls] matrix with ``curve(u_j) = B[j] @ Q``."""
    knots = np.asarray(knots, dtype=np.float64).reshape(-1)
    nc, p = int(num_controls), int(degree)
    if knots.shape[0] != nc + p + 1:
        raise ValueError("knot vector length %d != num_controls(%d)+degree(%d)+1"
                         % (knots.shape[0], nc, p))
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    return np.stack([_basis_row(float(u), nc, p, knots) for u in params], axis=0)


def bspline_derivative_operator(knots, num_controls: int, degree: int
                                ) -> np.ndarray:
    """``D_op`` with ``C'(u) = N_{p-1}^{U[1:-1]}(u) @ D_op @ Q``.

    Standard derivative-curve identity: for a degree-``p`` B-spline with ``C``
    controls on the knot vector ``U``, the derivative is a degree-``p-1``
    B-spline with ``C-1`` controls

        D_i = p / (U_{i+p+1} - U_{i+1}) * (Q_{i+1} - Q_i)

    on the reduced knot vector ``U[1:-1]``.  This is EXACT (no finite
    differences) and, because the reduced vector is clamped too, it evaluates
    the correct one-sided derivative at ``u = 0`` and ``u = 1``.
    """
    knots = np.asarray(knots, dtype=np.float64).reshape(-1)
    p, nc = int(degree), int(num_controls)
    op = np.zeros((nc - 1, nc), dtype=np.float64)
    for i in range(nc - 1):
        den = knots[i + p + 1] - knots[i + 1]
        coef = 0.0 if abs(den) < 1e-14 else float(p) / float(den)
        op[i, i] = -coef
        op[i, i + 1] = coef
    return op


def bspline_basis_derivative_matrix(knots, num_controls: int, degree: int,
                                    params) -> np.ndarray:
    """``[n_params, num_controls]`` matrix with ``curve'(u_j) = B'[j] @ Q``."""
    knots = np.asarray(knots, dtype=np.float64).reshape(-1)
    nc, p = int(num_controls), int(degree)
    reduced = knots[1:-1]
    low = bspline_basis_matrix(reduced, nc - 1, p - 1, params)   # [P, C-1]
    return low @ bspline_derivative_operator(knots, nc, p)       # [P, C]


@functools.lru_cache(maxsize=8)
def _cached_codec_numpy(knots_bytes: bytes, num_controls: int, degree: int,
                        curve_points: int):
    knots = np.frombuffer(knots_bytes, dtype=np.float64).copy()
    params = np.linspace(0.0, 1.0, int(curve_points))
    basis = bspline_basis_matrix(knots, num_controls, degree, params)
    interior = basis[:, 1:-1]
    pinv = np.linalg.pinv(interior)                     # [C-2, H]
    return knots, basis, pinv


def numpy_fit_curve_to_controls(knots, num_controls: int, degree: int,
                                p: np.ndarray, start: np.ndarray,
                                goal: np.ndarray, curve_points: int = 128
                                ) -> np.ndarray:
    """Endpoint-constrained LS projection of a dense curve onto controls.

    ``p [H,2]``, ``start/goal [2]`` -> ``q [num_controls, 2]``.
    The basis is the SAME basis the network decodes with (uniform parameters).
    """
    knots = np.asarray(knots, dtype=np.float64).reshape(-1)
    _, basis, pinv = _cached_codec_numpy(
        knots.astype(np.float64).tobytes(), int(num_controls), int(degree),
        int(curve_points))
    p = np.asarray(p, dtype=np.float64).reshape(-1, 2)
    start = np.asarray(start, dtype=np.float64).reshape(2)
    goal = np.asarray(goal, dtype=np.float64).reshape(2)
    Y = p - basis[:, 0:1] * start[None, :] - basis[:, -1:] * goal[None, :]
    q_interior = pinv @ Y                                # [C-2, 2]
    return np.concatenate([start[None, :], q_interior, goal[None, :]], axis=0)


# ---------------------------------------------------------------------------
# torch module
# ---------------------------------------------------------------------------


class BSplineCodec(nn.Module):
    """Fixed clamped B-spline codec; every heavy matrix is a registered buffer."""

    def __init__(self, degree: int = 3, num_controls: int = 32,
                 curve_points: int = 128, knots=None, knots_path: str | None = None,
                 endpoint_constrained: bool = True, dtype=torch.float32):
        super().__init__()
        self.degree = int(degree)
        self.num_controls = int(num_controls)
        self.curve_points = int(curve_points)
        self.endpoint_constrained = bool(endpoint_constrained)
        if knots is None:
            path = knots_path or _DEFAULT_KNOTS
            if not os.path.exists(path):
                raise FileNotFoundError(
                    "B-spline knot file not found: %s (config: bspline.knots)"
                    % path)
            knots = np.load(path)
        knots = np.asarray(knots, dtype=np.float64).reshape(-1)
        if knots.shape[0] != self.num_controls + self.degree + 1:
            raise ValueError(
                "knots length %d != num_controls %d + degree %d + 1"
                % (knots.shape[0], self.num_controls, self.degree))
        params = np.linspace(0.0, 1.0, self.curve_points)
        basis = bspline_basis_matrix(knots, self.num_controls, self.degree,
                                     params)                 # [H, C]
        derivative_op = bspline_derivative_operator(
            knots, self.num_controls, self.degree)           # [C-1, C]
        basis_t = torch.as_tensor(basis, dtype=torch.float64)
        interior = basis_t[:, 1:-1]                          # [H, C-2]
        pinv = torch.linalg.pinv(interior)                   # [C-2, H]
        self._knots64 = knots
        self.register_buffer("knots", torch.as_tensor(knots, dtype=dtype))
        self.register_buffer("basis", basis_t.to(dtype))      # [H, C]
        # persistent=False: derived, fixed geometry.  Keeping it out of the
        # state dict means every pre-existing checkpoint still loads with
        # strict=True.
        self.register_buffer("derivative_op",
                             torch.as_tensor(derivative_op, dtype=dtype),
                             persistent=False)
        self.register_buffer("interior_pinv", pinv.to(dtype))  # [C-2, H]
        # constant endpoint columns (kept explicit for readability)
        self.register_buffer("basis_start", basis_t[:, 0:1].to(dtype))
        self.register_buffer("basis_end", basis_t[:, -1:].to(dtype))

    # ------------------------------------------------------------------ basis
    def basis_at(self, params) -> torch.Tensor:
        """Basis rows ``N(u)`` at arbitrary parameters -> ``[P, C]``.

        Exact (Cox-de Boor), so it can subdivide the curve at any parameter
        without touching the 128-point sampling grid.
        """
        values = _as_param_array(params)
        rows = bspline_basis_matrix(self._knots64, self.num_controls,
                                    self.degree, values)
        return torch.as_tensor(rows, dtype=self.basis.dtype,
                               device=self.basis.device)

    def basis_derivative_at(self, params) -> torch.Tensor:
        """Derivative basis rows ``N'(u)`` -> ``[P, C]``, exact, no FD.

        Uses the derivative curve of degree ``p-1`` on the reduced (still
        clamped) knot vector ``U[1:-1]``, so ``u = 0`` and ``u = 1`` return the
        correct one-sided derivatives.
        """
        values = _as_param_array(params)
        rows = bspline_basis_derivative_matrix(
            self._knots64, self.num_controls, self.degree, values)
        return torch.as_tensor(rows, dtype=self.basis.dtype,
                               device=self.basis.device)

    def knot_span_boundaries(self) -> np.ndarray:
        """Distinct knot values inside ``(0, 1)`` of the clamped knot vector."""
        knots = np.asarray(self._knots64, dtype=np.float64).reshape(-1)
        u = np.unique(knots)
        return u[(u > 1e-12) & (u < 1.0 - 1e-12)]

    # ------------------------------------------------------------------ decode
    def decode_controls(self, q: torch.Tensor) -> torch.Tensor:
        """q [B,C,2] -> p [B,H,2] = B_128 q (differentiable)."""
        if q.shape[-2] != self.num_controls:
            raise ValueError("expected %d controls, got %s"
                             % (self.num_controls, tuple(q.shape)))
        basis = self.basis.to(dtype=q.dtype)
        return torch.einsum("hk,bkd->bhd", basis, q)

    # ------------------------------------------------------------------- fit
    def fit_curve_to_controls(self, p: torch.Tensor, start: torch.Tensor,
                              goal: torch.Tensor) -> torch.Tensor:
        """Endpoint-constrained LS: p [B,H,2], start/goal [B,2] -> q [B,C,2]."""
        if p.shape[-2] != self.curve_points:
            raise ValueError("expected %d curve points, got %s"
                             % (self.curve_points, tuple(p.shape)))
        basis_start = self.basis_start.to(dtype=p.dtype)
        basis_end = self.basis_end.to(dtype=p.dtype)
        pinv = self.interior_pinv.to(dtype=p.dtype)
        y = (p - basis_start[None] * start[:, None, :]
             - basis_end[None] * goal[:, None, :])
        q_interior = torch.einsum("kh,bhd->bkd", pinv, y)
        return torch.cat([start[:, None, :], q_interior, goal[:, None, :]],
                         dim=1)

    # -------------------------------------------------------------- utilities
    @staticmethod
    def hard_control_endpoints(q: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """q [B,C,2]; cond [B,2,2] -> q with q[:,0]=start and q[:,-1]=goal."""
        q = q.clone()
        q[:, 0] = cond[:, 0]
        q[:, -1] = cond[:, 1]
        return q


class TrajectoryToControlHead(nn.Module):
    """Trajectory -> control head (report section 4.2).

    First version: NOT a learned MLP.  It is the fixed, differentiable,
    endpoint-constrained least-squares projection onto the B-spline control
    polygon (``BSplineCodec.fit_curve_to_controls``).  It has zero parameters,
    so autograd only routes the gradient of the curve loss into the trunk.
    """

    def __init__(self, codec: BSplineCodec):
        super().__init__()
        self.codec = codec

    @property
    def num_controls(self) -> int:
        return self.codec.num_controls

    def forward(self, p: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.codec.fit_curve_to_controls(p, cond[:, 0], cond[:, 1])
