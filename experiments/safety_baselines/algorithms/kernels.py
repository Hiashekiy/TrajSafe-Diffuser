"""Shared linear algebra for the safety updates. g <= 0 feasible; particle axis is 0.

Only one thing lives here now. The Primal-Dual / ALM multiplier steps, the generic CBF QP
and the PDM inner loop that used to be in this file were replaced on 2026-11-14 by the
vendored official code in `algorithms/official/` (see docs/official_vs_ours_diff.md §C/§D):
the official update only ever needs the weighted row sum below, and every other helper had
become dead code that no runner imported.
"""
import numpy as np


def row_contribution(jac, weights, hinge=None):
    """sum_k w_k * J_k as [particles, D].

    `jac` is either the dense [particles, rows, D] tensor (vertex rows: 128 x D, small)
    or a per-particle sequence of rows (sparse [rows, D] matrices: 127 x 362 segment
    rows = 46k, where the dense tensor is ~370 MB and does not fit). The public
    Constrained Diffusers update only ever uses this weighted sum, never J itself
    (official: `autograd.grad(g.sum(), state)`, diffusion.py:306-322), so the sparse
    form is the faithful one. `weights` is [rows] (shared multiplier) or
    [particles, rows] (one multiplier per particle, which is the official
    `dual_vars[k, b, t]` layout). `hinge[b,k]` drops row k of particle b.
    """
    weights = np.asarray(weights, dtype=float)
    if isinstance(jac, np.ndarray):
        if weights.ndim == 1:
            return np.einsum('k,bkd->bd', weights, jac)
        return np.einsum('bk,bkd->bd', weights, jac)
    out = []
    for b, J in enumerate(jac):
        w = weights if weights.ndim == 1 else weights[b]
        if hinge is not None: w = w*hinge[b]
        out.append(np.asarray(J.T @ w).reshape(-1))
    return np.stack(out)
