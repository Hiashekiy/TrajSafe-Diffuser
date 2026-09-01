"""Zero-sum increment bridge helpers.

The diffusion variable is the zero-sum residual of position increments:
    N   = H - 1
    g   = goal - start
    base = g / N                                   (broadcast over the N increments)
    delta_k = p_k - p_{k-1}                        (increment k=1..N)
    z_k   = delta_k - base                         (residual)
and the hard start/goal constraint is  sum_k z_k = 0.

All coordinates here are in the SCENE frame [-1,1]^2.  Positions are
[.., H, 2]; increments/residuals are [.., N, 2] with N = H-1.
"""

import torch


def zero_sum(x):
    """Project x [B,N,...] onto the zero-sum subspace along the N axis."""
    return x - x.mean(dim=1, keepdim=True)


def integrate_positions(start, delta):
    """start [B,2], delta [B,N,2] -> positions [B,N+1,2] (p0 = start).

    p_k = start + sum_{i<=k} delta_i.
    """
    return torch.cat([start[:, None, :],
                      start[:, None, :] + torch.cumsum(delta, dim=1)], dim=1)


def positions_to_delta(pos):
    """pos [B,H,2] -> delta [B,H-1,2]."""
    return pos[:, 1:] - pos[:, :-1]


def compute_base(g, n):
    """g [B,2], int n -> [B,n,2]."""
    return g[:, None, :] / n


def compute_z0(pos, cond):
    """pos [B,H,2], cond [B,2,2] (start, goal) -> (z0, delta, base, g).

    z0 is numerically re-projected to enforce the zero-sum constraint.
    """
    start = cond[:, 0]
    goal = cond[:, 1]
    n = pos.shape[1] - 1
    g = goal - start
    base = compute_base(g, n)
    delta = positions_to_delta(pos)
    z0 = zero_sum(delta - base)
    return z0, delta, base, g


def state6_from_pos(pos):
    """Do NOT use: this module is position-only.  Kept out on purpose."""
    raise NotImplementedError("velocity/acceleration are removed; position only")


def check_zero_sum(z, atol=1e-5):
    return torch.allclose(z.sum(dim=1), torch.zeros_like(z.sum(dim=1)), atol=atol)


def check_endpoints(pos, start, goal, atol=1e-5):
    return torch.allclose(pos[:, 0], start, atol=atol) and torch.allclose(pos[:, -1], goal, atol=atol)
