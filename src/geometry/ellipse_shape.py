"""Shared ellipse-shape parameterisation (report sections 19 and 20).

The whole project must use exactly ONE mapping from the 4 raw head outputs to
the physical ellipse parameters, otherwise the offline GT mask and the training
loss would disagree.

    raw [l1, l2, u, v]
        -> log a = max(l1, l2), log b = min(l1, l2)
        -> (u, v) normalised, with a safe (1, 0) fallback for (0, 0)
        -> shape4 = [log a, log b, cos 2theta, sin 2theta]

The functions work on arbitrary trailing dimensions so they can be used by the
network head (``[B, H, 4]``), the dataset (``[H, 4]``) and the losses.
"""

from __future__ import annotations

import torch

__all__ = ["unit_direction", "raw_to_shape4", "raw_to_abtheta",
           "shape4_to_abtheta"]

_LOG_CLAMP = 8.0
_DIR_EPS = 1e-12


def unit_direction(uv: torch.Tensor, eps: float = _DIR_EPS) -> torch.Tensor:
    """Normalise ``uv`` to a unit vector, with a NaN-free (1, 0) fallback.

    The head can emit exactly (0, 0) at initialisation.  ``atan2(0, 0)`` has no
    derivative, so the degenerate direction must be replaced *before* any
    trigonometric function is evaluated.  ``torch.where`` keeps the unused branch
    out of the backward pass, and the numerator/denominator are both regular at
    zero, so no NaN/Inf can be produced.
    """
    norm2 = (uv * uv).sum(dim=-1, keepdim=True)
    safe = norm2 > float(eps)
    # Evaluate sqrt only on the safe entries: sqrt(0) has an infinite derivative,
    # and multiplying that by the (zero) fallback mask would still produce NaN.
    norm2_safe = torch.where(safe, norm2, torch.ones_like(norm2))
    unit = uv / torch.sqrt(norm2_safe)
    one = torch.ones_like(uv[..., :1])
    zero = torch.zeros_like(uv[..., :1])
    fallback = torch.cat([one, zero], dim=-1)
    return torch.where(safe, unit, fallback)


def raw_to_shape4(raw: torch.Tensor) -> torch.Tensor:
    """``[..., 4] = [l1, l2, u, v] -> [log a, log b, cos 2t, sin 2t]``.

    ``log a`` / ``log b`` are returned UNCLAMPED: the report explicitly says
    shape supervision uses the raw log-axis values.  Clamping only happens when
    the physical semi-axes are needed (see :func:`shape4_to_abtheta`).
    """
    log_a = torch.maximum(raw[..., 0], raw[..., 1])
    log_b = torch.minimum(raw[..., 0], raw[..., 1])
    direction = unit_direction(raw[..., 2:4])
    return torch.stack([log_a, log_b, direction[..., 0], direction[..., 1]],
                       dim=-1)


def shape4_to_abtheta(shape4: torch.Tensor):
    """``[..., 4] -> (a, b, theta)`` with ``a >= b > 0``.

    ``(cos 2t, sin 2t)`` is renormalised, so the returned angle is always
    consistent with the returned axes.  ``exp`` is clamped for numerical safety
    only; the label definition itself stays ``[log a, log b, cos 2t, sin 2t]``.
    """
    log_a = shape4[..., 0]
    log_b = shape4[..., 1]
    a = torch.exp(log_a.clamp(-_LOG_CLAMP, _LOG_CLAMP))
    b = torch.exp(log_b.clamp(-_LOG_CLAMP, _LOG_CLAMP))
    direction = unit_direction(shape4[..., 2:4])
    theta = 0.5 * torch.atan2(direction[..., 1], direction[..., 0])
    return a, b, theta


def raw_to_abtheta(raw: torch.Tensor):
    """Convenience: raw head output -> (a, b, theta)."""
    return shape4_to_abtheta(raw_to_shape4(raw))
