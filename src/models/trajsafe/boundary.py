"""FIXED (never trained) boundary decoder for the control polygon.

The network predicts a free control polygon ``Q~ [B,C,2]``.  The boundary
decoder then applies a deterministic, constant correction profile so that the
first and the last control are EXACTLY the start / goal:

    e_s = S - Q~_0              e_g = G - Q~_{C-1}
    w^s = [1, 0.75, 0.5, 0.25, 0, ...]        (first ``span`` controls)
    w^g = [..., 0, 0.25, 0.5, 0.75, 1]        (mirrored at the end)

    Q*_i = Q~_i + w^s_i e_s + w^g_i e_g

so ``Q*_0 = S`` and ``Q*_{C-1} = G`` hold by construction for ANY prediction,
while the neighbouring controls move coherently with the endpoints instead of
being teleported.  Because the knot vector is clamped, this is exactly the curve
endpoint condition ``C(0) = S``, ``C(1) = G``.

The profile is a constant buffer: it has NO parameters, is never trained and is
never touched by the optimiser.  ``BoundaryDecoder.weights()`` is the single
source of the profile for both this correction and the boundary loss, so the
supervised local shape and the applied correction can never drift apart.

The module is token-count agnostic: ``span`` is clamped to ``C // 2`` so that a
small control count degenerates gracefully to plain endpoint hard-conditioning.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

__all__ = ["BoundaryDecoder", "boundary_targets", "DEFAULT_BOUNDARY_PROFILE"]

DEFAULT_BOUNDARY_PROFILE: tuple = (1.0, 0.75, 0.5, 0.25)


def boundary_targets(q_gt: torch.Tensor, cond: torch.Tensor):
    """GT local control-polygon shape translated to THIS sample's endpoints.

    ``T^s_i = S + (Q^GT_i - Q^GT_0)`` and
    ``T^g_j = G + (Q^GT_j - Q^GT_{C-1})``.
    One single definition, shared by the boundary loss and by the metrics.
    """
    C = q_gt.shape[-2]
    off_s = (q_gt - q_gt[:, 0:1]) + cond[:, 0][:, None, :]
    off_g = (q_gt - q_gt[:, C - 1:C]) + cond[:, 1][:, None, :]
    return off_s, off_g


class BoundaryDecoder(nn.Module):
    """Parameter-free endpoint correction on the control polygon."""

    def __init__(self, profile: Sequence[float] | None = None,
                 span: int | None = None, dtype=torch.float32):
        super().__init__()
        if profile is None:
            n = int(span) if span else len(DEFAULT_BOUNDARY_PROFILE)
            if n < 1:
                raise ValueError("boundary span must be >= 1")
            base = list(DEFAULT_BOUNDARY_PROFILE)
            profile = (base[:n] + [0.0] * max(0, n - len(base)))
        prof = torch.as_tensor(list(profile), dtype=dtype).reshape(-1)
        if prof.numel() < 1:
            raise ValueError("boundary profile must not be empty")
        if float(prof[0]) != 1.0:
            raise ValueError(
                "boundary profile[0] must be 1.0 so that the first control is "
                "exactly the start (got %s)" % float(prof[0]))
        # derived from the config: never stored in the state dict
        self.register_buffer("profile", prof, persistent=False)
        self.span = int(prof.numel())

    # ------------------------------------------------------------------ utils
    def weights(self, num_controls: int, device=None, dtype=None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(w^s, w^g)`` constant vectors of length ``num_controls``.

        ``w^s[0] == 1`` and ``w^g[C-1] == 1`` so both curve endpoints are exact.
        """
        C = int(num_controls)
        if C < 2:
            raise ValueError("num_controls must be >= 2, got %d" % C)
        dtype = dtype or self.profile.dtype
        device = device or self.profile.device
        k = max(1, min(self.span, C // 2))
        w = self.profile[:k].to(device=device, dtype=dtype)
        ws = torch.zeros(C, device=device, dtype=dtype)
        wg = torch.zeros(C, device=device, dtype=dtype)
        ws[:k] = w
        wg[C - k:] = torch.flip(w, dims=[0])
        return ws, wg

    def boundary_targets(self, q_gt: torch.Tensor, cond: torch.Tensor
                         ) -> torch.Tensor:
        """Per-control targets ``T^s`` (start window) / ``T^g`` (goal window).

        Convenience wrapper that also blanks the two vectors outside their own
        window and merges them, for callers that want ONE target tensor per
        control; the boundary loss uses the raw tuple instead.
        """
        C = q_gt.shape[-2]
        ws, wg = self.weights(C, device=q_gt.device, dtype=q_gt.dtype)
        off_s, off_g = boundary_targets(q_gt, cond)
        m_s = (ws > 0)[None, :, None].to(q_gt.dtype)
        m_g = (wg > 0)[None, :, None].to(q_gt.dtype)
        return m_s * off_s + m_g * off_g + (1.0 - m_s) * (1.0 - m_g) * q_gt

    # ---------------------------------------------------------------- forward
    def forward(self, q_raw: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """q_raw [B,C,2], cond [B,2,2] -> q [B,C,2] with exact endpoints."""
        if q_raw.dim() != 3 or q_raw.shape[-1] != 2:
            raise ValueError("q_raw must be [B,C,2], got %s"
                             % (tuple(q_raw.shape),))
        C = int(q_raw.shape[-2])
        ws, wg = self.weights(C, device=q_raw.device, dtype=q_raw.dtype)
        e_s = cond[:, 0] - q_raw[:, 0]
        e_g = cond[:, 1] - q_raw[:, C - 1]
        return (q_raw + ws[None, :, None] * e_s[:, None, :]
                + wg[None, :, None] * e_g[:, None, :])


def modulate_heading(decoder, q, cond, heading, heading_valid=None, *, enabled=True,
                     apply_goal=True, min_length=0.01, valid_goal=None):
    """Rotate the endpoint tangents using the same local profile as V5."""
    if not enabled or heading is None:
        return q
    B, C, _ = q.shape
    dev, dt = q.device, q.dtype
    valid = (heading_valid if heading_valid is not None
             else torch.ones(B, 2, device=dev, dtype=torch.bool))
    valid = valid.bool()
    ws, _ = decoder.weights(C, device=dev, dtype=dt)
    # profile starting at the SECOND control (Q0 must stay exact)
    prof = ws.clone()
    prof[0] = 0.0
    q_out = q
    # The DIRECTION is the constraint, the segment length is the network's:
    # keeping the current |Q1-Q0| (clipped) means the soft profile blend
    # rotates the tangent instead of fighting the learned control spacing.
    seg = torch.linalg.norm(cond[:, 1] - cond[:, 0], dim=-1, keepdim=True)
    hi = 0.30 * seg
    def _len(cur):
        return torch.maximum(torch.minimum(cur, hi),
                             torch.full_like(cur, float(min_length)))
    # start tangent: target for Q1
    t0 = heading[:, 0] / heading[:, 0].norm(dim=-1, keepdim=True).clamp_min(1e-9)
    l0 = _len((q_out[:, 1] - q_out[:, 0]).norm(dim=-1, keepdim=True))
    tgt1 = cond[:, 0] + l0 * t0
    d1 = (tgt1 - q_out[:, 1]) * valid[:, 0][:, None].to(dt)
    q_out = q_out + prof[None, :, None] * d1[:, None, :]
    if apply_goal and (valid_goal is None or valid_goal):
        t1 = heading[:, 1] / heading[:, 1].norm(dim=-1, keepdim=True).clamp_min(1e-9)
        l1 = _len((q_out[:, C - 1] - q_out[:, C - 2]).norm(dim=-1, keepdim=True))
        tgt2 = cond[:, 1] - l1 * t1
        d2 = (tgt2 - q_out[:, C - 2]) * valid[:, 1][:, None].to(dt)
        prof_g = torch.flip(prof, dims=[0])
        q_out = q_out + prof_g[None, :, None] * d2[:, None, :]
    # the two endpoints stay exact
    q_out = q_out.clone()
    q_out[:, 0] = cond[:, 0]
    q_out[:, C - 1] = cond[:, 1]
    return q_out
