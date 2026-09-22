"""Historical safety feedback (上一轮 ALM 修正) for the control-space planner.

The diffusion state is the C-control B-spline polygon, and the inference loop is

    Q0_raw(t)  ->  ALM  ->  Q0_safe(t)  ->  DDIM  ->  q_{t-1}

The network is blind to what the ALM just did: the next reverse step only sees
``q_{t-1}``, so it can (and does) re-predict a curve that violates the very
corridor the ALM just projected into.  This module gives the denoiser an explicit
memory of the previous step::

    feedback_control [B,C,2]   Q0_safe_prev   (or Q0_raw_prev when it was safe)
    feedback_delta   [B,C,2]   Q0_safe_prev - Q0_raw_prev
    feedback_valid   [B]       1 = the previous ALM result passed verification

so every control token carries the 5-vector ``[x, y, dx, dy, valid]``.

Two modules, both deliberately tiny:

``FeedbackEncoder``
    per-control MLP ``5 -> hidden -> d_model``.  The feedback is indexed by the
    control polygon itself (one vector per control point), so no Transformer /
    attention is needed: token ``i`` of the feedback stream only ever describes
    control ``i``.

``FeedbackFusion``
    gated residual into the control feature stream::

        gate = sigmoid(Linear([h_ctrl ; h_fb])) * valid
        h_ctrl <- h_ctrl + gate * h_fb

    The network decides per control point and per feature channel how much of
    the historical ALM information to trust, instead of a plain
    ``h_ctrl + h_fb`` that would let a wiggly ALM output overwrite the
    denoiser's own representation.

Both are zero-initialised by default (``zero_init``): with the encoder's last
layer at zero, ``h_fb == 0``, so a checkpoint that predates this module keeps
its EXACT previous behaviour - and the ``valid`` gate makes the fallback exact
even after training, which is what the sampler's warm-up phase relies on.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["FeedbackEncoder", "FeedbackFusion", "FEEDBACK_INPUT_DIM"]

# [x_safe, y_safe, dx, dy, valid]
FEEDBACK_INPUT_DIM = 5


class FeedbackEncoder(nn.Module):
    """Per-control MLP over ``[x, y, dx, dy, valid]`` -> ``[B,C,d_model]``."""

    def __init__(self, d_model: int, hidden: int = 64, zero_init: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        layers = [nn.Linear(FEEDBACK_INPUT_DIM, int(hidden)), nn.SiLU(),
                  nn.Linear(int(hidden), self.d_model)]
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        self.net = nn.Sequential(*layers)
        if zero_init:
            last = self.net[-1] if not isinstance(self.net[-1], nn.Dropout) \
                else self.net[-2]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features [B,C,5] -> h_fb [B,C,D]."""
        if features.shape[-1] != FEEDBACK_INPUT_DIM:
            raise ValueError("feedback features must be [B,C,%d], got %s"
                             % (FEEDBACK_INPUT_DIM, tuple(features.shape)))
        return self.net(features)


class FeedbackFusion(nn.Module):
    """Gated residual fusion of the historical feedback into ``h_ctrl``."""

    def __init__(self, d_model: int, gate_bias: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        self.gate = nn.Linear(2 * self.d_model, self.d_model)
        if float(gate_bias) != 0.0:
            nn.init.constant_(self.gate.bias, float(gate_bias))

    def forward(self, h_ctrl: torch.Tensor, h_fb: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        """h_ctrl / h_fb [B,C,D], valid [B] (or [B,1]) -> [B,C,D].

        ``valid == 0`` makes the update the EXACT identity: the gate is zeroed
        per sample, so a model trained with feedback degrades to the plain
        diffusion model during warm-up and on samples without a reliable
        corridor.
        """
        if h_ctrl.shape != h_fb.shape:
            raise ValueError("h_ctrl %s and h_fb %s must match"
                             % (tuple(h_ctrl.shape), tuple(h_fb.shape)))
        gate = torch.sigmoid(self.gate(torch.cat([h_ctrl, h_fb], dim=-1)))
        v = valid.to(device=h_ctrl.device, dtype=h_ctrl.dtype).reshape(
            h_ctrl.shape[0], -1)
        v = v.amax(dim=1) if v.shape[1] > 0 else torch.zeros(
            h_ctrl.shape[0], device=h_ctrl.device, dtype=h_ctrl.dtype)
        gate = gate * v[:, None, None]
        return h_ctrl + gate * h_fb
