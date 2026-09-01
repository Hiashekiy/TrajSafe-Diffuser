"""Automatic multi-task loss weighting (from training records, no hand-set weights).

- DWA (Dynamic Weight Averaging): adjusts weights by relative loss descent rate,
  so a task that is learning slowly gets a larger weight automatically.
- Lagrangian: treats the collision loss as a constraint L_col <= delta and
  dynamically adjusts lambda_col based on the constraint violation.
"""

import numpy as np


class DWA:
    """Dynamic Weight Averaging for a subset of losses. Average weight ~1 per task."""

    def __init__(self, keys, T=2.0):
        self.keys = list(keys)
        self.T = T
        self.history = {k: [] for k in self.keys}
        self.w = {k: 1.0 for k in self.keys}

    def weights(self, losses):
        """losses: {key: loss_tensor}.  Update history and return {key: weight}."""
        for k, v in losses.items():
            self.history[k].append(float(v.detach().item()))
        rs = {}
        for k in self.keys:
            h = self.history[k]
            r = 1.0
            if len(h) >= 3:
                r = h[-2] / (h[-3] + 1e-8)      # L(t-1) / L(t-2)
            rs[k] = r / self.T
        exps = {k: float(np.exp(rs[k])) for k in self.keys}
        denom = sum(exps.values())
        K = len(self.keys)
        self.w = {k: K * exps[k] / denom for k in self.keys}
        return self.w

    def state_dict(self):
        return {"keys": list(self.keys), "T": float(self.T),
                "history": {k: list(v) for k, v in self.history.items()}, "w": dict(self.w)}

    def load_state_dict(self, sd):
        self.keys = list(sd["keys"])
        self.T = float(sd["T"])
        self.history = {k: list(v) for k, v in sd["history"].items()}
        self.w = dict(sd["w"])


class Lagrangian:
    """Dynamically adjust lambda_col so that L_col stays near / below delta."""

    def __init__(self, delta, eta, init=0.1, max_lam=1.0):
        self.delta = float(delta)
        self.eta = float(eta)
        self.lam = float(init)
        self.max_lam = float(max_lam)

    def step(self, L_col):
        """Update and return the current lambda_col (clipped to [0, max_lam])."""
        self.lam = min(max(0.0, self.lam + self.eta * (float(L_col.detach().item()) - self.delta)), self.max_lam)
        return self.lam

    def state_dict(self):
        return {"delta": self.delta, "eta": self.eta, "lam": self.lam, "max_lam": self.max_lam}

    def load_state_dict(self, sd):
        self.delta = float(sd["delta"])
        self.eta = float(sd["eta"])
        self.lam = float(sd["lam"])
        self.max_lam = float(sd.get("max_lam", 1.0))

