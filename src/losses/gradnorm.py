"""GradNorm: automatic multi-task weighting by balancing gradient magnitudes.

For each task loss L_i with weight lambda_i, we measure its gradient norm
G_i = || grad_W(lambda_i * L_i) || w.r.t. a shared layer W.  We then adjust
lambda_i so that every task gradient norm moves toward the target
G_i* = Gbar * r_i^alpha, where Gbar is the mean gradient norm and r_i is the
relative learning speed (how far each task is from its initial loss).
"""

import torch


class GradNorm:
    def __init__(self, keys, alpha=1.0, lr=1e-4, init=1.0, min_w=0.1):
        self.keys = list(keys)
        self.alpha = alpha
        self.lr = lr
        self.w = {k: float(init) for k in self.keys}
        self.min_w = float(min_w)
        self.L0 = {k: None for k in self.keys}

    def weights(self, losses, shared_params):
        """losses: {key: loss_tensor}; shared_params: list of tensors (shared layer W).

        Returns {key: weight} after updating each weight by GradNorm."""
        sp = list(shared_params)
        K = float(len(self.keys))
        # record initial loss per task
        for k in self.keys:
            if self.L0[k] is None:
                self.L0[k] = float(losses[k].detach().item())
        # gradient norm per task
        Gs = {}
        for k in self.keys:
            g = torch.autograd.grad(self.w[k] * losses[k], sp, retain_graph=True)[0]
            Gs[k] = float(g.detach().norm())
        Gbar = sum(Gs.values()) / K
        # relative learning speed
        rel = {k: float(losses[k].detach().item()) / (self.L0[k] + 1e-8) for k in self.keys}
        rel_mean = sum(rel.values()) / K
        rs = {k: rel[k] / (rel_mean + 1e-8) for k in self.keys}
        # target gradient norm
        Gstar = {k: Gbar * (rs[k] ** self.alpha) for k in self.keys}
        # update weights
        for k in self.keys:
            delta = Gs[k] - Gstar[k]
            self.w[k] = max(self.min_w, self.w[k] - self.lr * delta / (Gbar + 1e-8))
        return dict(self.w)

    def state_dict(self):
        return {"keys": list(self.keys), "alpha": self.alpha, "lr": self.lr,
                "w": dict(self.w), "L0": dict(self.L0)}

    def load_state_dict(self, sd):
        self.keys = list(sd["keys"])
        self.alpha = float(sd["alpha"])
        self.lr = float(sd["lr"])
        self.min_w = float(sd.get("min_w", 0.1))
        self.w = dict(sd["w"])
        self.L0 = dict(sd["L0"])
