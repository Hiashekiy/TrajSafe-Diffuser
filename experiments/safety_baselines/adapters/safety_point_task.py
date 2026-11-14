"""Task adapter for the native-point safety model (`models/safety_points.py`).

The constraint machinery is inherited unchanged from `NativePointTask` (one row per free
waypoint, exact map distance, separable CBF QP, per-point projection) -- that is the whole
point of going back to a native trajectory state: none of the seven baselines needs a new
solver.

Only `clean` differs: this model's denoiser needs the scene's candidate topologies (the
ellipse queries anchor on the selected candidate) and the per-step `sqrt(alphabar_t)` as a
[B] tensor, and it returns `points` instead of `trajectory`.
"""
import numpy as np
import torch

from experiments.safety_baselines.adapters.native_points import NativePointTask


class SafetyPointTask(NativePointTask):
    def __init__(self, model, schedule, occupancy, cond, points=128, radius=0.,
                 field_kind='exact', candidates=None):
        super().__init__(model, schedule, occupancy, cond, points, radius, field_kind)
        if candidates is None:
            raise ValueError('SafetyPointTask needs the candidate topology tensors')
        dev = self.cond.device
        expect = {'candidate_xy': 3, 'candidate_mask': 1, 'geometry': 3, 'geometry_lengths': 1}
        self.cand = {}
        for k, v in candidates.items():
            t = v.detach().to(dev) if torch.is_tensor(v) else torch.as_tensor(np.asarray(v), device=dev)
            # accept any number of leading batch dims of size 1
            while k in expect and t.dim() > expect[k] and t.shape[0] == 1:
                t = t[0]
            t = t.bool() if 'mask' in k else t.float()
            if k == 'geometry_lengths':
                t = t.long()
            if k in expect and t.dim() != expect[k]:
                raise ValueError('%s must be %dD, got %s' % (k, expect[k], tuple(t.shape)))
            self.cand[k] = t.reshape(1, *t.shape)
        self._sqrt_ab = schedule.sqrt_alphas_cumprod.detach().to(dev, dtype=torch.float32)

    @torch.inference_mode()
    def clean(self, x, t, ve=False):
        x = np.atleast_2d(x)
        state = x * np.sqrt(self.ab[t]) if ve else x
        p = torch.as_tensor(self.unpack(state), device=self.cond.device, dtype=torch.float32)
        n = len(x)
        tt = torch.full((n,), int(t), device=p.device, dtype=torch.long)
        out = self.model.forward_from_memory(
            p, self.memory.expand(n, -1, -1), self.cond.expand(n, -1, -1), tt,
            self._sqrt_ab[int(t)].expand(n),
            self.cand['candidate_xy'].expand(n, -1, -1, -1),
            self.cand['candidate_mask'].expand(n, -1),
            self.cand['geometry'].expand(n, -1, -1, -1),
            self.cand['geometry_lengths'].expand(n, -1))
        pred = out['points']
        if not torch.isfinite(pred).all():
            raise RuntimeError('nonfinite point denoiser')
        return pred[:, 1:-1].reshape(n, -1).cpu().numpy().astype(float)
