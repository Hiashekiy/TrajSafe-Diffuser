"""Task adapters for the native-point safety model (`models/safety_points.py`).

Two classes, differing ONLY in the constraint object, exactly as in the point-model runs:

    SafetyPointTask        one row per waypoint  (the seven papers' own definition)
    SafetySegmentPointTask one row per SEGMENT   (PDM App.C.3 circle-segment, the
                                                 task-adapted "continuous curve" variant)

Both inherit the whole constraint/solver machinery from `NativePointTask` /
`SegmentPointTask` unchanged -- one row per free waypoint, exact map distance, separable
CBF QP, per-point projection. That is the point of going back to a native trajectory state.

Only `clean` differs from those base classes: this model's denoiser needs the scene's
candidate topologies (the ellipse queries anchor on the selected candidate) and the
per-step `sqrt(alphabar_t)` as a [B] tensor, and it returns `points` instead of
`trajectory`.

History: the first version of this file provided only the vertex adapter AND
`algorithms/runner.py` selected `SegmentPointTask` only for `state_adapter='native_points'`,
so `segment_constraints_for=['PDM']` was silently ignored on this base. That is why PDM
delivered paths with every waypoint free but segments crossing walls
(docs/vertex_vs_segment_audit.md).
"""
import numpy as np
import torch

from experiments.safety_baselines.adapters.native_points import NativePointTask
from experiments.safety_baselines.adapters.segment_points import SegmentPointTask

_EXPECT = {'candidate_xy': 3, 'candidate_mask': 1, 'geometry': 3, 'geometry_lengths': 1}


class _CandidateCleanMixin:
    """Candidate-aware denoiser call, shared by the vertex and the segment task."""

    def _init_candidates(self, schedule, candidates):
        if candidates is None:
            raise ValueError('the safety-point task needs the candidate topology tensors')
        dev = self.cond.device
        self.cand = {}
        for k, v in candidates.items():
            t = v.detach().to(dev) if torch.is_tensor(v) else torch.as_tensor(np.asarray(v), device=dev)
            while k in _EXPECT and t.dim() > _EXPECT[k] and t.shape[0] == 1:
                t = t[0]                       # accept pre-batched [1,...] tensors
            t = t.bool() if 'mask' in k else t.float()
            if k == 'geometry_lengths':
                t = t.long()
            if k in _EXPECT and t.dim() != _EXPECT[k]:
                raise ValueError('%s must be %dD, got %s' % (k, _EXPECT[k], tuple(t.shape)))
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


class SafetyPointTask(_CandidateCleanMixin, NativePointTask):
    def __init__(self, model, schedule, occupancy, cond, points=128, radius=0.,
                 field_kind='exact', candidates=None):
        NativePointTask.__init__(self, model, schedule, occupancy, cond, points, radius, field_kind)
        self._init_candidates(schedule, candidates)


class SafetySegmentPointTask(_CandidateCleanMixin, SegmentPointTask):
    """PDM App.C.3 segment rows: a path is judged on the SEGMENTS, not on its waypoints."""

    def __init__(self, model, schedule, occupancy, cond, points=128, radius=0.,
                 field_kind='exact', candidates=None, margin=0.0):
        SegmentPointTask.__init__(self, model, schedule, occupancy, cond, points, radius,
                                  field_kind, margin)
        self._init_candidates(schedule, candidates)
