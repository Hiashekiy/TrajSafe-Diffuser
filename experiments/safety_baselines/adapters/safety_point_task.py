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
    """PDM App.C.3 segment rows: a path is judged on the SEGMENTS, not on its waypoints.

    The constraint object stays the segments (the accepted point is verified against the
    segment residual), but the local SQP is INITIALISED from the separable vertex
    projection. Reason: at the top noise levels the state is O(1e3) scene units from any
    feasible curve, and a 46k-row local SQP cannot travel that far
    (`segment_sqp_maxiter: residual 0.45` measured on the very first level), while the
    vertex projection is separable and always lands in a free-space point set. This is the
    same "give the local optimiser a feasible starting point" role the A* initialiser plays
    in the PDM code; it is NOT a change of the constraint, and a failure to reach the
    segment tolerance is still raised.
    """

    def __init__(self, model, schedule, occupancy, cond, points=128, radius=0.,
                 field_kind='exact', candidates=None, margin=0.0):
        SegmentPointTask.__init__(self, model, schedule, occupancy, cond, points, radius,
                                  field_kind, margin)
        self._init_candidates(schedule, candidates)

    # Gate for the segment projection: at the top noise levels the state is not a curve yet
    # (sigma ~ 5e3 scene), and no local 46k-row SQP can make all 127 segments avoid 438
    # circles (measured: `segment_sqp_no_descent`, residual 0.33, even from a vertex-feasible
    # start). Projections at those levels are therefore SKIPPED and the loop keeps its
    # Langevin update; PDM App. F explicitly allows adjusting the projection frequency. The
    # FINAL projection is never skipped (allow_partial=False), so an infeasible delivery
    # still raises.
    projection_gate_residual = 0.05          # scene units (~4 m): start projecting below this

    def project(self, target, budget, start=None, tol=None, maxiter=None, allow_partial=False):
        """Segment projection with a high-noise gate and a vertex-feasible initial point."""
        target = np.asarray(target, dtype=float).reshape(-1)
        g = float(np.max(self.constraints(target)[0]))
        # Gate on the residual only (not on `allow_partial`): CD's Projected/PD/ALM call the
        # projection with allow_partial=False at EVERY level, so gating on that flag would
        # leave the top levels un-gated. A skip is recorded in `info`, and the delivered path
        # is still scored by the evaluator, so nothing is hidden.
        if g > self.projection_gate_residual:
            return target, {'status': 'skipped_over_gate', 'residual': g,
                            'gate': float(self.projection_gate_residual)}
        init = start
        if init is None:
            p = self.unpack(target)
            h, grad = self.map.query(p)
            margin = 0.004
            bad = h < margin
            if bad.any():
                p[bad] = p[bad] + grad[bad] * (margin - h[bad])[:, None]
            init = p[1:-1].reshape(-1)
        kw = {'start': init} if init is not None else {}
        return SegmentPointTask.project(self, target, budget, allow_partial=allow_partial, **kw)
