"""S-C48 control-space task: 48 B-spline controls, constraints on the decoded curve.

S-C48 is the historical model `outputs/models/oneshot_k4p_c48_s/ckpt/best_task.pt`
(epoch 65, trained without the feedback rollout). Its planner predates the current
HEAD API: it exposes `forward_all(q, occ, cond, t, ab, candidate_xy, candidate_mask,
geometry, geometry_lengths)` and `scene_tokens(occ)` instead of
`encode_scene` / `forward_from_memory`, and the forward pass needs the scene's
candidate topologies. This adapter supplies exactly those, and otherwise reuses
`CurveControlTask` (state unpacking, B-spline basis, curve-point constraints,
projection, nominal update) unchanged.

Everything the seven loops touch is identical to the point-space tasks: `D` free
variables, `constraints(x) -> (g, J)`, `score(x,t,ve)`, `clean(x,t,ve)`, `unpack`,
`points`, `nominal`, `project`. The only difference is that the constraint Jacobian is
mapped through the B-spline basis (`J_Q = -(B[:,1:-1] * grad_p)`), which is the exact
adjoint of `p = B Q`, i.e. the safety rows are on the delivered curve and not on the
control polygon.
"""
import numpy as np
import torch
from experiments.safety_baselines.adapters.curve_control import CurveControlTask
from experiments.safety_baselines.adapters.constraints import MapConstraint

class SC48ControlTask(CurveControlTask):
    """One scene of S-C48. `candidates` holds the scene's candidate topology tensors."""

    def __init__(self, model, schedule, occupancy, cond, points, radius=0.,
                 field_kind='bilinear', candidates=None, cbf_solver='legacy', slack_weight=None):
        if field_kind not in ('exact', 'bilinear'):
            raise ValueError('unapproved constraint field ' + field_kind)
        self.model = model; self.schedule = schedule
        occ = np.asarray(occupancy)
        self.occ = torch.as_tensor(occ, device='cuda', dtype=torch.float32).reshape(1, 1, *occ.shape)
        self.cond = torch.as_tensor(np.asarray(cond), device='cuda', dtype=torch.float32).reshape(1, 2, 2)
        self.memory = None                      # this planner takes occ directly
        self.endpoints = np.asarray(cond, dtype=float)
        self.C = int(model.num_controls); self.D = (self.C - 2) * 2
        self.B = model.bspline.basis_at(np.linspace(0, 1, points)).detach().cpu().numpy().astype(float)
        base = MapConstraint(occ, extent=2., radius=radius) if field_kind == 'bilinear' else None
        if base is None:
            from experiments.safety_baselines.adapters.exact_map import ExactMapConstraint
            base = ExactMapConstraint(occ, extent=2., radius=radius)
        self.map = base
        self.ab = schedule.alphas_cumprod.detach().cpu().numpy()
        self.betas = schedule.betas.detach().cpu().numpy()
        self.sigma = np.sqrt((1 - self.ab) / self.ab)
        # forward_all wants the per-sample scalar sqrt(alphabar_t) of shape [B], not the
        # whole schedule: the ellipse bias uses ab**2 = alphabar_t, so passing [T] here
        # silently broadcasts the batch to T rows (16) and the attention reshape fails.
        self._sqrt_ab = schedule.sqrt_alphas_cumprod.detach().to('cuda', dtype=torch.float32)
        c = candidates or {}
        self.cand = {k: (torch.as_tensor(v, device='cuda', dtype=torch.bool).reshape(1, *np.shape(v))
                         if 'mask' in k else
                         torch.as_tensor(v, device='cuda', dtype=torch.float32).reshape(1, *np.shape(v)))
                     for k, v in c.items()}
        gi = self.cand['geometry_lengths']
        self.cand['geometry_lengths'] = gi.to(torch.long).reshape(1, -1)
        # Solver selection is explicit: `legacy` keeps `cbf_velocity` (the repo default and
        # what the committed 20-scene table used); `osqp` binds the soft-constraint QP.
        self.cbf_solver = str(cbf_solver)
        self.slack_weight = None if slack_weight is None else float(slack_weight)
        if self.cbf_solver == 'osqp':
            self.cbf_solve = self.cbf_solve_osqp
        elif self.cbf_solver != 'legacy':
            raise ValueError('unknown cbf_solver %r' % cbf_solver)

    # ------------------------------------------------------------------ projection
    def project(self, target, budget, tol=None, maxiter=None, allow_partial=False, start=None):
        """Nearest feasible curve as a QP in control space, with relinearisation.

        With p = B Q the occupancy field h(p) is bilinear inside a grid cell and p is
        affine in Q, so h(B Q) is QUADRATIC (not affine) in Q within a cell; the QP below
        solves the locally LINEARISED problem and the accepted point is re-checked against
        the true constraint (i.e. this is a local SQP, not a single convex QP).
        SLSQP stalls on the same problem (measured residual 0.87 / 8.5 / 42.9 m); the
        QP version with a trust region and monotone line search reaches 0.87 m.
        """
        import osqp
        from scipy import sparse
        tol = float(self.projection_tol if tol is None else tol)
        iters = int(self.projection_maxiter if maxiter is None else maxiter)
        t = np.asarray(target, dtype=float).reshape(-1)
        q = t.copy() if start is None else np.asarray(start, dtype=float).reshape(-1).copy()
        g, J = self.constraints(q)
        if float(np.max(g)) <= tol:
            return q, {'status': 'identity', 'iterations': 0, 'residual': float(np.max(g))}
        best = None; used = 0
        trust = float(getattr(self, 'projection_trust', 0.05))
        for k in range(max(1, iters)):
            budget.check()
            # linear model  g(q+dq) ~= g + J dq <= 0  <=>  (-J) q >= g - J q_at_current
            A = sparse.csc_matrix(-J); b = g - J @ q
            n = len(q)
            row = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).reshape(-1))
            sc = 1.0 / np.maximum(row, 1e-8)
            A2 = sparse.diags(sc) @ A; b2 = b * sc
            # one scalar slack column, heavily weighted, absorbs momentarily infeasible rows
            A2 = sparse.hstack([A2, sparse.csc_matrix(np.full((len(b2), 1), 1e-3))], format='csc')
            pad = sparse.csc_matrix((1, n + 1)); pad[0, -1] = -1.0        # slack >= 0
            A2 = sparse.vstack([A2, pad], format='csc')
            P = sparse.eye(n + 1, format='csc'); P[-1, -1] = 1.0
            solver = osqp.OSQP()
            solver.setup(P=P, q=np.r_[-t, 0.], A=A2, l=np.r_[b2, 0.],
                         u=np.r_[np.full(len(b2), np.inf), np.inf],
                         verbose=False, eps_abs=1e-9, eps_rel=1e-9, max_iter=20000,
                         time_limit=budget.left(), polish=True)
            res = solver.solve(raise_error=False)
            budget.check()
            if res.x is None:
                break
            # Globalise the SQP step: the bilinear field is only piecewise affine, so an
            # unbounded QP step cycles (measured: residual frozen at 0.87 m over 200
            # iterations). Trust region + monotone line search on the TRUE residual.
            dq = np.asarray(res.x[:n], dtype=float) - q
            biggest = float(np.max(np.abs(dq))) if dq.size else 0.0
            if biggest > trust:
                dq *= trust / biggest
            base = float(np.max(g)); step = 1.0; bestq = q + dq; bestr = None
            for _ in range(8):
                cand = q + step * dq
                gr = float(np.max(self.constraints(cand)[0]))
                if bestr is None or gr < bestr:
                    bestq, bestr = cand, gr
                if gr < base:
                    break
                step *= 0.5
            q = bestq
            g, J = self.constraints(q)
            res_max = float(np.max(g)); used = k + 1
            if best is None or res_max < best[1]:
                best = (q.copy(), res_max)
            if res_max <= tol:
                return q, {'status': 'qp_relaunched', 'iterations': used, 'residual': res_max}
            if base - res_max < 1e-12:      # no measurable progress: stop early
                break
        q, res_max = best if best is not None else (q, float(np.max(g)))
        if allow_partial and res_max <= 1e-2:
            return q, {'status': 'partial', 'iterations': used, 'residual': res_max}
        raise RuntimeError('sc48_projection_failed:%d:residual=%.4g' % (used, res_max))

    # ------------------------------------------------------------------ CBF QP
    def cbf_solve_osqp(self, desired, J, offset, w, budget, tol=1e-5):
        """eq.(13) in control space, solved as an explicit soft-constraint QP:

            min .5||v-desired||^2 + (rho/2)||s||^2   s.t.  J v + offset + s >= 0, s >= 0

        `rho` follows the SAME convention as `cbf_velocity`, whose slack term is
        ||min(Jz+offset,0)||^2/(2w^2): rho = 1/w^2. So w=100 -> rho=1e-4 (slack almost
        free), w=0.01 -> rho=1e4 (nearly hard). Nothing here is a hidden constant.

        Only bound to the task when `cbf_solver == "osqp"` (see __init__); the default is
        the repo's `cbf_velocity`, which is what the committed 20-scene table used.
        """
        import osqp
        from scipy import sparse
        budget.check()
        d = np.asarray(desired, dtype=float).reshape(-1); n = len(d)
        Jm = np.asarray(J, dtype=float); off = np.asarray(offset, dtype=float).reshape(-1)
        m = len(off)
        rho = float(getattr(self, 'slack_weight', 0.) or (1.0 / max(float(w), 1e-6) ** 2))
        # (J v + off + s) >= 0  ->  [J I] z >= -off ;   s >= 0  ->  [0 I] z >= 0
        top = sparse.hstack([sparse.csc_matrix(Jm), sparse.eye(m)], format='csc')
        bot = sparse.hstack([sparse.csc_matrix((m, n)), sparse.eye(m)], format='csc')
        A = sparse.vstack([top, bot], format='csc')
        l0 = np.r_[-off, np.zeros(m)]
        rn = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).reshape(-1))
        sc = 1.0 / np.maximum(rn, 1e-8)
        A = sparse.diags(sc) @ A
        l0 = l0 * sc
        P = sparse.diags(np.r_[np.ones(n), np.full(m, rho)], format='csc')
        solver = osqp.OSQP()
        solver.setup(P=P, q=np.r_[-d, np.zeros(m)], A=A, l=l0, u=np.full(2 * m, np.inf),
                     verbose=False,
                     eps_abs=max(tol * .1, 1e-9), eps_rel=max(tol * .1, 1e-9),
                     max_iter=20000, time_limit=budget.left(), polish=True)
        res = solver.solve(raise_error=False); budget.check()
        status = str(res.info.status); status_val = int(res.info.status_val)
        # OSQP: 1 solved, 2 solved inaccurate. Anything else means the returned x is the
        # last iterate of a non-converged run and must NOT be reported as a solution.
        if res.x is None or status_val not in (1, 2):
            raise RuntimeError('sc48_cbf_qp_not_converged:%s' % status)
        v = np.asarray(res.x[:n], dtype=float); s = np.asarray(res.x[n:], dtype=float)
        gap = Jm @ v + off
        info = {'qp_status': status, 'qp_status_val': status_val,
                'cbf_residual': float(np.min(gap)),
                'cbf_residual_soft': float(np.min(gap + s)),
                'primal_residual': float(res.info.prim_res), 'dual_residual': float(res.info.dual_res),
                'velocity_change_norm': float(np.linalg.norm(v - d)),
                'velocity_ratio': float(np.linalg.norm(v) / max(np.linalg.norm(d), 1e-12)),
                'slack_norm': float(np.linalg.norm(s)), 'slack_max': float(s.max()) if m else 0.,
                'rho': rho, 'iterations': int(res.info.iter)}
        return v, s, info

    @torch.inference_mode()
    def clean(self, x, t, ve=False):
        x = np.atleast_2d(x)
        state = x * np.sqrt(self.ab[t]) if ve else x
        q = self.unpack(state)
        qt = torch.as_tensor(q, device='cuda', dtype=torch.float32)
        n = len(x)
        time = torch.full((n,), int(t), device='cuda', dtype=torch.long)
        out = self.model.forward_all(
            qt, self.occ.expand(n, -1, -1, -1), self.cond.expand(n, -1, -1), time,
            self._sqrt_ab[int(t)].expand(n),
            self.cand['candidate_xy'].expand(n, -1, -1, -1),
            self.cand['candidate_mask'].expand(n, -1),
            self.cand['candidate_geometry'].expand(n, -1, -1, -1),
            self.cand['geometry_lengths'].expand(n, -1))
        value = out['control'][:, 1:-1].reshape(n, -1).cpu().numpy().astype(float)
        if not np.isfinite(value).all():
            raise RuntimeError('nonfinite_score_model')
        return value
