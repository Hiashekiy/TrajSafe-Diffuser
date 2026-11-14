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
        """Nearest feasible point, solved as an SQP with an explicit per-row slack.

        With p = B Q the occupancy field h(p) is bilinear inside a grid cell and p is
        affine in Q, so h(B Q) is QUADRATIC (not affine) in Q within a cell: the QP below
        solves the locally LINEARISED problem and the accepted point is re-checked against
        the true constraint (local SQP, not one convex QP).

        Three defects fixed on 2026-11-14 (review):
          A. feasibility was tested on the OUTPUT argument and an already-feasible
             warm start was returned as `identity`, silently discarding `target`; it is
             now tested on `target`, and a feasible target is the only identity case.
          B. the slack row was `-s >= 0`, i.e. s <= 0, which TIGHTENS instead of
             relaxing the linearised rows; the slack is now per-row with s >= 0 and a
             diagonal penalty 0.5*rho*s^2 (a single shared slack let one bad row relax
             every row, so the design was wrong as well as the sign).
          C. the OSQP result was accepted whenever `res.x is not None`; only status 1/2
             is accepted now, and primal/dual residuals + the slack are reported.
        """
        import osqp
        from scipy import sparse
        tol = float(self.projection_tol if tol is None else tol)
        iters = int(self.projection_maxiter if maxiter is None else maxiter)
        t = np.asarray(target, dtype=float).reshape(-1)
        # (A) identity is decided by the TARGET, never by the warm start.
        g_t, _ = self.constraints(t)
        if float(np.max(g_t)) <= tol:
            return t.copy(), {'status': 'identity', 'iterations': 0, 'residual': float(np.max(g_t)),
                              'slack_norm': 0., 'slack_max': 0.}
        q = t.copy() if start is None else np.asarray(start, dtype=float).reshape(-1).copy()
        rho = float(getattr(self, 'projection_slack_weight', 1e3))
        trust = float(getattr(self, 'projection_trust', 0.05))
        g, J = self.constraints(q)
        best = None; used = 0; last = {}

        def solve_qp(A_hard, b_hard, n, with_slack, rho_s):
            """min .5||x-t||^2 (+ .5*rho*||s||^2) s.t. A_hard x >= b_hard (+ s, s>=0)."""
            if not with_slack:
                A2, l2 = A_hard, b_hard
            else:
                A2 = sparse.hstack([A_hard, sparse.eye(len(b_hard))], format='csc')
                A2 = sparse.vstack([A2, sparse.hstack([sparse.csc_matrix((len(b_hard), n)),
                                                       sparse.eye(len(b_hard))])], format='csc')
                l2 = np.r_[b_hard, np.zeros(len(b_hard))]
            rn = np.sqrt(np.asarray(A2.multiply(A2).sum(axis=1)).reshape(-1))
            sc = 1.0 / np.maximum(rn, 1e-8)
            A2 = sparse.diags(sc) @ A2; l2 = l2 * sc
            P = (sparse.eye(n, format='csc') if not with_slack else
                 sparse.diags(np.r_[np.ones(n), np.full(len(b_hard), rho_s)], format='csc'))
            solver = osqp.OSQP()
            solver.setup(P=P, q=(np.r_[-t] if not with_slack else np.r_[-t, np.zeros(len(b_hard))]),
                         A=A2, l=l2, u=np.full(A2.shape[0], np.inf), verbose=False,
                         eps_abs=1e-9, eps_rel=1e-9, max_iter=40000, time_limit=budget.left(),
                         polish=True)
            return solver.solve(raise_error=False)

        for k in range(max(1, iters)):
            budget.check()
            n = len(q)
            # linear model  g(q+dq) ~= g + J dq <= 0  <=>  (-J) x >= g - J q  (x = q+dq)
            A = sparse.csc_matrix(-np.asarray(J, dtype=float)); b = g - J @ q
            # exact projection first (well conditioned); the paper's relaxation only as a
            # fallback when the linearised rows are momentarily inconsistent.
            res = solve_qp(A, b, n, False, 0.)
            used_slack = False
            if res.x is None or int(res.info.status_val) not in (1, 2):
                res = solve_qp(A, b, n, True, rho); used_slack = True
            budget.check()
            status_val = int(res.info.status_val); status = str(res.info.status)
            if res.x is None or status_val not in (1, 2):
                last = {'status': 'osqp_%s' % status, 'iterations': k,
                        'residual': float(np.max(g)), 'osqp_status': status}
                break
            x_qp = np.asarray(res.x[:n], dtype=float)
            s_qp = np.asarray(res.x[n:], dtype=float) if used_slack else np.zeros(len(b))
            dq = x_qp - q
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
            last = {'status': 'qp_relaunched', 'iterations': used, 'residual': res_max,
                    'osqp_status': status, 'used_slack': used_slack,
                    'slack_norm': float(np.linalg.norm(s_qp)),
                    'slack_max': float(np.max(s_qp)) if s_qp.size else 0.,
                    'osqp_primal_residual': float(res.info.prim_res),
                    'osqp_dual_residual': float(res.info.dual_res),
                    'distance_to_target': float(np.linalg.norm(q - t))}
            if best is None or res_max < best[1]:
                best = (q.copy(), res_max, dict(last))
            if res_max <= tol:
                return q, dict(last, status='qp_relaunched')
            if base - res_max < 1e-12:
                break
        if best is not None:
            q, res_max, last = best[0], best[1], best[2]
        else:
            res_max = float(np.max(g))
        # Near-feasible stalls (residual a few ulps above tol) are finished with an
        # active-set Gauss-Newton step: solve J_a dq = -g_a in the least-squares sense and
        # line-search it on the TRUE residual. Same idea as the segment projector's
        # finish_feasible; it never widens the tolerance, it just converges further.
        if tol < res_max <= 1e-3:
            for _ in range(8):
                budget.check()
                g_cur, J_cur = self.constraints(q)
                act = g_cur > 0
                if not act.any():
                    res_max = float(np.max(g_cur)); break
                Ja = np.asarray(J_cur, dtype=float)[act]; ga = g_cur[act]
                try:
                    dq = -Ja.T @ np.linalg.solve(Ja @ Ja.T + 1e-12 * np.eye(act.sum()), ga)
                except np.linalg.LinAlgError:
                    break
                base = float(np.max(g_cur)); step = 1.0; improved = False
                for _ in range(20):
                    cand = q + step * dq
                    if float(np.max(self.constraints(cand)[0])) < base:
                        q = cand; improved = True; break
                    step *= 0.5
                g_new = float(np.max(self.constraints(q)[0]))
                res_max = g_new; used += 1
                last = dict(last, status='gauss_newton_polish', residual=res_max, iterations=used)
                if res_max <= tol or not improved:
                    break
        if res_max <= tol:
            return q, dict(last, status=last.get('status', 'qp_relaunched'))
        if allow_partial and res_max <= 1e-2:
            return q, dict(last, status='partial', residual=res_max)
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
