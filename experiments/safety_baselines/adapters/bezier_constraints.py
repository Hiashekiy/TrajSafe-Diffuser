"""Use Ours' exact geometry pack, without using Ours' ALM optimizer.

The corridor source is intentionally external: this adapter neither searches a
route nor silently imports Ours' predicted corridor into an independent baseline.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
from experiments.safety_baselines.algorithms.solvers import cbf_velocity, SolverFailure

@dataclass
class BezierControlConstraints:
    matrix: np.ndarray
    bound: np.ndarray
    endpoints: np.ndarray
    num_controls: int

    @classmethod
    def from_pack(cls, pack, endpoints, sample=0):
        def array(t):
            return t.detach().cpu().numpy()
        E = array(pack.extraction[sample]).astype(float)
        A = array(pack.piece_A[sample]).astype(float)
        b = array(pack.piece_b[sample]).astype(float)
        active = array(pack.piece_mask[sample])[:, None, None] & array(pack.face_mask[sample])[:, None, :]
        active = np.broadcast_to(active, (len(E), 4, A.shape[1]))
        if not active.any():
            raise ValueError('empty corridor pack is not a safety constraint')
        # Preserve Ours constraint_state ordering: piece, Bezier control, face.
        full = np.einsum('pfd,prc->prfcd', A, E)[active].reshape(-1, E.shape[-1]*2)
        rhs = np.broadcast_to(b[:, None, :], active.shape)[active].copy()
        endpoints = np.asarray(endpoints, dtype=float).reshape(2, 2)
        rhs -= full[:, :2] @ endpoints[0] + full[:, -2:] @ endpoints[1]
        return cls(full[:, 2:-2], rhs, endpoints.copy(), E.shape[-1])

    def unpack(self, x):
        x = np.asarray(x)
        q = np.empty((*x.shape[:-1], self.num_controls, 2))
        q[..., 0, :] = self.endpoints[0]
        q[..., -1, :] = self.endpoints[1]
        q[..., 1:-1, :] = x.reshape(*x.shape[:-1], self.num_controls-2, 2)
        return q

    def constraints(self, x):
        return self.matrix @ np.asarray(x) - self.bound, self.matrix

    def feasible_start(self, budget):
        result = linprog(np.zeros(self.matrix.shape[1]), A_ub=self.matrix,
            b_ub=self.bound, bounds=[(None, None)]*self.matrix.shape[1],
            method='highs', options={'time_limit':budget.left()})
        budget.check()
        if not result.success:
            raise SolverFailure('bezier_corridor_infeasible:'+str(result.status))
        if np.max(self.constraints(result.x)[0])>1e-7:
            raise SolverFailure('bezier_corridor_feasibility_residual')
        return result.x

    def project(self, target, budget, tol=1e-7):
        # min 0.5 ||x-target||^2 subject to the SAME Gx<=d, a convex QP.
        value, _, info = cbf_velocity(np.asarray(target), -self.matrix,
            self.bound, 0., budget, tol)
        return value, {'status':'convex_bezier_control_projection', **info}
