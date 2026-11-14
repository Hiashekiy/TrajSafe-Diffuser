"""Optimize the delivered B-spline, without lifted residuals or output fitting."""
import numpy as np
from experiments.safety_baselines.adapters.paper_task import PaperTask
from experiments.safety_baselines.algorithms.solvers import nearest_projection

class CurveControlTask(PaperTask):
    def projection_verification_jac(self,x):
        if hasattr(self.map,'scalar_query'):
            _,grad=self.map.scalar_query(self.points(x))
            return -(self.B[:,1:-1,None]*grad[:,None,:]).reshape(len(self.B),-1)
        return self.constraints(x)[1]
    def initial(self, rng, particles=1, ve=False):
        return rng.normal(size=(particles, self.D)) * (self.sigma[-1] if ve else 1.0)

    def points(self, x):
        return self.B @ self.unpack(x)

    def nominal(self, x, t, clean):
        if t == 0:
            return clean
        noise = (x - np.sqrt(self.ab[t]) * clean) / np.sqrt(1 - self.ab[t])
        return np.sqrt(self.ab[t - 1]) * clean + np.sqrt(1 - self.ab[t - 1]) * noise

    def max_point_speed(self, velocity):
        return float(np.linalg.norm(self.B[:, 1:-1] @ np.asarray(velocity).reshape(-1, 2), axis=1).max())

    def project(self, target, budget):
        # Pure paper projection objective. No extra prior draws, old guidance,
        # feasibility penalty, selected model prediction or feasible fallback.
        value, info = nearest_projection(target,
            lambda q: self.constraints(q)[0], self.projection_verification_jac,
            None, budget,
            self.projection_maxiter, self.projection_tol,kkt_jac=self.projection_verification_jac)
        return value, info
