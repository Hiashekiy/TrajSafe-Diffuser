# Exact control constraints shared with Ours geometry

Current implementation: adapters/bezier_constraints.py. It consumes an explicit BSplineConstraintPack and endpoints. It does not call Ours ALM, predict geometry, search a route, or automatically use Ours corridor.

For each responsibility interval, subdivide at the union of interval boundaries and original spline knots. Ours exact extraction gives beta[l,r] = E[l,r] Q. A convex cell A_j p <= b_j gives A_j E[l,r] Q <= b_j for all four Bezier controls. This is a sufficient continuous-curve condition inside the supplied convex cells; Bezier controls are an exact linear representation, not a least-squares fit.

The adapter flattens this exact pack to G x <= d after eliminating the two fixed endpoint controls. Constraint values and Jacobians match src.diffusion.bspline_alm.constraint_state to floating-point precision. Projection is a convex minimum-control-displacement QP with the original constraint set. Empty packs and infeasible endpoint conditioning are rejected.

Validation: 44 tests passed, including equality with Ours constraints, finite-difference Jacobian, independent dense curve check, endpoint invariance, and projection fixed point. These are geometry/solver tests, not CARLA campaign acceptance. No new full campaign has launched.

Critical unresolved integration: corridor source must be explicit. A pack from Ours learned geometry/searched skeleton is an additional conditional input to external methods; it cannot silently replace map-only baseline constraints in a claimed independent-method comparison. The existing direct-SDF validation runs have not established successful complete solves for the CBF and projection branches. Do not promote them to a full run.
