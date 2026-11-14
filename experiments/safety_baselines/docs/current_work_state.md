# USER HANDOFF / 2026-11-14

User cancelled further debugging and requested a report for DSH. Handoff completed in docs/DSH_HANDOFF_20261114.md. No experiment/training process found at handoff. Do NOT automatically resume prior research, tests or training; user has handed work off. All7 paper methods remain unvalidated; refer to the report and original evidence.

# LATEST SMALL VALIDATION COMPLETE / 2026-11-14

results/native_points_validation_20261114_011528 completed160/160:20val scenes,1seed, seven new algorithms + plain point Diffuser. All7 return raw128point trajectories, no solver failure, representation identity checked. All7 had20/20 full-polyline collisions; Projected/PDM0vertex collisions, rest20vertex collisions. Plain Diffuser13/20collisions. Do NOT publish this as native-paper quality or launch full yet: unconstrained sampler compatibility diagnostic at val0 gives DDIM RMSE5.949m, CD-Langevin19.090m, PDM-VE34116.744m. This shows unresolved finite-budget noise/sampler compatibility. NativePointTask score VP/VE formula tests pass, but trained cosine16 x0 prior differs from original score-matching model. PDM paper Alg1 gamma differs from official generic ALD epsilon/sigma_min schedule; source audit needed, do not silently switch to improve outcomes. CBF step-refinement audit shows point violation drops (RoS0.2056m at base ->0.0548m/4 ->0.0140m/16). Numerical integration not exact certificate. User asked small validation, not auto full. Current subtask completed and report/assessment saved.

# LATEST / 2026-11-14

Direct-point comparison base training COMPLETE100/100 (2455s), best_task epoch96. User now explicitly requests small validation, preserving paper methods. Native point adapter and evaluator implemented; no control points/spline in point model/adapter/output.25 focused tests passed. Active validation: results/native_points_validation_20261114_011528,20val scenes x1seed x8methods =160. Unified exec session84882. All seven branches return point trajectories in completed cases. Projected/PDM vertex feasibility is separate from polyline collision; do not confuse it with path safety. Safe branches have finite-step CBF residuals; investigate numerical step convergence before claiming theorem safety. PDM uses native VE parameterization with exact VP-to-VE score conversion; 10x100 unchanged. All task adaptations and coarse16 levels must be disclosed. Do not run old baseline full, do not substitute old correction routines, no extra prior draws (previous auto-review rejected such a change). No full run started.

# ACTIVE OVERRIDE / 2026-11-14

Latest user explicitly abandoned control-space debugging and requested comparison base directly output trajectory points; only comparison base, never full Ours. Implemented point architecture and started fresh100epochs: results/map_only_points_l1_final8_e100_20261114_000521. PID32460. Point outputs[B,128,2],1+8 blocks,16noise levels, no spline/control operations. See docs/direct_point_baseline.md for implementation, tests, smoke and run details. This overrides all prior old-control adapters/legacy-reuse proposals and pending questions. New seven-algorithm full test must wait for the new point base and small validation. Do not use old invalid results, old checkpoint or experimental direction-field/extra-initialization modifications in the new comparison.

--- Historical notes below ---

# Current work state / 2026-11-13

User objective: fix baseline bugs, validate a small batch, then run full comparison. Latest steering: reference Ours ALM control constraints.

Old full run results/seven_direct_full_t16_20261113_111204 completed 11340 records, but is INVALID for paper-method ranking. validity_assessment.json/md records the lifted point-state/output-spline mismatch. Original results preserved; summary marked invalid. Never reuse those records in a corrected run.

All training is completed and locked: map-only base1+8/100epochs/16noise levels and full Ours1+8/100epochs/16noise levels. Do not retrain or change to256 noise steps.

CurveControlTask removes residual lifting/output fitting and constrains the actual delivered48-control spline. Four fixed validation cases (indices0,516,1032,1549) were tested in results/seven_curve_control_validation*_20261113. Representation identity is fixed, but five CBF/projection branches did not complete a solve in these cases. PD/ALM complete but collide. Do not launch full merely because unit tests pass. Independent LP checks in docs/control_cbf_feasibility_audit.json found positive minimum feasibility slack for RoS/TVS constraints; coupled linearization really can be infeasible, not simply a constraint-count argument.

Solvers: exact elimination of ReS unconstrained relaxation avoids w100 conditioning; row normalization keeps QP unchanged; independent Farkas/phase-I feasibility verification distinguishes real infeasibility from OSQP nonconvergence. Projection phase-I initial guess uses only pretrained model/map; never replaces mandatory nearest-point/KKT checks, never uses A* or Ours ALM fallback. Campaign dump retries Windows file locks. New --small-validation and --from-small options provide validation-to-full without S2; full gate requires actual completed outputs for each external method, not collision-free outcomes. Final full gate must also check solver_execution_gate.

Latest addition: adapters/bezier_constraints.py consumes the exact BSplineConstraintPack from src/geometry/bspline_constraints.py. Converts A_j E_l,r Q <= b_j to G x <= d with fixed endpoints eliminated, uses independent convex QP projection, and verifies feasible set. No calling Ours optimizer. Tests compare exactly with Ours constraint_state and dense curve evaluation. Latest S0:44 tests passed. See docs/bezier_constraint_adapter.md.

Critical next integration issue: explicitly determine a valid, fair corridor source. Do not silently hand external methods Ours learned/searched corridor while calling them map-only. No new full job is running. All modifications confined to experiments/safety_baselines; preexisting user changes elsewhere preserved. No subagents permitted.

## Latest user boundary (overrides speculative fixes)
Only borrow the control-to-curve constraint mapping. Do NOT modify seven-paper algorithms, initial distribution, objective, update state, random budget, dual/slack/rho equations or add old correction loops. Old engineering full comparison is NOT requested.

Removed from active CurveControlTask: old penalty warm start, clean-model initial guess selection, phase-I penalty and bounds, numerical safety offset. The proposed extra8 diffusion prior draws was rejected by automatic review and was not applied/executed; do not retry it indirectly. Legacy direction-field experimental config is rejected by PaperTask dispatch; use actual scalar derivatives for paper methods. Only constraints are g(Q)=d_safe-d(BQ), Jacobian by chain rule, fixed endpoints. Projection retains its original Euclidean target and constraints. QP exact elimination/scaling preserve the mathematical problem; residual verification cannot turn a failed solve into a trajectory.

Latest unit tests46 passed. Current small batch: seven_constraint_mapping_only_validation_20261113, using configs/seven_reused_constraints_t16.json (actual old scalar SDF, no old correction methods). Other diagnostics are not formal paper results. New full run has NOT started. Need disclose all preexisting task adaptations (DDIM nominal for Safe, coarse16 legal score levels, control-space nonconvex constraints; cannot claim unchanged original benchmark).
