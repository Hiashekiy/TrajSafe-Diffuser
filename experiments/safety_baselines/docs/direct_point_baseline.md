# Direct-point comparison base / 2026-11-14

User explicitly replaced the comparison base representation with direct trajectory points. Full Ours stays unchanged. Old control-space adapter debugging is stopped. No full seven-method comparison is launched before this new base is trained and validated.

Architecture: map_only_point_diffuser_v1, map/endpoints only;128 trajectory tokens with2D coordinates;1 backbone block and8 final denoiser blocks, hidden128,4 heads, map memory16x16. Parameters4,366,146. Output trajectory[B,128,2] is delivered directly. No BSplineCodec, control prediction, spline decode, least-squares fit, learned safety features, skeleton, topology, corridor or correction-history input. Fixed parameter-free endpoint profile is retained and exact endpoint pinning applied.

Data loader reads only curve_gt.npy (the128-point targets already used as pos in previous evaluation), occupancy.npy and conditions.npy. It does not read control_gt or any candidate/safety labels. The target cache is reused as point supervision; no spline operation occurs in training or sampling. Pointwise x0/coarse MSE, finite-difference smoothness and boundary losses use the same mathematical forms and weights as the old common losses, applied directly to trajectory points.16-level diffusion schedule, AdamW2e-4, batch16 and100epochs unchanged. Fresh initialization, no loading old control checkpoint.

Files: models/map_only_points.py,models/point_data.py,models/point_losses.py,adapters/point_sampler.py,scripts/train_map_only_points.py,configs/map_only_points_l1_final8_e100.json.

Validation:2 targeted tests passed: forward/backward gradients,128-point output, exact endpoints, deterministic point DDIM, strict checkpoint round trip, absence of spline/control attributes, and a minimal dataset with no control or skeleton files. Real CUDA smoke:32train/32val samples,2training batches,1epoch completed with finite loss and saved checkpoints.

Full fresh100epoch training started: results/map_only_points_l1_final8_e100_20261114_000521. First epoch completed26.9s; actual throughput may vary. Training3850 and validation1550 samples. Existing Ours and old model/results not overwritten. Model training is in progress, not completed.

Next: load only the point architecture checkpoint into new paper adapters; optimize and return the same128points. Preserve the paper algorithms. Independent evaluation must check the connecting trajectory as well as vertices, without fitting to a B-spline or introducing auxiliary residual variables. Old invalid comparison records cannot be reused.

## Training and native validation update
Training completed100/100 epochs in2455.05s. Best task epoch96 (zero based). NativePointTask now directly consumes/returns128points with exact nearest free-cell-union point projection and an independently tested exact separable solution of the same CBF QP. VP/VE score conversion and pinned endpoints tested. No spline, fitting, residual lifting, additional prior samples or old correction routines. Twenty fixed validation scenes, one seed, eight methods (seven paper methods plus plain point Diffuser) are running in results/native_points_validation_20261114_011528.25 focused tests passed. Constraints act on the128vertices; evaluation separately checks vertices and every cell traversed by the connecting polyline. This is not a continuous-path guarantee of the constraint itself. No full campaign has launched.
