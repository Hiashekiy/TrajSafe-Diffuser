# 表 1 不同安全规划方法的安全性与轨迹质量

| method | collision_pct | clearance_m | detour_ratio | nrmse_pct | kappa_max_inv_m | kappa2_mean_inv_m2 | n | clearance_valid_n |
|---|---|---|---|---|---|---|---|---|
| Unconstrained Diffusion | 61.19 | 2.899 | 1.09 | 13.2 | 0.3313 | 0.004119 | 1260 | 489 |
| Fixed Penalty Guidance | 19.05 | 1.71 | 1.467 | 12.07 | 1.189 | 0.04596 | 1260 | 1020 |
| Primal-Dual Guidance | 23.1 | 1.774 | 1.234 | 11.9 | 0.7393 | 0.01764 | 1260 | 969 |
| SDF-ALM Guidance | 23.89 | 1.763 | 1.168 | 11.91 | 0.5872 | 0.01154 | 1260 | 959 |
| TrajSafe-Diffuser | 0 | 3.255 | 1.325 | 8.759 | 1.087 | 0.02182 | 1260 | 1260 |
