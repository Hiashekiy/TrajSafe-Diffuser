# TrajSafe-Diffuser 论文实验表格（自动汇总）

由 `scripts/experiments/summarize.py` 从各实验的原始结果生成；所有数字均来自实际运行，未做任何人工填写。

## 实验 1：不同安全规划方法对比与计算效率

| Method | Collision (%) | Min clearance (m) | Detour ratio | NRMSE (%) | Max curvature (1/m) | Sq. curvature (1/m2) | N |
|---|---|---|---|---|---|---|---|
| Unconstrained Diffusion | 61.19 | 2.899 | 1.0896 | 13.202 | 0.3313 | 0.00412 | 1260 |
| Fixed Penalty Guidance | 19.05 | 1.710 | 1.4665 | 12.066 | 1.1893 | 0.04596 | 1260 |
| Primal-Dual Guidance | 23.10 | 1.774 | 1.2337 | 11.904 | 0.7393 | 0.01764 | 1260 |
| SDF-ALM Guidance | 23.89 | 1.763 | 1.1679 | 11.905 | 0.5872 | 0.01154 | 1260 |
| TrajSafe-Diffuser | 0.00 | 3.255 | 1.3246 | 8.759 | 1.0869 | 0.02182 | 1260 |

**Table 1b — efficiency (reused historical interleaved timing, speed-up 3.03x)**

| Configuration | DDIM steps | Base regions | Time (ms/traj) | Collisions |
|---|---|---|---|---|
| Standard | 16 | 128 | 82.26 | 0/420 |
| Fewer steps | 4 | 128 | 51.41 | 0/420 |
| Shared regions | 16 | 17 | 56.44 | 0/420 |
| Combined | 4 | 17 | 27.14 | 0/420 |

## 实验 2：关键模块消融与历史反馈作用分析

**Table 2 — component ablation (A0–A3)**

| Config | Safety cond. | Hist. feedback | ALM update | Collision (%) | Min clearance (m) | Detour | NRMSE (%) | N |
|---|---|---|---|---|---|---|---|---|
| A0: Baseline | N | N | N | 61.19 | 2.899 | 1.0896 | 13.202 | 1260 |
| A1: + Safety | Y | N | N | 25.16 | 3.120 | 1.2278 | 7.911 | 1260 |
| A2: + Feedback | Y | Y | N | 3.33 | 3.225 | 1.3154 | 8.734 | 1260 |
| A3: Full Model | Y | Y | Y | 0.00 | 3.255 | 1.3246 | 8.759 | 1260 |

**Table 2b — paired process statistics, forward 5–16**

| Quantity | Without FB | With FB | Diff | Std. diff | Wilcoxon p | Pairs |
|---|---|---|---|---|---|---|
| Pre-correction violation (m) | 0.2749 | 0.1167 | -0.1582 | 0.6459 | 0.000 | 1260 |
| ALM correction (m) | 0.4609 | 0.2488 | -0.2121 | 0.4146 | 0.000 | 1260 |

## 实验 3：ALM 迭代预算与安全性-计算效率权衡

**Table 3 — ALM iteration budget: safety and trajectory quality**

| Config | Continuous feasible (%) | Collision (%) | Collisions/N | Min clearance (m) | Detour | NRMSE (%) | Max curvature (1/m) | Sq. curvature (1/m2) |
|---|---|---|---|---|---|---|---|---|
| I2 | 79.60 | 0.24 | 3/1260 | 3.221 | 1.3139 | 8.576 | 1.0349 | 0.02087 |
| I4 | 95.32 | 0.00 | 0/1260 | 3.315 | 1.3209 | 8.642 | 1.0499 | 0.02068 |
| I6 | 98.10 | 0.16 | 2/1260 | 3.329 | 1.3216 | 8.659 | 1.0471 | 0.02065 |
| I8 | 98.57 | 0.00 | 0/1260 | 3.321 | 1.3211 | 8.668 | 1.0406 | 0.02057 |
| I10 | 99.13 | 0.00 | 0/1260 | 3.318 | 1.3212 | 8.668 | 1.0436 | 0.02058 |
| Default | 98.81 | 0.00 | 0/1260 | 3.319 | 1.3209 | 8.668 | 1.0397 | 0.02055 |

**Table 3b — cost and solver diagnostics**

| Config | Mean iters | P95 iters | At budget (%) | ALM ms/call | E2E ms (b=32, med) | E2E ms (b=1, mean) | E2E ms (b=1, med) | E2E P95 (b=1) |
|---|---|---|---|---|---|---|---|---|
| I2 | 2.000 | 2.00 | 27.8 | 25.10 | 65.69 | 242.24 | 241.01 | 252.82 |
| I4 | 3.730 | 4.00 | 16.5 | 26.29 | 70.44 | 306.05 | 302.75 | 325.01 |
| I6 | 5.257 | 6.00 | 9.9 | 31.56 | 73.21 | 303.04 | 299.76 | 326.98 |
| I8 | 6.583 | 8.00 | 7.2 | 39.67 | 75.89 | 302.97 | 298.44 | 332.03 |
| I10 | 7.762 | 10.00 | 8.0 | 45.51 | 78.59 | 301.91 | 298.95 | 329.65 |
| Default | 6.811 | 10.00 | 7.1 | 37.95 | 77.11 | 300.24 | 296.72 | 320.45 |

**Table 3c — paired bootstrap vs Default (95% CI, resampled by task)**

| Config | Quantity | Mean diff | CI low | CI high | Tasks |
|---|---|---|---|---|---|
| I2 | constraint_feasible | -0.1921 | -0.2278 | -0.1579 | 420 |
| I2 | collision | 0.0024 | 0.0000 | 0.0071 | 420 |
| I2 | detour_ratio | -0.0070 | -0.0086 | -0.0054 | 420 |
| I2 | nrmse_pct | -0.0911 | -0.1339 | -0.0509 | 420 |
| I4 | constraint_feasible | -0.0349 | -0.0548 | -0.0198 | 420 |
| I4 | collision | 0.0000 | 0.0000 | 0.0000 | 420 |
| I4 | detour_ratio | -0.0000 | -0.0007 | 0.0008 | 420 |
| I4 | nrmse_pct | -0.0256 | -0.0506 | -0.0042 | 420 |
| I6 | constraint_feasible | -0.0071 | -0.0167 | 0.0024 | 420 |
| I6 | collision | 0.0016 | 0.0000 | 0.0048 | 420 |
| I6 | detour_ratio | 0.0007 | 0.0003 | 0.0014 | 420 |
| I6 | nrmse_pct | -0.0090 | -0.0246 | 0.0019 | 420 |
| I8 | constraint_feasible | -0.0024 | -0.0111 | 0.0040 | 420 |
| I8 | collision | 0.0000 | 0.0000 | 0.0000 | 420 |
| I8 | detour_ratio | 0.0002 | 0.0001 | 0.0004 | 420 |
| I8 | nrmse_pct | 0.0003 | -0.0001 | 0.0010 | 420 |
| I10 | constraint_feasible | 0.0032 | 0.0000 | 0.0079 | 420 |
| I10 | collision | 0.0000 | 0.0000 | 0.0000 | 420 |
| I10 | detour_ratio | 0.0003 | -0.0001 | 0.0010 | 420 |
| I10 | nrmse_pct | 0.0007 | -0.0011 | 0.0026 | 420 |

