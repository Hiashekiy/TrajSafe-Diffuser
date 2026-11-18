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

## 实验 3：ALM 迭代预算对安全约束修正效果的影响

**Table 4 — ALM iteration budget vs constraint-correction quality** (main set: calls whose cached input already violated the corridor)

| Max iters | Feasible calls (%) | 95% CI | Residual violation (scene) | P95 | Correction (m) | Actual iters | Calls |
|---|---|---|---|---|---|---|---|
| I1 | 32.20 | [26.71, 37.92] | 0.00697 | 0.00999 | 0.35936 | 1.00 | 5694 |
| I2 | 50.32 | [44.69, 56.02] | 0.00266 | 0.00460 | 0.40980 | 1.68 | 5694 |
| I4 | 87.82 | [84.18, 90.91] | 0.00056 | 0.00130 | 0.42160 | 2.45 | 5694 |
| I6 | 95.01 | [93.14, 96.65] | 0.00048 | 0.00110 | 0.41893 | 2.65 | 5694 |
| I8 | 96.92 | [95.49, 98.11] | 0.00044 | 0.00098 | 0.41787 | 2.73 | 5694 |
| I10 | 97.38 | [96.01, 98.51] | 0.00042 | 0.00090 | 0.41725 | 2.79 | 5694 |

Coverage: cached calls 16380 (all valid), baseline-satisfied 65.24%, main set 5694 calls / 183 scenes / 515 trajectories; parity check max |diff| = 0.0.

Model: `D:\ProjectDirectory\Neural-IRISDiffuser\outputs\models\v4_map16_c48\ckpt\best_task.pt` (sha256 `614c9208559c6af7`), branch `v4-map16-fbdelta-safeloss`.

