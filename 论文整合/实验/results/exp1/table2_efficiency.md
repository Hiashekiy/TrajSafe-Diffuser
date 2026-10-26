# 表 2 完整方法的采样步数与凸区域共享效率（沿用既有测时）

| configuration | ddim_steps | base_regions | time_ms | collisions | n | feasible_rate |
|---|---|---|---|---|---|---|
| Standard configuration | 16 | 128 | 82.26 | 0 | 420 | 1 |
| Fewer sampling steps only | 4 | 128 | 51.41 | 0 | 420 | 1 |
| Shared convex regions only | 16 | 17 | 56.44 | 0 | 420 | 1 |
| Combined acceleration | 4 | 17 | 27.14 | 0 | 420 | 1 |
