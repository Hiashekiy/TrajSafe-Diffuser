# PDG-Focal A* 使用说明（供后续调用）

栅格搜索模块：**用扩散模型逆扩散早期的中间 x0 作软引导，加速安全栅格上的 A\* 搜索**。
不依赖、也不修改 `src/diffusion`、`src/models`、`train.py`、`sample.py`、`evaluate.py`。

- 算法与实验结论：`docs/PDG_Focal_Astar_实验报告.md`
- 项目接口核查（地图语义、坐标系、x0 从哪取）：`docs/PDG_Focal_Astar_S0_接口核查.md`
- 端点朝向约束设计（扩散模型侧，已定稿、尚未实施）：`docs/PDG_Focal_Astar_端点朝向约束设计.md`
- 试验脚本：`scripts/grid_search/`
- 结果与图：`outputs/grid_search/`（主结果在 `run_pdg/`）

---

## 1. 最小调用（3 步）

```python
import numpy as np
from src.grid_search import (
    GridSpec, build_safe_grid, build_neighbor_table,
    prepare_reference_path, build_progress_guidance, baselines, validate_path)

# ① 地图 + 安全栅格（occupancy: [H,W] uint8，0=free 1=obstacle，row0 = scene_y=-1）
spec = GridSpec.from_resolution(256, 256, 256, scene_to_meter=80.0)  # cell=0.625 m
safe = build_safe_grid(occupancy, spec,
                       robot_radius_m=0.8, safety_margin_m=0.3,
                       already_inflated_cells=4.0)   # 数据侧已腐蚀 4 格时可填 4
nbr  = build_neighbor_table(safe.free, spec.cell_m)   # 8 邻域 + 禁穿角

# ② 扩散参考轨迹（scene 坐标 [-1,1] 的折线，例如 x0 解码出的 128 点）
ref   = prepare_reference_path(x0_curve_world, spec, start_world, goal_world)
field = build_progress_guidance(ref, spec, J=3)        # 全图每格 3 个投影候选

# ③ 搜索
s = int(spec.cell_to_index(spec.world_to_cell(start_world)))
g = int(spec.cell_to_index(spec.world_to_cell(goal_world)))
res = baselines.pdg_focal_astar(safe, s, g, field, w=1.2)   # Ours
# res.status ∈ {SUCCESS, NO_PATH, INVALID_START, INVALID_GOAL}
# res.path → [n] 栅格索引；res.cost_m → 真实路径长度（米）；res.stats → 统计
check = validate_path(safe, res.path, s, g, res.cost_m)     # 独立复核，必须跑
```

只要 dxf 出一条路径？把 `pdg_focal_astar` 换成 `baselines.astar(safe, s, g, nbr)`。
参考轨迹无效（空 / NaN / 越界 / 退化）时 `pdg_focal_astar` **自动退化为无引导
Focal A\***（`fallback_secondary="clearance"`），不会报错也不会规划失败。

---

## 2. 核心对象

| 对象 | 作用 | 关键字段 |
|---|---|---|
| `GridSpec(h, w, cell_m, scene_to_meter)` | 栅格 ↔ 场景坐标 | `cell_m`、`world_to_cell()`、`cell_to_world()`、`cell_to_index()`、`index_to_rc()` |
| `build_safe_grid(occ, spec, ...)` → `SafeGrid` | 占据语义 + 障碍膨胀 | `free[H,W]`、`clearance_m[H,W]`、`inflation_cells`、`notes` |
| `build_neighbor_table(free, cell_m)` → `NeighborTable` | 8 邻域表（对角需两侧自由） | `idx[N,8]`（-1 表示不可走）、`cost[8]` |
| `prepare_reference_path(pts_world, spec, start, goal)` → `ReferencePath` | 端点校正、去重、弧长重采样、共线合并 | `points`(cell 坐标)、`length_m`、`valid`、`reason`、`diagnostics` |
| `build_progress_guidance(ref, spec, J=3)` → `GuidanceField` | 全图最近 J 个线段投影 | `dist_m[N,J]`、`prog[N,J]`、`n_segments`、`coverage`、`build_ms` |
| `SearchResult` | 搜索结果 | `status`、`path`、`cost_m`(真实长度)、`stats`、`expanded`、`snapshots` |

坐标约定（与 `evaluate.py` 完全一致）：
`world_to_cell(p) = (p+1)/2*res - 0.5`；`col = x`、`row = y`、`row 0 ↔ scene_y = -1`；
`cell_m = 2*scene_to_meter/res`（160 m 裁剪 / 256 格 = 0.625 m）。

---

## 3. 方法一览（`baselines.run_method(name, ...)`）

| name | 说明 | 额外参数 |
|---|---|---|
| `dijkstra` | 无启发式 | — |
| `astar` | octile 启发式，最优 | — |
| `focal_clearance` | Focal A\*，FOCAL 内按"障碍间隙大→f0→索引" | `w` |
| `focal_index` | Focal A\*，FOCAL 内按稳定索引（弱基线） | `w` |
| **`pdg`** | **Ours：FOCAL 内按 `H_D`（距离+进度）排序** | `w`、`alpha_d`、`alpha_p`、`sigma_m`、`L_ref_m` |
| `guided_astar` | 【已评估、不采用】把引导项直接加进 g 的融合版 | `lam`（权重）、`alpha_p` |

默认参数：`w=1.2`（FOCAL 阈值）、`alpha_d=1.0`、`alpha_p=1.0`、
`sigma_m=3*cell_m`、`L_ref_m=参考轨迹长度`、`J=3`。

**`guided_astar` 为什么不用**：把惩罚加进 g 后 octile 启发式不再解释剩余代价
（λ=1 时惩罚可达路径本身的数倍），A\* 退化成 Dijkstra 且进度项让代价依赖已走距离，
破坏了"每格展开一次"的前提。实测 100 个 CARLA case：A\* 355 格 → 融合版 954~3131 格，
路径还长 0.7~4.3%；难场景上融合版最好 1644 格（A\* 2270、Ours 1384）。详见报告 §7.6。

---

## 4. 批量评估（自己造 case 也用它）

```python
from src.grid_search.benchmark import (Case, prepare_case, evaluate_case,
                                       aggregate, write_csv)

case = Case("demo", occupancy, spec, start_world, goal_world, ref_world,
            meta={"sample": 0})
prep = prepare_case(case, robot_radius_m=0.8, safety_margin_m=0.3,
                    already_inflated_cells=4.0)
rec  = evaluate_case(prep, "pdg", "pdg", w=1.2, alpha_d=1.0, alpha_p=1.0)
# rec 里是可直接进 CSV 的扁平字段：expanded_nodes / search_time_ms /
# path_cost_m / path_valid / guide_penalty_m / guidance_build_time_ms / ...
```

`prepare_case` 内部会一次性算好安全图、邻域表和引导场，并分别计时
（`prep.timings = {preprocess_ms, reference_ms, guidance_ms}`），批量评估时复用同一个
`prep` 跑多个方法即可。

---

## 5. 现成脚本（`scripts/grid_search/`）

| 脚本 | 用途 |
|---|---|
| `extract_diffusion_x0.py` | 用现有 ckpt 跑纯 DDIM，缓存 K∈{1..5} 的中间 x0 曲线 |
| `run_pdg_experiment.py` | 主对比 + G0/G1/G2 消融 + K/w/α_p 扫描 + 合成场景 |
| `summarize_pdg_results.py` | 由 `records_*.csv` 生成 markdown 表（`REPORT_TABLES.md`） |
| `plot_paths_and_cost.py` | **论文用两张图**：路径对比 + 代价对比 |
| `plot_pdg_results.py` | 统计图：主对比/消融/K/w/α_p |
| `run_hard_suite.py` | 难场景（迷宫/房间/螺旋/死胡同/长墙）× 4 种参考质量 |
| `run_fused_guidance.py` + `plot_fused_tradeoff.py` | 融合代价 λ 扫描（对照实验，已弃用） |

```bash
# 抽取 x0（不训练、不改模型）
python scripts/grid_search/extract_diffusion_x0.py \
    --config configs/config_160k4p_c48_oneshot.yaml \
    --ckpt outputs/models/oneshot_k4p_c48/ckpt/best_task.pt \
    --split test --limit 420 --out outputs/grid_search/diffusion_x0/test

# 主实验
python scripts/grid_search/run_pdg_experiment.py \
    --cache outputs/grid_search/diffusion_x0/test/diffusion_x0_test_0_420.npz \
    --processed data/processed/160m-k4p-c48 --split test \
    --limit-main 200 --limit-sweep 80 --K 3 --w 1.2 \
    --out outputs/grid_search/run_pdg

# 两张图 / 难场景 / λ 对照
python scripts/grid_search/plot_paths_and_cost.py \
    --cache outputs/grid_search/diffusion_x0/test/diffusion_x0_test_0_420.npz \
    --run outputs/grid_search/run_pdg --samples 125 168 47
python scripts/grid_search/run_hard_suite.py --out outputs/grid_search/run_hard
python scripts/grid_search/run_fused_guidance.py \
    --cache outputs/grid_search/diffusion_x0/test/diffusion_x0_test_0_420.npz \
    --limit 100 --out outputs/grid_search/run_fused
```

---

## 6. 使用注意

1. **引导只用来排序，不改代价**：`pdg` 不进入 `g`、边权或 FOCAL 阈值，路径仍受
   `w·C*` 有界次优保护；想改代价请用 `guided_astar`，但结论是更慢更绕。
2. **必须跑 `validate_path`**：它是独立复核（越界/障碍/边合法/穿角/代价复算），
   实验里 `path_valid` 必须为 1 才算成功。
3. **`cost_m` 是重建路径的真实长度**；`stats["search_cost_m"]` 是 Focal 界对应的 g
   （目标可能比另一个 FOCAL 成员先被扩展，两者会差一点，报告里分列）。
4. **无效参考不报错**，退化成 `focal_clearance`；想强制 A\* 就显式调用 `astar`。
5. **障碍膨胀不要重复算**：数据侧已腐蚀（`05_erode_occupancy.py`，k=4）时把
   `already_inflated_cells=4` 传进去，否则会要求两次安全余量。
6. **引导场构建约 20–25 ms/次**（256×256，全图精确投影），单次查询端到端会因此比
   A\* 慢；可复用时（同一张图/同一段参考多次查询）才划算 —— 报告 §0 有实测。
7. 单元测试：`python -m pytest src/grid_search/tests -q`（54 passed）。

---

## 7. 目录

```
src/grid_search/
  grid_map.py          占据语义、GridSpec、欧氏膨胀、clearance
  grid_neighbors.py    8 邻域表、禁穿角
  diffusion_path.py    x0 → 栅格折线（端点校正/去重/重采样/共线合并）
  progress_guidance.py 全图 J=3 线段投影 + H_D 打分
  focal_astar.py       anchor A* + 动态 FOCAL + 次排序（含 λ 融合开关）
  baselines.py         dijkstra / astar / focal_* / pdg / guided_astar
  path_validation.py   独立复核、代价复算、转弯数、间隙
  visualization.py     单 case 8 类过程图
  scenes.py            T1–T12 合成验收场景
  hard_scenes.py       迷宫/房间/螺旋/死胡同/长墙 + 4 种参考质量
  benchmark.py         Case/PreparedCase、批量评估、聚合、CSV
  tests/               54 个单元测试
```
