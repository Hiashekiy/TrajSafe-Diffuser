# 路径搜索性能瓶颈审计（v5 · PDG-Focal A*）

**日期**：2026-11-18
**对象**：`configs/config_v5_c32q128.yaml` + `outputs/models/v5_c32q128/ckpt/final_best.pt`
**口径**：test split、batch=1、`steps=1`（只跑第 1 步 → 只触发搜索分支，隔离掉网络与 ALM）、
RTX 4070、`E:\CondaEnvData\envs\GGMPC`（torch 2.9.1+cu126 / scipy 1.15.3 / skimage 0.25.2）。
**复现**：`scripts/debug/profile_search_bottleneck.py`、`bench_search_overheads.py`、
`profile_search_focal_band.py`、`exp_focal_threshold.py`、`exp_geodesic_heuristic.py`。

审计结果先说一句：**搜索慢不是"某一段代码写慢了"，而是两件事叠加**——
26% 的样本扩展了 2 千～5 万个节点（吃掉 75% 的搜索时间），
而每扩展一个节点要付 ~66 µs 的 Python/numpy 代价（其中 55 µs 是引导场打分）。
中位样本（221 个节点）则几乎全是**每样本固定开销**：32.5 ms 的栅格/邻居表/居中。

自检：本次测得的扩展中位数 **221**，与 `new_framework_FINAL_SUMMARY.md` 报的 220.5 一致，
说明测的就是线上那条链路。

---

## 1. 全 test（418 样本）分布

| 指标 | p50 | p90 | p99 | max |
|---|---|---|---|---|
| `expanded`（扩展节点数） | **221** | 4 764 | 31 411 | **52 073**（= 82% 个网格） |
| `search_ms`（prepare+引导+搜索+居中） | **52.2 ms** | 434 ms | 2 613 ms | 3 783 ms |
| `grid_ms`（安全栅格 + 邻居表） | 18.7 ms | 22.5 ms | — | — |
| `guidance_score` 累计 | 21.2 ms | 235 ms | 1 399 ms | 2 027 ms |

* **搜索占 `grid+search` 的 91%（均值）**；中位样本 52.2 ms，占整链路 111.6 ms 的 ~47%。
* **双峰分布**：`expanded > 2000` 的样本 **109 / 418 = 26%**，却吃掉 **75% 的搜索时间**（62.4 s / 82.9 s）。
* `guide_valid` 418/418 全为 True、`fallback` 全空 → **"引导失效退化到 clearance 打分"这条路径完全没发生**，不是慢的原因（这个假设我一开始也怀疑过，实测否掉）。

---

## 2. 瓶颈逐项（按贡献排序）

### B1 — 引导场打分：占搜索时间 54%（中位样本 41%）

`src/grid_search/lazy_guidance.py:83 score()`

* **每次扩展调用一次**（418 样本共调用 843k 次 = 平均每样本 2 021 次 ≈ 扩展次数）。
* 每次传入的是**整个 FOCAL 带**：中位 264 个格、最大 736 个格；
  全 test 累计查询 **155.9 M 个格**，而真正需要算距离的（缓存未命中）只有 **0.73 M 个格**
  → **99.5% 的打分是在重算"距离已缓存、只是 u_g 变了"的格**。
* 单次实测（真实地图、128 段参考）：

  | 带大小 | 冷（要填缓存） | 热（全命中） |
  |---|---|---|
  | 264 | 1 534 µs | **25.8 µs** |
  | 736 | 2 370 µs | **48.0 µs** |
  | 3 000 | 19 623 µs | 187.7 µs |

  也就是说：**每新增一个格要付 5.8 µs**（算 127 段只留 J=3，见 B5），
  而**每个扩展节点固定要付 ~26–48 µs** 的重打分。

### B2 — 扩展数爆炸：26% 样本吃掉 75% 时间

发现两个叠加的机制，都有实测证据。

**(a) FOCAL 阈值 `thr = w · f_min` 是正反馈**（`focal_astar.py:286`，`w=1.2`）

`f_min` 是**活的** open 集最小 f0。引导规则偏偏会挑 `f0 > f_min` 的节点（这正是 FOCAL 的用途），
于是低 f0 节点被慢慢消耗、`f_min` 一路抬高，阈值 `1.2·f_min` 跟着抬高，放进来的节点更多 → 自我放大。
样本 71008 的逐次扩展记录（`profile_search_focal_band.py`）：

* `f_min`：**167.8 m → 202.9 m**（+21%，单调上爬）
* **93.2%** 的扩展选了 `f0 > 1.01·f_min` 的节点，中位"松弛" **10.25 m**（≈16 格绕路）
* 结果：扩展 53 826 次 = **82.1% 的整张网格**、动到 56.6% 的自由格

**(b) octile 启发式松 ~20%，`f0 == C*` 的平台极大**

`focal_astar.py:245` 用 octile 直线距离当 h，但路径必须绕障碍：C* ≈ 202.9 m 而 h(start)=167.8 m（松 21%）。
后果是"次优代价带"很宽，**即使纯 A* 也要 12 944 次扩展**（同一样本，见 §3 对照）。
所以 FOCAL 的 4.3× 放大是在一个本来就很贵的平台上再乘一次。

### B3 — 每节点的固定 Python/numpy 开销 ~66 µs

cProfile（样本 71008，52 073 次扩展）：`focal_search` 自身 1.60 s、`score` 2.84 s、`_tiebreak_pick` 0.48 s。

| 组成 | 每次扩展 | 说明 |
|---|---|---|
| `guidance.score` | ~55 µs | 见 B1（26–48 µs 只是热态重算） |
| `_tiebreak_pick` | ~9 µs | `min`+`flatnonzero`+`lexsort` |
| 循环体自身 | ~31 µs | 每轮对 open 数组做 4–6 趟全量 numpy 操作（`np.where`/`.min()`/布尔索引/`in_open[...].sum()` 统计） |

同样的迭代量，用标量堆实现的 A* 大约 2–5 µs/节点 → **这里还有 10× 以上的余量**。

### B4 — 每样本固定开销 32.5 ms（决定中位样本，与扩展数无关）

| 项 | 实测（中位/样本） | 现状 | 可做到 |
|---|---|---|---|
| `build_neighbor_table`（`grid_neighbors.py:44`） | **12.2–12.6 ms** | 每格 8 次 move、每次十几趟 65 536 元素 numpy + 多个 int64 大临时数组 | **1.42 ms**：2D 切片 padded 实现，输出**逐位相同**（已实测 `identical=True`） |
| 4 次 `scipy EDT` | 9.8 ms | `build_safe_grid` 2 次（膨胀 + clearance_m，都必要）＋ `center_path` 2 次 | 第 4 次是 `path_centering.py:69-70` 的 `distance_transform_edt(~free, return_indices=True)`，**`iy/ix` 从头到尾没被用过 → 纯浪费 2.06 ms** |
| `center_path`（含上项） | 12.8 ms | 40 迭代 × 3 次 `_bilinear`；单次调用 **66 µs**（128 点上的 ~15 个 numpy 小算子） | 三个场（clear/gx/gy）合并成一次 gather → 122 次调用降到 ~41 次，省 ~4–5 ms |
| `prepare_reference_path` | 1.2 ms | — | 可不做 |

### B5 — 引导场冷填充 42× 冗余

`lazy_guidance.py:55 _values()` 对 `N × 127` 段全算距离，再 `argpartition` 只留 `J=3` 列。
按全 test 计：算 13.5 M 次"点到线段"只为保留 0.32 M。改成"先粗定位最近段、只精算 ±k 段"，或直接用
`center_path` 已经算过的 EDT 索引做投影，可以把冷填充从 5.8 µs/格压到 ~0.2 µs/格。

---

## 3. 已实测的修复 A/B（全部在真实样本上，非估算）

### 3.1 `focal_w: 1.2 → 1.0`（= 留引导，但只在精确 f0 平台上做 tie-break）

33 个样本（`--limit 420 --stride 13`，覆盖全 split）：

| 变体 | 扩展 p50 | 总耗时 | 路径代价 / 纯 A* 最优 | 最差 |
|---|---|---|---|---|
| 现状 `w=1.2` | 860 | **16.52 s** | 1.022 | **1.065**（长 6.5%） |
| `w=1.0` + 引导 tie-break | 958 | **6.68 s**（2.47×） | **1.000** | 1.000 |
| 纯 A*（对照上限） | 1 644 | 2.47 s（6.68×） | 1.000 | 1.000 |

**`w=1.2` 是被支配的**：既慢 2.5×，路径还比最优长 2.2%（最差 6.5%）。
注意到纯 A* 的扩展数是 `w=1.0`+引导的 1.7×，说明**引导 tie-break 本身是有用的**（省平台探索），
真正有害的是 `w=1.2` 那个会自我放大的阈值。

### 3.2 启发式 octile → 精确测地场（`skimage.graph.MCP_Geometric`）

12 个重样本（含 8 个 >1 万扩展的）：

| h | 扩展 p50 | 循环总耗时 | 路径代价 |
|---|---|---|---|
| octile | 10 836 | 8.35 s | 196.8 |
| **测地（MCP）** | **631** | **1.20 s**（7×） | 196.8（**逐位相同**） |

MCP 场构造成本 **2.5–8.4 ms/样本**（均值 5.7），且它在本网格的移动代价下**可采纳且一致**
（MCP 允许对角穿角、我们的规则不允许 → MCP 距离 ≤ 我们的最优 → 仍是合法下界）。
个别样本仍剩 2 400–4 000 扩展，那是对角穿角带来的乐观偏差，可再用"禁穿角"掩码收紧。

### 3.3 组合效果（12 个重样本，实测相加）

| 阶段 | 总耗时 |
|---|---|
| 现状（w=1.2 + octile） | **45.6 s** |
| w=1.0 + 引导 tie-break | 13.9 s（−70%） |
| 再把 h 换成测地场 | **1.2 s + 0.07 s 建场 ≈ 1.3 s（−97%）** |

### 3.4 被否掉的方向（有实测，别再做）

* **静态阈值 `w · h(start)`**：想用"不涨的阈值"掐死正反馈，但 h(start) 只有最优的 82%，
  1.2×h(start) 常常**小于最优代价** → 33 个样本里 **9 个直接 NO_PATH**。不可用。
* 引导场"批量多边形化"：`FINAL_SUMMARY.md §5` 已实测只有 1.0–1.4×，不复述。

---

## 4. 建议落地顺序（按风险分级）

**P0 零风险（输出逐位相同，只改实现）**
1. `build_neighbor_table` 换 2D 切片的 padded 实现：**−11.2 ms/样本**（已验证 `identical=True`）。
2. 删掉 `path_centering.py:69-70` 那个没人用的 `distance_transform_edt(return_indices=True)`：**−2.06 ms/样本**。
3. `center_path` 里 clear/gx/gy 三次 `_bilinear` 合成一次：**−4~5 ms/样本**。
   → 合计 **每样本省 ~17–18 ms**（≈ 111.6 ms 中位延迟的 16%）。

**P1 参数级（会改变 `active_path`，必须重跑 eval 复核 0 碰撞 + RMSE）**
4. `grid_search.focal_w: 1.2 → 1.0`：搜索 **2.5×**、路径代价同时变好（1.022 → 1.000）。
   `FINAL_SUMMARY.md` 里"focal_w=1.2"的表述要一起改。

**P2 启发式（同样要重跑 eval；收益最大，主要吃尾部）**
5. h 换成测地场（MCP，+3–8 ms/样本）：重样本扩展 ÷17，p90/p99 延迟塌缩。
   建议**先按需触发**（例如先用 octile 跑，`expanded > 800` 时再建场重跑一次），中位样本零成本。

**P3 重写内环（要回归测试：扩展数/路径代价不变）**
6. 内环改标量堆 A*（2–5 µs/节点）或整批向量化 focal；顺带把 `score` 的"热门格"缓存改成
   `j* = argmin_j d_j` 预存，热态从 26–48 µs 降到 ~5 µs。**注意**：`FINAL_SUMMARY.md §5` 已经点过
   open-list 维护“风险较高、要先加扩展数/代价不变的回归测试”，`src/grid_search/tests/` 里已有
   `test_focal_properties.py` 等 11 个用例可以当护栏。

**建议不做**：`J`（3）的调整——它只影响冷填充的 `argpartition`，对总时间影响 <2%；
以及张罗"引导失效回退"——418/418 样本根本没走那条路。

---

## 5. 一句话总结

* 想要**中位延迟**下来（111.6 → ~94 ms）：只能靠 P0 的固定开销（邻居表 12.6→1.4 ms、死代码 EDT、居中向量化）。
* 想要**尾部**下来（p90 581 ms / p99 2.6 s）：P1 + P2 是主力，
  实测 12 个重样本可以从 45.6 s 压到 ~1.3 s；P3 再把每节点的 66 µs 压到 5–10 µs，是最后的 5–10×。

---

# 第二部分：实施结果（2026-11-18 已落地）

## A. 改了什么

| # | 位置 | 改动 | 性质 |
|---|---|---|---|
| 1 | `grid_neighbors.build_neighbor_table` | flat 版 → **2D 切片 + padded mask** | 逐位相同，12.6 → **1.31 ms** |
| 2 | `grid_map.inflate_obstacles / SafeGrid` | 新增 `return_distance=True`、`raw_clearance_m` | 逐位相同，省掉居中里重复的 EDT（~2.6 ms） |
| 3 | `path_centering` | 删掉没人用的 `distance_transform_edt(return_indices=True)`；clear/gx/gy 三次 `_bilinear` → `_bilinear3` 一次 gather；支持传入 `clearance` | 逐位相同，10.2 → **~4.5 ms** |
| 4 | **新** `grid_search/geodesic_heuristic.py` | `skimage.graph.MCP_Geometric` 精确测地距离当 h（可采纳 + 一致） | 改变扩展顺序，路径代价不变 |
| 5 | `focal_astar` | 新增 `secondary="guidance_h"`（progress-first，扩散分只在近并列里用）+ `h_tie_tol_cells`；`w==1` 派发到堆后端 | 算法级 |
| 6 | **新** `grid_search/focal_fast.py` | 堆 + 标量 tie-break + 批量冷填充（`w==1` 的后端） | 等价于旧内环（测试逐位比对） |
| 7 | `configs/config_v5_c32q128.yaml` | `heuristic: geodesic`、`secondary: guidance_h`、`h_tie_tol_cells: 0.0`；**`focal_w` 保持 1.2** | 参数级 |

`focal_w` **没有改**：换上强启发式后 band 不再是瓶颈（w 扫描 1.2/1.4/1.6 扩展数与路径长度完全一致）。

## B. 怎么验的

| 验证 | 结果 |
|---|---|
| `tests/test_grid_search_hotpath.py`（新） | 邻居表 6 种掩码 / corner-cut / allow_corner_cut 与旧实现 **逐位相同**；`_bilinear3` 与逐通道 **逐位相同**；`center_path` 与旧实现 **逐位相同**；共享 EDT **逐位相同**且不改动传入数组 |
| `tests/test_focal_fast.py`（新） | 堆后端与向量化后端 **路径代价相同**（w=1 最优）、路径合法（无穿角/无碰撞）、扩展数同量级；w>1 不走堆；测地 h 不增加扩展；无效引导回退 clearance；端点阻塞报告为 INVALID_* |
| `src/grid_search/tests`（原有 54 个） | 全过，含 `test_heap_focal_equals_naive_full_scan`（堆 = 朴素全扫描，**扩展数完全相同**） |
| 全仓 `pytest tests src/grid_search/tests` | **263 passed / 13 skipped / 1 error**（该 error 是 matplotlib 测试的临时目录权限，与本改动无关） |
| 全 test 420 样本 eval | **碰撞 0 / 420**（128 点与 512 点都为 0），端点误差 0，RMSE 0.0943 |

## C. 数字

**C1 同进程配对 A/B（`scripts/debug/exp_search_configs.py`，80 样本，同网格同输入）**

| 配置 | 总耗时 | p50 | p90 | p99 | max | 扩展 p50 | 路径长 p50 |
|---|---|---|---|---|---|---|---|
| 旧：w1.2 + octile + guidance | 11.32 s | 23.6 | 506.7 | 1168.6 | 1358.6 | 143 | 87.8 m |
| **新：w1.2 + geodesic + guidance_h(tol=0)** | **1.96 s** | 26.6 | **34.7** | **54.6** | **60.0** | 138 | 87.8 m |
| 中间档：tol=1 cell（扩散引导重新参与） | 3.58 s | 45.2 | 67.1 | 111.3 | 130.3 | 138 | 89.1 m |

* 总耗时 **−83%**（5.8×）；p90 **−93%**、p99 **−95%**、max **−96%**。
* p50 只高 3 ms，正好等于测地场本身的开销（~3 ms）：中位样本本来就快，强启发式在中位上是净零、在尾部是净赚。
* 路径长度/居中后 clearance 与旧版**一致**（87.8 m / 6.87 m）。

**C2 端到端同进程配对（`scripts/debug/exp_end_to_end.py`，60 样本，同 batch 同 seed）**

| 指标 | 旧 | 新 | 差 |
|---|---|---|---|
| `total_ms` p50 | 180.92 | **151.50** | **−29.4 ms（57/60 样本更快）** |
| `search_ms` p50 | 57.28 | **27.94** | −29.35 ms |
| `expanded` p50 | 188.5 | 140.5 | −25% |
| 碰撞样本 | 0/60 | **0/60** | 一致 |
| RMSE 均值 | 0.0444 | 0.0451 | **+1.6%** |

（此对照里 `grid_ms` 两边相同，因为两个变体共用同一份 P0 之后的栅格代码；栅格侧的收益见 C3。）

**C3 单样本固定开销（隔离微基准 `scripts/debug/bench_search_overheads.py`）**

| 项 | 旧 | 新 |
|---|---|---|
| `build_neighbor_table` | 12.59 ms | **1.31 ms** |
| `center_path(iters=40)` | 10.24 ms | **~4.5 ms**（含复用 EDT、融合 bilinear） |
| `center_path` 里那个没用的 `distance_transform_edt(return_indices=True)` | 2.06 ms | **0** |

**C4 全 test 420 eval（`outputs/v5_fastsearch/`）**

| 指标 | 旧（`v5_final_clean`） | 新 |
|---|---|---|
| 碰撞样本（128 / 512 点） | 0 / 420 | **0 / 420** |
| `search_ms` 中位 | 46.25 | **31.56** |
| `grid_ms` 中位 | 16.64 | **13.53** |
| `expanded` 中位 | 220.5 | **149** |
| 曲线 RMSE | 0.0924 | 0.0943（+2.0%） |
| 端点误差 | 0.0 | 0.0 |
| 输出来源 | model 413 / last_verified 4 / search_path 3 | model 416 / last_verified 2 / search_path 2 |

> ⚠️ **绝对延迟不要横向比**：本次 eval 期间这台机器有稳定的“跑久了 GPU 前向变慢”现象——
> 同一份未改动的模型代码，在一次 420 样本 run 里前 60 个样本 `net` 中位 **50.6 ms**、
> 之后 360 个样本 **111.9 ms**；两次独立 run 的 `net` 也在 51 ms / 113 ms 之间漂。
> 所以上表的 `total_ms`（156 ms）不可与旧报告的 111.6 ms 直接比。可信的端到端倍率见 C2（同进程配对）。

## D. 两个必须知道的结论

1. **`h_tie_tol_cells = 0` 时，扩散引导在搜索里完全不参与排序。**
   实测：58 个样本 8657 次扩展，`_tiebreak_pick` 被调用 **0 次**。
   原因：测地 h 是连续实值，`f0` 平台内的**精确** h 并列几乎不存在。
   也就是说这份配置等价于「测地启发式 A\* + 居中」，PDG 的"扩散引导"退化成理论上的 tie-break。
   * 要让引导重新参与：`h_tie_tol_cells: 1.0`（把 1 格内的 h 视为并列，再用扩散分排序），
     实测慢 **1.8×**（1.96 → 3.58 s）、路径长 89.1 m（+1.5%）。
   * 想完全回到"扩散驱动排序"：`secondary: "guidance"`，实测相对旧版只有 ~1.1× 收益。
2. **RMSE +2%**：搜索路径变了 → `active_path` 初值变了 → 第 2–4 步的 conditioning 变了。
   碰撞仍是 0，端点仍精确；但论文里的 RMSE 数字需要按新结果更新（或按 D1 的 tol 取舍）。

## E. 复现

```bash
# 单元测试（含新增的逐位等价 / 堆等价测试）
python -m pytest tests/test_grid_search_hotpath.py tests/test_focal_fast.py -q
python -m pytest src/grid_search/tests -q

# 配对 A/B（搜索配置；同进程、同网格、同输入）
python scripts/debug/exp_search_configs.py --limit 80

# 端到端配对 A/B（同 batch、同 seed）
python scripts/debug/exp_end_to_end.py --limit 60

# 单样本固定开销
python scripts/debug/bench_search_overheads.py

# 全量评估
python eval_v5.py --config configs/config_v5_c32q128.yaml \
       --ckpt outputs/models/v5_c32q128/ckpt/final_best.pt \
       --split test --limit 420 --batch 1 --steps 4 --figures 0 \
       --out outputs/v5_fastsearch
```

## F. 遗留

* `focal_fast`（堆后端）只在 `w == 1` 启用；当前配置 `w=1.2` 走的是改进后的向量化内环。
  它服务于 `astar()`/`dijkstra()` 与 `w=1` 的配置（测试里逐位比对过）。
* 引导场冷填充仍是 **5.8 µs/格**（算 127 段只留 3 段，42× 冗余）；
  当前配置 tol=0 下几乎不触发，若要启用 tol>0 值得先把它降到 ~0.7 µs/格。
* `center_path` 仍是最大单项（~4.5 ms，40 次迭代的 Python 循环），未做向量化重写。

