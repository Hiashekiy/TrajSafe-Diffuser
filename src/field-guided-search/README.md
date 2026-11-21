# field-guided-search

独立最小包：**占用栅格地图 + 一条引导路径（可以穿墙）→ 全局引导场 `F(x)` → 在该场上的路径搜索结果**。

只依赖 `numpy`（`scipy` 可选，装了场构建快 20 倍以上）。复制整个目录即可独立使用：
`pip install -e .`，或直接把 `src/field_guided_search` 放进你项目的 `sys.path`。

```python
from field_guided_search import GuidanceField

gf    = GuidanceField(occupancy)      # (H, W)，0 = 自由
field = gf.build(trajectory)          # (N, 2) 折线，[row, col]；允许穿过障碍
res   = gf.search(start, goal)        # 纯引导场；alpha=0.5 融合目标距离

res["path"]      # (M, 2) 整数栅格路径 [row, col]
res["length"]    # 路径长度（步数）
res["expanded"]  # 扩展节点数
res["status"]    # success | unreachable | cutoff | invalid
```

---

## 安装

```bash
pip install -e .              # 只依赖 numpy
pip install -e ".[fast]"      # 加 scipy：场构建快 20 倍以上（推荐）
```

`scipy` 缺失时自动回落到纯 Python 实现，结果完全相同（实测最大差 6e-16）。

## 最小示例

```python
import numpy as np
from field_guided_search import GuidanceField

occ = np.zeros((64, 64), np.int8)          # 0 = 自由，1 = 障碍
occ[0, :] = occ[-1, :] = 1
occ[:, 0] = occ[:, -1] = 1
occ[20, 5:50] = 1

trajectory = np.array([[10., 10.], [10., 45.], [45., 50.]])   # 引导路径
start, goal = (5, 5), (55, 55)

gf = GuidanceField(occ, lam=0.02, weight="linear")
gf.build(trajectory)

print("A*         ", gf.astar(start, goal)["expanded"], "expanded")
print("field-only ", gf.search(start, goal)["expanded"], "expanded")
```

## 端到端推荐配方（实测最优，含场侧参数）

如果只想照抄一组能用的参数，用这一组。它在仓库自带 A/B/C/D 四族 20 张图 × 4 个轨迹等级
（80 组成对比较）上调出，并在**另外 20 张未见种子的地图**上复核过；成功率 100%。

```python
from field_guided_search import GuidanceField

gf = GuidanceField(
    occupancy,             # (H, W)，0 = 自由；要"硬"净空保证就先按机器人半径膨胀
    weight="quadratic",    # w(u) = u²
    lam=0.01,              # 目标 lam*L ≈ 1.3–1.6（L = 引导路径长度，格）；换尺度按 lam ≈ 1.5/L
    distance="manhattan",  # geodesic 只会慢 5–6 倍，净空收益无差别
    source_method="drop",  # = M0；严重碰撞时 project(M1) 同量级
    num_samples=64,
)
gf.build(trajectory)       # 引导折线允许穿墙

res = gf.search(
    start, goal,
    alpha=0.15, gamma=1.0, adaptive=False,                        # 搜索侧
    clearance_beta=0.10, clearance_mode="quadratic_hinge",
    clearance_threshold=3.0,                                      # d_s = 3 格
)
gf.clearance().distance()  # EDT 每张图只算一次；只换 beta/d_s 时只重算风险图（~0.2 ms）
```

实测（20 张未见地图 × 4 等级 = 79 组配对，`preprocess+search`，不含场构建）：

| 配置 | 内段贴墙率 | mean D | p10 D | L/L\* | 扩展/A\* | 单次 |
|---|---|---|---|---|---|---|
| 包默认 `linear, lam=0.02, alpha=0.25`，净空关 | 0.563 | 2.59 | 1.03 | 1.030 | 0.153 | 1.16 ms |
| 包默认 + `ds=3, beta=0.10` | 0.332 | 2.87 | 1.43 | 1.045 | 0.169 | 1.39 ms |
| **推荐配方** | **0.319** | **2.80** | **1.49** | **1.028** | **0.131** | **1.21 ms** |
| 推荐配方 + `ds=5, beta=0.25`（要更大余量） | 0.309 | 3.44 | 1.87 | 1.072 | 0.230 | 1.82 ms |
| A\*（最短路径，无场） | 0.614 | 2.37 | — | 1.000 | 1.000 | 2.58 ms |

也就是说：**比 A\* 贴墙少一半、长度只多 2.8%、快 2.1 倍（1.21 vs 2.58 ms），而且比"包默认参数、净空关闭"还快。**

* 想更远离障碍：先加 `clearance_threshold`（**只加 `beta` 无法让路径离开 `d_s` 之外**），
  再加 `beta`：`ds=5, beta=0.25` → mean D 3.44 / p10 1.87；`ds=8, beta=0.4~1.5` → mean D 3.7–3.9（接近本类地图上限）。
* **不要这样配**：`lam=0.05`（场太局部，净空收益塌到 0.47）、`alpha >= 0.5`（退回距离贪心）、
  `distance="geodesic"`（慢 5–6 倍）、`exponential`/`logistic` 作默认（同净空下更贵 1.5–2.5 倍）。
* 只关心"不贴墙"而不在乎平均距离时，`ds=3, beta=0.10` 已经到收益拐点；`ds>=5` 主要提升均值而非贴墙率。
* 数据来源与全部消融表：工作区 `docs/clearance_aware_search.md`（§6.1 调参、§6.2 端到端）；
  这些是**合成引导轨迹**的实测，不是 Diffusion 网络推断，也不含下游安全椭圆/ALM 验证。

## API

### `GuidanceField(occupancy, ...)`

| 参数 | 默认 | 说明 |
|---|---|---|
| `occupancy` | — | `(H, W)`，`0` 自由，其它值障碍 |
| `lam` | `0.02` | 衰减率：越大越"只在路径附近有效"，越小越全局。经验区间 `lam*L ≈ 1…4`；**推荐 0.01**（见上面的端到端配方，`lam*L ≈ 1.5`） |
| `weight` | `"linear"` | `uniform` / `linear`（w=u）/ `quadratic`（w=u²）；**推荐 `quadratic`**（同样 80 组配对上扩展数 0.134 vs 0.204 × A\*） |
| `num_samples` | `64` | 引导路径采样点数 |
| `source_method` | `"drop"` | `drop`（= M0，核心实验方案）：丢弃障碍内采样点、保留其余原始进度；`project`（= M1）：移到 `projection_radius` 内最近自由格 |
| `projection_radius` | `3` | `project` 的搜索半径 |
| `method` | `"auto"` | 场后端：`auto` / `scipy` / `heap` |
| `distance` | `"manhattan"` | 目标距离模型：`manhattan`（忽略障碍，**发表实验所用**）或 `geodesic`（自由空间 BFS 距离，会绕墙） |

| 方法 | 返回 |
|---|---|
| `build(trajectory)` | `(H, W)` 引导场，障碍与不可达处为 `NaN`，同时存入 `gf.field` / `gf.sources` |
| `search(start, goal, alpha=0.0, gamma=1.0, adaptive=False, ...)` | 搜索结果 dict |
| `astar(start, goal)` | 最短路径基线 dict |
| `is_valid(path, start, goal)` | `{'valid', 'length'}` 或 `{'valid': False, 'reason'}` |
| `normalised()` | `{'F_hat', 'field'}`，搜索实际使用的标量图 |
| `clearance()` | `ClearanceRiskMap`：本图的距离图 / 风险图缓存 |

搜索结果的时间分三段，避免把预处理算成搜索开销：

| 键 | 含义 |
|---|---|
| `preprocess_ms` | 目标距离、场归一化、优先级图构建（开启净空项时还包含 EDT 与风险图） |
| `search_ms` | 优先队列搜索本身（`time_ms` 是它的别名，保留兼容） |
| `total_ms` | 两者之和 |

只在 `clearance_beta > 0` 时额外出现三个键：`clearance_edt_ms`（本次调用真正计算距离图的时间，
命中缓存时为 0）、`clearance_risk_ms`、`clearance_source`（`computed` / `cache` / `provided`）。
`beta=0` 时结果键与旧版逐键一致。

### 搜索优先级

```
J(n) = alpha(n) · d̂_G(n)^gamma − (1 − alpha(n)) · F̂(n)
```

| 用法 | 含义 |
|---|---|
| `alpha=0`（默认） | 纯引导场跟随 |
| `alpha=1, gamma=1` | 纯目标距离贪心 |
| `0 < alpha < 1` | 线性融合。已发表配置是 `linear`+`lam=0.02`+`alpha=0.25`（扩展数 0.20× A\*、路径 +4%）；**端到端重扫后更优的是 `quadratic`+`lam=0.01`+`alpha=0.15`（0.13×、+1.7%）**，见上面"端到端推荐配方" |
| `gamma=0.5` | 非线性融合：靠近目标时更看重距离 |
| `adaptive=True` | `alpha` 随剩余距离从 `alpha_far=0.25` 变到 `alpha_near=0.75` |

`alpha` / `gamma` 作用在**归一化后**的 `F̂`、`d̂_G` 上，所以同一组参数在不同地图上含义一致。只有 `astar` 声称最短路径最优。

### 净空感知（可选，默认关闭）

```
J(n) = alpha·d̂_G(n)^gamma − (1−alpha)·F̂(n) + beta·R(D(n))
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `clearance_beta` | `0.0` | 风险权重。**0 = 原版搜索**（路径、trace、expanded、结果键全部一致，且不计算距离图） |
| `clearance_mode` | `"quadratic_hinge"` | `quadratic_hinge` / `exponential` / `logistic` |
| `clearance_threshold` | `3.0` | hinge 与 logistic 的 `d_s`（单位：格） |
| `clearance_sigma` | `2.0` | exponential 的 `sigma` |
| `clearance_tau` | `0.7` | logistic 的陡度 `tau` |
| `clearance_penalty_map` | `None` | 直接传入已算好的风险图 `(H, W)`，跳过 EDT（此时不需要 scipy） |

```
R_hinge(D)    = [max(0, 1 − D/d_s)]²        # D ≥ d_s 时精确为 0
R_exp(D)      = exp(−D/sigma)
R_logistic(D) = 1 / (1 + exp((D − d_s)/tau))
```

`D` 是到最近障碍格中心的欧氏距离（格为单位，地图外视为障碍）；风险值域 `[0, 1]`，随 `D` 单调不增。

```python
gf = GuidanceField(occ)
gf.build(trajectory)
res = gf.search(start, goal, alpha=0.25, clearance_beta=0.10)   # 首次：算 EDT + 风险图
res = gf.search(start, goal, alpha=0.25, clearance_beta=0.10)   # 再查：命中缓存
res = gf.search(start, goal, alpha=0.25, clearance_beta=0.25,
                clearance_threshold=2.0)                        # 换参数：只重算风险图
dist = gf.clearance().distance()                                # 本图的 D（只算一次）
```

**边界与注意事项：**

* 这是**节点优先级上的软惩罚**，不是沿路径累计的代价：仍然只弹最小 `J`，仍然首次发现即定父节点，
  因此不声称最优，也不声称最小化整条路径的风险。
* `beta` 无法保证路径处处 `D ≥ d_s`；单格宽通道、贴墙的起终点仍然会贴墙——那是地图决定的。
* `D` 是**障碍格中心**的欧氏距离，只是相对指标，没考虑机器人尺寸。要做轨迹安全，先按安全裕度膨胀占据图
  再算净空；输入本身已膨胀就不要重复膨胀。
* 缓存绑定地图：占位图变了请新建 `Grid` / `GuidanceField`，或调用 `gf.clearance().invalidate()`；
  把别的地图的缓存传进来会直接报错，不会静默复用。
* 不开启净空时不导入 scipy、不分配距离图；开启但没装 scipy 会抛 `ImportError` 并提示
  `pip install -e '.[fast]'`（或自己传 `clearance_penalty_map`）。
* 实测（20 张图 × 4 个轨迹等级 = 80 组成对比较）：用**包默认场参数**（`linear, lam=0.02, alpha=0.25`）时
  `hinge ds=3, beta=0.10` 把内段贴墙率从 0.644 降到 0.352、p10 净空 +0.39 格，代价是长度 +2.6%、
  扩展 +14.9%、热启动重规划 +19%；换用上面推荐的场参数后是 0.563 → 0.319、`L/L*` 1.028、扩展 0.131× A\*。
  迷宫族（单格走廊）在任何参数下都没有变化——净空由地图几何决定。
  方法与原始数据见工作区 `docs/clearance_aware_search.md`；这些是**合成引导轨迹**的结果，不是 Diffusion 网络推断。

### 一次性函数

```python
from field_guided_search import field_from_trajectory, guided_search

field = field_from_trajectory(occ, trajectory, lam=0.02, weight="linear")
res   = guided_search(occ, start, goal, field=field, alpha=0.5)
res   = guided_search(occ, start, goal, trajectory=trajectory, alpha=0.5)

# 已经持有 Grid / 场时用底层的 Grid 级引擎（同名会混淆，故单独导出）
from field_guided_search import guided_search_grid
res = guided_search_grid(grid, start, goal, field, alpha=0.5)
```

### 与原实验代码的数值一致性

`tests/test_package_parity.py`（在工作区）逐项对照本包与 `src/guidance_field`：M0 源点、
场值、`d̂_G`、A\* 以及 α=0/0.25/0.5/1 的引导搜索 **expanded 计数完全相同**。改动本包后
务必重跑该测试，它是防止两者悄悄漂移的唯一保障。

## 原理

```
F(x) = max_i [ w(u_i) − lam · d_free(x, s_i) ]
```

* `u_i` 是第 i 个采样点在**原始路径**上的归一化弧长；处理碰撞时只删/移采样点，**不重新归一化** `u`，否则会凭空造出路径从未有过的进度；
* `d_free` 是四邻域测地距离，场不会穿墙（障碍格永远是 `NaN`）；
* 场等价于"沿路径衰减的势"，搜索沿场的等高线前进，所以比纯目标距离更快收敛到终点方向。

场构建：把 `C(x) = min_i[d + (w_max − w_i)/lam]` 展开成一次多源最短路，再用 `w_max − lam·C` 还原。默认后端是一个虚拟超级源的稀疏 Dijkstra（精确）；`method="heap"` 是等价的纯 Python 多源最大堆。

## 注意

* 坐标一律 `(row, col)`；matplotlib 需要 `(col, row)`。
* 引导路径若一个合法采样点都留不下，`build()` 直接报错而不是给出空场——那种"看起来能跑"的结果其实是盲搜。只剩 1 个采样点是允许的，对应退化场 `F(x) = w_1 − λ·d(x, s_1)`。
* 飞出地图边界的采样点会被**丢弃并计数**（`sources['num_out_of_bounds']`），不会被裁剪到边缘——裁剪会凭空造出路径从未经过的源点。
* `lam`、`weight`、`distance`、`num_samples`、起终点坐标都会校验，非法值抛 `ValueError`。
* `drop` 保留真实位置但会在进度轴留洞；`project` 保持采样完整但位置轻微偏移。碰撞严重时 `project` 更稳。
* 场与地图绑定（缓存了 CSR 邻接矩阵），换地图请新建对象；同一地图多次查询可复用场。
* **拷到别的项目后先自检拿到的是哪份代码**：如果目标环境里曾装过旧版本，`import field_guided_search`
  可能命中 site-packages 里的旧副本，此时 `gf.search(..., clearance_beta=...)` 会报
  `AttributeError: 'GuidanceField' object has no attribute 'clearance'`。自检：

  ```bash
  python -c "import field_guided_search as f; print(f.__file__, f.__version__, hasattr(f,'ClearanceRiskMap'))"
  ```

  把 `src/` 放在 `sys.path` 首位，或 `pip install -e .` 覆盖旧版即可。同上，导入路径必须是
  `field-guided-search/src`，而不是上层仓库目录。

## 变更记录

* **1.1.0** — 新增可关闭的障碍净空风险项（`clearance.py`：padded EDT + `quadratic_hinge` /
  `exponential` / `logistic` 三种风险函数、参数与外部风险图校验、按地图缓存），
  `GuidanceField.search()` / `guided_search()` / `guided_search_grid()` 全部透传；
  `clearance_beta=0` 与原版逐键一致。同时新增 34 项测试。
* **1.0.0** — 首次发布：占用栅格 + 引导路径 → 引导场 → GBFS 搜索。

## 测试

```bash
python -m unittest discover -s tests      # 62 项（28 旧 + 34 净空）
```

## 文件

```
src/field_guided_search/
    __init__.py   公开 API
    api.py        GuidanceField + 一次性函数
    grid.py       占用栅格、四邻域表、BFS 距离
    sources.py    路径重采样、障碍采样点投影/丢弃
    field.py      场构建（scipy 稀疏 Dijkstra / 纯 Python 多源堆）
    search.py     A*、引导搜索、归一化、路径校验、净空风险叠加
    clearance.py  padded EDT、三种风险函数、参数/风险图校验、按地图缓存
```
