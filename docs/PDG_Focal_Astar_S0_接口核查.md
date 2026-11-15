# S0 接口核查：PDG-Focal A* 栅格搜索复用清单

**对象：** `PDG_Focal_Astar_Implementation_Design.md`（Progress-aware
Diffusion-Guided Focal A*）  
**范围：** 只新增 `src/grid_search/`、`scripts/grid_search/`、测试与文档；
**未改动** `src/diffusion/*`（sampler / corridor / ALM / scheduler）、
`src/geometry/*`、`src/models/*`、`train.py`、`sample.py`、`evaluate.py`。

所有结论来自本机实跑，不是按文档推测；命令附在每节末尾。

> **相关设计（扩散模型侧，已定稿、尚未实施）：** 端点由"只钉位置"扩展为"位置 + 朝向"，
> 见 [`PDG_Focal_Astar_端点朝向约束设计.md`](PDG_Focal_Astar_端点朝向约束设计.md)。
> 本文 §3 描述的是该设计实施**之前**的现状（`cond` 仅含位置）。

---

## 1. 数据与地图

| 项 | 实测值 | 来源 |
|---|---|---|
| 处理后数据根 | `data/processed/160m-k4p-c48` | `configs/config_160k4p_c48_oneshot.yaml: data.processed_root` |
| 划分 | `train=3850 / val=1550 / test=420` | `preprocess_report.json` |
| 占据栅格 | `occupancy.npy` `[N,256,256] uint8` | `CarlaSplineDataset` |
| 取值语义 | **实测只有 `{0,1}`**，`0 = free`，`1 = obstacle` | `np.unique(occ[:20])` |
| 朝向 | **canonical：已 `flipud`**，row 0 ↔ `scene_y = -1` | `docs/数据流水线与数据格式说明.md:23-25,43` |
| 场景尺寸 | 160 m × 160 m，`scene_to_meter = 80`（`[-1,1]` 为 160 m） | `data/raw/160m-256px/dataset_config.json: size_m=160`；`src/geometry/sdf_field.py: build_occupancy_sdf(scene_to_meter=80.0)` 默认值 |
| 栅格边长 | `cell_m = 2 * 80 / 256 = 0.625 m` | 同上 |
| test 自由率 | `0.204`（均值） | `(occ == 0).mean()` |

### 1.1 地图已被人为“拓宽”，不是机器人膨胀

`data/processed/160m-k4p-c48/erode_report.json`：`k=4`、`border_mode="protect"`、
`free_before_mean 0.186 → free_after_mean 0.283`。即
`scripts/data/carla_full/05_erode_occupancy.py` 用 `binary_erosion` **腐蚀障碍 4 格
（2.5 m）** 来拓宽 4–8 m 的窄车道，属于**数据侧的可行驶性拓宽**，与机器人半径无关。

因此安全栅格的做法是：

```
safe = inflate(processed_occupancy, max(0, r_robot + margin)/cell_m - already_inflated_cells)
```

`already_inflated_cells = 4`（已在数据侧给出的拓宽量，避免重复要求同一份安全余量），
`r_robot + margin = 0.8 + 0.3 = 1.1 m = 1.76 格`，故实际追加膨胀 `0` 格
（`1.76 - 4 < 0`）——**这一事实在日志与报告里显式记录**，不能静默。
为了让安全栅格不退化，实验脚本允许 `--robot-radius/--safety-margin/
--already-inflated-cells` 覆盖；连通性敏感性实测（60 个 test 样本）：

| 追加膨胀 | 可解样本 |
|---|---|
| 0 格（默认） | 60/60 |
| 1 格 | 60/60 |
| 2 格（=1.25 m 追加，等价忽略 4 格拓宽） | 60/60 |
| 3 格 | 60/60 |

结论：地图余量充足，追加膨胀不会把问题变成“不可解”，因此报告同时给出
`0 格 / 2 格` 两组膨胀下的结果（见实验报告）。

---

## 2. 坐标系

`evaluate.py` 的碰撞判定就是仓库的权威口径：

```python
px = np.rint((points[:,0] + 1.0) / 2.0 * res - 0.5).astype(int)   # col
py = np.rint((points[:,1] + 1.0) / 2.0 * res - 0.5).astype(int)   # row
occ[py, px]
```

`src/grid_search/grid_map.py` 采用**完全相同**的映射：

```
world_to_cell(p) = (p + 1) / 2 * res - 0.5        # 连续 (col, row)
cell_to_world(c) = (c + 0.5) * 2 / res - 1.0
rc_to_index(rc)  = rc[...,0] * width + rc[...,1]  # (row, col) 顺序
```

实跑核对（`test` 第 0 个样本）：`cond = [(168,26), (168,138)]`，两个端点栅格值均为
`0`；GT 曲线的 128 点在占据图上 `0/128` 碰撞（前 50 个样本全部为 0），说明
row/col、x/y 与图像上下翻转的对应关系是正确的。

---

## 3. Diffusion 的“中间干净状态”

* 配置：`diffusion.prediction_type = "x0"`（`configs/config_160k4p_c48_oneshot.yaml`），
  **网络直接预测 x0**，因此不需要 ε/velocity 的反推公式。
* 状态变量只有控制点 `Q_t [B,C,2]`（`C = num_controls = 48`）。一步反扩散里
  `out["control"]` 是**经过固定 BoundaryDecoder 的干净控制点预测**，
  `model.bspline.decode_controls(q)` 得到 128 点曲线。**不存在 128 点扩散状态**。
* `src/diffusion/sampler.py::sample(..., return_trace=True)` 的每一步 trace 都带
  `final = decode_controls(q0_ddim)`，即该步的 x0 曲线 → 直接作为“逆扩散第 K 步的
  中间干净状态”，不需要改采样器。
* 抽取时使用 `guidance="none"`（纯 DDIM，无走廊、无 ALM、无反馈），这样才能得到
  “可能穿障/回头”的原始 x0，与本任务的软引导设定一致。
  `src/diffusion/sampler.py` 明确：`guidance='none'` 时 `alm_enabled=False`，
  `q0_ddim = out["control"]`。

抽取脚本：`scripts/grid_search/extract_diffusion_x0.py`
（`--guidance none`，K∈{1,2,3,4,5} + 末步，缓存 `curves [S,K,128,2]`、
`cond [S,2,2]`、`batch_ms`、`times`）。

实测（`outputs/grid_search/diffusion_x0/test`，420 样本，16 步全长反向）：
`41 ms/sample`（RTX/单卡，含 `torch.cuda.synchronize`），x0 曲线长度中位数
≈ 97 m。

---

## 4. 碰撞与膨胀：复用了什么

| 复用对象 | 位置 | 用法 |
|---|---|---|
| 栅格语义 + 坐标映射 | `evaluate.py::_collides/_free` | `grid_search.grid_map` 逐位等价实现；`path_validation.world_collision_rate` 复刻同一判定，用于报告“原始地图碰撞率” |
| 形态学算子 | `scipy.ndimage.distance_transform_edt` / `binary_erosion`（`scripts/data/carla_full/05_erode_occupancy.py`、`src/geometry/thinning.py`） | 膨胀/间隙场用同一库，几何口径一致 |
| 连续 SDF | `src/geometry/sdf_field.py::OccupancySDF / build_occupancy_sdf` | **不重复造**：它服务于 ALM/SDF baseline 的连续约束，栅格搜索需要的是布尔安全图 + 离散 8 邻域，语义不同，故不改也不替换 |
| 数据集 | `src/datasets/carla_spline_dataset.py::CarlaSplineDataset` | 占据图、cond、candidate 全部从它取 |

新增（不复制已有判定）：`grid_search/grid_map.py`、`grid_neighbors.py`、
`diffusion_path.py`、`progress_guidance.py`、`focal_astar.py`、`baselines.py`、
`path_validation.py`、`visualization.py`、`scenes.py`、`benchmark.py`。

---

## 5. 现有搜索代码

`src/`、`scripts/` 全量 grep `astar|dijkstra|focal` → **无命中**（网格搜索是本仓库
没有的部分），因此不存在“两套不一致判定”的风险；唯一需要对齐的是上面的坐标
映射与占据语义，已逐位对齐。

---

## 6. 计时工具

* 仓库口径：`src/diffusion/sampler.py` 的 `timing` 全部用
  `time.perf_counter()` 并在 GPU 上 `torch.cuda.synchronize()` 前后夹逼。
* 本任务沿用同一口径：`search_time_ms` 只包搜索循环；`guidance_build_time_ms`
  只包投影候选构建；`preprocess_time_ms` 只包安全图 + 8 邻域表；
  `validation_time_ms` 只包独立复核；`warmup_ms_est = 反向单步均值 × K`
  （由抽取脚本实测、按样本摊销）。端到端 = 四者之和 + warmup，
  报告中**分开列搜索耗时与端到端耗时**，不重复计时。

---

## 7. 参数决定（第一版，全部可配置）

| 参数 | 值 | 说明 |
|---|---|---|
| `robot_radius_m` | 0.8 | CARLA 车辆半宽量级 |
| `safety_margin_m` | 0.3 | 安全余量 |
| `already_inflated_cells` | 4 | 数据侧腐蚀量（见 §1.1），报告同时给 0/2 格配置 |
| `J`（投影候选数） | 3 | 文档 §4.1 建议值 |
| `sigma_d` | `3 * cell_m = 1.875 m` | 文档 §4.3 建议值 |
| `alpha_d` | 1.0 | 第一轮固定 |
| `alpha_p` | 1.0（消融 0/0.5/1/2） | 文档 §7.3 |
| `w` | 1.2（扫描 1.0/1.1/1.2/1.3/1.5） | 文档 §5.2 |
| `L_ref` | `L_D`（参考轨迹弧长） | 文档 §4.2；记录在每条结果里 |
| 重采样步长 | `2 * cell_m = 1.25 m`，相邻共线（<5°）合并 | 文档 §3.3 + §4.1 |
| 起终点吸附 | 关闭（`snap_start_goal=false`） | 文档 §2.4，越界/障碍直接报 `INVALID_*` |

---

## 8. 复现命令

```bash
# 1) 抽取现有 checkpoint 的中间 x0（不训练、不改模型）
python scripts/grid_search/extract_diffusion_x0.py \
    --config configs/config_160k4p_c48_oneshot.yaml \
    --ckpt outputs/models/oneshot_k4p_c48/ckpt/best_task.pt \
    --split test --limit 420 --out outputs/grid_search/diffusion_x0/test

# 2) 单元测试（T1–T12 + Focal 正确性 + 引导场 + 独立复核）
python -m pytest src/grid_search/tests -q

# 3) 主对比 / 消融 / 参数实验 + 真实数据可视化
python scripts/grid_search/run_pdg_experiment.py \
    --cache outputs/grid_search/diffusion_x0/test/diffusion_x0_test_0_420.npz \
    --processed data/processed/160m-k4p-c48 --split test \
    --out outputs/grid_search/run_pdg
```
