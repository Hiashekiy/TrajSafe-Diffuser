# CARLA + 32-control B-spline 数据处理契约（v1）

本文件是 `scripts/data/carla/*` 与 `src/datasets/carla_spline_dataset.py` 之间的**唯一接口约定**。
任何一方修改字段名/形状前必须先改这里。

工程根：`D:\ProjectDirectory\Neural-IRISDiffuser`
Python：`E:\CondaEnvData\envs\GGMPC\python.exe`（torch 2.9.1+cu126, numpy 2.2, scipy, networkx）
注意：bash 工具的 `workdir` 参数在本机不可用；请用 `cd /d/ProjectDirectory/Neural-IRISDiffuser && ...`。

原始数据（**只读，绝对不要修改**）：

```
data/carla_v1/
  samples.jsonl          # 每条 sample 一行（含 file / split / episode_id / town / sample_id）
  episodes.jsonl
  dataset_report.json
  bspline_knots.npy      # [36]，clamped uniform，degree=3，32 controls
  samples/<split>/*.npz  # occupancy[256,256] u8 / trajectory_128[128,2] f32 /
                         # bspline_controls[32,2] / start[2] / goal[2] / episode_id
```

场景坐标：`scene ∈ [-1,1]^2`，80 m ↔ 2.0。1 scene 单位 = 40 m。
**CARLA 栅格方向**：原始 `occupancy` 的 row 0 ↔ `y_local = +40 m`；仓库几何约定 row 0 ↔
`scene_y = -1`。因此进入任何几何/网络代码前必须 `occupancy = np.flipud(occupancy).copy()`，
并且**只保存/使用 flip 后的 canonical occupancy**（preprocessing 与训练用同一份）。

## 1. processed cache 目录（`data/carla_processed/`）

```
clean_manifest.jsonl      # 清洗后每条 sample 一行
cleaning_report.json      # 扫描/清洗统计（缺失、损坏、reason 聚合）
preprocess_report.json    # 03 验证输出
train/  val/  test/       # 每个 split 一份定长数组，顺序 = clean_manifest 中该 split 的顺序
```

每个 `<split>/` 目录的数组（`N` = 该 split 的样本数，`M=4`，`H=128`，`C=32`）：

| 文件 | shape | dtype | 含义 |
|---|---|---|---|
| `conditions.npy` | `[N,2,2]` | f32 | `[start, goal]`，scene |
| `control_gt.npy` | `[N,32,2]` | f32 | endpoint-constrained 重拟合控制点，`q[0]=start, q[-1]=goal` |
| `curve_gt.npy` | `[N,128,2]` | f32 | 原始 `trajectory_128`（scene） |
| `occupancy.npy` | `[N,256,256]` | u8 | **canonical（已 flipud）** 占据栅格，0=free 1=obstacle |
| `episode_id.npy` | `[N]` | i64 | episode（split 的单位，禁止跨 split 泄漏） |
| `sample_id.npy` | `[N]` | i64 | 原始 sample_id |
| `candidate_xy.npy` | `[N,4,128,2]` | f32 | 网络输入 Skeleton `S_m` |
| `candidate_mask.npy` | `[N,4]` | bool | 候选是否有效 |
| `candidate_lengths.npy` | `[N,4]` | f32 | 候选弧长（scene） |
| `candidate_geometry.npy` | `[P,2]` | i16 | dense 安全曲线 cell 索引（flat），`(px+0.5)*cell-1` 还原 scene |
| `candidate_geometry_offsets.npy` | `[N,5]` | i64 | 每个候选在 flat 中的切片 |
| `candidate_geometry_lengths.npy` | `[N,4]` | i32 | 每个候选 dense 点数 |
| `topology_best.npy` | `[N]` | i64 | `argmin_m nDTW(curve_gt, S_m)`，无候选时 0 |
| `ellipse_shape4_gt.npy` | `[N,128,4]` | f32 | `[log a, log b, cos2θ, sin2θ]`，在 fixed center 上查询 |
| `shape_valid.npy` | `[N,128]` | bool | 该 center 是否有安全椭圆标签 |

另有可选 `candidate_branch_counts.npy [N,4] i32` 仅作诊断。

**禁止**出现 `progress_gt.npy` / `ellipse_center_gt.npy`。

## 2. 关键几何定义

- Skeleton graph：`src.geometry.skeleton_graph.build_skeleton_graph(occ_canonical,
  safety_dilation_cells=cfg["skeleton"]["safety_dilation_cells"],
  thinning_backend=cfg["skeleton"]["thinning_backend"],
  pure_cycle_aux_nodes=cfg["skeleton"]["pure_cycle_aux_nodes"])`
- 候选：`src.geometry.skeleton_paths.generate_candidates(graph, start, goal,
  CandidateConfig.from_dict(cfg["topology"], strict=False))`（与旧 Maze2D 完全同一实现）
- `topology_best = argmin_m normalized_dtw(curve_gt, cands.metric_polyline(m))`（仅用有效候选）
- fixed progress：`s = np.linspace(0.0, 1.0, 128)`；ellipse center
  `c_i = Gamma_{m*}(s_i)` = `skeleton_paths.interpolate_path(dense_scene, s)`（**按弧长插值 dense 曲线**
  = 对 `candidate_geometry` 的 cell 索引还原出的 scene 折线，用 `(px+0.5)*cell-1`，
  `cell = 2/256`，并把首尾点覆盖为 `start/goal`）
- GT ellipse 形状：在**同一个 canonical occupancy** 上，用
  `scripts/data/03_build_ellipse_labels.py` 的算法（36 orientations / local_radius=0.25 /
  dilation=1 / boundary=128 / interior_rings=4 / interior_angles=32 / binary_iters=12 /
  min_semi_axis=1e-3）在每个固定 center 上求最大安全椭圆，写 `[log a, log b, cos2θ, sin2θ]`。

## 3. B-spline codec（`src/geometry/bspline.py`，已实现）

```python
from src.geometry.bspline import BSplineCodec, numpy_fit_curve_to_controls
codec = BSplineCodec(degree=3, num_controls=32, curve_points=128,
                     knots_path=cfg["bspline"]["knots"])
p = codec.decode_controls(q)                    # [B,32,2] -> [B,128,2]
q = codec.fit_curve_to_controls(p, start, goal) # [B,128,2],[B,2],[B,2] -> [B,32,2]
q_np = numpy_fit_curve_to_controls(knots, 32, 3, p_np, start_np, goal_np, 128)
```

`basis` [128,32] 与 `interior_pinv` [30,128] 都是 buffer；均匀参数 `u_j=j/127`；
clamped knot 保证 `decode(q)[:,0] == q[:,0]`、`decode(q)[:,-1] == q[:,-1]`。

## 4. 环境注意

- 16 逻辑核；`cv2` **没有** ximgproc → thinning 走 numpy 后端（单样本 ~0.05 s，正常）。
- 单样本 shape label 全量计算约 5~6 s；请用 `multiprocessing`（spawn）+ `--workers` 并行。
- 脚本必须容忍坏样本：单条失败 → 记录 reason → 跳过，不得中断整夜任务。
- 必须支持 `--resume`：per-sample 缓存（`<split>/_cache/<i>.npz`），存在且完整则跳过；
  全部完成后组装成大数组（组装可重复执行）。
