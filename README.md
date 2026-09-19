# TrajSafe-Diffuser — V3（报告一致的骨架引导安全扩散规划器）

本仓库实现 **TrajSafe-Diffuser V3**：以 Maze2D 占据地图上的**骨架拓扑**为几何
先验，用一条轨迹扩散链生成起终点之间的安全路径，并预测沿路径分布的安全椭圆。

**唯一架构规格**（source of truth）：

> [`docs/TrajSafe-Diffuser_网络结构设计报告_实现细化版.md`](docs/TrajSafe-Diffuser_网络结构设计报告_实现细化版.md)

V1 / V2 的代码、配置、入口、脚本、测试与设计文档已在本分支删除。当前仓库只包含
V3 及其实际依赖的共享组件（Skeleton 几何、推理期 ALM 引导、数据准备流水线）。

---

## 1. 核心思想

只有一个 diffusion state：`P_t ∈ R^{B×H×2}`。椭圆**不是**扩散状态，椭圆圆心由
选中的骨架曲线给出（`c_i = Γ_m(s_i)`），没有 center head：

```
P_t
  -> H_traj                    Trajectory Backbone
  -> {R_m}                     Skeleton Transformer（共享 MatchBlock）
  -> pi = TopologyHead         m = argmax pi（每个 reverse step 重选）
  -> H_prog -> s               ProgressHead（严格单调）
  -> c = Γ_m(s)
  -> H_ell -> shape4           EllipseShapeHead（稳定参数化 a>=b>0）
  -> H_clean -> P0_hat         Final Denoiser
  -> DDIM 一步
```

粗解码 `P~_0` 与最终解码 `P^_0` 共用同一个 `Head_P`，只有 `P^_0` 进入 DDIM。
训练损失：

```
L = 1.00 L_traj + 0.50 L_coarse + 0.08 L_smooth + 0.25 L_topo
  + 5.00 L_align + 0.08 L_shape + 0.25 L_iou + 0.15 L_safe
```

其中 `L_align = mean SmoothL1(Γ(s_i), p_i^GT)` **直接对 GT 轨迹**，没有 GT 骨架投影、
没有 `progress_gt`、没有 `ellipse_center_gt`；`L_shape` 只作用于椭圆形状头，不含圆心。

---

## 2. 环境

```bash
E:/CondaEnvData/envs/GGMPC/python.exe --version    # Python 3.10 + torch 2.9.1+cu126
```

所有路径 / 超参数集中在 `configs/config_v3_skeleton.yaml`，代码不硬编码数值。

---

## 3. 目录结构

```
configs/
    config_v3_skeleton.yaml     训练 / 数据 / 模型 / 损失
    config_v3_alm.yaml          推理期凸区域 + ALM 引导参数
src/
    models/skeleton_v3/         V3 网络（planner / blocks / encoders / ellipse / fusion / geometry / heads）
    models/joint/               共享 AdaLN / MHA（joint_blocks）与 SceneCNN（scene_cnn）
    models/position_encoding.py SpatialPE / PE_1D / timestep embedding
    datasets/skeleton_dataset_v3.py
    diffusion/                  schedule / sampler_v3 / alm_guidance
    losses/v3_losses.py
    geometry/                   骨架图与候选路径、细化、椭圆几何、凸区域、数据准备几何
    utils/                      config / checkpoint / seed
train_v3.py  sample_v3.py  evaluate_v3.py
scripts/data/{01,02,08,10,13,14}_*.py     数据准备流水线
scripts/debug/v3_*.py                     候选召回 / 单步性能 / 标签可视化
tests/                                    V3 + 共享几何测试
diffusion-dashboard/                      V3-only 交互式可视化（见其 README）
docs/                                     架构报告、V3 说明、ALM、数据说明
```

---

## 4. 数据

```bash
# V3 骨架 / 候选 / 椭圆标签（在已提供的 data/processed_scene_v1 上运行）
python scripts/data/10_build_skeletons.py            --config configs/config_v3_skeleton.yaml
python scripts/data/13_build_skeleton_candidates_v3.py --config configs/config_v3_skeleton.yaml
python scripts/data/14_build_ellipse_labels_v3.py    --config configs/config_v3_skeleton.yaml
```

* 轨迹 / 条件 / 地图：`data/processed_scene_v1`（仓库已提供）；
* V3 候选缓存与椭圆标签：`data/processed_scene_v3`；
* 训练只使用 large：`data.mazes: ["large"]`。

从 d4rl hdf5 重建 `processed_scene_v1` 的 V1 数据准备脚本已在本分支删除
（需要时从 git 历史取回）。

---

## 5. 训练 / 采样 / 评估

```bash
python train_v3.py    --config configs/config_v3_skeleton.yaml
python sample_v3.py   --config configs/config_v3_skeleton.yaml --ckpt outputs/ckpt_v3_skeleton/best.pt
python evaluate_v3.py --config configs/config_v3_skeleton.yaml --ckpt outputs/ckpt_v3_skeleton/best.pt
```

`sample_v3.py` / `evaluate_v3.py` 支持 `--steps`（DDIM 子采样）与 `--device`；
`evaluate_v3.py` 输出 `traj_collision`、`center_free`、`progress_violations`、
`recall@M`、`sel_best_rate` 等指标。

---

## 6. 测试

```bash
python -m pytest tests/test_v3_model.py tests/test_v3_dataset.py tests/test_v3_geometry.py \
                 tests/test_thinning.py tests/test_alm_guidance.py \
                 tests/test_convex_corridor_validity.py -q
```

---

## 7. 交互式 Dashboard

`diffusion-dashboard/` 提供 V3-only 的在线可视化：在线生成候选骨架、回放 16 步反向
扩散、叠加 verified convex region 与 ALM 修正前后的 `x̂₀`。启动方式见
[`diffusion-dashboard/README.md`](diffusion-dashboard/README.md)。

---

## 8. 推理期扩展：凸区域 + ALM 修正

可选开启（`configs/config_v3_alm.yaml`）：对 `t <= start_t` 的帧，用预测椭圆在线构造
verified convex region，并用 `alm_correct` 一阶修正 `x0` 后再走 DDIM。该扩展只发生在
推理期，不改变网络、checkpoint 与训练损失。
