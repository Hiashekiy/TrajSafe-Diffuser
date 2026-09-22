# TrajSafe-Diffuser（报告一致的骨架引导安全扩散规划器）

本仓库实现 **TrajSafe-Diffuser**：以占据地图（CARLA v1 快照）上的**骨架拓扑**为
几何先验，用一条**控制点空间**的扩散链生成起终点之间的安全 B 样条路径，
并预测沿路径分布的安全椭圆。

**架构规格**（source of truth）：

> 当前实现（**优先**）：[`docs/CONTROL_SPACE_REFACTOR.md`](docs/CONTROL_SPACE_REFACTOR.md)
> 重构前设计报告：[`docs/TrajSafe-Diffuser_网络结构设计报告_实现细化版.md`](docs/TrajSafe-Diffuser_网络结构设计报告_实现细化版.md)
> CARLA 数据流水线：[`docs/CARLA_BSPLINE_PIPELINE_SPEC.md`](docs/CARLA_BSPLINE_PIPELINE_SPEC.md)
>
> 设计报告描述的是重构前的 128 曲线点 token 版本，两者冲突时以控制点空间重构说明为准。

旧版本的代码、配置、入口、脚本、测试与设计文档已删除。当前仓库只包含
本实现及其依赖的共享组件（Skeleton 几何、B 样条编解码、推理期 ALM 引导、数据准备流水线）。

---

## 1. 核心思想

只有一个 diffusion state：B 样条**控制多边形** `Q_t ∈ R^{B×C×2}`。轨迹主变量从头到尾
都是这 `C` 个控制点 token，**网络内部不再出现 128 轨迹点张量**；`128` 只作为骨架 /
椭圆安全 query 数 `Q` 存在。椭圆**不是**扩散状态，椭圆圆心由选中骨架的固定进度给出
（`c_i = Γ_m(i/(Q-1))`），没有 center head、也没有 progress head：

```
Q_t [B,C,2]                       扩散状态，端点硬条件（clamped knot）
  -> H_ctrl = ControlEncoder      planner.traj_encoder
  -> H_ctrl = ControlBackbone     planner.traj_backbone，N_T 个 AdaLN block
  -> H_ctrl = FeedbackFusion      [可选] 融合上一轮 ALM 的 (Q0_safe, Δ, valid)（见 §10）
  -> Q~_coarse = head_p           [B,C,2]，直接出控制点（无 LS 拟合、不解码）
  -> Q_coarse = BoundaryDecoder   固定、零参数 -> B_128 @ Q_coarse（仅绘图/回退）
  -> H_S = SkeletonEncoder        [B,M,L,D]; MatchBlock(H_ctrl, H_S) -> R [B,M,C,D]
  -> pi = TopologyHead            m = m*（训练）/ argmax(pi)（推理）
  -> H_path = PathFeatureHead(R[m])             [B,C,D]
  -> H_safety = SafetyQueryHead(H_S[m])         [B,Q,D]
  -> c_i = Gamma_m(i/(Q-1))       固定进度 buffer，无 head
  -> EllipseGeometry + EllipseShapeHead         -> [B,Q,*]
  -> A_safety = SafetyControlFusion             cross attention 控制 <- 安全
  -> FusionMLP([H_ctrl, H_path, A_safety]) -> FinalDenoiser -> head_p
  -> Q~_final [B,C,2] -> BoundaryDecoder -> Q_final [B,C,2]
  -> B_128 @ Q_final -> 曲线                    仅绘图 / 指标 / 控制器
```

`C = model.num_controls`（与 `bspline.num_controls` 必须一致，默认 32）是**唯一的轨迹
表示**；`Q = model.num_safety_queries`（= `topology.candidate_points`，默认 128）只用于
Skeleton / 椭圆安全 query。`head_p` 同时给出 coarse 与 final 的 raw 控制点；只有
`Q_final` 进入 DDIM。`out["control"] = Q_final`（端点精确）、
`out["q_raw_final"] = Q~_final`（网络自由预测），解码曲线 `out["final"]` 只在最末端生成。

训练损失（8 项）：

```
L = 0.20 L_ctrl + 0.50 L_coarse + 0.08 L_smooth + 0.20 L_boundary
  + 0.25 L_topo + 0.08 L_shape + 0.25 L_iou + 0.15 L_safe
```

`L_ctrl` / `L_coarse` 用 Boundary Decoder **之前**的 raw 控制点做内部 MSE（端点硬条件）；
`L_smooth` 是控制多边形的二阶/三阶差分（按 GT 控制点平均步长缩放，不解码任何轨迹点）；
`L_boundary` 让网络自己学会端点附近的控制多边形局部形状（
`T^s_i = S + (Q^GT_i - Q^GT_0)`，`T^g_j = G + (Q^GT_j - Q^GT_{C-1})`）。`L_traj` 与
`L_align` 已删除：没有 128 点解码曲线、没有 GT 骨架投影、没有 `progress_gt`、
没有 `ellipse_center_gt`；`L_shape` 只作用于椭圆形状头，不含圆心。

---

## 2. 环境

```bash
E:/CondaEnvData/envs/GGMPC/python.exe --version    # Python 3.10 + torch 2.9.1+cu126
```

所有路径 / 超参数集中在 `configs/config.yaml`，代码不硬编码数值。

---

## 3. 目录结构

```
configs/
    config.yaml     训练 / 数据 / 模型 / 损失 + 推理期 ALM 段
src/
    models/trajsafe/         网络（planner / blocks / encoders / boundary / ellipse / fusion / geometry / heads）
    models/common/               共享 AdaLN / MHA（blocks）与 SceneCNN（scene_cnn）
    models/position_encoding.py SpatialPE / PE_1D / timestep embedding
    datasets/                skeleton_dataset / carla_spline_dataset
    diffusion/                  schedule / sampler / alm_guidance / bspline_alm
    losses/losses.py
    geometry/                   骨架图与候选路径、细化、B 样条编解码与约束、椭圆几何、凸区域与安全走廊
    utils/                      config / checkpoint / seed
train.py  sample.py  evaluate.py
scripts/data/{01,02,03}_*.py               scenes 数据准备流水线
scripts/data/carla/{00,01,02,03}_*.py      CARLA v1 数据准备流水线
scripts/debug/*.py                        候选召回 / 单步性能 / 标签可视化
tests/                                    模型 / 数据 / 几何测试
diffusion-dashboard/                      交互式可视化（见其 README）
docs/                                     架构报告、控制点空间重构说明与流水线规格
```

---

## 4. 数据

训练使用 CARLA v1 处理快照（`data.root: data/carla_v1` -> `data.processed_root:
data/carla_processed`），由 `CarlaSplineDataset` 读取。离线控制点标签
（`control_gt.npy`）是**数据**：改 `bspline.num_controls` 后必须重跑流水线。

```bash
python scripts/data/carla/00_clean_dataset.py        --config configs/config.yaml
python scripts/data/carla/01_build_candidates.py     --config configs/config.yaml
python scripts/data/carla/02_build_ellipse_labels.py --config configs/config.yaml
python scripts/data/carla/03_validate_processed.py   --config configs/config.yaml
```

* 原始轨迹 / 条件 / 占据图：`data/carla_v1`（仓库已提供）；
* 候选缓存 / 椭圆标签 / 控制点 GT：`data/carla_processed/{train,val,test}`；
* 详情见 [`docs/CARLA_BSPLINE_PIPELINE_SPEC.md`](docs/CARLA_BSPLINE_PIPELINE_SPEC.md)。
  旧的 Maze2D（`data/scenes` / `data/skeleton`）工具链仍保留在 `scripts/data/` 下，
  但默认配置不再使用。

---

## 5. 训练 / 采样 / 评估

```bash
python train.py    --config configs/config.yaml
# 用已经训练好的 checkpoint 直接采样 / 评估（--arch auto 会自动识别架构）
python sample.py   --config configs/config.yaml     --ckpt outputs/bspline_carla/ckpt/best_task.pt --split test --num 6
python evaluate.py --config configs/config.yaml     --ckpt outputs/bspline_carla/ckpt/best_task.pt --split test --runs 4
```

`sample.py` / `evaluate.py` 支持 `--steps`（DDIM 子采样）、`--device` 与
`--arch {auto,control,legacy}`（默认 `auto`，见第 9 节）；
`evaluate.py` 输出 `traj_collision`、`center_free`、`progress_violations`、
`recall@M`、`sel_best_rate` 等指标。

---

## 6. 测试

```bash
python -m pytest tests/test_model.py tests/test_dataset.py tests/test_geometry.py \
                 tests/test_thinning.py tests/test_alm_guidance.py \
                 tests/test_convex_region.py -q
```

---

## 7. 交互式 Dashboard

`diffusion-dashboard/` 提供在线可视化：在线生成候选骨架、回放 16 步反向
扩散、叠加 verified convex region 与 ALM 修正前后的 `x̂₀`。启动方式见
[`diffusion-dashboard/README.md`](diffusion-dashboard/README.md)。

---

## 8. 推理期扩展：安全走廊 + ALM 修正

可选开启（`configs/config.yaml` 的 `alm` / `corridor` 段）：`sample()` 内部是三阶段
状态机 —— `WARMUP` -> `TRY_ACTIVATE`（用候选骨架构造安全走廊，检查区域重叠 / 间隙
桥接 / 冻结）-> `GUIDED`，在反向扩散的每个 step 用 `bspline_alm_correct` 对**控制点**
做一阶修正（B 样条约束包只构建一次后冻结，只刷新网络的 clean 预测）。旧 waypoint
语义的 `alm_guidance` 参数已被 `sample()` 拒绝。该扩展只发生在推理期，不改变网络、
checkpoint 与训练损失。

---

## 9. 控制点空间重构 / 旧 checkpoint 兼容

* 轨迹表示只有控制点：`C = model.num_controls`（`bspline.num_controls` 必须与之一致，
  默认 32）；`Q = model.num_safety_queries`（默认 = `topology.candidate_points` = 128）
  只用于 Skeleton / 椭圆安全 query。改 `C` 是配置改动（`bspline.knots: "auto"` 会按
  `(C, degree)` 生成 clamped 均匀节点向量），不是代码改动。
* 网络内部不再出现 128 轨迹点；`B_128` 解码只在最末端出现一次。`L_traj` / `L_align`
  已删除，损失固定为 8 项，曲线端点由**固定、零参数、不训练**的 Boundary Decoder 保证。
* 旧 checkpoint 仍可直接使用：`sample.py` / `evaluate.py` 的 `--arch auto`（默认）会
  检测 checkpoint 架构，重构前的模型自动走 `forward_curve_tokens` 的 128 曲线 token
  链路原样回放；`--arch control` / `--arch legacy` 可强制指定。dashboard 引擎与
  preview / inspect 脚本同样默认 `arch="auto"`。
* 已验证：`outputs/bspline_carla/ckpt/best_task.pt`（epoch 449）在 legacy 链路上与重构前
  逐位一致（曲线 / 控制点 / 椭圆中心 / shape4 / pi 的 max |diff| = 0.0）。
* 细节（新前向链路、固定 Boundary Decoder、8 项损失、配置项、改 `C` 的完整步骤）见
  [`docs/CONTROL_SPACE_REFACTOR.md`](docs/CONTROL_SPACE_REFACTOR.md)；与设计报告冲突时
  以该文为准。

---

## 10. 历史安全反馈：把上一轮 ALM 的结果喂回 Diffusion

`model.feedback.enabled: true`（见 `configs/config_160k8p.yaml`）时，每个控制点额外携带
上一轮 ALM 的信息 `[x_safe, y_safe, dx, dy, valid]`（`Q0_safe_prev` / `Δ = Q0_safe − Q0_raw`
/ 验证标志），经 **门控残差** `h_ctrl ← h_ctrl + sigmoid(...) * valid * h_fb` 注入控制点
特征流，因此影响下游全部模块。`valid = 0`（warmup / 无可靠走廊）时是**精确恒等**，
旧 checkpoint 在开启该开关后仍然逐位一致。

训练不再是单步：每个 batch 跑一次真实的 `网络 → ALM → DDIM → feedback → 网络` 两步
rollout，梯度只回到**第二次网络自己的原始预测**，监督来自
`L_feedback_safe`（与推理 ALM 同一套连续 Bézier 约束包，`config_160k8p.yaml` 的
`lambda_feedback_safe`）与 `L_curve_smooth`（解码曲线的二阶/三阶差分，
`lambda_curve_smooth`），刻意**不**加 `||Q_next − Q_safe_prev||` 蒸馏项。
训练日志里的 `fb_valid_rate / fb_raw_violation / fb_correction` 用于确认反馈分支真的
被训练到了。

* 实现细节、配置项、消融方式、逐 step 诊断字段：见
  [`docs/HISTORICAL_SAFETY_FEEDBACK.md`](docs/HISTORICAL_SAFETY_FEEDBACK.md)。
* 测试：`python -m pytest tests/test_feedback.py tests/test_alm_state_machine.py -q`。
