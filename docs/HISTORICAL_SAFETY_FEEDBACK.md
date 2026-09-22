# 历史安全反馈（Historical Safety Feedback）实现说明

> 对应设计文档：`TrajSafe_Safety_Feedback_Diffusion_Design.md`
> （TrajSafe-Diffuser：ALM 安全反馈扩散改造方案）。
> 本文记录**当前代码里已经落地的东西**：张量契约、融合方式、推理缓存状态机、
> 训练期两步真实 rollout、新增损失、配置项，以及已验证的结果与取舍。
> 与设计文档冲突时，以本文（代码）为准。

---

## 0. 一句话

推理期每一步 ALM 产出的 `Q0_safe` 与修正量 `Δ = Q0_safe − Q0_raw` 被缓存下来，
作为**下一步** Diffusion 的额外条件；训练期用与推理**同一条** `网络 → ALM → DDIM`
两步 rollout 监督第二次网络**自己**的原始预测（安全 + 平滑 + 专家轨迹），
因此网络不会把安全责任推给 ALM，也不会去“抄” ALM 的输出。

---

## 1. 张量契约

每个控制点 5 维：`[x_safe, y_safe, dx, dy, valid]`。

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| `feedback_control` | `[B,C,2]` | 上一轮的安全控制点；若上一轮原始预测本来就安全，则为 `Q0_raw_prev` |
| `feedback_delta` | `[B,C,2]` | `Q0_safe_prev − Q0_raw_prev`（上述“本来就安全”情形为 0） |
| `feedback_valid` | `[B]`（也接受 `[B,1]` / `[B,C,1]`） | `1` = 上一轮 ALM 输出通过了冻结的连续约束包 |

约定：

* 三个参数全部可选，默认 `None` = “没有可靠历史”；
* `feedback_valid = 0` 的行会在进入 encoder **之前**被清零，
  陈旧张量不可能泄漏进网络；
* `valid = 0` 时融合是**精确恒等**（`torch.equal` 级别，见
  `tests/test_feedback.py::test_feedback_fusion_is_an_exact_identity_when_invalid`）。

---

## 2. 网络侧

新增文件 `src/models/trajsafe/feedback.py`：

```python
FeedbackEncoder:  [B,C,5] -> Linear(5, hidden) -> SiLU -> Linear(hidden, d_model)
FeedbackFusion:   gate = sigmoid(Linear([h_ctrl ; h_fb])) * valid
                  h_ctrl <- h_ctrl + gate * h_fb
```

`src/models/trajsafe/planner.py` 的接入点（控制点链路 `forward_controls`）：

```python
h_ctrl = self.encode_trajectory(q_t, c_g, h_t)
h_fb, fb_valid = self.feedback_features(h_ctrl, feedback_control,
                                        feedback_delta, feedback_valid)
if h_fb is not None:
    h_ctrl = self.feedback_fusion(h_ctrl, h_fb, fb_valid)   # 替换，不是并联
q_coarse_raw = self.head_p(h_ctrl)
```

* `h_ctrl` 被**替换**，因此反馈影响下游全部模块：`q_coarse`、`MatchBlock`、
  `TopologyHead`、`PathFeatureHead`、安全 → 控制 cross attention、`FusionMLP`、
  `FinalDenoiser`（设计文档 §8）。
* 保持不变：椭圆圆心仍来自选中骨架的固定进度 `s_i = i/(Q−1)`；历史轨迹不参与
  定义骨架；没有新的 `u-s` 映射。
* `model.feedback.zero_init: true`（默认）时 encoder 最后一层零初始化，`h_fb ≡ 0`，
  于是**旧 checkpoint + 新代码 = 逐位一致**，可以边微调边用旧模型推理。
* `forward_all()` 新增 `feedback_control / feedback_delta / feedback_valid`
  三个关键字参数（默认 `None`）；legacy 曲线链路接受但忽略它们，只为了接口一致。
* **per-sample 拓扑路由**：`select_index` 传入 `[B]` 张量时，**负值表示“这一行用
  `argmax(pi)`”**（`_route_topology`）。采样器用它把“已冻结骨架的行”和“还在
  激活的行”混在一个 batch 里（见下一节）。

---

## 3. 推理侧（`src/diffusion/sampler.py`）

只有 `model.feedback_enabled`（即 `model.feedback.enabled: true`）为真时，采样器才
维护并传递反馈；否则采样器与旧版本行为完全相同。

### 3.1 per-sample 冻结骨架（修正）

每个反向 step 的 `select_index` 不再是“整批都 guided 才用 frozen_idx”，而是逐样本：

```python
sel = torch.where(guided, frozen_idx, torch.full_like(frozen_idx, -1))
```

即 **已激活的行永远用它自己冻结的骨架**，其余行走 `argmax(pi)`。旧写法
（`all_guided`）在混合 batch 下会把已冻结的行重新路由到别的骨架，而 ALM 仍在用
**冻结骨架的走廊**做投影 —— 网络按骨架 B 生成、ALM 按骨架 A 约束。
`result["selected_idx"]` / `progress_alignment` 也改为逐样本（未激活行不再被误报成
`frozen_idx` 的默认 0）。

回归测试：`tests/test_alm_state_machine.py::test_mixed_batch_keeps_the_frozen_skeleton_per_sample`。

### 3.2 缓存状态机（`feedback_step()`，设计文档 §4）

| 本轮情况 | `feedback_control` | `feedback_delta` | `feedback_valid` |
| --- | --- | --- | --- |
| warmup / 未 guided | 0（保持） | 0（保持） | 0（保持） |
| 原始预测本来就安全 | `Q0_raw` | `0` | `1` |
| ALM 修正后满足约束 | `Q0_safe` | `Q0_safe − Q0_raw` | `1` |
| ALM 修正后仍不满足 | 保留上一次已验证的反馈，**不覆盖** | 同上 | 保持 `1` |

判定阈值：`alm.feedback_accept_tol`（默认回落到 `alm.constraint_tol`）。
`configs/config_160k8p.yaml` 里设为 `1e-2`：实测训练好的模型上 ALM 收敛后的残余
违反量 max ≈ `0.004` scene（1 scene = 80 m，即约 0.3 m），沿用 `1e-3` 会把
“实际上已经安全”的结果判为失败，导致反馈永远不激活。**请以训练日志中的
`fb_valid_rate` 为准调整该值**。

语义提醒：放宽后它是“**可接受的近似可行反馈**”，不等于严格安全保证 —— 带少量
残余违反的 `Q0_safe` 也会被当作历史条件喂给下一步。要严格判定就把
`feedback_accept_tol` 设回 `constraint_tol`，并接受反馈激活率下降。

### 3.3 Trace / 结果字段（设计文档 §18 需要的逐 step 曲线）

```text
result["feedback"]        = {enabled, valid, control, delta, history}
trace[i]["feedback_valid_in"]   进入本次 forward 的 valid
trace[i]["feedback_valid"]      本次 step 结束后发布的 valid
trace[i]["feedback_control"/"feedback_delta"]
trace[i]["raw_violation"]       本 step 原始预测的最大违反量
trace[i]["alm_correction"]      本 step ALM 的曲线修正量（scene）
trace[i]["curve_smoothness_raw"/"curve_smoothness_safe"]  解码曲线二阶差分均值
```

关闭方式（三选一，都会回到旧行为）：

* `model.feedback.enabled: false`（模块不构建，采样器不传反馈）；
* 采样器侧给 `model.feedback_enabled = False`（等价于设计文档里的消融 A）；
* warup 阶段天然 `valid = 0`。

---

## 4. 训练侧

### 4.1 两步真实 rollout（`train.py::feedback_rollout`）

```text
q_t ──network──▶ Q0_raw(t) ──真实 ALM──▶ Q0_safe(t) ──真实 DDIM──▶ q_s
                                                      │
                              feedback = (Q0_safe, Δ, valid)
                                                      ▼
                          network(q_s, feedback) ──▶ Q0_raw(s) ──▶ LOSS
```

* **时间步**：启用 rollout 时第一步只从 `t ∈ [1, T)` 采样
  （`rollout_timesteps()`）。`t = 0` 时 `s = t = 0`，DDIM 给出 `q_s ≡ q_t`，
  第二次前向只是“换个 feedback 的重复计算”，反传不到任何真实的下一次去噪。
  时间步对由 `rollout_pair()` 统一给出（当前恒为 `s = t − 1`）；将来若要支持
  DDIM 跳步，应在这里改成从 `src.diffusion.sampler.pick_times` 的同一套 schedule
  采样 `(t, s)`，而不是继续写死 `t − 1`。
* 第一段（ALM / DDIM / feedback 构造）在 `torch.no_grad()` 下执行，
  梯度**只**回到第二次网络预测；
* 走廊直接读离线缓存 `alm_cell_a/b/valid`，用
  `build_constraint_pack_from_regions()` 转成与推理 ALM **完全相同**的
  `BSplineConstraintPack`（精确 Bézier 提取 + 线性不等式），
  因此不需要在训练循环里重建 `SafetyCorridor`（~240 ms/sample）；
* `feedback_valid` 是真实判定结果，不是常数；`feedback.drop_prob` 额外把一部分
  样本强制置 0，模拟 warmup / 无走廊分布；`simulate_warmup: true` 则整批置 0（消融）。
* **第二次前向的骨架**由 `train.feedback.topology` 选择：

  | 取值 | 含义 | 风险 |
  | --- | --- | --- |
  | `"expert"`（默认） | 缓存的 `m* = argmin nDTW(curve_gt, S_m)`，离线走廊正是在它上面建的 | 与推理路由不完全一致（推理用的是 `argmax(pi)` + 尝试候选后的回退） |
  | `"pi"` | 第一次前向自己的 `argmax(pi)`，更接近推理路由 | 离线走廊**未必**属于被选中的骨架，可能给网络一个与它自己骨架不匹配的约束 |

  两种模式的差距由日志里的 `fb_topo_match`（第二步选中的骨架与 `m*` 的一致率）
  与 `fb_valid_rate` 体现。另外 `topology: "pi"` 时会额外计算
  `fb_topo_corridor_fit`（`corridor_fit()`）：被选中骨架的稠密折线有多少比例的
  采样点落在**离线走廊内部**。它在 `"expert"` 下应接近 1（走廊本来就建在 `m*`
  上），在 `"pi"` 下若明显小于 1，就说明网络自选的骨架**不在**离线走廊里 ——
  这时 ALM 会把它硬拽回 `m*` 的走廊，反馈语义已经失真，应先回到 `"expert"`，
  或让推理侧也使用 `m*` 的走廊来源。**建议先用 `"expert"` 跑通，再把它当成对照
  实验。**

### 4.2 损失（`src/losses/losses.py`）

第二次网络**自己的原始输出**上新增两项：

* `Lfbsafe = feedback_safety_loss(out2["control"], pack)`：
  连续 Bézier 约束包上的**最大**违反量（`pack_violation(..., reduction="max")`，
  与推理 ALM 同一对象、同一公式），可行时严格为 0；
* `Lcurve = curve_smoothness_loss(out2["control"], q_gt, codec.basis)`：
  先 `B_128 @ Q` 解码，再对**曲线**算二阶/三阶差分（按 detached 的 GT 曲线步长
  归一化）。复制“安全但歪”的 ALM 输出同样会被惩罚。

第二步的 `L_ctrl / L_coarse / L_boundary` 以 `train.feedback.step2_weight`
加权后并入原 key（日志里 `Lctrl=...` 是两步之和）。**没有**
`L_distill = ||Q_next − Q_safe_prev||`（设计文档 §14）。

**空约束包的边界情况**：整个 batch 的走廊都失败时，约束包是 `[B,0,...]`
（没有 piece / 没有 face）。此时

* `pack_violation / pack_max_violation / feedback_safety_loss` 返回**与计算图相连的
  零**（可以 `backward()`，梯度为 0），不会对空维度做 `amax`；
* `bspline_alm_correct()` 是 no-op（`q_safe = q_ref`），其 stats 全部有限
  （`_violation_stats` / `lambda_max` 都有退化分支）。

测试：`test_empty_pack_is_a_differentiable_zero_and_never_raises`、
`test_alm_on_an_empty_pack_is_a_noop_with_finite_stats`、
`test_training_rollout_survives_a_batch_without_any_corridor`（`batch_size=1`）。

**显存**：`stats["loss_parts"] = [total_step1, total_step2]` 两段图互不相交，
训练循环对它们分别 `backward()`（梯度等价，峰值激活显存接近减半）。

---

## 5. 配置项（`configs/config_160k8p.yaml`）

```yaml
model:
  feedback:            # 网络侧：采样器/evaluate/dashboard 都从这里读
    enabled: true
    hidden: 64
    zero_init: true
    dropout: 0.0

loss:
  lambda_feedback_safe: 0.3
  lambda_curve_smooth: 0.1
  feedback_margin: 0.0                 # 约束包上的附加余量
  feedback_curve_acc_weight: 0.25
  feedback_curve_jerk_weight: 1.0

train:
  feedback:            # 训练侧 rollout
    rollout: true
    drop_prob: 0.25
    step2_weight: 0.5
    simulate_warmup: false
    topology: "expert"                 # "expert" | "pi"
    # accept_tol: 1e-2                 # 覆盖 alm.feedback_accept_tol
    # alm_inner_steps: 10              # 覆盖 alm.inner_steps

alm:
  feedback_accept_tol: 1.0e-2
```

复现命令（微调当前模型）：

```bash
python train.py --config configs/config_160k8p.yaml \
    --resume outputs/bspline_carla_160k8/ckpt/best_task.pt \
    --lr 5e-5
```

`--resume` 会提示 `feedback_encoder.* / feedback_fusion.*` 为 fresh（6 个张量），
这是预期行为：零初始化让它们从“恒等”开始学。

---

## 6. 消融与诊断

| 目的 | 做法 |
| --- | --- |
| 关掉反馈（旧模型基线） | `model.feedback.enabled: false` 或加载旧 checkpoint（零初始化等价） |
| 只训练两步、不看反馈 | `train.feedback.drop_prob: 1.0` 或 `simulate_warmup: true` |
| 训练路由换成网络自选骨架 | `train.feedback.topology: "pi"` |
| 反馈但无安全监督 | `loss.lambda_feedback_safe: 0.0` |
| 反馈但无平滑监督 | `loss.lambda_curve_smooth: 0.0` |
| 看反馈是否被训练到 | `fb_valid_rate`（0 = 分支完全没信号） |
| 看是否真的变安全 | `fb_raw_violation` ↓（最深违反）、`fb_mean_violation` ↓（平均正违反，**仅诊断**）、`fb_safe_violation` ↓、`fb_correction` ↓ |
| 看训练路由与专家骨架的一致率 | `fb_topo_match` |
| 看自选骨架是否落在离线走廊内 | `fb_topo_corridor_fit`（`"pi"` 模式下自动计算） |
| 看两步 rollout 的时间步 | `fb_t_min`（启用 rollout 时应 ≥ 1） |
| 看逐 step 行为（§18） | `sample(..., return_trace=True)` 的 `raw_violation` / `alm_correction` / `curve_smoothness_*` |

期望曲线：`raw_violation ↓`、`alm_correction ↓`、`curve_smoothness` 不劣化。
若 `fb_valid_rate` 长期为 0，说明 ALM 从未给出“通过验证”的结果 →
调大 `alm.feedback_accept_tol` 或 `train.feedback.alm_inner_steps` / `alm.max_curve_step_scene`。

关于 `fb_mean_violation`：训练损失仍然只惩罚**最大**违反量（它对应 ALM 真正
要压下去的量，梯度也更聚焦）。如果后续观察到“最大违反量降了、但轨迹上仍有
多处贴着走廊边界”，再看这个平均值 —— 目前它只是诊断，不参与反传。

---

## 7. 已验证 / 未验证

**已验证**

* `python -m pytest tests -q` → **136 passed**。
* 真实数据推理冒烟（`best_task.pt`，val 8 样本 × 16 步）：`alm_status=guided`，
  `feedback` 字段齐全；这批样本上原始预测已满足约束，走“already safe”分支
  （`delta = 0`），说明 §3.2 的退化路径与逐 step 诊断正确。
* 真实数据训练冒烟（`--overfit 8 --batch-size 4 --max-batches 2 --resume
  best_task.pt`，仅 4 个 epoch、非正式训练）：两步 rollout 正常反传，日志出现
  `Lfbsafe / Lcurve`，`fb_valid_rate=0.375~0.875`、`fb_raw_violation≈0.047 →
  fb_safe_violation≈0.016`、`Lfbsafe 0.047→0.024`。
* **完整 16 步闭环**：`tests/test_feedback.py::test_full_reverse_loop_with_the_real_planner_feeds_feedback_back`
  用真实 planner + 真实 ALM + 真实约束包 + 真实 feedback 缓存跑 4 个反向 step
  （走廊用恒真盒子替换，避免依赖随机小模型的椭圆），验证 warmup 无历史、
  最后一个 step 有历史、`delta = 0` 分支与逐 step 诊断字段。

**尚未验证（不能凭上面的冒烟就下结论）**

* 正式的完整训练与完整 16 步推理对照（`raw violation ↓ / ALM correction ↓ /
  smoothness` 三条曲线是否真的改善）；
* `topology: "pi"` 模式下离线走廊与网络自选骨架的相容性（需要真实数据统计
  `fb_topo_match` 与 `fb_valid_rate`）；
* 训练走廊来自**离线 GT 走廊**（m*），推理走廊来自网络自己预测的椭圆 ——
  两者分布并不完全一致。

---

## 8. 取舍与后续

* 按设计文档 §15/§16，第一版**没有**动 `L_alm`，也**没有**把 ALM 自身的平滑目标
  并进来，方便单独判断提升来源；
* 仍未做（设计文档明确留到后续）：最终的 ALM 曲线级平滑目标、
  跨多步（>2）rollout 的反馈链、把推理时的**在线走廊构建**搬进训练
  （当前只能读离线 GT 走廊，这是 §4.1 里 `topology` 选项存在的原因）。

---

## 9. 变更文件

```text
src/models/trajsafe/feedback.py         新增：FeedbackEncoder / FeedbackFusion
src/models/trajsafe/planner.py          接入融合点、forward_all/forward_controls 参数、
                                        _route_topology 的 per-sample 负索道路由、断言
src/models/trajsafe/__init__.py         导出
src/diffusion/sampler.py                per-sample 冻结骨架、feedback 缓存状态机、
                                        trace/result 诊断
src/diffusion/bspline_alm.py            空约束包的退化分支（_violation_stats / lambda_max）
src/geometry/bspline_constraints.py     build_constraint_pack_from_regions（离线走廊 → 连续约束包）
src/losses/losses.py                    pack_violation / pack_max_violation /
                                        feedback_safety_loss / curve_smoothness_loss
train.py                                rollout_timesteps / rollout_pair、两步真实 rollout、
                                        第二步骨架模式、新损失项、loss_parts、fb_* 日志
configs/config_160k8p.yaml              model.feedback / loss / train.feedback /
                                        alm.feedback_accept_tol
tests/test_feedback.py                  新增 20 项
tests/test_alm_state_machine.py         反馈缓存契约、per-sample 冻结骨架、trace 字段
```
