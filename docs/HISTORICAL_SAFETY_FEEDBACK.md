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

---

## 3. 推理侧（`src/diffusion/sampler.py`）

只有 `model.feedback_enabled`（即 `model.feedback.enabled: true`）为真时，采样器才
维护并传递反馈；否则采样器与旧版本行为完全相同。

缓存状态机（`feedback_step()`，设计文档 §4）：

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

`sample()` 的返回值/Trace 新增（设计文档 §18 需要的逐 step 曲线）：

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

* 第一段（ALM / DDIM / feedback 构造）在 `torch.no_grad()` 下执行，
  梯度**只**回到第二次网络预测；
* 走廊直接读离线缓存 `alm_cell_a/b/valid`，用
  `build_constraint_pack_from_regions()` 转成与推理 ALM **完全相同**的
  `BSplineConstraintPack`（精确 Bézier 提取 + 线性不等式），
  因此不需要在训练循环里重建 `SafetyCorridor`（~240 ms/sample）；
* `feedback_valid` 是真实判定结果，不是常数；`feedback.drop_prob` 额外把一部分
  样本强制置 0，模拟 warmup / 无走廊分布；`simulate_warmup: true` 则整批置 0（消融）。

### 4.2 损失（`src/losses/losses.py`）

第二次网络**自己的原始输出**上新增两项：

* `Lfbsafe = feedback_safety_loss(out2["control"], pack)`：
  连续 Bézier 约束包上的违反量（`pack_max_violation`，与推理 ALM 同一对象、
  同一公式），可行时严格为 0；
* `Lcurve = curve_smoothness_loss(out2["control"], q_gt, codec.basis)`：
  先 `B_128 @ Q` 解码，再对**曲线**算二阶/三阶差分（按 detached 的 GT 曲线步长
  归一化）。复制“安全但歪”的 ALM 输出同样会被惩罚。

第二步的 `L_ctrl / L_coarse / L_boundary` 以 `train.feedback.step2_weight`
加权后并入原 key（日志里 `Lctrl=...` 是两步之和）。**没有**
`L_distill = ||Q_next − Q_safe_prev||`（设计文档 §14）。

显存：`stats["loss_parts"] = [total_step1, total_step2]` 两段图互不相交，
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
| 反馈但无安全监督 | `loss.lambda_feedback_safe: 0.0` |
| 反馈但无平滑监督 | `loss.lambda_curve_smooth: 0.0` |
| 看反馈是否被训练到 | 训练日志 `fb_valid_rate`（0 = 分支完全没信号） |
| 看是否真的变安全 | `fb_raw_violation` ↓、`fb_safe_violation` ↓、`fb_correction` ↓ |
| 看逐 step 行为（§18） | `sample(..., return_trace=True)` 的 `raw_violation` / `alm_correction` / `curve_smoothness_*` |

期望曲线：`raw_violation ↓`、`alm_correction ↓`、`curve_smoothness` 不劣化。
若 `fb_valid_rate` 长期为 0，说明 ALM 从未给出“通过验证”的结果 →
调大 `alm.feedback_accept_tol` 或 `train.feedback.alm_inner_steps` / `alm.max_curve_step_scene`。

---

## 7. 已验证

* `python -m pytest tests -q` → **127 passed**
  （新增 `tests/test_feedback.py` 12 项 + `tests/test_alm_state_machine.py` 反馈缓存 2 项）。
* 真实数据训练冒烟（`--overfit 8 --batch-size 4 --max-batches 2 --resume best_task.pt`）：
  两步 rollout 正常反传，日志出现
  `Lfbsafe / Lcurve` 与 `fb_valid_rate=0.375~0.875`、`fb_raw_violation≈0.047 →
  fb_safe_violation≈0.016`、`Lfbsafe 0.047→0.024`（4 个 epoch 内下降）。
* 真实推理冒烟（`best_task.pt`，val 8 个样本 × 16 步）：`alm_status=guided`，
  `feedback` 字段齐全；这批样本上原始预测已满足约束，走的是“already safe”分支
  （`delta = 0`），说明 §4.2 的退化路径正确。

---

## 8. 取舍与后续

* 训练走廊来自**离线 GT 走廊**（`04_build_alm_constraints.py`），推理走廊来自
  网络自己预测的椭圆 → 两者分布不完全一致，但 `L_fbsafe` 用的是与推理同一套
  `BSplineConstraintPack` 数学对象；
* 一步 rollout 的训练成本约 ×2（多一次前向 + 一次 ALM），显存通过
  `loss_parts` 分开反传控制；
* 按设计文档 §15/§16，第一版**没有**动 `L_alm`，也**没有**把 ALM 自身的平滑目标
  并进来，方便单独判断提升来源；
* 仍未做（设计文档明确留到后续）：最终的 ALM 曲线级平滑目标、
  以及跨多步（>2）rollout 的反馈链。

---

## 9. 变更文件

```text
src/models/trajsafe/feedback.py         新增：FeedbackEncoder / FeedbackFusion
src/models/trajsafe/planner.py          接入融合点、forward_all/forward_controls 参数、断言
src/models/trajsafe/__init__.py         导出
src/diffusion/sampler.py                feedback 缓存状态机 + trace/result 诊断
src/geometry/bspline_constraints.py     build_constraint_pack_from_regions（离线走廊 → 连续约束包）
src/losses/losses.py                    pack_max_violation / feedback_safety_loss / curve_smoothness_loss
train.py                                两步真实 rollout、新损失项、loss_parts、fb_* 日志
configs/config_160k8p.yaml              model.feedback / loss / train.feedback / alm.feedback_accept_tol
tests/test_feedback.py                  新增 12 项
tests/test_alm_state_machine.py         反馈缓存契约 2 项 + trace 字段
```
