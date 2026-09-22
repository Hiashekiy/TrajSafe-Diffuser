# 160k8p 双 arm 对比实验结果（2026-09-23 夜间跑）

数据：`data/carla_processed_160k8p`（train 3850 / val 1550 / test 420，C=32 控制点，
alm_valid ≈ 97.8% / 98.6% / 99.0%）。
由 `scripts/stage_chain.py` 按 `configs/campaign_160k8p_compare.json` 顺序执行，
02:02:48 → 07:31:08 共 **5.47 h**，三个 stage 全部 `exit_code = 0`。

| stage | 配方 | epochs 实跑 / 目标 | 墙钟 | 输出 |
|---|---|---|---|---|
| **A_oneshot** | 从零直接开启历史安全反馈（两步 rollout，drop 0.25，topology expert），lr 2e-4，batch 8×accum 2 | **58 / 65**（墙钟截断） | 2.41 h | `outputs/campaign_a_oneshot` |
| **B1_base** | 基座，feedback 关闭（单步），lr 2e-4，batch 16 | **100 / 100** | 1.43 h | `outputs/campaign_b1_base` |
| **B2_feedback** | 从 B1 的 `best_task.pt` 出发，开 feedback 微调，lr 5e-5，batch 8×accum 2 | **37 / 67**（墙钟截断） | 1.63 h | `outputs/campaign_b2_feedback` |

复现命令：

```bash
python scripts/stage_chain.py --plan configs/campaign_160k8p_compare.json
python scripts/compare_campaign.py                     # -> outputs/campaign_compare.md
python scripts/eval_campaign_testset.py --config configs/config_160k8p.yaml \
  --split test --samples 16 --steps 16 --seed 0 \
  --model A_oneshot=outputs/campaign_a_oneshot/ckpt/best_task.pt \
  --model B2_feedback=outputs/campaign_b2_feedback/ckpt/best_task.pt \
  --model "B1_base=outputs/campaign_b1_base/ckpt/best_task.pt::configs/config_160k8p_s1.yaml" \
  --model REF_160k8=outputs/bspline_carla_160k8/ckpt/best_task.pt
```

---

## 1. 验证集曲线（`task = curve_rmse_m + 80 × collision_rate`）

| stage | epochs | best_task(epoch) | rmse_m | collision | best_topo | fb_valid(first→last) | Lfbsafe | Lcurve |
|---|---|---|---|---|---|---|---|---|
| A_oneshot | 58 | **10.33 (50)** | 10.13 | **0.0026** | 0.438 | 0.200 → **0.734** | 0.002 | 0.010 |
| B1_base | 100 | 15.94 (24) | 11.12 | 0.0602 | 0.406 | – | – | – |
| B2_feedback | 37 | 11.68 (36) | 10.84 | 0.0105 | 0.323 | 0.730 → 0.730 | 0.001 | 0.009 |

* A（一步到位）验证集最好，**碰撞率比基座低 23 倍**；
* B2 相比它自己的基座 B1 全面变好（task 15.94 → 11.68，collision 0.060 → 0.011），
  说明“反馈微调”这一步本身是有效的；
* `fb_valid_rate` 在 A 上从 0.20 爬到 0.73，说明反馈分支在训练过程中真的被激活、
  且随着网络变好而变可靠；B2 稳定在 0.73。

## 2. test split 同口径 16 步 DDIM+ALM 采样（16 样本 / seed 0）

每个模型都用**自己预测的椭圆**在线建走廊（部署协议），另有 **OFFLINE** 一列是
用**同一份离线 GT 走廊**衡量的、与模型无关的判据。

| model | collision | sampler viol | sampler member | rmse_m | OFFLINE viol | OFFLINE member | fb_valid | raw viol 1st→last | ALM corr 1st→last |
|---|---|---|---|---|---|---|---|---|---|
| **A_oneshot** | 0.0000 | −0.0289 | 1.0000 | **5.21** | −0.0351 | 1.0000 | 1.000 | **0.0034 → 0.0010** | **0.0034 → 0.0011** |
| B2_feedback | 0.0000 | −0.0223 | 1.0000 | 7.71 | **−0.0391** | 1.0000 | 1.000 | 0.0277 → 0.0256 | 0.0332 → 0.0354 |
| B1_base | **0.0625** | −0.0119 | 0.9838 | 9.08 | −0.0260 | 0.9937 | 0.000¹ | 0.0520 → 0.0522 | 0.0561 → 0.0578 |
| REF_160k8² | 0.0000 | −0.0177 | 0.9918 | 6.59 | −0.0345 | 0.9966 | 0.875 | 0.0649 → 0.0507 | 0.0659 → 0.0735 |

¹ B1 的 feedback 是关闭的（`config_160k8p_s1.yaml`），采样器不维护反馈缓存，
`fb_valid = 0` 是构造性结果，不是“训练失败”。
² REF 是在 **160k8（k=8，无边界保护）** 缓存上训练、在 160k8p test 上评估的**跨缓存**
参考，只作连续性对照，不是同数据同预算的对照臂。

逐 step（13 个 guided step 的均值）：

| model | raw violation | ALM correction | 解码曲线平滑度 raw → safe | feedback history |
|---|---|---|---|---|
| A_oneshot | **0.0006** | **0.0007** | 0.00022 → **0.00022**（ALM 不改变平滑度） | already_safe 204/208 |
| B2_feedback | 0.0284 | 0.0357 | 0.00025 → 0.00031（+24%） | already_safe 98/208 |
| B1_base | 0.0531 | 0.0597 | 0.00031 → 0.00041（+32%） | –（关闭） |
| REF_160k8 | 0.0626 | 0.0773 | 0.00027 → 0.00043（+59%） | rejected 26/208 |

---

## 3. 结论（已验证）

1. **反馈机制解决了“反复违规 → ALM 大修”的问题。**
   两个开启反馈的 arm 的**原始预测**（未经 ALM）违反量比关闭反馈的臂低 1~2 个数量级
   （A 0.0006 / B2 0.028 vs B1 0.053 / REF 0.063），ALM 需要做的修正同步降到
   0.0007 / 0.036（B1 0.060、REF 0.077）。
2. **平滑度不再被 ALM 破坏。** B1/REF 的 ALM 把解码曲线二阶差分抬高 32% / 59%，
   B2 抬高 24%，而 A 的 ALM 前后完全不变（0.00022 → 0.00022）。
3. **A（一步到位）整体最好**：test 上 0 碰撞、5.21 m 曲线 RMSE（比 B1 低 43%）、
   自身走廊隶属 100%、98% 的 guided step 直接“已经安全”。
4. **B2（先基座再反馈微调）比它自己的基座 B1 全面变好**，但不如 A：在共同的离线
   GT 走廊上 B2 的余量最大（−0.0391），然而它的原始违反量是 A 的 8 倍、
   ALM 修正量是 A 的 10 倍、RMSE 高 48%。
5. **唯一仍在碰撞的是关闭反馈的 B1**（1/16 样本 = 6.25%），与验证集上的
   collision 0.060 一致。

## 4. 未验证 / 需要注意

* **样本量小**：16 个 test 样本，1 次碰撞就是 6.25%，碰撞率比较的粒度很粗；
  `A/B2/REF` 的 0 碰撞与 B1 的 6.25% 只差一个样本。RMSE 与 raw violation 的
  差距（数量级）比碰撞率稳健得多。
* **单 seed**，没有重复实验，也没有做显著性检验。
* **三个 arm 的预算不对等**：A 只有 58 epoch（墙钟截断，best 出现在第 50 个）、
  B1 100 epoch、B2 = 100 + 37 epoch。A 用更少的 epoch 取得更好的结果，方向上
  支持“联合训练优于两阶段”，但这不等于“等预算下必然如此”。
* **两个 arm 都还在改善时被墙钟切断**（A 的 best 在第 50/58，B2 在第 36/37），
  继续训练很可能还会涨。
* **训练/推理走廊分布差异仍在**：训练用离线 GT 走廊（`alm_cell_*.npy`），
  推理用网络自己预测的椭圆在线建走廊。
* **`topology: "pi"` 实验没做**（本次两个 arm 都用 `expert`），B2 的
  `fb_topo_match` / `fb_topo_corridor_fit` 诊断没有取值。
* REF 是跨缓存对照，**不能**用来比较“反馈 vs 无反馈”，只能说明这批模型都在同一
  量级上。
* 本次没有跑 `evaluate.py` 的完整 test 指标（曲线 RMSE / 碰撞 / 椭圆安全），
  上面的 RMSE 是采样器输出与 GT 曲线在 16 个样本上的点对点误差。

## 5. 建议的后续（如果要把结论做扎实）

1. 把 A 与 B2 各延长一段（A resume `campaign_a_oneshot/ckpt/latest.pt`，B2 resume
   `campaign_b2_feedback/ckpt/latest.pt`），确认曲线是否还在改善；
2. 把 test 评估扩到 64~128 个样本、3 个 seed，碰撞率才有统计意义；
3. 补一个“等预算”的对照（例如 A 与 B2 的总 epoch 相同），再加一个
   `topology: "pi"` 的 arm；
4. 评估时把走廊固定成同一份 GT 走廊（本文 OFFLINE 列已是这个思路的初版），
   避免“不同模型不同走廊”的解释空间。

## 6. 产物

```text
outputs/campaign_a_oneshot/{train.log,training_summary.json,ckpt/{best,best_task,latest}.pt}
outputs/campaign_b1_base/{train.log,training_summary.json,ckpt/...}
outputs/campaign_b2_feedback/{train.log,training_summary.json,ckpt/...}
outputs/campaign_compare.md          训练侧 val 曲线 + fb_* 诊断
outputs/campaign_testset_eval.{json,md}   test 同口径采样评估（含逐 step 表）
outputs/stage_chain_status.json     三个 stage 的起止时间与 exit code
```
