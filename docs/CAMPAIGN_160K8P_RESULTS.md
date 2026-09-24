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

## 2. test split 同口径 16 步 DDIM+ALM 采样（**16 样本 / seed 0，已作废**）

> **⚠️ 本节结论已被 §2b 的全量 420 样本评估推翻，仅作过程记录保留。**
> 16 个样本下 1 次碰撞 = 6.25%，RMSE 也被少数难样本主导：当时看到的
> "A 不开 ALM 零碰撞""A 的 RMSE 比 B1 低 43%"在 420 样本上都不成立
> （见 §2b）。

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

## 2b. 全量 test（420 样本，seed 0，各模型配对同一噪声）

`--samples 0 --chunk 32`（分块聚合：比率按样本、RMSE 按平方误差和、
逐 step 序列按 guided 加权），四种模型 × {ALM 开, ALM 关}：

| model | **ALM 开**：collisions | rate | rmse_m | OFFLINE 走廊 viol / member | **ALM 关**：collisions | rate | rmse_m | OFFLINE viol / member |
|---|---|---|---|---|---|---|---|---|
| **A_oneshot** | **2/420** | **0.48%** | 17.50 | −0.0313 / 0.9885 | **38/420** | **9.05%** | 17.38 | −0.0257 / 0.9688 |
| B2_feedback | 7/420 | 1.67% | 17.05 | −0.0300 / 0.9898 | 69/420 | 16.43% | 16.98 | −0.0194 / 0.9535 |
| B1_base | 24/420 | 5.71% | 17.72 | −0.0155 / 0.9771 | 165/420 | 39.29% | 18.41 | **+0.0133** / 0.8670 |
| REF_160k8 | 19/420 | 4.52% | 16.57 | −0.0203 / 0.9784 | 123/420 | 29.29% | 16.98 | −0.0030 / 0.9031 |

**结论（全量口径）**

1. **每个模型都需要 ALM**：ALM 把碰撞率压低 3.2~6.9 倍
   （A 9.05%→0.48%、B2 16.4%→1.67%、B1 39.3%→5.71%、REF 29.3%→4.52%）。
   16 样本时" A 不开 ALM 也零碰撞"是样本量假象，**不成立**。
2. **A 仍然是最好的 arm**，两种模式下都领先：开 ALM 0.48% vs 1.67/5.71/4.52%；
   关 ALM 9.05% vs 16.4/39.3/29.3%。
3. **反馈训练确实有用**：两个开 feedback 的 arm（A、B2）开 ALM 后 0.48%/1.67%，
   而没训过 feedback 的 B1 是 5.71%、跨缓存的 REF 是 4.52% —— 差 3~9 倍；
   模型无关的 GT 走廊隶属率也同序（0.9885/0.9898 > 0.9771/0.9784）。
4. **RMSE 不能用来区分这几个模型**：420 样本下四者都在 16.6~18.4 m
   （差异 ~10%，且难样本主导），16 样本时的" A 比 B1 低 43%"是噪声。
5. 稳健的是**逐步诊断的量级差**：A/B2 的 raw violation 与 ALM 修正量比
   B1/REF 低 1~2 个数量级（见 §2 的逐 step 表），碰撞率排序与之一致。

多 seed（1、2）的全量评估在后台补跑，用于给出配对的多 seed 均值/方差。

## 3. 结论（16 样本阶段，已被 §2b 修正）

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

## 5b. 去噪步数消融（A_oneshot，全量 420 样本，seed 0）

`sample(steps=k)` 用 `pick_times(16, k)` 对 16 级训练 schedule 做均匀子采样
（`steps=4` → `t = 15, 10, 5, 0`）。warmup 是按**已执行的反向前向次数**数的，
所以短 schedule 必须同时调低 `warmup_reverse_steps`，否则引导段会被吃掉。

| schedule | 反向前向 | 碰撞 | 率 | rmse_m | OFFLINE member | sampler viol |
|---|---|---|---|---|---|---|
| 16 步（w3，基线） | 16 | 2/420 | 0.48% | 17.50 | 0.9885 | −0.0298 |
| 8 步（w3） | 8 | 2/420 | 0.48% | 17.48 | 0.9890 | −0.0302 |
| 8 步（w1） | 8 | 2/420 | 0.48% | 17.50 | 0.9889 | −0.0299 |
| **4 步（w1）** | **4** | **2/420** | **0.48%** | 17.47 | 0.9888 | −0.0298 |
| **4 步（w2）** | **4** | **2/420** | **0.48%** | 17.39 | 0.9898 | −0.0303 |

* 碰撞的**始终是同 2 个样本**，其余指标差异都在噪声内 → 在这条链路上，
  DDIM 跳步几乎无损（x̂₀ 本身就准，剩下交给逐步 ALM）；把步数从 16 降到 4
  不会牺牲任何指标。
* 单样本延迟（GPU 空闲时）：16 步 ≈ 1.05 s、8 步 ≈ 0.83 s、4 步 ≈ 0.71 s、
  2 步 ≈ 1.3 s（首次含 CUDA 核启动）。**注意不是线性下降**：在线建走廊 +
  激活尝试 + 单样本前向的固定开销约 0.3-0.5 s 是下限。
* 结论：dashboard 默认用 **4 步**（`sample.py --steps 4` 同样可用），
  需要"完整 16 帧回放"时再切 16 步。

`sample()` 现在还支持显式非均匀 schedule：`times=[15,14,13,12,9,6,3,0]`
（预热段保持细步长、激活后跳步），见 `src/diffusion/sampler.py` 的文档。

## 6. 图（`python scripts/plot_campaign.py --samples 4 --steps 16`）

```text
outputs/figures/campaign_panels.png     4 样本 x 4 模型 轨迹面板：
                                        占据栅格 + 各模型自己的走廊 + GT 曲线 +
                                        预测曲线 + 预测椭圆 + 起终点
outputs/figures/campaign_per_step.png   逐步诊断：raw violation / ALM 修正量 /
                                        ALM 引起的曲线粗糙化(%) vs 反向步
outputs/figures/campaign_val_curves.png 训练曲线：val task / curve RMSE /
                                        collision / fb_valid_rate vs epoch
```

图里能直接看到的：
* 逐步诊断图上 A_oneshot（蓝）的 raw violation 比 B1/REF 低 1~2 个数量级，
  且 ALM 修正量同步变小；粗糙化那一栏 A 贴 0%，B2 ≈ +20%，B1 ≈ +32%，
  REF ≈ +65%。
* 训练曲线上 A 的 task 从 epoch 20 起就压在 10~12，B1 停在 15~20 且抖动大，
  B2 在第 25 个 epoch 接上 B1 后继续下探到 ~12；`fb_valid_rate` 在 A 上前
  15 个 epoch 从 0.20 爬到 0.73。
* 轨迹面板同时说明 4 个样本**不足以**给模型排名：例如 sample 2 上 A 的余量只有
  −0.001 而 RMSE 9.3 m，B2/REF 在同一格反而更好（−0.036/4.8 m、−0.043/5.8 m）。

## 7. ALM 消融：关掉走廊与 ALM（ablation A，纯扩散）

同口径（test 16 样本 / 16 步 / seed 0），只把 ALM 关掉（`--ablation A`，
`sample()` 直接返回网络自己的 x̂₀）：

| model | ALM 关：collision | rmse_m | GT 走廊隶属 | ALM 开：collision | rmse_m |
|---|---|---|---|---|---|
| **A_oneshot** | **0.0000** | **4.73** | **1.000** | 0.0000 | 5.21 |
| B2_feedback | 0.1875 | 7.52 | 0.964 | 0.0000 | 7.71 |
| B1_base | 0.3125 | 10.05 | 0.892 | 0.0625 | 9.08 |
| REF_160k8 | 0.3125 | 7.47 | 0.899 | 0.0000 | 6.59 |

（GT 走廊隶属 = 用**同一个离线 GT 走廊**衡量的逐点隶属率，与模型无关。）

结论：

1. **A 可以不依赖 ALM 部署**：关掉 ALM 后仍然 0 碰撞、RMSE 4.73 m（比开 ALM 的
   5.21 m 还低，因为 ALM 会把曲线往走廊余量方向推、牺牲一点 GT 贴合度）、
   100% 落在 GT 走廊内。它的"原始预测"就是安全的。
2. **B2 / REF 必须靠 ALM**：关掉后碰撞率 18.75% / 31.25%，开 ALM 才回到 0。
3. **B1（无 feedback）ALM 也救不回来**：关 31.25% → 开 6.25%，四种配置里唯一
   开着 ALM 仍会碰撞的。
4. 注意训练期 val 的 `collision_rate`（B2 只有 0.0105）比这里的**部署回路**指标
   乐观得多：val 是在随机 t 上做单步 x̂₀ 预测后解码 128 点算碰撞，而部署要跑完
   16 步 DDIM，两者不是同一个量。以后判断"能不能不用 ALM"必须以部署回路为准。

复现：

```bash
python scripts/eval_campaign_testset.py --config configs/config_160k8p.yaml     --split test --samples 16 --steps 16 --seed 0 --ablation A     --model A_oneshot=outputs/campaign_a_oneshot/ckpt/best_task.pt ...     --out outputs/campaign_testset_eval_noalm.json     --md  outputs/campaign_testset_eval_noalm.md
python sample.py --config configs/config_160k8p.yaml     --ckpt outputs/campaign_a_oneshot/ckpt/best_task.pt     --split test --num 4 --seed 0 --ablation A --no-trace-plot     --out outputs/figures/A_oneshot_noalm          # 规划效果图（无 ALM）
```

图：`outputs/figures/{A_oneshot,B2_feedback,B1_base,REF_160k8}_noalm/`
（`samples_test_0_s0_ablA_*.png`，标题里 `alm=disabled`）。

## 8. 产物

```text
outputs/campaign_a_oneshot/{train.log,training_summary.json,ckpt/{best,best_task,latest}.pt}
outputs/campaign_b1_base/{train.log,training_summary.json,ckpt/...}
outputs/campaign_b2_feedback/{train.log,training_summary.json,ckpt/...}
outputs/campaign_compare.md          训练侧 val 曲线 + fb_* 诊断
outputs/campaign_testset_eval.{json,md}   test 同口径采样评估（ALM 开，含逐 step 表）
outputs/campaign_testset_eval_noalm.{json,md}  ALM 关（ablation A）的同口径评估
outputs/figures/*_noalm/                  无 ALM 的规划效果图（sample.py --ablation A）
outputs/stage_chain_status.json     三个 stage 的起止时间与 exit code
```

---

## 9. 附加：更严格的 p4 测试缓存 `data/carla_processed_160k4p`（仅 test）

**动机.** `160k8p` 用 `05_erode_occupancy.py --k 8` 把障碍腐蚀 8 格（1 格 = 0.625 m，
通道两侧各让出 5 m、合计 **+10 m**），test 自由面积从 18.5% 抬到 35.7%。上表里所有
模型都是在这个"宽走廊"上评估的，所以"碰撞 2/420"可能只是几何太宽松。为了看模型在
**更紧的真几何**下是否仍然安全，额外用 `--k 4`（两侧各 2.5 m、合计 **+5 m**）生成
一套**只含 test** 的缓存，其余流程与 `160k8p` 完全一致（同一个
`configs/config_160k8p.yaml`，走廊配置逐字段相同 —— 已核对，与
`160k8p/alm_constraints_report.json` 的 `corridor_cfg` 完全相等）。

**构建**（`scripts/build_k4p_test.sh`，实测 ~4 分钟 / 48 MB）：

```bash
python scripts/data/carla_full/05_erode_occupancy.py     --source data/carla_processed_160 \
    --out data/carla_processed_160k4p --k 4 --border-mode protect --splits test   # 0.6 s
python scripts/data/carla/01_build_candidates.py         --processed data/carla_processed_160k4p \
    --config configs/config_160k8p.yaml --splits test                             # 8.4 s
python scripts/data/carla/02_build_ellipse_labels.py     --processed data/carla_processed_160k4p \
    --config configs/config_160k8p.yaml --splits test                             # 172 s
python scripts/data/carla_full/04_build_alm_constraints.py --processed data/carla_processed_160k4p \
    --config configs/config_160k8p.yaml --splits test                             # 51 s
python scripts/data/carla/03_validate_processed.py       --processed data/carla_processed_160k4p \
    --config configs/config_160k8p.yaml --splits test                             # VALID=True, 0 error
```

**test（420 样本）逐缓存对比** —— 由 `scripts/check_k4p_cache.py` 实测：

| 缓存 | 自由面积 mean (min) | 直线起终点碰撞 | GT 逐点自由率 | GT vs 自身走廊 可行 / max | `alm_valid` | cells/sample | `shape_valid` |
|---|---|---|---|---|---|---|---|
| `carla_processed_160`（原始） | 0.1854 (0.0508) | 261/420 | 1.0000 | 295/420 / 1.2222 | 0.6762 | 86.6 | 0.9949 |
| **`..._160k4p`（本次）** | **0.2792 (0.0811)** | **234/420** | 1.0000 | 407/420 / 0.9148 | 0.9929 | 127.1 | 0.9994 |
| `..._160k8p`（训练/评估用） | 0.3573 (0.1094) | 220/420 | 1.0000 | 406/420 / 0.8839 | 0.9905 | 126.8 | 0.9997 |

（"直线碰撞" = 起点→终点 128 点直线落在障碍上的样本数；"GT vs 自身走廊" = 用**该缓存
自己的**离线 ALM 走廊衡量的 GT 曲线最大违反量 / 可行样本数 —— 走廊本身是从 GT 路线
提的，所以这里 GT 仍有 ~13 个样本 max≈0.9 m，是走廊拼接处的固有缝隙，k8p 上同样
存在，不是 p4 的问题。）

**结论**

1. **监督信号合法**：三种缓存里 GT 逐点自由率都是 **1.0000**（GT 端点距裁剪边 ≥8.4 格，
   而 `protect` 保证"原图自由 ⇒ 新图自由"，实测 `raw_free ⊆ k4p_free` 逐格成立），
   所以 k4p 上 GT 依旧可行、可直接用来评估。
2. **k4p 确实比 k8p 严格**：自由面积 0.2792 vs 0.3573，直线碰撞 234 vs 220；逐样本
   比较 **234/420 更难、0 更容易**（直线自由率平均低 5.97 个百分点）。顺序单调：
   raw(0.5551) < k4p(0.6451) < k8p(0.7047)。
3. **两套缓存不是嵌套关系**：k4p 整体自由面积更小，但有 72 270 格（0.26%）在 k4p 自由、
   在 k8p 是障碍。原因已定位：`--border-mode protect` 把外侧 **k** 格恢复成原图，k=8
   保护 0–7 环、k=4 只保护 0–3，所以多出来的格子**全部**落在 4≤d≤7 环带内
   （逐带计数 16300/16600/19700/19670）。远离边界（d≥8）时严格嵌套
   （k8p 障碍 ⊆ k4p 障碍，实测 True）。这是保护规则本身的性质，不是数据错误。
4. 两个困难样本在 k4p 上更难（直线自由率 `test_0167` 0.234、`test_0291` 0.336，
   k8p 为 0.313 / 0.430），但它们的 GT 依旧全自由、且 100% 落在自身走廊内。

**在 k4p 上评估**（无需新建 config，`--processed` 覆盖 `data.processed_root`）：

```bash
python scripts/eval_campaign_testset.py --config configs/config_160k8p.yaml \
    --processed data/carla_processed_160k4p --split test --samples 0 --steps 4 --seed 0 \
    --model "A_oneshot=outputs/campaign_a_oneshot/ckpt/best_task.pt" \
    --model "B2_feedback=outputs/campaign_b2_feedback/ckpt/best_task.pt" \
    --out outputs/eval_k4p_testset.json --md outputs/eval_k4p_testset.md
```

冒烟已验证：2 样本 / 4 步 / ALM 开 → `A_oneshot` collision 0，`OFFv=-0.0331`、隶属 1.000。

**注意**：本次**只构建了 test**（train/val 未生成）。若要在 k4p 上训练，把上面的
`--splits test` 去掉重跑（train 3850 + val 1550，按本次速率约 35–40 分钟）。

### 9.1 k4p 实测（只跑了 A_oneshot，与 k8p 同口径：test 420 / 16 步 / seed 0 / ALM 开）

```bash
python scripts/eval_campaign_testset.py --config configs/config_160k8p.yaml \
    --processed data/carla_processed_160k4p --split test --samples 0 --chunk 32 \
    --steps 16 --seed 0 \
    --model "A_oneshot=outputs/campaign_a_oneshot/ckpt/best_task.pt" \
    --out outputs/eval_k4p_A_oneshot_alm.json --md outputs/eval_k4p_A_oneshot_alm.md
```

| 指标 | k8p (宽走廊) | **k4p (紧走廊)** |
|---|---|---|
| collision | 2/420 = 0.48% | **11/420 = 2.62%** |
| free_rate | 0.9997 | 0.9981 |
| curve_rmse_m | 17.50 | 17.90 |
| 自身走廊隶属 | 0.9970 | 0.9859 |
| max_constraint_violation | −0.0298 | **−0.0122** |
| 离线 GT 走廊隶属 / maxviol | 0.9885 / −0.0313 | 0.9796 / −0.0090 |

逐样本（`per_sample_collision`）：

- **k8p 上碰撞的 167 / 291，在 k4p 上一个都没被"修好"**，而且更糟：free_rate
  0.9551→0.9258、0.9316→0.8672；
- **新增 9 个碰撞**：{2, 10, 189, 206, 285, 307, 313, 325, 342}，全部是**边缘擦碰**
  （free_rate 0.84–0.998，maxviol +0.02 ~ +0.14 m）。

结论：k8 腐蚀带来的宽走廊**贡献了相当一部分"安全性"**。机制是余量而不是优化失败 ——
ALM 照常收敛（`guided=1.00`，`fb_valid=0.964`），但在紧走廊上收敛后的约束余量从
−0.030 m 缩到 −0.012 m，几乎贴着零，于是 1 cm 级的数值余量不足就无法再兜住碰撞。
换句话说：**"ALM 打开后 0.48%"这个数字里，有 5.5 倍是几何给的**。A_oneshot 是在 k8p 上
训的，所以这个差值同时混着"几何更紧"和"训练/测试分布不一致"两个因素；要分开需要
在 k4p 的 train+val 上重训（本次未构建 train/val）。

### 9.2 dashboard 里切换这两套缓存

`diffusion-dashboard` 的左侧面板新增「处理缓存（地图难度）」下拉（`engine_carla.DATASETS`），
可在 `160k8p` / `160k4p` 之间切换；同一 `test_0167` 在两套缓存上是两张不同的地图，所以
切换后样本与已生成序列都会作废。`dataset` 已加入 HTTP 缓存键与 payload（format 6，
`backend_carla.PAYLOAD_FORMAT`），实测切缓存必然 `cache_hit=false`、同一设置重复才 `HIT`。
`160k4p` 只有 test，划分下拉自动只提供 test。详见 `docs/PROJECT_REPORT.md` 第 6.3 节。

---

## 10. 附加：在「腐蚀前」数据集上重训 OneShot（RAW160_oneshot）

动机：上面所有 arm 都在 k=8 的宽走廊上训。把同一个 recipe（Arm A：feedback 一步到位）
原样搬到**未腐蚀**的 `data/carla_processed_160`（k=0，自由面积 0.185），即可把“安全性里
有多少是几何给的”变成单变量对照。

配置：`configs/config_160raw_oneshot.yaml`（base 继承 `config_160k8p.yaml`，逐 key 只差
`data.processed_root`、`train.ckpt_dir`、`train.epochs/max_hours`）；产物全新目录
`outputs/oneshot_raw160/`，**不覆盖**任何已有 run。65 epoch 跑满，`best_task` 在 epoch 59
（task 12.94），用时 4662 s。

同口径评测（test 420 / 16 步 / seed 0，`scripts/eval_raw160_oneshot.sh`）：

| 评估地图 | ALM | collision | rmse_m | 自身走廊违约 | 走廊隶属 | guided | fb_valid |
|---|---|---|---|---|---|---|---|
| raw160（自己的） | 开 | **122/420 = 29.1%** | 25.03 | **+0.1028（不可行）** | 0.8867 | 0.78 | 0.619 |
| raw160（自己的） | 关 | **219/420 = 52.1%** | 19.23 | +0.0228 | 0.7416 | — | — |
| k8p（跨缓存） | 开 | **3/420 = 0.71%** | 18.07 | −0.0085 | 0.9958 | 0.99 | 0.960 |
| k8p（跨缓存） | 关 | **15/420 = 3.57%** | 18.08 | −0.0315 | 0.9864 | — | — |
| k8p（对照：A_oneshot，k8p 训练） | 开 | 2/420 = 0.48% | 17.50 | −0.0298 | 0.9970 | 1.00 | 0.964 |
| k8p（对照：A_oneshot） | 关 | 38/420 = 9.05% | 17.38 | −0.0313 | 0.9885 | — | — |

（raw160 test 的直线起终点基线：261/420 = 62% 碰撞；GT 自身 100% 自由。）

结论：

1. **窄地图上训练让裸预测明显更稳**：在 k8p 上关掉 ALM，RAW160 模型 3.57% vs A_oneshot
   9.05%（好 2.5 倍）；开着 ALM 时两者同级（3 vs 2，噪声量级）。即“更难地图上训出来的
   策略对宽地图更鲁棒”，而 ALM 需要做的修正也更少。
2. **在窄地图自身上，当前 recipe 远远不够**：29.1%（开 ALM）/ 52.1%（关），虽然比直线
   基线 62% 好，但离“安全”很远。
3. **ALM 在窄地图上收敛不了**：修正后 max violation = **+0.1028**（正的 = 曲线在走廊外），
   说明冻结走廊上的精确证书在 k=0 几何下不成立；ALM 仍把碰撞从 52.1% 降到 29.1%，但
   它输出的是“尽力而为”的不可行解。这是为什么 raw160 训练集里只有 56.6% 的样本能建出
   GT 走廊（k8p 是 97.8%）——同一个瓶颈。
4. dashboard 已注册 `RAW160_oneshot:best_task/latest`（用
   `configs/config_160raw_oneshot.yaml`），并把 `raw160` 作为第三张地图加进数据集下拉。

## 11. 附加：k=4 腐蚀数据集做成完整三 split 并重训 OneShot（K4P_oneshot）

§9 的 k=4 缓存当时只建了 test。这一节把它扩成 train/val/test 全套并在其上重训，
回答「k=4 这条中间道路（不腐蚀 0.185 / k=4 0.279 / k=8 0.357 自由面积）值不值得」。

### 11.1 数据集

`scripts/k4p_finish_and_train.sh`
（labels → ALM → validate → 训练，全程可断点续跑，逐样本缓存）。

- 腐蚀：`05_erode_occupancy.py --k 4 --border-mode protect`，三个 split 都做。
  自由面积：train 0.186 → 0.283，val 0.339 → 0.448，test 0.185 → 0.279；
  通道两侧各 +2.5 m（1 格 = 0.625 m，共 +5 m）。
- 边界回归检查：外圈 w=1/2/4/8/12/16 的障碍率在 k=4 后 `regressed=0`，
  即 `protect` 语义生效，腐蚀没有吃掉裁剪边界。
- 之后重建全部派生缓存：候选几何、椭圆标签、ALM 约束（`alm_*`）。
  离线 GT 走廊覆盖率（`alm_valid`）test = **99.3%**（§9 记录）。
- `03_validate_processed.py`：`VALID = True`，errors 0 / warnings 0。
  control_fit_rmse 0.0053 m（val）/ 0.0067 m（test），shape_valid 99.96% / 99.94%。

### 11.2 训练

`outputs/oneshot_k4p/`，02:45 → 08:30，**200/200 epoch，5 h 45 min**（20698 s），
3850 train / 1550 val，batch 8 × accum 2，lr 2e-4，`best_task.pt` 在第 **98** 轮
（任务分 10.28；对照 `RAW160_oneshot` 是 ep59 / 12.94，验证集不同不可直接比）。

**200 轮对这个任务是过量的**：验证集总损失的最小值出现在第 5 轮（0.4932），
而它是被拓扑项的过拟合造出来的假信号 —— `val Ltopo` 从第 20 轮起单调恶化
（1.3 → 7.3，末轮 7.32），`ctrl_rmse` 则稳定在 9.5–11.1 m 不再改善。
选点机制挑到的是第 98 轮，所以没有落到坏模型上，但后 100 轮基本是白烧的。
同类训练以后可以把上限压到 ~100 轮，或者给拓扑头加正则/早停。

### 11.3 test 全量评测（420 样本 / 16 步 / seed 0 / chunk 32，**自身腐蚀缓存**）

| 模型 | 测试地图 | ALM 开 | ALM 关 |
|---|---|---|---|
| **K4P_oneshot**（k=4 自训） | `160k4p` | **5/420 = 1.19%** | 50/420 = 11.90% |
| A_oneshot（k=8 训练，§9 记录） | `160k4p` | 11/420 = 2.62% | — |

K4P_oneshot + ALM 的细节：`curve_rmse_m` 17.78 m，`final_max_constraint_violation`
**−0.0177（负 = 可行）**，走廊归属率 0.995，引导成功率 0.998，历史反馈有效率 0.981。

### 11.4 结论

1. **在自己这张腐蚀图上，k=4 自训是目前这条线上最好的**：5/420，比拿 k=8 地图训的
   同一 recipe（11/420）好一倍；加上 ALM 后把裸预测的 11.90% 压到 1.19%（10 倍）。
2. **ALM 在 k=4 上收敛良好**（残差 −0.0177），和 §10 里 k=0 的 **+0.1028（不可行）**
   正好形成对照：k=4 已经把走廊几何改善到「精确证书成立」的程度，k=0 不行。
   这和离线覆盖率一致（k=0 的 GT 走廊只有 56.6% 的样本能建出来，k8p 是 97.8%）。
3. **但碰撞是按「缩小后的障碍物」统计的**：k=4 把每个障碍物在每个方向缩了 2.5 m，
   贴着真墙跑 2.4 m 在这张图上算无碰撞。同一个模型放到未腐蚀地图上是
   184/420（ALM 开）/ 168/420（关）—— 模型确实学到了「离真障碍很近也算安全」。
   也就是说 k=4 提升了**自身一致性**，但没有直接提升**真实障碍下的安全性**；
   要的是后者的话，收尾还得在真地图上做，或者把「到障碍的距离」写进约束，
   而不是靠腐蚀来放宽。
4. 200 轮过量（见 11.2），同类实验建议 ~100 轮。

### 11.5 产物

- 数据：`data/carla_processed_160k4p/{train,val,test}`（含 `erode_report.json`、
  `alm_constraints_report.json`、`preprocess_*_report.json`）
- 模型：`outputs/oneshot_k4p/ckpt/{best_task.pt(ep98), best.pt(ep5), latest.pt(ep200)}`
  + `training_summary.json` + `train.log`
- 评测：`outputs/eval_k4p_own_alm.{json,md}`、`outputs/eval_k4p_own_noalm.{json,md}`
  （脚本 `scripts/eval_k4p_oneshot.sh`，日志 `outputs/logs/eval_k4p_own.log`）
- dashboard：注册 `K4P_oneshot:best_task / latest`（`configs/config_160k4p_oneshot.yaml`），
  数据集下拉里的 `160k4p` 可直接与它配对。

