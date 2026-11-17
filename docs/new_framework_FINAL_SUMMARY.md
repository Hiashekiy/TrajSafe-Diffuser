# TrajSafe-Diffuser 新框架改造：最终总结

**日期**：2026-11-17
**分支**：`v4-map16-fbdelta-safeloss`
**设计依据**：`TrajSafe-Diffuser 新框架整体改造实施方案.md`
**配套文档**：`new_framework_S0_audit.md`（现状审查）、`new_framework_STATUS.md`（实施/偏差/bug 全过程）、`new_framework_RESULTS.md`（数字明细）

---

## 1. 一句话结论

方案里的新框架（单路径、无 q_coarse、无 TopologyHead、C=32、Q=128、端点朝向、四步闭环推理、
局部/完整凸走廊 + ALM + 全局验证、`last_verified` 输出选择）**已经落地并跑通**，
test 420 个样本的最终输出**碰撞 0 个**（128 点与 512 点两种检查粒度都为 0），
中位单样本延迟 **111.6 ms**（batch=1）。同时在实施过程中发现并修掉了 7 个真实缺陷，
其中 3 个是"看起来通过、其实没生效"的静默失败。

---

## 2. 最终模型与指标

**模型**：`outputs/models/v5_c32q128/ckpt/final_best.pt`
（C=32 控制点 / Q=128 安全椭圆查询 / 128 解码点 / 1 层 Control Backbone / 4 层 Final Denoiser /
4.05 M 参数；训练到 ep73，best = **ep65**，val 0.04298）

### 2.1 test 420（batch=1，居中路径 + 不插点 + `last_verified`）

| 指标 | 值 |
|---|---|
| **碰撞样本（128 点）** | **0 / 420** |
| **碰撞样本（512 点稠密）** | **0 / 420** |
| 越界点 | 0 |
| 曲线 RMSE（vs GT） | **0.0924** |
| 端点误差 | 0.0 |
| 起点 / 终点朝向 cos | 0.968 / 0.965 |
| PDG-Focal A\* 成功 / 回退 | **100% / 0%** |
| 扩展节点中位数 | 220.5 |
| ALM 执行率 → 全局验证通过率 | 22.1% → **100%** |
| `active_path` 更新率 | 22.1% |
| 输出来源 | model 413 / last_verified 4 / search_path 3 |
| 曲线长度中位数 | 96.3 m |

### 2.2 延迟（batch=1，单卡 RTX 4070）

| 项 | 中位数 | p90 | 最大 |
|---|---|---|---|
| **整条链路** | **111.6 ms** | 581 ms | 3493 ms |
| 其中 PDG-Focal A\* 搜索（含引导构建与居中） | 46.2 ms | 375 ms | 3437 ms |
| 其中安全栅格构建 | 16.6 ms | 22.5 ms | 47.7 ms |

尾部（p90）几乎全被少数样本的搜索吃掉（扩展上千节点）。**注意**：所有对外数字必须按
batch=1 报；用 batch=8 时单样本会被摊薄约 8 倍（早期报告里 2164 ms 就是整批时间被误当单样本）。

### 2.3 与 Q=64 的对照（同一套最终代码、同一 test、batch=1）

| 指标 | Q=64（ep138） | **Q=128（ep65）** |
|---|---|---|
| **碰撞样本（128 / 512 点）** | **0 / 420** | **0 / 420** |
| 曲线 RMSE | **0.0870** | 0.0924 |
| 端点误差 | 0.0 | 0.0 |
| ALM 通过率 | 100% | 100% |
| 搜索中位数 | 45.3 ms | 46.2 ms |
| 整链路中位数 | 119.7 ms | 111.6 ms |

**诚实结论**：Q=128 **没有**带来精度收益（RMSE 0.0924 反而略差于 Q=64 的 0.0870），
延迟也基本一样。差别只在训练量（Q=64 训了 138 epoch，Q=128 只到 ep73）以及
ALM 走廊闭合率（训练期 Q=128 明显更好：通过率 0.60 vs 0.35）。也就是说，
**"输出 0 碰撞"是居中路径 + 局部走廊跳退化 + `last_verified` 这套机制带来的，不是 Q 的密度**；
若只追求精度，Q=64 更划算。

---

## 3. 实施内容（逐项对照方案）

| 方案要求 | 落地位置 | 状态 |
|---|---|---|
| 删 q_coarse / TopologyHead | `src/models/trajsafe/planner_v5.py`（新链路零引用；legacy `planner.py` 仅保留用于旧 ckpt 复现） | ✅ |
| 单路径输入 `active_path [B,128,2]` | 同上 + `src/diffusion/sampler_v5.py` | ✅ |
| C=32 / Q=128 / 128 解码点 / 1 层 backbone / 4 层 final | `configs/config_v5_c32q128.yaml` | ✅ |
| 第 1 步无路径无安全无反馈（空 token + absent 嵌入，不是"全零冒充"） | `planner_v5.py` | ✅ |
| 一次 PDG-Focal A\*（失败回退 focal→astar，带统计） | `sampler_v5._search_once` | ✅ |
| 全量椭圆 → 局部/完整凸走廊（重叠 + 桥接） | `src/geometry/safety_corridor.py`（legacy 原样复用） | ✅ |
| ALM 投影 + 全局稠密验证 | `src/diffusion/alm_local.py`（逐行复用 legacy 调用序列） | ✅ |
| 验证通过才更新 `active_path` 并产生反馈 | `sampler_v5` / `train_v5` | ✅ |
| 端点位置精确 + 端点朝向 | `planner_v5.apply_heading`（方向约束，|Q1−Q0| 保持网络自身的长度） | ✅ |
| 训练 rollout 与推理同一闭环 | `src/diffusion/alm_local.py::alm_closed_loop` 被 `train_v5` 与 `sampler_v5` 共用 | ✅ |
| 搜索路径居中（你对根因的判断） | `src/grid_search/path_centering.py` | ✅ |
| 输出安全选择 `last_verified` | `sampler_v5`（逐样本，512 点门） | ✅ |

---

## 4. 这轮发现并修掉的 7 个缺陷（都是实测定位）

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | `alm_ran_rate=0`，ALM 从不生效 | `sampler_v5` 里残留旧副本覆盖了 `alm_local` 的导入，旧副本引用了已删的 `EllipseRegionBuilder` | 删掉 161 行重复实现 |
| 2 | `search_fallback_rate=99.3%` | `_search_once` 重采样后仍用旧 `cum` 做 `np.interp` | 只做一次弧长重采样 |
| 3 | `--steps 16` 直接崩 | `pick_times` 在 steps≥T 时返回 None | 用 `range(T)` 兜底 |
| 4 | 单个退化凸单元废掉整条走廊 | `build_safety_corridor` 遇 `invalid_base_region` 直接 return 空 | **跳过该 cell 并重连相邻 cell**（`skipped_cells` 记录） |
| 5 | 居中把路径推进障碍（raw 0.000 → 居中 0.328） | 落点"吸附最近自由栅格"会瞬移穿墙 | 改为**回退上一步位置** + 限制在场景内 + 安全闸门（不得比输入更差） |
| 6 | 128 点检查漏判 2 个擦角样本 | 采样粒度不足 | 输出选择改用 **512 点稠密检查** |
| 7 | 搜索静默退化成直线（3 个样本） | `LazyGuidance` 在 `n_segments < J` 时列数不匹配抛异常，被 `except` 吞掉 | edge-padding 到 J 列；并把截断 traceback 记入 `search_traceback` |

---

## 5. 性能优化（诚实版）

| 优化 | 手段 | 同进程 A/B 实测 |
|---|---|---|
| 引导场惰性化 | 原实现为**全网格 65536 格 × 128 段**预计算（28 ms/样本）；改为只算搜索碰到的 ~220 个节点 | 引导构建 28 → **1.2 ms** |
| 启发式查表 | `octile_heuristic` 每次扩展每邻居调 Python 函数（≈1.2 万次/样本）→ 一次向量化生成全网格 H 表 | 该部分开销归零 |
| **合计** | | 搜索 **90.4 → 73.6 ms/样本（1.23×）** |

被否掉的方向（有实测）：
* **批处理多边形化**：仓库里早已实现过 `vertices_pairwise_batch`，实测 0.95×（pad 到最大面数使复杂度从 `ΣM³` 变 `N·M_max³`）；我又做了一版黄金测试逐位一致，但真实 cell 上仅 1.4×，端到端≈1× → 放弃并删除原型。
* 还想继续压尾部的话，要动 `focal_astar.py` 的 open-list 维护（每轮全量 argpartition/lexsort）——风险较高，需要先加"扩展数/路径代价不变"的回归测试。

---

## 6. 数据现状与注意事项

| 项 | 现状 |
|---|---|
| 数据根 | `data/processed/160m-k4p-c32q64`（train 3850 / val 1550 / test 420） |
| 控制点 | `control_gt.npy` (N,32,2)，decode RMSE 0.003–0.004 m，端点误差 0 |
| 椭圆标签 | `ellipse_shape4_gt.npy` **(N,128,4)**，有效率 99.6% / 97.1% / 99.4%，半轴均值 a≈19 m、b≈7–11 m；旧的 64 查询版本备份为 `*_q64.npy` |
| 路径缓存 | `pdg_path_points.npy` **已居中**（clearance 均值 6.25→8.82 m），原始 A\* 备份为 `pdg_path_points_raw.npy`；居中后碰撞 0/400（三个 split 都验过） |
| heading | GT 曲线端点切线的**代理方向**（非车辆 yaw），`heading_valid` 全 1 |
| 地图 | k4p 的障碍已被 `binary_erosion(k=4)` 腐蚀 2.5 m，**这不是安全膨胀**；搜索再按 r=0.8+0.3 m 膨胀。净效果仍比真实障碍薄约 2.2 格，**不能声称安全边界精确** |
| 脚本 | `scripts/data/new_framework/{10..14}_*.py`（C=32 / heading / PDG 路径 / 128 椭圆 / 路径居中），均支持 `--limit` 自检与 `--shard` 并行 |

---

## 7. 复现命令

```bash
# 数据（如需从零重建）
python scripts/data/new_framework/10_make_c32_dataset.py
python scripts/data/new_framework/11_build_heading_labels.py
python scripts/data/new_framework/12_build_pdg_paths.py
python scripts/data/new_framework/14_center_pdg_paths.py --root data/processed/160m-k4p-c32q64
python scripts/data/new_framework/13_build_ellipse64_labels.py \
       --root data/processed/160m-k4p-c32q64 --num-queries 128 --shard K 8   # K=0..7
python scripts/data/new_framework/13_build_ellipse64_labels.py \
       --root data/processed/160m-k4p-c32q64 --num-queries 128 --assemble-only

# 训练（Q=128，4 步闭环 rollout）
python train_v5.py --config configs/config_v5_c32q128.yaml \
       --batch 16 --epochs 200 --max-hours 72 \
       [--resume outputs/models/v5_c32q128/ckpt/latest.pt]

# 评估（务必 batch=1 才是真实延迟）
python eval_v5.py --config configs/config_v5_c32q128.yaml \
       --ckpt outputs/models/v5_c32q128/ckpt/final_best.pt \
       --split test --limit 420 --batch 1 --steps 4 --figures 10 \
       --out outputs/v5_final_clean

# 单元测试
python -m pytest tests/test_v5_framework.py -q      # 7 passed
```

---

## 8. 已知限制 / 未完成

1. **训练未跑满**：ep73 / 200 停止（ep40 后平台期，best ep65）。继续训预期再降 2–5%。
2. **训练用的路径缓存来自"修 bug 前"的居中实现**（约有 5% 样本带越界/贴墙点）；推理侧已是修正版，
   数据侧也已重新居中，但**模型本身是在旧缓存上训的**。若要完全一致，需要在修正后的缓存上微调一轮。
3. **地图**：k4p 已腐蚀 4 格，安全边界不可宣称精确（见 §6）。
4. **朝向标签**是代理方向，不是物理 yaw。
5. **反馈分支**（`feedback_delta`）在消融里几乎没有影响（RMSE 0.0985 vs 0.0991），因为训练时
   ALM 有效比例只有 ~60%，反馈多为 invalid；要让这一支真正起作用需要更高的训练侧 ALM 通过率。
6. **尾部延迟**：p90 581 ms 由少数大扩展搜索样本贡献，未优化。
7. **S7 全量消融**（4 步 vs 16 步、朝向开关、反馈开关）已做；**未做**：Q 扫描、局部 vs 全量走廊
   （这一项在 Q=64 阶段做过，Q=128 因关闭插点已不适用）、以及与 v4 旧框架的逐样本对齐比较。

---

## 9. 产物清单

```
outputs/models/v5_c32q128/ckpt/{best_task.pt(=ep65), final_best.pt, latest.pt(=ep73), train_log.jsonl}
outputs/v5_final_clean/            # 主结果（summary_test.json / eval_test.csv / curves_test.npz / 10 张图）
outputs/v5_q128_noheading/         # 消融：关朝向
outputs/v5_q128_nofb/              # 消融：关反馈
outputs/v5_q128_steps16/           # 消融：16 步（120 样本子集）
outputs/v5_q64_final/              # Q=64 对照（同代码同 test）
docs/new_framework_{S0_audit,STATUS,RESULTS,FINAL_SUMMARY}.md
```

其余中间产物（旧评估目录、重复 checkpoint）已按 §10 清理。

---

## 10. 清理记录

见本次清理输出的 DELETE/KEEP 列表（保留：本轮 Q=128 的全部 checkpoint、主结果与 4 个对照/消融目录、
Q=64 的 `final_best_ep138.pt`；删除：所有被后续修复覆盖的中间评估目录、Q=64 的重复权重副本、
以及 `.tmp/` 下的诊断脚本）。
