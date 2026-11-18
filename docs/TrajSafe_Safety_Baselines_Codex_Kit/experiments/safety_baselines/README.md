# Safety Diffusion Baselines（Codex 工作包）

此目录用于 TrajSafe-Diffuser 的 **7 种已发表安全扩散算法复现实验**；是现有仓库内的独立实验区，并非已实现的算法代码。

## 放置与开始

1. 将本 ZIP **解压到 TrajSafe-Diffuser 仓库根目录**，使任务书路径为 `experiments/safety_baselines/Codex_TrajSafe_Safety_Baselines_Experiment_Plan.md`。
2. 先确认 `papers/` 中的三篇原文 PDF 完整存在，见下文的文件列表与 `SHA256SUMS.txt`。
3. 在仓库根目录启动 Codex，把任务书发给它。第一阶段只执行 Phase 0/1：只读审查、原文逐式对照、测试基线与目录评审，**不得直接实现 7 个算法或跑正式实验**。
4. 审查本地工作树状态，确保不覆盖用户的未提交修改。审查通过后，在独立分支 `experiment/safety-baselines` 开发；全部新实验实现放此目录，Ours 原有代码、checkpoint 与结果原则上只读。
5. 实现通过 S0/S1/S2 阶段验收后，再讨论 S3 正式测试。

## 原文（原名保持不变）

| 论文 | 所对应方法 | 文件 |
|---|---|---|
| SafeDiffuser (ICLR 2025) | RoS-diffuser, ReS-diffuser, TVS-diffuser | `papers/ICLR-2025-safediffuser-safe-planning-with-diffusion-probabilistic-models-Paper-Conference(1).pdf` |
| Constrained Diffusers (NeurIPS 2025) | Projected, Primal-Dual, ALM | `papers/NeurIPS-2025-constrained-diffusers-for-safe-planning-and-control-Paper-Conference.pdf` |
| Constrained Synthesis with Projected Diffusion Models (NeurIPS 2024) | PDM | `papers/NeurIPS-2024-constrained-synthesis-with-projected-diffusion-models-Paper-Conference.pdf` |

## 建议首次发给 Codex 的话

> 请先阅读 `experiments/safety_baselines/Codex_TrajSafe_Safety_Baselines_Experiment_Plan.md`，以及同目录 `papers/` 的三篇原文，严格按 Phase 0 与 Phase 1 执行。先检查本地代码、checkpoint、采样器与论文公式的兼容性，产出本目录 `docs/baseline_audit.md`、`docs/baseline_source_map.md`。目前不要实现算法，不要跑大量实验，不要动原有 Ours 的代码或结果。审查前请先确认 Git 工作树有无未提交修改，不能自动清理。

## 提醒

本工作包仅提供**更新后的详细任务书和三篇 PDF 原文**；没有宣称外部算法已经完成复现。论文 PDF 为用户提供的研究资料，建议保留在本地，不要直接提交到 Git。代码目录由 Codex 按任务书创建。
