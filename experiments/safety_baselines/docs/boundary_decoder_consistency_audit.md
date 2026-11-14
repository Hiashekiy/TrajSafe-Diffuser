# BoundaryDecoder 训练/推理一致性审计

结论先说：**没有发现破坏性不一致。** BoundaryDecoder 让 x0 预测在每个噪声等级上略好 0.3–1.8%，没有引入拉伸或端点漂移。

## 1. 事实

- 训练损失作用在 `trajectory_raw` 上（`models/point_losses.py: point_x0_loss(out['trajectory_raw'], q)`，另有 `coarse_raw`、`boundary`、`smooth` 项）。
- 推理时 `NativePointTask.clean()` 用的是 `forward_from_memory(...)['trajectory']`，即
  `pin_endpoints(BoundaryDecoder(final), cond)`。
- 因此"训练直接优化的量"与"用于算 score 的量"相差一个 BoundaryDecoder（span=4 的端点修正）+ 端点钉死。

## 2. 量化（`results/phase_a_base_model/sampler_steps_best_task.jsonl` 的 `score` 段，20 案例、同一噪声向量）

|t|σ（scene）|x0 RMSE raw (m)|x0 RMSE decoded (m)|decoded/raw|x0 误差差 (m)|
|---:|---:|---:|---:|---:|---:|
|0|0.037|4.13|4.07|0.985|−0.06|
|64|0.40|3.90|3.83|0.982|−0.07|
|128|1.00|6.15|6.11|0.993|−0.04|
|192|2.54|8.05|8.00|0.994|−0.05|
|255|5194|12.71|12.67|0.997|−0.04|

- 端点误差：两者都被 `pin_endpoints` 钉死，恒为 0（`metrics.endpoint_error_m`），不存在"decoder 把端点拉歪"的问题。
- 平滑性/曲率：decoded 与 raw 的差别只有端点邻域 ±4 个点，二阶差分平均差别 < 0.01 m（同一 JSONL 的 `mean_second_difference_m`）。
- 碰撞：在 DDIM-256 端到端结果里，顶点碰撞率 14/20（best_task）与 raw/decoded 的差异无关 —— 改用 decoded 或 raw 不会改变碰撞结论（碰撞由路径穿越决定，见 `base_model_quality_audit.md`）。

## 3. 判断

1. **不需要删除 BoundaryDecoder**，也不需要把损失改到 `trajectory` 上：decoder 项已被 boundary loss 约束（`lambda_boundary=0.2`、`boundary_coarse_weight=0.5`），实测偏差方向是"更准"。
2. 仍然存在一个**口径事实**需要写进论文：score 用的 x̂0 是 `trajectory`，不是损失直接作用的那一项。这属于"工程后处理 + 端点硬约束"，不是论文算法；报告中应披露。
3. 如果将来要追求严格一致，可选项（未实施，需重训）：让损失直接作用于 `trajectory`（而不是 `trajectory_raw`），或把 boundary loss 权重设为唯一监督。当前无证据表明需要这样做。

## 4. 复现

```powershell
$py='E:/CondaEnvData/envs/GGMPC/python.exe'
& $py -B -W ignore experiments/safety_baselines/scripts/audit_base_sampler.py --steps 16 `
  --indices 0,81,163,244,326,407,489,570,652,733,815,896,978,1059,1141,1222,1304,1385,1467,1549
# x0_rmse_raw_m / x0_rmse_decoded_m 在输出 JSONL 的 score 段
```
