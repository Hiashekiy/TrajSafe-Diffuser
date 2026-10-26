# scripts 目录说明

所有脚本用 `E:/CondaEnvData/envs/GGMPC/python.exe` 运行，配置从 `configs/*.yaml` 读。
**2026-10 清理后的实际内容**（旧 Maze2D 流水线脚本已删除）。

## 1. 闭环仿真（当前主线）

| 脚本 | 作用 |
|---|---|
| `carla_online_closed_loop.py` | **CARLA 在线闭环主入口**：`--mode inspect/observe/drive/audit`，输出到 `outputs/closed_loop/<时间戳>/` |
| `start_dashboard_carla.cmd` | 启动 diffusion-dashboard（CARLA 链路） |
| `probe_carla_visibility.py` | 相机可见性排查（35 m / 100 m 高度下车是否可见） |
| `inspect_carla_lane_candidates.py` | 车道候选快速查看 |
| `audit_carla_handover.py` | 换轨（handover）审计 |
| `audit_carla_speed_boundaries.py` | 参考速度边界审计 |

## 2. 数据流水线（重建缓存唯一途径）

```
scripts/data/carla/          80 m / 原始 carla_v1 快照
  00_clean_dataset.py        清洗 + 生成控制点 GT
  01_build_candidates.py     候选通路 S_m / Γ_m
  02_build_ellipse_labels.py 椭圆标签
  03_validate_processed.py   校验 processed 缓存形状

scripts/data/carla_full/     160 m / 256 px 快照
  00_build_processed.py       生成 160 m processed 缓存
  04_build_alm_constraints.py ALM 约束包
  05_erode_occupancy.py       障碍腐蚀（--k / --border-mode protect）
  06_refit_controls.py        控制点重拟合（C=32 → C=48）
  run_pipeline_160.ps1/.sh    一键跑完整 160 m 流水线
```

当前在用的缓存：`data/processed/160m-k4p-c48`（k=4 腐蚀保边 + C=48）。

## 3. 训练与评估

| 脚本 | 作用 |
|---|---|
| `night_train.py` | 训练驱动（deadline / 断点续训 / 自动重启），`--config` `--ckpt-dir` |
| `eval_chapter4.py` | 第 4 章统一评估（方法 = checkpoint × guidance × feedback 四元组） |
| `eval_random_endpoints.py` | 随机端点评估 |
| `audit_paper_constraints.py` / `audit_paper_offline_preflight.py` | 论文口径的约束/预检审计 |

仓库根的 `train.py` / `sample.py` / `evaluate.py` 是训练、采样、评估的主入口。

## 4. 论文实验

| 目录 / 脚本 | 作用 |
|---|---|
| `experiments/` | 实验框架：`common.py`（采样与计时）、`metrics.py`、`exp1_methods.py`（主对比）、`exp2_ablation.py` / `exp2_feedback_process.py`（消融与历史反馈）、`exp3_alm_budget.py` / `exp3_ellipse_prior.py`（ALM 预算、椭圆先验）、`summarize.py` |
| `run_paper_modified_campaign.py` | RA-L 论文 A/D/S 三组 campaign（结果在 `outputs/paper_ral/`） |
| `run_modified_model_experiment.py` | modified model 消融（结果在 `outputs/ablation_hpath/`） |
| `ch4_*.py` / `ch4_*.sh` / `ch4_*.cmd` | 中文论文第 4 章：campaign、性能计时、绘图、报告（结果在 `outputs/chapter4/`） |

## 5. 出图与论文写作

| 脚本 | 作用 |
|---|---|
| `plot/plot_c48_failure_342.py` / `plot_c48_regions_342.py` | C=48 失败样本、区域示意 |
| `plot/make_diffusion_report_figures.py` | 扩散报告配图 |
| `plot_ch4_closed_loop.py` | 闭环过程曲线（读 `outputs/chapter4/`） |
| `plot_dashboard_case293.py` / `plot_random_endpoints.py` | 看板案例、随机端点出图 |
| `paper/make_zh_docx.py` / `sync_zh_md.py` / `sync_figures.py` | 中文论文 docx 生成、正文与插图同步 |
| `generate_thesis_docx.py` | 学位论文 docx 生成 |

## 6. 诊断与剖析（一次性为主）

| 脚本 | 作用 |
|---|---|
| `debug/profile_planner.py` / `profile_stages.py` | 单次在线规划的耗时拆解（网络 / 走廊 / 硬投影） |
| `debug/profile_step.py` | 单步 forward/backward 计时 |
| `_diag_c48_*.py`（5 个） | C=48 版本的激活、评估、失败、命中原因、投影范围诊断 |
| `diagnose_four_step_collision.py` | 4 步采样碰撞诊断 |
| `compare_modified_geometry.py` / `summarize_modified_geometry.py` / `summarize_square_filter.py` | 几何修改消融的对比与汇总 |
| `hpath_diagnostics.py` / `geometry_audit_utils.py` | hpath 分支诊断、几何审计工具 |
| `check_window_regression.py` / `rank_dashboard_samples.py` | 窗口回归检查、看板样本排序 |
| `update_paper_completed_results.py` / `update_paper_no_projection.py` | 论文正文数字回填 |
| `paper_modified_endpoints.py` / `paper_modified_geometry.py` / `paper_modified_timing.py` | 论文三项补充实验 |
| `launch_detached.ps1` | 后台脱离启动 |
