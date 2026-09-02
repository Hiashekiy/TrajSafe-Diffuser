# 项目文件结构与作用说明

## 一、顶层入口
- train.py：单场景训练入口，读一个 config，训对应迷宫，存 epoch_N.pt / best.pt / train.log / train_metrics.csv（核心）
- train_joint.py：多迷宫批级混合训练（旧，顺序遍历）（可归档）
- train.py：多迷宫样本级混合训练（混合数据 + per-sample map/SDF）
- sample.py：读 ckpt 采样（单场景），输出 world state + 指标（核心）
- evaluate.py：读 ckpt 评估测试集，输出 evaluate_report.json（核心）

## 二、configs
- config.yaml：唯一主配置（混合三场景 / 离线 IRIS 标签；所有参数已注释）
- （原 experiment_medium.yaml / experiment_large.yaml 已删除，几何并入 config.yaml 的 mazes: 列表）

## 三、src/geometry
- d4rl_coordinates.py：统一坐标约定（obs 帧、body offset、墙中心、AABB-SDF、碰撞）
- d4rl_geometry.py：重导出 d4rl_coordinates + build_occupancy_grid
- maze2d_env.py：纯 Python 迷宫环境 + MAZES
- maze_occupancy.py：全局 occ + 局部 128×128 patch
- sdf_utils.py：SDF 网格 + 可微采样
- ellipse_utils.py：椭圆参数/Q/掩码/网格
- iris_solver.py：真正离线 IRIS MVIE 求解器（椭圆标签）
- offline_iris_wrapper.py：离线 IRIS wrapper（04 用）
- convex_region.py：椭圆+障碍地图→凸安全区域（超平面生成 / 障碍点筛选 / 半空间转顶点；世界坐标投影）
- （neural_iris_adapter.py / iris_wrapper.py 已归档 _archive/neural_iris_legacy/ 与 _archive/legacy_scripts/；下游解析凸区域构造已重新实现于 src/geometry/convex_region.py）

## 四、src/datasets
- normalization.py：state 归一化
- mixed_dataset.py：三迷宫混合 Dataset + collate（per-sample map/SDF）

## 五、src/diffusion
- schedule.py：DDPM beta schedule
- sampler.py：反向扩散采样 + 端点 inpainting
- conditioning.py：apply_endpoint_condition（端点条件）

## 六、src/models
- scene_encoder.py：占据图 CNN → scene 特征（conv 40×40）
- trajectory_encoder.py：轨迹 token 编码（motion+pos+index+time，无条件）
- position_encoding.py：2D 位置/timestep/index embedding
- local_scene_sampler.py：按 waypoint grid_sample 局部窗口
- point_scene_attention.py：waypoint 查局部场景
- safety_fusion.py：cat(F_traj,A_t)->MLP(LN)
- ellipse_head.py：椭圆头（center/radii/dir）
- trajectory_decoder.py：TransformerDecoderLayer
- trajectory_head.py：轨迹头 -> [B,H,6]
- planner.py：端到端模型

## 七、src/losses
- trajectory_loss.py：L_diff（时变项）/L_collision/L_smooth
- ellipse_loss.py：L_param/L_iou/L_ecol/L_anchor（L_traj_ellipse 已弃用）
- total_loss.py：汇总（含 warmup 分支）

## 八、src/utils
- config.py：load yaml（支持 base 继承）
- seed.py：固定种子
- logger.py：日志
- checkpoint.py：保存/加载
- metrics.py：稠密轨迹碰撞/clearance
- visualization.py：可视化

## 九、scripts（按类别分子目录）
scripts/data/（数据准备 + mixed 构建）
- 01_prepare_map.py：重建地图+校验
- 02_prepare_trajectories.py：切窗轨迹+conditions+归一化
- 04_generate_ellipse_labels.py：离线 IRIS 椭圆标签
- 05_validate_processed_data.py：校验数据
- 06_visualize_processed_sample.py：可视化样本
- build_mixed_dataset.py：从 maze2d_* 一条龙合成 mixed/（缩放+距离变换 SDF+合并分切）

scripts/plot/（模型采样/绘图）
- plot_test.py：混合数据集测试/可视化
- plot_test_samples.py：模型采样 + 轨迹/椭圆可视化（SDF 指标）
- plot_reverse_diffusion.py：模型逆向扩散可视化
- plot_training_curves.py：画训练曲线（读 logs/train.log）
- plot_sample_visuals.py：地图+轨迹+GT 椭圆可视化

scripts/debug/
- overfit_diff.py：只训 L_diff overfit（排查）

## 十、文档
- README.md：简介
- REPORT.md：旧报告
- RESTORE_REPORT.md：本次排查/恢复报告
- PITFALLS.md：踩坑记录
- DATASET_USAGE.md：数据说明
- METHOD.md：方法文档
- DSH_实施指南_Maze2D安全扩散规划器_v2_含数据集踩坑.md：实施指南
- PROJECT_STRUCTURE.md：本文件

## 十一、已归档（_archive/）
- scripts/prepare_mixed.py、plot_medium_samples.py、plot_unified_samples*.py（依赖旧中间层/单场景）
- scripts/03_benchmark_iris.py、src/geometry/neural_iris_adapter.py（Neural-IRIS 网络，_archive/neural_iris_legacy/）
- train_joint.py（旧批级混合）

## 十二、核心保留（当前主线：混合训练 + 离线 IRIS 标签）
- train.py（训练核心）/ sample.py / evaluate.py
- configs/config.yaml
- scripts/data/04_generate_ellipse_labels.py、scripts/data/build_mixed_dataset.py、scripts/plot/plot_test.py
- src/geometry/（d4rl_coordinates/d4rl_geometry/maze2d_env/maze_occupancy/iris_solver/offline_iris_wrapper/sdf_utils/ellipse_utils）
- src/datasets/（normalization/mixed_dataset/data_io）
- src/models/、src/diffusion/、src/losses/、src/utils/

## 十三、数据（data/）
- data/d4rl/*.hdf5：原始 D4RL 数据
- data/processed/maze2d_{umaze,medium,large}：各迷宫处理数据（含离线 IRIS 椭圆标签；scripts/data/04 重生成标签的源）
- data/processed/mixed：混合三场景 [0,8]^2 数据（train 使用；由 build_mixed_dataset.py 一步生成，无中间层）
