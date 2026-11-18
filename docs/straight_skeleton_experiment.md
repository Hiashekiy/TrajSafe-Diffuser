# 直线骨架生成实验

从项目根目录运行（使用已有的 PyTorch 环境）：

```powershell
& 'E:/CondaEnvData/envs/GGMPC/python.exe' scripts/sample_straight_skeleton.py
```

每次以相同概率选择 train / val / test，再均匀随机抽取该集合的一行地图及其原始起终点。默认种子随机产生，打印并保存在 JSON 中。随机抽样可能重复。

脚本不读取预搜索的候选骨架，只读取 conditions.npy 和 occupancy.npy。将稀疏候选 candidate_xy 与稠密几何 candidate_geometry 均替换成一条起点到终点的等距直线，仅有一个有效候选。整个扩散过程使用 guidance=none：无 ALM、无最终投影、无修正反馈。椭圆形状仍由网络预测，中心依模型原有逻辑沿替换后的直线分布。

默认配置 configs/config_160k4p_c48_v4_map16.yaml，权重取该配置 train.ckpt_dir 下的 best_task.pt，16 步采样。传入 --config、--ckpt 可以使用其他匹配的模型；脚本拒绝权重缺失或尺寸适配，避免未训练参数干扰实验。

输出至 outputs/straight_skeleton/：PNG 只画地图、起点、终点、生成轨迹和预测椭圆；NPZ 保存轨迹、控制点、椭圆参数、直线和地图；JSON 保存配置路径、模型路径、数据集、索引、种子等。文件名含时间，避免覆盖结果。

复现实验：

```powershell
& 'E:/CondaEnvData/envs/GGMPC/python.exe' scripts/sample_straight_skeleton.py --seed 42
& 'E:/CondaEnvData/envs/GGMPC/python.exe' scripts/sample_straight_skeleton.py --split test --index 293 --seed 42
```

椭圆较密时，追加 --ellipse-stride 4 可每隔四个画一个，不影响生成。--processed-root 可覆盖数据目录，--out 可指定输出目录。
