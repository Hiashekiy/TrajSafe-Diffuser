# scripts 目录说明

按用途分子目录，均通过 `--config configs/config.yaml` 读取唯一配置。

## scripts/data/ —— 数据准备 + mixed 构建
| 脚本 | 作用 |
|---|---|
| `01_prepare_map.py` | 从原始 hdf5 重建单迷宫地图 + 校验（occupancy/sdf/obstacle_mask） |
| `02_prepare_trajectories.py` | 切窗轨迹 + conditions + 归一化（per-maze） |
| `04_generate_ellipse_labels.py` | **离线 IRIS MVIE** 椭圆标签（`--maze` 可选，默认全部迷宫） |
| `05_validate_processed_data.py` | 校验 processed 数组 |
| `06_visualize_processed_sample.py` | 可视化地图+waypoint+椭圆 |
| `build_mixed_dataset.py` | 从 `maze2d_*` 一条龙合成 `data/processed/mixed`（缩放+距离变换 SDF+合并分切） |

## scripts/plot/ —— 模型采样 / 绘图
| 脚本 | 作用 |
|---|---|
| `plot_test.py` | 模型在指定迷宫上采样并画轨迹+椭圆（SDF 指标） |
| `plot_test_samples.py` | 模型多采样 + 轨迹/椭圆可视化 |
| `plot_reverse_diffusion.py` | 模型逆向扩散（noise→clean）可视化 |
| `plot_training_curves.py` | 画训练曲线（读 `logs/train.log`） |
| `plot_sample_visuals.py` | 地图+轨迹+GT 椭圆可视化 |

## scripts/debug/
| 脚本 | 作用 |
|---|---|
| `overfit_diff.py` | 只训 L_diff 做 overfit 排查 |

## 用法示例
```bash
# 重生成 umaze 椭圆标签
python scripts/data/04_generate_ellipse_labels.py --config configs/config.yaml --maze umaze

# 重建 mixed 数据集
python scripts/data/build_mixed_dataset.py --config configs/config.yaml

# 模型采样/评估
python sample.py --config configs/config.yaml --maze umaze --ckpt outputs/ckpt/best.pt
python evaluate.py --config configs/config.yaml --maze umaze --ckpt outputs/ckpt/best.pt
python scripts/plot/plot_test.py --config configs/config.yaml --maze umaze --ckpt outputs/ckpt/best.pt
```
