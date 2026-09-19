# scripts 目录说明

所有脚本通过 `--config configs/config.yaml` 读取配置。

## scripts/data/ —— 数据准备

| 脚本 | 作用 |
|---|---|
| `01_build_skeletons.py` | 构建骨架图并缓存 `data/skeleton/skeletons` |
| `02_build_candidates.py` | 离线候选路径 + topology 标签（`S_m` / `Γ_m`） |
| `03_build_ellipse_labels.py` | 椭圆形状标签（`ellipse_shape4_gt` / `shape_valid`） |

```bash
python scripts/data/01_build_skeletons.py               --config configs/config.yaml
python scripts/data/02_build_candidates.py  --config configs/config.yaml
python scripts/data/03_build_ellipse_labels.py       --config configs/config.yaml
```

`data/scenes`（轨迹 / 条件 / 地图）已随仓库提供。从 d4rl hdf5 重建它的
V1 数据准备脚本已删除，需要时从 git 历史取回。

## scripts/debug/ —— 诊断

| 脚本 | 作用 |
|---|---|
| `candidate_recall.py` | 候选路径对 GT 轨迹的召回率（离线标签体检） |
| `profile_step.py` | 单步 forward/backward 计时探针（不训练） |
| `visualize_labels.py` | 可视化标签：候选路径 / 选中拓扑 / 椭圆 |

## scripts/plot/

采样图由 `sample.py` 直接输出到 `--out`。
