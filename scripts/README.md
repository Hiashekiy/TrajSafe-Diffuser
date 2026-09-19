# scripts 目录说明

所有脚本通过 `--config configs/config_v3_skeleton.yaml` 读取配置。

## scripts/data/ —— 数据准备

| 脚本 | 作用 |
|---|---|
| `10_build_skeletons.py` | 构建骨架图并缓存 `data/processed_scene_v3/skeletons` |
| `13_build_skeleton_candidates_v3.py` | 离线候选路径 + topology 标签（V3 的 `S_m` / `Γ_m`） |
| `14_build_ellipse_labels_v3.py` | V3 椭圆形状标签（`ellipse_shape4_gt` / `shape_valid`） |

```bash
python scripts/data/10_build_skeletons.py               --config configs/config_v3_skeleton.yaml
python scripts/data/13_build_skeleton_candidates_v3.py  --config configs/config_v3_skeleton.yaml
python scripts/data/14_build_ellipse_labels_v3.py       --config configs/config_v3_skeleton.yaml
```

`data/processed_scene_v1`（轨迹 / 条件 / 地图）已随仓库提供。从 d4rl hdf5 重建它的
V1 数据准备脚本已删除，需要时从 git 历史取回。

## scripts/debug/ —— 诊断

| 脚本 | 作用 |
|---|---|
| `v3_candidate_recall.py` | 候选路径对 GT 轨迹的召回率（离线标签体检） |
| `v3_profile_step.py` | 单步 forward/backward 计时探针（不训练） |
| `visualize_v3_labels.py` | 可视化 V3 标签：候选路径 / 选中拓扑 / 椭圆 |

## scripts/plot/

V1 / V2 的绘图脚本已删除；V3 采样图由 `sample_v3.py` 直接输出到 `--out`。
