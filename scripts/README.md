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

## scripts/plot/ —— 出图

| 脚本 | 作用 |
|---|---|
| `plot_safety_stack.py` | 对指定样本画 2×2 四宫格：① 提取的骨架 ② 沿骨架的 128 个固定进度椭圆 ③ 由每个椭圆生成的局部凸区域 ④ 安全走廊 + **示意**的修正对比（灰色虚线=起点直连终点的直线，落在通道外的部分用品红高亮；红色实线=沿走廊的修正轨迹）。前三格单色线稿，只有第四格上色 |

```bash
python scripts/plot/plot_safety_stack.py --config configs/config.yaml \
    --split test --ids 59 23 41 17 82 109 10 42 \
    --out outputs/bspline_carla/safety_stack
# --shape-source pred --ckpt <ckpt> 改用网络自己预测的椭圆与 argmax(pi) 候选
# --no-illustration 只画通道，不画示意轨迹
```

输出：`sample_<idx>_stack.png`（每样本一张 2×2 四宫格）、`stage{1..4}_*.png`（每阶段一张 2×4 网格）、
`safety_stack.json`（每样本的骨架规模 / 区域有效数 / 走廊 base+bridge / overlap min·mean）。

采样图由 `sample.py` 直接输出到 `--out`。
