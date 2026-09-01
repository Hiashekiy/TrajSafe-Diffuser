# TrajSafe-Diffuser — 顺序式「先轨迹后椭圆」安全扩散规划器

本仓库（原名 Neural-IRISDiffuser）实现基于 **零和桥式扩散** 的安全规划框架：
先生成轨迹，再在轨迹点附近用局部几何特征预测**安全椭圆**，并在第三阶段让
椭圆反过来指导轨迹（`Trajectory ⇄ Safety Ellipse`）。

默认混合三张迷宫（umaze / medium / large）在 `data/processed_scene` 上训练。
网络结构和损失设计见 [`docs/顺序式网络方案.md`](docs/顺序式网络方案.md)。

所有超参数/路径集中在唯一配置 `configs/config.yaml`，后端代码不硬编码数值。

---

## 环境

统一使用 conda 环境 GGMPC：

```bash
E:/CondaEnvData/envs/GGMPC/python.exe --version
```

如果换了新的强机器，建议新建同名环境并安装依赖：

```bash
conda create -n GGMPC python=3.10 -y
conda activate GGMPC
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy matplotlib pyyaml h5py
pip install cvxpy   # 离线 IRIS MVIE 求解器用
```

---

## 目录结构

```
configs/config.yaml              # 唯一主配置（参数全部注释）
data/d4rl/*.hdf5                 # 原始 D4RL 数据
data/processed/maze2d_{umaze,medium,large}   # 单迷宫源（含离线 IRIS 标签）
data/processed_scene           # 场景归一化([-1,1]^2)的混合三场景数据（训练用）
scripts/data/                    # 数据准备 / 标签生成 / mixed 构建
scripts/plot/                    # 模型采样 / 绘图
scripts/debug/                   # 调试工具
src/                             # 算法/模型/数据组件（见 src/README.md）
train.py / sample.py / evaluate.py
```

实现过程中踩过的坑与处理见 [PITFALLS.md](docs/PITFALLS.md)。

---

## 如何训练

### 1. 数据准备（如需重建单迷宫源与椭圆标签）

如果 `data/processed_scene` 已经存在，可跳过这一步直接 `train.py`。
重新生成需要按顺序执行（都会读 `configs/config.yaml`）：

```bash
E:/CondaEnvData/envs/GGMPC/python.exe scripts/data/01_prepare_map.py --config configs/config.yaml
E:/CondaEnvData/envs/GGMPC/python.exe scripts/data/02_prepare_trajectories.py --config configs/config.yaml
E:/CondaEnvData/envs/GGMPC/python.exe scripts/data/04_generate_ellipse_labels.py --config configs/config.yaml
E:/CondaEnvData/envs/GGMPC/python.exe scripts/data/05_validate_processed_data.py --config configs/config.yaml
E:/CondaEnvData/envs/GGMPC/python.exe scripts/data/06_visualize_processed_sample.py --config configs/config.yaml
E:/CondaEnvData/envs/GGMPC/python.exe scripts/data/build_mixed_dataset.py --config configs/config.yaml
```

- `04` 只重生成某迷宫标签时可加 `--maze umaze`（默认全部）。
- 生成标签用**离线 IRIS MVIE**（`src/geometry/iris_solver.py`），不依赖 Neural-IRIS 网络。

### 2. 训练（三阶段）

本项目采用**顺序式三阶段**训练：

- **Phase 1 `traj`**：只训练轨迹规划 `L_traj = λ_z·L_Z + λ_p·L_p + λ_s·L_smooth`（**不含避障**，跳过椭圆分支）。
- **Phase 2 `ellipse`**：加载 Phase 1 轨迹权重，**冻结轨迹骨干**并设为 `eval()`（保证 `p̂` 确定），只训练椭圆分支（`EllipseAggregator + EllipseHead + 相对 PE`），并微调 local decoder 最后两层。
- **Phase 3 `joint`**：全部解冻，**不 `detach(p̂)`**，联合训练 `L_traj + λ_E·L_E + λ_align·L_align`，`λ_E` 按 `joint_ellipse.ramp_ratio`（默认 0.2）线性爬坡。

每个阶段的 checkpoint 保存到**独立目录**，互不覆盖：

| 阶段 | 输出目录 |
|---|---|
| traj    | `outputs/ckpt/traj/`    |
| ellipse | `outputs/ckpt/ellipse/` |
| joint   | `outputs/ckpt/joint/`   |

**阶段一（轨迹预训练）：**

```bash
E:/CondaEnvData/envs/GGMPC/python.exe train.py --config configs/config.yaml --phase traj --epochs 100
```

**阶段二（椭圆预训练，加载阶段一 best）：**

```bash
E:/CondaEnvData/envs/GGMPC/python.exe train.py --config configs/config.yaml --phase ellipse --epochs 100 --resume outputs/ckpt/traj/best.pt
```

**阶段三（联合训练，加载阶段二 best）：**

```bash
E:/CondaEnvData/envs/GGMPC/python.exe train.py --config configs/config.yaml --phase joint --epochs 100 --resume outputs/ckpt/ellipse/best.pt
```

**参数说明：**

- `--phase`：`traj` / `ellipse` / `joint`，默认 `joint`。
- `--config`：配置路径，默认 `configs/config.yaml`。
- `--epochs`：训练轮数（默认取配置里的 `train.epochs`）。
- `--resume`：**跨阶段**续训时只加载模型权重（尤其 Phase 2 不恢复 optimizer，因为各阶段参数组不同）；**同阶段**续训会同时恢复 model + optimizer + epoch。
- `--log-interval`：每多少个 batch 打印一条日志（默认 10）。

**训练行为要点：**

- 数据：`data/processed_scene`，三迷宫 sample 级混合，每个 batch 带各自 map/SDF。
- 保存目录：`outputs/ckpt/{phase}/`，每个阶段独立的 `best.pt`、`epoch_N.pt`、`train.log`。
- 由 `--phase` 决定冻结策略：Phase 1/3 全部可训练；Phase 2 冻结轨迹骨干并 `eval()`。
- **阶段损失**：Phase 1 只学 `L_Z + L_p + L_smooth`，**不含避障**；Phase 3 `joint` 才加入 `lambda_collision` 防撞项与椭圆项。
- **平滑**：`lambda_smooth`（默认 0.2）从第 1 个 epoch 生效。
- **验证**用固定中间噪声级别（`t = num_timesteps // 2`），训练时仍用随机 `t`。

**在强机器上可调项（`configs/config.yaml` 的 `train:` 段）：**

```yaml
train:
  batch_size: 32        # 可增大到 64/128（显存足够时）
  num_workers: 0        # Windows 建议 0；Linux 可设 4/8
  epochs: 100           # 训练轮数
  lr: 1.0e-4            # 学习率
  traj_lr: 1.0e-4       # Phase1 轨迹模块学习率
  ellipse_lr: 1.0e-4    # Phase2 椭圆模块学习率
  local_decoder_lr: 1.0e-5  # Phase2 local decoder 微调学习率
  save_every: 10        # 每 N 轮保存一次
  ckpt_dir: "outputs/ckpt"   # 各阶段会再加 phase 子目录
```

---

## 采样

```bash
E:/CondaEnvData/envs/GGMPC/python.exe sample.py --config configs/config.yaml --maze umaze --ckpt outputs/ckpt/joint/best.pt --n 4
E:/CondaEnvData/envs/GGMPC/python.exe sample.py --config configs/config.yaml --maze medium --ckpt outputs/ckpt/joint/best.pt --n 4
E:/CondaEnvData/envs/GGMPC/python.exe sample.py --config configs/config.yaml --maze large --ckpt outputs/ckpt/joint/best.pt --n 4
# 可复现：指定随机种子
E:/CondaEnvData/envs/GGMPC/python.exe sample.py --config configs/config.yaml --maze umaze --ckpt outputs/ckpt/joint/best.pt --n 4 --seed 42
```

- `--maze`：`umaze` / `medium` / `large`。
- `--n`：采样多少条轨迹 / 多少个测试案例。
- `--steps`：扩散去噪次数；不传时用完整 schedule（默认 64）。
- `--seed`：可选。不传则每次随机换一组 test 案例并随机采样；传固定值（如 `--seed 42`）则复现。
- 输出：`outputs/samples/sampled_world_state.npy` + `conditions.npy` + `maze_id.npy`。

## 评估

```bash
E:/CondaEnvData/envs/GGMPC/python.exe evaluate.py --config configs/config.yaml --maze umaze --ckpt outputs/ckpt/joint/best.pt --n 32
E:/CondaEnvData/envs/GGMPC/python.exe evaluate.py --config configs/config.yaml --maze medium --ckpt outputs/ckpt/joint/best.pt --n 32
E:/CondaEnvData/envs/GGMPC/python.exe evaluate.py --config configs/config.yaml --maze large --ckpt outputs/ckpt/joint/best.pt --n 32
# 可复现：指定随机种子
E:/CondaEnvData/envs/GGMPC/python.exe evaluate.py --config configs/config.yaml --maze umaze --ckpt outputs/ckpt/joint/best.pt --n 32 --seed 42
```

- `--n`：评估多少个测试案例。
- `--steps`：扩散去噪次数（默认完整 schedule）。
- 输出：`outputs/evaluate_report_{maze}.json` + `outputs/evaluated_world_state_{maze}.npy`。
- 指标：collision_rate / mean_clearance（在 shared SDF 上算，`[0,8]^2`）。
- `--seed`：同采样，不传随机、传固定值复现。

## 可视化

```bash
E:/CondaEnvData/envs/GGMPC/python.exe scripts/plot/plot_test.py --config configs/config.yaml --maze umaze --ckpt outputs/ckpt/joint/best.pt --n 4
E:/CondaEnvData/envs/GGMPC/python.exe scripts/plot/plot_test_samples.py --config configs/config.yaml --maze umaze --ckpt outputs/ckpt/joint/best.pt --n 4
E:/CondaEnvData/envs/GGMPC/python.exe scripts/plot/plot_sample_visuals.py --n 6
E:/CondaEnvData/envs/GGMPC/python.exe scripts/plot/plot_sample_visuals.py --idxs 0,5,100,500
E:/CondaEnvData/envs/GGMPC/python.exe scripts/plot/plot_reverse_diffusion.py --config configs/config.yaml --maze umaze --ckpt outputs/ckpt/joint/best.pt
E:/CondaEnvData/envs/GGMPC/python.exe scripts/plot/plot_reverse_diffusion.py --config configs/config.yaml --maze umaze --ckpt outputs/ckpt/joint/best.pt --mode video
E:/CondaEnvData/envs/GGMPC/python.exe scripts/plot/plot_reverse_diffusion.py --config configs/config.yaml --maze umaze --ckpt outputs/ckpt/joint/best.pt --mode video --no-ellipses
E:/CondaEnvData/envs/GGMPC/python.exe scripts/plot/plot_reverse_diffusion.py --config configs/config.yaml --maze umaze --ckpt outputs/ckpt/joint/best.pt --mode video --video-out outputs/rev.gif --fps 10 --video-every 2
E:/CondaEnvData/envs/GGMPC/python.exe scripts/plot/plot_training_curves.py --log outputs/ckpt/joint/train.log --out outputs/training_curves.png
```

- 输出：`outputs/test_{maze}.png`、`outputs/test_sample_visual.png`、`outputs/test_reverse_diffusion.png`、`outputs/training_curves.png`。
- `--seed`：同上，不传随机、传固定值复现（`plot_reverse_diffusion.py` 用 `--test-idx` 指定案例，`--seed` 复现采样噪声）。
- 所有地图类可视化都会用 `src/utils/visualization.py` 的 `set_map_limits()` 把坐标轴固定成地图范围，超出地图的轨迹/椭圆会被裁剪掉。
- 轨迹绘制已改为“时间渐变线段”，底层用 `src/utils/visualization.py` 的 `draw_traj()`；可通过 `marker_every` / `arrow_every` / `lw` 调整，其中 `marker_every=0` 隐藏中间散点、`arrow_every=0` 隐藏方向箭头（当前默认两者都为 0，只画渐变线 + 起终点）。
- `plot_reverse_diffusion.py` 的 `--mode image|video`：`--mode video` 会把每一步去噪过程录成 GIF/MP4（`--video-out`、`--fps`、`--video-every` 可调）；`--no-ellipses` 可关闭椭圆绘制。
- `plot_test.py` / `plot_test_samples.py` / `plot_sample_visuals.py` 的子图数量按 `--n` 动态生成（默认 2 列；奇数个会隐藏最后一行空面板）。`plot_sample_visuals.py` 还可用 `--idxs 1,5,100` 指定样本索引。
- `plot_training_curves.py` 默认读 `logs/train.log`；要看当前 100 epoch 的训练曲线，用 `--log outputs/ckpt/joint/train.log`（还可用 `--out` 指定输出图）。

---

## 交互式路径规划

可视化之外，新增了一个交互式程序，可以自己画障碍/迷宫、选起点/终点，再运行训练好的模型并动态展示去噪过程。

```bash
E:/CondaEnvData/envs/GGMPC/python.exe scripts/interactive/plan_interactive.py --config configs/config.yaml --ckpt outputs/ckpt/joint/best.pt
```

用法：

- 左键从一角拖到另一角：**框选填充一个矩形墙**（黑色）。
- 右键从一角拖到另一角：**框选擦除矩形**；按 `e` 后左键拖动也进入擦除矩形模式。
- 按 `s` 后点击：设定起点（绿色 `*`）。
- 按 `g` 后点击：设定终点（红色 `*`）。
- 按 `p`：用当前地图运行模型，动态展示逐步去噪，最后显示最终轨迹。
- 按 `c`：清空地图和起终点。
- 按 `q`：退出。

> **关于快捷键**：这个程序已经把 matplotlib 自带快捷键全部禁掉了，避免冲突：
> - `s`（保存）禁用
> - `q`（退出）禁用
> - `p`（平移）禁用
> - `g`（网格）禁用
> - `xscale` / `yscale` 禁用
>
> 所以这些键完全归程序自己的交互逻辑用：
> - `s` → 设起点
> - `g` → 设终点
> - `p` → 规划
> - `c` → 清空
> - `q` → 退出
> - `w` / `e` → 画墙 / 擦除矩形模式
> - 左键拖框 → 矩形墙；右键拖框 → 矩形擦除
>
> 重新运行就不会再弹保存窗口了：
> ```bash
> E:/CondaEnvData/envs/GGMPC/python.exe scripts/interactive/plan_interactive.py \
>   --config configs/config.yaml --ckpt outputs/ckpt/joint/best.pt
> ```

可选参数：

- `--steps`：去噪次数（默认 32）。
- `--grid-size`：地图网格大小（默认 80）。
- `--fps-anim`：动画帧率（默认 30）。
- `--no-ellipses`：不画预测椭圆。
- `--seed`：指定随机种子可复现采样。

## 坐标约定

所有几何/地图数据采用 **D4RL observation/qpos 帧**（见
`src/geometry/d4rl_coordinates.py`）。轨迹状态通道顺序固定为
`[ax, ay, x, y, vx, vy]`，其中 `[x, y]` 为该帧位置。数据集使用
`[0,8]^2` 目标坐标系（`EXTENT`），碰撞/clearance 指标在 shared SDF 上计算。

## 备注

- 椭圆标签由离线 IRIS MVIE 求解器生成（`src/geometry/iris_solver.py`），不依赖 Neural-IRIS 网络。
- 原 RGG 对照 base（`third_party/leekwoon_maze2d_*`）已删除；本网络不依赖任何第三方预训练权重。
- 采样/评估/绘图脚本默认随机抽 test 案例并随机采样；传 `--seed 42` 可复现。
- 代码组件说明见 [`src/README.md`](src/README.md)。

## 完整一键示例（从零到图）

```bash
set -e
E:/CondaEnvData/envs/GGMPC/python.exe scripts/data/01_prepare_map.py --config configs/config.yaml
E:/CondaEnvData/envs/GGMPC/python.exe scripts/data/02_prepare_trajectories.py --config configs/config.yaml
E:/CondaEnvData/envs/GGMPC/python.exe scripts/data/04_generate_ellipse_labels.py --config configs/config.yaml
E:/CondaEnvData/envs/GGMPC/python.exe scripts/data/build_mixed_dataset.py --config configs/config.yaml
E:/CondaEnvData/envs/GGMPC/python.exe train.py --config configs/config.yaml --phase traj --epochs 100
E:/CondaEnvData/envs/GGMPC/python.exe train.py --config configs/config.yaml --phase ellipse --epochs 100 --resume outputs/ckpt/traj/best.pt
E:/CondaEnvData/envs/GGMPC/python.exe train.py --config configs/config.yaml --phase joint --epochs 100 --resume outputs/ckpt/ellipse/best.pt
E:/CondaEnvData/envs/GGMPC/python.exe evaluate.py --config configs/config.yaml --maze umaze --ckpt outputs/ckpt/joint/best.pt
E:/CondaEnvData/envs/GGMPC/python.exe scripts/plot/plot_test.py --config configs/config.yaml --maze umaze --ckpt outputs/ckpt/joint/best.pt
```