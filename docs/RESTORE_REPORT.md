# Maze2D 安全扩散规划器 —— 问题排查与恢复报告

> 版本：2026 年，针对“联合/统一训练后椭圆塌缩、large 穿墙、训练损失低但测试差”的问题

## 1. 问题现象

联合统一（umaze/medium/large 缩放+padding 到 [0,8]²）训练后：
- **训练损失很低**（train total≈2.0，L_col≈0.12、L_ecol≈0.43、L_anchor≈0.04）；
- **但测试很差**：
  - umaze：0 碰撞，但椭圆是“大圆环”；
  - medium：0~25% 穿墙（后单场景改善）；
  - large：**68%~92% 穿墙**，轨迹基本直线穿迷宫。
- **共享安全特征/椭圆头退化**：
  - A_t 的 waypoint 标准差 0.14（point-scene 几乎不区分 waypoint）；
  - S_t 标准差 0.27；
  - 模型预测椭圆 r1≈r2≈0.73，ratio≈1.03（圆形），而 GT 椭圆 ratio≈3.5~5.4（细长）。

## 2. 排查过程

### 2.1 先排除“标签错误”
统一数据里 GT 椭圆统计：
```
umaze  r1≈2.28 r2≈0.66 ratio≈3.49
medium r1≈1.33 r2≈0.36 ratio≈3.94
large  r1≈1.30 r2≈0.24 ratio≈5.37
```
→ **GT 标签是贴走廊的细长椭圆**，不是圆。

### 2.2 检查“数据/SDF 一致性”（关键 bug）
用统一数据里的真实轨迹 + 统一 SDF 算穿墙：
```
GT umaze  collision=0.0000
GT medium collision=0.0009
GT large  collision=0.5192   ← 真实轨迹一半在“墙”里！
```
- **原因**：统一 SDF 用“几何法”（把原始墙中心缩放+偏移到 [0,8]² 再 box_sdf）生成，对 large 是**错的**，把真实自由点标成墙；
- 而**统一占据图（重采样版）是对的**（用重采样占据图算：large collision=0.0000）。
- **修复**：统一 SDF 改为**从重采样占据图做 signed distance transform**（`free>0, wall<0`），与地图/轨迹完全一致。修复后：
```
GT umaze 0.0033 / medium 0.0009 / large 0.0000
```

### 2.3 检查“椭圆监督是否在引导”
单批 forward 显示：
```
L_iou=1.00（IoU=0）  L_param=6.67  L_anchor=9.68  L_ecol_raw=37.2
pred r1≈0.73   GT r1≈2.66
```
→ **椭圆损失在非常强地引导**，但模型仍没学会，说明是“局部最优/信息不足”，不是“没监督”。

### 2.4 检查“模型结构是否破坏一致性”
联合/统一期间给 SceneEncoder 加了：
```python
self.pool = nn.AdaptiveAvgPool2d((32, 32))
```
- 这会把卷积输出（原本 40×40，umaze）强制缩成 32×32；
- 但 `map_pos_embed`（地图位置编码）是按 40×40 网格训练的，`scene_xy`/局部采样窗口空间关系随之改变；
- 结果：**用旧 umaze checkpoint（40×40 训练）加载到 32×32 模型时，位置编码/采样对不上**，模型行为错乱。

## 3. 最终根因
1. **统一 SDF（几何法）错误** → large 数据内部不一致 → 模型学不到 large、训练损失低但测试穿墙；
2. **AdaptiveAvgPool2d 改变了 scene 分辨率** → 与旧 checkpoint 的位置编码不一致，破坏模型；
3. **椭圆头输入信息不足**（A_t / S_t waypoint 方差过低）→ 椭圆头困在“安全圆”局部最优。

## 4. 恢复措施
### 4.1 代码恢复
- `src/models/scene_encoder.py`：**去掉 `AdaptiveAvgPool2d((32,32))`**，恢复卷积自然输出（umaze 80×80 → 40×40 scene）。

### 4.2 使用旧 checkpoint
- 采用 `outputs/ckpt/epoch_100.pt`（当时 umaze 100 epoch 好模型）；
- `outputs/ckpt/best.pt` 后来被覆盖，但 `epoch_100.pt` 一直保留。

### 4.3 恢复后 umaze 测试结果
```
test #0  coll=0.000  clear=0.412  p05=0.372
test #1  coll=0.000  clear=0.411  p05=0.373
test #2  coll=0.000  clear=0.412  p05=0.375
test #3  coll=0.000  clear=0.396  p05=0.373
```
- **全部 0 碰撞**，`mean_clearance≈0.40–0.41`；
- 轨迹沿 U/J 走廊从 start 到 goal；
- **椭圆贴走廊、细长**（不再是圆环）。

## 5. 结论
1. **单场景 umaze 模型（epoch_100）已恢复**，效果与之前高光时刻一致；
2. **统一/联合方案目前不可行**，因为：
   - 统一 SDF 曾用错方法，需改为距离变换；
   - 改变 scene 尺寸（AdaptivePool）破坏了与旧 checkpoint 的一致性；
   - 椭圆头信息不足，单靠调参/加信息解决不了，必须增强 point-scene 相对位置 + 椭圆头位置/方向输入。

## 6. 建议
### 路线 A（可靠）—— 单场景各自训练
- 每个迷宫单独训练，用各自地图/坐标/归一化；
- SceneEncoder 保持 conv 自然输出（不统一 scene 尺寸）；
- medium 单场景测试：coll 0.4%~1.2%，`clear≈0.26–0.43`，安全性很好（但椭圆仍偏圆，需后续修椭圆头）。

### 路线 B（联合）—— 若要真正多地图联合，需先满足：
1. 所有迷宫地图**统一到相同像素尺寸**（如都 80×80），并**保持 scene 位置编码一致**（不要用 AdaptivePool 改分辨率）；
2. 统一 **SDF 用距离变换**（与占据图一致）；
3. 增强椭圆分支：point-scene 加“相对位置” + 椭圆头喂 `p_world`，并提高 `lambda_r/lambda_theta/lambda_iou/lambda_anchor`；
4. 平衡 large 采样权重 / 拉长训练。

## 7. 关键产物
- 恢复模型：`outputs/ckpt/epoch_100.pt`
- 恢复后 umaze 图：`outputs/umaze_restored.png`
- 统一数据集构建脚本：`scripts/data/build_mixed_dataset.py`（已改为距离变换 SDF）
- medium 单场景：`outputs/ckpt_medium/best.pt`、`outputs/medium_test.png`
