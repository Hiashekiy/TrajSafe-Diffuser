# 二维占据地图准备工具

将已有二维占据地图准备成 **DMA 骨架 + 稀疏曲线拓扑图**。这是独立离线工具，不接入训练、Diffusion 网络或现有骨架模块。

最终采用的流程：

```text
0/1 占据地图
  → DMA 提取（同时计算精确 EDT、最近背景像素索引）
  → scikit-image 细化工作副本
  → 原版 sknw 多重图
  → JSON / NPZ / 可选预览图片
```

## 安装与运行

在项目根目录运行。已有实验依赖已整理到项目级 `.venv-map-preparation` 环境：

```powershell
.venv-map-preparation/Scripts/python.exe -m src.map_preparation
```

默认递归读取 `data/processed/**/occupancy.npy`，固定种子抽取16张不同占据图，使用 **δ=3px、0.625m/px**，输出到 `outputs/map_preparation/`。分辨率须与数据匹配；0.625仅适用于当前160m/256px缓存。δ=3是本次地图上的实用选择，不是普适最优值。

其他环境可安装：

```powershell
python -m pip install -r src/map_preparation/requirements.txt
python -m src.map_preparation --help
```

指定数据与参数：

```powershell
python -m src.map_preparation --input data/processed --count 16 --seed 20261113 --delta-px 3 --resolution 0.625 --output outputs/map_preparation
```

指定单个批量文件及样本索引（例如之前第8张地图）：

```powershell
python -m src.map_preparation --input data/processed/160m-k4p-c48/val/occupancy.npy --indices 1220 --output outputs/map_preparation_one
```

也支持单张 `H×W` 的 `.npy`，使用 `--indices 0`。批量格式支持 `N×H×W` 和 `N×1×H×W`。目录输入只查找 `occupancy.npy`，不会误读其他特征数组。`--indices` 仅用于单文件，覆盖随机抽样数量；普通抽样按原占据图SHA256去重。`--no-plots` 只导出数据。非空输出目录默认拒绝写入，显式 `--overwrite` 才允许覆盖同名文件（不会自动删除其他文件）。

## Python 接口

完整准备流程：

```python
from src.map_preparation import prepare_occupancy_map
from src.map_preparation.export import save_prepared_map

result = prepare_occupancy_map(occupancy, delta_px=3, resolution_m_per_pixel=0.625)
save_prepared_map(result, 'outputs/map_preparation/map_0000')

# result.occupancy         原占据地图，0自由/1障碍
# result.skeleton         原始DMA二值骨架，不覆盖
# result.thin_skeleton    构图用细化副本
# result.distance_px      H×W精确距离变换，像素单位
# result.nearest_background_rc  2×H×W最近背景像素坐标
# result.graph            {'nodes': ..., 'edges': ..., 'audit': ...}
```

只从已有骨架构图：

```python
from src.map_preparation import build_skeleton_graph
graph, thin = build_skeleton_graph(skeleton, distance_px, resolution=0.625)
```

只提取骨架：

```python
from src.map_preparation import DeltaMedialAxis
extractor = DeltaMedialAxis(occupancy == 0)
skeleton = extractor.extract(delta=3)
distance_px = extractor.distance[1:-1, 1:-1]
```

## 文件组织

|文件|职责|
|---|---|
|`delta_medial_axis.py`|DMA 连续线段判定和骨架提取|
|`skeleton_graph.py`|直接调用 scikit-image/sknw；整理节点、曲线和物理特征|
|`pipeline.py`|组合两阶段，复用 EDT，返回 `PreparedMap`|
|`export.py`|准备结果保存入口|
|`visualization.py`|原DMA/细化副本/图三联预览与批量汇总|
|`__main__.py`|输入读取、抽样、命令行和批量处理|

## 数据与坐标约定

- 输入严格是二维0/1二值占据图，**0=自由，1=障碍**。不额外添加road mask。
- 内部坐标是 `[row,column]`，`origin=lower`显示；无转置或翻转。
- 物理坐标 `x=column×resolution`、`y=row×resolution`，左下像素中心是局部零点。这不是CARLA世界坐标，也不是scene归一化坐标；转换到世界坐标需要每帧姿态。
- δ输入始终为像素；图长度、clearance和width为米。EDT与最近背景索引仍为像素单位。
- 地图外部补一圈障碍。最近背景索引已转换回原图坐标，因此可以为-1或H/W，表示图外背景。
- clearance为到**背景像素中心**的距离，width=2×clearance是通道宽度估计。不是精确的连续障碍单元边界距离，也不直接保证机器人安全。

## 图结构与输出

每张图输出 `map_XXXX.json`、`map_XXXX.npz` 和可选 `map_XXXX_preview.png`。批次输出 `manifest.json`、`statistics.csv` 和可选 `summary.png`。文件编号是本次处理顺序，原始路径与索引在source/manifest中保存。

JSON节点保存 `node_id`、`node_type`、代表点 `pixel_rc/x/y`、`pixel_cluster`、`centroid_rc`、`degree` 和clearance。节点代表点在sknw节点簇内选取最靠近质心的像素。类型包括endpoint、junction、isolated、loop_anchor；自环degree计2。

JSON边保存 `edge_id`、`source/target`、`multigraph_key`、有序 `curve_pixels_rc/curve_points`、`length`、`mean/min/max_clearance`、`mean/min_width`。长度沿实际曲线逐段累加；均值为采样点算术平均。平行边及自环保留。

sknw参数固定为 `multi=True, iso=True, ring=True, full=False`。`full=False` 不用节点质心替换边端点，因此边端点属于对应节点的**pixel_cluster**，不一定恰好等于节点代表点；节点区域内部不虚构直线连接。最终边保留细化曲线的全部采样，原始厚骨架另外保存，不强行遍历其全部像素。

NPZ使用 `allow_pickle=False` 可读取：

|数组|内容|
|---|---|
|`occupancy`, `raw_skeleton`, `thin_skeleton`|原图、DMA和构图工作副本|
|`distance_px`, `nearest_background_rc`|复用的EDT和最近背景索引|
|`node_xy`, `node_types`, `node_degree`|节点物理坐标、类型及度数|
|`node_cluster_rc`, `node_cluster_offsets`|节点簇原像素的拼接数组与区间边界|
|`edge_index`|2×E source/target，可有重复节点对与自环|
|`curve_pixels_rc`, `curve_offsets`|有序曲线拼接数组与区间边界|
|`edge_features`, `edge_feature_names`|E×6边特征及列名|
|`resolution_m_per_pixel`|米/像素|

节点/边ID等于数组顺序。第i条边的曲线是 `curve_pixels_rc[curve_offsets[i]:curve_offsets[i+1]]`，乘分辨率并交换列即可得到米制xy。

## 算法来源和边界

DMA依据 Marie等2016论文公式(3)–(5)与Algorithm 4；连续线段检查采用Algorithm 3的距离函数跳步，不做均匀采样。使用精确KDTree连续距离替代论文基于猜想的四投影加速，数值停滞时用解析圆盘/线段区间覆盖判定同一连续条件。严格采用伪代码的 `D(z)>δ`，容差1e-10px；不声称与作者软件逐像素一致。仅实现DMA，未实现FDMA。

构图直接使用上游sknw 0.15，第三方源码未修改。细化仅作用于骨架工作副本，使用scikit-image 0.26.0的 `skeletonize(method='zhang')`。不做旧版自写拆边、不按距离合并节点、不删短支路、不补断裂、不平滑曲线。

必要的一致性检查包括：细化前后8连通分量与数字孔洞一致、所有细化像素均被节点或曲线覆盖、曲线步长合法、图分量与骨架一致。审计记录图cycle rank与数字孔洞差异，并打印警告。**已知sknw可能在八邻域局部三角形上输出微小自环**：之前第12张图即有此情况，保留并报告，不自动删除。不能保证所有输入语义路口都只产生一个节点。

原实验第8张图的十字路口已确认是一个degree=4节点；整图V=14/E=15。迁移时用同一地图核对DMA数组及sknw节点/曲线与已接受版本一致。

来源：

- DMA：https://doi.org/10.1016/j.patcog.2016.02.011
- sknw：https://github.com/Image-Py/sknw
- skeletonize：https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.skeletonize
