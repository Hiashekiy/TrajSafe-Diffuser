# V5 场引导搜索接入

V5 在线搜索已使用随仓库提供的 `src/field-guided-search`，替换 PDG-Focal A*。
旧 PDG 实验模块保留用于历史实验复现；V5 不再调用它。

## 地图与坐标

- 网络仍接收数据集原始 `occupancy`，不在原张量上做膨胀。
- 搜索先按 `robot_radius_m + safety_margin_m` 做圆形障碍膨胀，默认
  `0.8 + 0.3 = 1.1 m`，地图外也视作障碍。数据中障碍腐蚀不抵消膨胀。
- 新包只接收膨胀后的占据图：源点合法性、引导场、目标距离、EDT 净空、
  风险图和 A* 回退均基于这一张图。曼哈顿目标距离忽略绕障，正是包内推荐模型。
- 项目 scene `(x,y)` 通过 GridSpec 转为包内 `(row,col)`，输出再转回。
- 路径保持 128 点接口，重采样后按最多 0.1 格的采样间隔检查线段，
  若穿过膨胀障碍则报告失败。该检查是离散验证，并非连续几何安全证明。
- 无合法引导源时回退到同图 A*；起终点受阻、不可达或重采样碰撞时显式失败，
  不返回冒充安全解的直线，不移动端点。

圆形尺寸模型沿用原有项目设定；车辆长方形的方向相关足迹不在此模型内。
现有下游 ALM 和最终模型输出碰撞检查仍使用原始地图；本次足迹膨胀保证范围是搜索链路。

## 参数

三份 V5 配置使用包内 README 的端到端推荐配方：
`weight=quadratic, lam=0.01, distance=manhattan, source_method=drop,
num_samples=64, alpha=0.15, gamma=1, adaptive=false,
clearance_beta=0.10, clearance_mode=quadratic_hinge, clearance_threshold=3`。
这些参数在包内合成地图上筛选，并非本项目 checkpoint 的全参数调优结果。
不再运行旧版额外路径居中，避免其使用原始地图距离场。

## 入口与缓存

- 在线：`src/diffusion/sampler_v5.py` → `src/grid_search/field_search.py`。
- 直接从随附目录加载包，避免环境中旧 pip 版本覆盖它；需要现有环境中的 SciPy。
- `scripts/data/new_framework/12_build_pdg_paths.py` 同样调用新的共享搜索接口。
  为兼容数据加载器保留 `pdg_path_*.npy` 文件名；断点文件改为 `_field_partial*`，
  防止混用历史 PDG 断点。
- 已有离线路径和椭圆标签没有自动重写。当前 checkpoint 可直接用于新在线搜索；
  如要重新训练，需要先重建离线路径，再重建以路径为中心的椭圆标签。
  旧的 `14_center_pdg_paths.py` 是历史后处理，不属于新推荐流程。

## 验证

新增集成测试覆盖膨胀与边界、原始输入不变、字段与距离图语义、坐标转换、
绕障和明确失败。与原 V5 测试共 11 项通过；包内 62 项测试通过。
真实 checkpoint 的评估产物保存在 `outputs/field_search_smoke` 和
`outputs/field_search_test420`；统计以各目录中的 `summary_test.json` 为准。

本次 420 样本（现有 Q128 epoch65 checkpoint，4 步，batch=1）：搜索成功
420/420、A* 回退 0 次；搜索中位耗时 28.35 ms，整链路中位耗时 134.875 ms，
扩展节点中位数 199。现有最终碰撞检查记录 0/420 碰撞，RMSE 0.09746。
最终来源 model 416、last_verified 2、search_path 2。最终碰撞指标仍按原有地图
检查，不能解释成全部模型输出通过了新增的足迹膨胀图检查，也不能据此断言
比历史 PDG 评测更快（本次未做同环境配对测速）。
