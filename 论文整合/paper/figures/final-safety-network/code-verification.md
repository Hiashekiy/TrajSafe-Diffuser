# 最新配置下的代码核对（2026-10-10 修订）

使用当前工作区，而非旧提交。planner.py 与 fusion.py 文件修改时间为 2026-10-08，论文主配置 paper_modified.yaml 为 2026-10-09。

配置链：paper_modified.yaml → ablation_hpath_without.yaml → config_160k4p_c48_oneshot.yaml → config_160k4p_oneshot.yaml → config_160k8p.yaml。通过 load_config 解析实际继承结果：path_features.enabled=false、safety_query.center_encoding=false，C=48、Q=128、D=128、4 heads、3 final blocks。

1. PathFeatureHead 不构建，h_path=None。安全分支保留独立 safety_query_head，用于骨架安全查询；该模块不产生 H_path。
2. planner.py 中 query=h_ctrl，SafetyControlFusion 计算 H_fuse=H_ctrl+CrossAttention(Q=projection(AdaLN(H_ctrl,h_t)), K/V=projection(H_ell))，H_fuse 为 48×128。图中 H_fuse 对应代码变量 h_safe_ctrl / 输出 A_safety。
3. FusionMLP 的 num_streams=1，直接读取 H_fuse；LayerNorm(128)→Linear(128,256)→GELU→Linear(256,128)。无拼接，无 H_path。
4. FinalDenoiser 依然为3个独立参数 TrajBlock，每块包含三个各自带残差的 AdaLN 子层：相对索引偏置自注意力、全局地图交叉注意力、128→512→128 GELU FFN。各 AdaLN 调制投影独立。
5. head_p 为共享 Linear(128,2)。BoundaryDecoder 为固定无训练参数的端点补偿：Q_bdry=Q_raw+w_s*(S-Q_raw[0])+w_g*(G-Q_raw[-1])，两端各4点的权重是 [1,.75,.5,.25]。
6. B 样条解码48个控制点至128个曲线采样点；避障ALM在下游。paper_modified.yaml 还关闭末端硬投影（final_project=false），本图未加入该模块。
7. 历史反馈若启用，已经在上游 H_ctrl 中。图展示有效候选路径；无候选时最终曲线回退 coarse。

上一版错误原因：使用旧 base 配置，未追踪最新论文配置覆盖 path_features 开关。现图已删除 H_path、三路拼接与 LN(384)，恢复实际单流融合。
