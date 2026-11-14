# 新共同基座：地图条件 1+8 去噪网络

用户授权：沿用刚完成的完整Ours 1层主干/8层末端框架训练对比基座。该模型从零训练，不使用Janner预训练权重，不加载Ours或旧D-C48权重。

保留公共模块实现：SceneCNN单16x16地图特征、控制点与位置时间编码、1个TrajBlock、8个FinalDenoiser块、同一个coarse/final共享输出头、48控制点三次B样条、固定端点decoder（profile不变）。隐藏维度128/4头/FFN512等继承。FusionMLP按删除分支后改成单流。

彻底不构建/不调用：骨架编码、候选匹配、拓扑选择、椭圆预测、安全特征融合、走廊、ALM和反馈。forward只有(q_t,occ,cond,t)。公共数据加载器仍会读取既有样本缓存中的其他字段，但模型、损失和验证不读取候选/拓扑/安全标签；没有重做数据集或使用测试GT。

训练：train3850/val1550，batch16，seed42，AdamW lr2e-4、weight_decay1e-4、grad_clip1，100轮，原cosine16步x0噪声设置。val_batches12沿用Ours。端点仍通过输入端点固定与输出固定decoder处理。

损失：保留原第一阶段 control/coarse/smooth/boundary 的实现和权重；删除 topology/shape/IoU/continuous safety。反馈第二阶段及仅该阶段计算的Lcurve随反馈一起移除，不人为添加新正则。GT控制点仅用于train/val去噪监督。不能声称模型不了解障碍：地图和示范本来就含障碍信息。

参数4,366,146。独立模型类 map_only_control_diffuser_v1 与load_baseline严格加载接口；checkpoint含配置、优化器、RNG状态。best_task采用训练内部 noisy-state 去噪验证 RMSE+80*点碰撞率，与正式完整采样评价不同。100轮不保证充分收敛，训练后还需检查验证曲线，七种方法可共用最终冻结的基座。

既有旧D-C48的兼容性阻塞保留在审计历史，新实验改用这一新训练基座，不绕过旧权重不匹配。

正式启动run_id：map_only_l1_final8_e100_20261112_212929。GPU trainer PID40348。两批次train/val冒烟通过；结构、公共参数形状、端点、梯度与严格权重往返2项测试通过。原src/configs/train.py未修改。
