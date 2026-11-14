# 末端去噪器 3 -> 8 层试验

用户明确指定：控制点主干保持1层，末端去噪器从3层增至8层。当前试验在完整 v4_map16_l1 上进行，保留骨架、几何安全与反馈，不是新纯基座训练。只有 model.final_blocks 与隔离输出路径变化。参数量由约3.98M增加到5.796295M。

配置：configs/v4_map16_l1_final8.yaml。继承原训练协议：200 epochs，batch16，seed42，lr2e-4，最长11小时；从零训练，不加载旧模型或冒烟权重。使用现有 E:/CondaEnvData/envs/GGMPC/python.exe，torch2.9.1+cu126，RTX4070。

已完成：模型实例化，验证结构为主干1/骨架2/末端8；仅上述配置差异；32条train/32条val、2批训练及验证完成，输出在 results/depth_l1_final8_smoke。该结果只证明训练可运行，不用于评价效果。

正式训练使用 scripts/train_final8.py --run-id <唯一ID>，保存配置快照、manifest、日志及checkpoint。训练失败和11小时预算未完成200轮均单独标记，不伪称训练完成。训练后再用best_task进行同协议420×3seed评估，并与原1+3比较；未获得结果前不判断8层更好。


## 2026-11-12 用户更改预算

用户明确确认：完整Ours，主干1层/末端8层，100轮，从零重新训练。原200轮任务已停止，输出全部保留并标记stopped_for_restart。新配置 configs/v4_map16_l1_final8_e100.yaml，新run_id depth_l1_final8_e100_20261112_190325；未传resume，未使用旧checkpoint。11小时仍仅为原有保险上限。
