# 基线审计 2026-11-12

原分支 v4-map16-fbdelta-safeloss，HEAD b7e8665fff4cf523d1453ca12ef24747bd76d2a5。存在大量用户未提交论文、图表与 scripts/experiments 改动；不 stash/reset/覆盖，不切换分支；新增实验目录隔离。没有自动提交。

用户已明确选 v4_map16_l1 作为 Ours，配置 configs/config_160k4p_c48_v4_map16_l1.yaml；外部模型 D-C48 配置 configs/config_160k4p_c48_d_std.yaml，checkpoint outputs/models/oneshot_k4p_c48_d/ckpt/best_task.pt（76311740 bytes）。资产哈希见 audit_manifest.json。

配置证据：C=48，三次clamped B样条，端点固定；继承 configs/config_160.yaml 的 timesteps=16 / squaredcos_cap_v2。scripts/ch4_campaign.py 默认 seeds=0,1,2 / steps=16；这只是现有脚本协议，尚未将已有正式结果逐记录对应验证。160m/256格，0.625m/格，scene_to_meter=80；坐标[-1,1]，row随+y。k4p数据已有障碍膨胀，不能再次无说明加安全半径。旧 bspline.py 模块说明写32，不能用该旧注释覆盖配置48。

D-C48 禁安全几何特征与反馈，但保留地图生成的骨架候选。不能称完全无骨架先验。src/diffusion/schedule.py 提供VP q_sample及epsilon/x0转换；新score适配遵循相同alpha_bar。Langevin循环不等价DDIM；PDM原VE时间表到VP基座的适配尚需验证。

src/geometry/sdf_field.py 现有梯度采用最近异类格点方向，不是双线性插值值的严格梯度；因此没有直接拿来做公式梯度。独立 constraints.py 使用插值解析梯度并附有限差分测试。外部方法不读取Ours走廊、椭圆或测试GT。

Ours状态机存在激活失败、空约束、ALM不收敛、NaN、边界和端点问题；统一Runner尚未实现，尚不能保证将所有状态映射到失败统计。

环境：默认 Python 3.14 无torch/pytest；Python 3.12有torch/pytest但缺scipy。依赖隔离安装到 .deps312，不改核心环境。初次三组回归测试收集失败（scipy缺失），补齐后结果见 validation.txt。未运行全量pytest，以相关几何/ALM/控制点测试代替。

尚缺：最终有效继承配置快照与checkpoint结构交叉检查、420 scene_id/seed逐项验证、数据split哈希、设备预热计时验证及完整失败映射。不得把这些项目误标完成。

## 已验证结果及阻塞

新公式/梯度测试8项通过；包含既有关键回归共38项通过（6.07s）。独立入口复跑8项通过（0.75s）。

严格加载：用户选定的 TrajSafe-Diffuser v4_map16_l1 与当前配置完全兼容。D-C48 被拒绝：missing=[]、adapted=[]，但 unexpected 为 scene_cnn.up_fuses.* 和 scene_cnn.merge.*（20个权重键），架构自动检测 legacy_curve。当前 src/models/common/scene_cnn.py 已移除这些旧解码层；不能假设忽略它们保持基座语义一致。需恢复与训练时一致的基座网络实现，或提供与当前网络严格匹配的无安全条件checkpoint；不可改Ours、重新训练或默认允许partial-load。该项触及任务书§9 checkpoint/适配红线，S1–S3不得运行。完整错误见 locked_assets.json。

测试集样本数 420，唯一 sample_id 420。这里只冻结ID，不以此断言1260已有记录的种子配对完成。
