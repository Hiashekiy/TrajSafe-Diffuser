# 七种论文安全基线复现：DSH交接报告

日期：2026-11-14（北京时间）。工作区：`D:/ProjectDirectory/Neural-IRISDiffuser`。

> **2026-11-14 后续：** 本报告第6节的 P0-2/P0-3 已按新证据处理，P1 部分仍开放。修复内容、测试与全部数字见
> [DSH_FIX_REPORT_20261114.md](DSH_FIX_REPORT_20261114.md)。要点：
> ① PDM 的 γ 约定与 VP/VE 换算修好后，无约束 RMSE 34204 m → 17.8 m；CD 的 19–39 m 主要是冻结日程的噪声下限（16 级 12.4 m）而非公式错误。
> ② "论文循环效果不如旧 SDF 修正器"已量化到根因：waypoint 间距 0.95 m vs 16 级日程最低训练噪声 8.77 m（9.2×），使所有以 p_0 边缘分布收尾的采样器必然交付 ~13× 锯齿折线（实测 CD 21×、PDM 14.4×，无约束 DDIM 0.97×）；旧修正器跑在 DDIM 骨架 + 48 控制点样条上，所以光滑。
> ③ 按"若训练配方必须改变应先给出来源和具体依据"的要求，把共同基座重训为 256 级（`results/map_only_points_t256_e100`，best_task 21.83 优于 16 级的 22.55，网络/数据/损失/100 轮/AdamW 均不变）：锯齿 21–52× → 2.4–5.4×，planning success 0–1/20 → 2–8/20（无约束基线 10/20）。RoS/TVS 未改善，因为它们的 ε/horizon 参数按 N·dt 定义、不能从 16 级平移，列为下一项。
> ④ PDM 的原文整段线段约束已实现并接入正式 runner，但在本任务上预算内解不出可行投影（8 m 级扰动即失败）。
> ⑤ 七个方法仍未通过验收（顶点/线段碰撞仍高）；未做任何 git 提交，Ours 代码与权重未改动。

## 1. 接手先读：当前没有合格的七方法正式比较

**七种方法全部尚未通过完整复现验收。现有高碰撞率不能作为原论文方法差的证据。** 之前一次11,340次的全量运行存在表示转换错误，已撤回方法排名结论；新点模型的160次小批量也暴露出约束遗漏和采样匹配问题。模型训练完成、单元测试通过、循环跑完，都不等于论文方法正确复现。

用户已经停止让当前助手继续排查，要求写交接报告，由DSH接手。本报告生成时未发现命令行包含`safety_baselines`的活动Python进程，没有训练或全量测试在运行。不要自动续跑旧任务。

当前Git HEAD：`b7e8665fff4cf523d1453ca12ef24747bd76d2a5`。工作树有大量用户既有未提交修改；没有为本任务提交代码。**不要reset、clean或覆盖这些改动。** 本轮新增/修改主要位于`experiments/safety_baselines/`；Ours核心实现未因本次点基座改造而修改。

## 2. 用户最终要求（优先于历史方案）

- 要比较七种新论文方法：RoS-diffuser、ReS-diffuser、TVS-diffuser、Projected、Primal-Dual、ALM、PDM。不能拿旧工程SDF修正器改名顶替。
- **只改对比基座**，让它直接输出128个二维轨迹点；不再经过48控制点、B样条解码或回拟合。完整Ours保持现状。
- 对比基座仍采用主干1层、末端去噪器8层、地图特征16×16、100轮训练。新模型已经训练完成。
- 用户允许借鉴旧代码的“约束如何作用于网络输出”这一表示逻辑，**不允许擅自改论文算法**：不能换更新状态、加旧修正循环、加额外先验采样兜底或偷偷换投影目标。
- 要按论文核对约束及其使用方式；不要求全部轨迹零碰撞，但结果不能由遗漏原文机制或错误适配造成。
- 先完成正确性审查与小批量验证，通过后再全量。不要再设置重复中规模测试；可复用的结果必须确认资产、算法、种子和评价完全一致。
- 不用别人训练好的模型冒充本项目训练；不能把Ours的搜索骨架、预测走廊、ALM或GT偷偷提供给其他方法。

原始方案：[Codex_TrajSafe_Safety_Baselines_Experiment_Plan.md](D:/ProjectDirectory/Neural-IRISDiffuser/docs/TrajSafe_Safety_Baselines_Codex_Kit/experiments/safety_baselines/Codex_TrajSafe_Safety_Baselines_Experiment_Plan.md)。其中控制点表示及S1→S2→S3分级计划已被用户后续指令部分覆盖。

## 3. 三篇论文与官方来源

PDF均已在仓库内，不需要用户再次上传：

- [ICLR-2025-safediffuser-safe-planning-with-diffusion-probabilistic-models-Paper-Conference(1).pdf](D:/ProjectDirectory/Neural-IRISDiffuser/docs/TrajSafe_Safety_Baselines_Codex_Kit/experiments/safety_baselines/papers/ICLR-2025-safediffuser-safe-planning-with-diffusion-probabilistic-models-Paper-Conference(1).pdf)
- [NeurIPS-2024-constrained-synthesis-with-projected-diffusion-models-Paper-Conference.pdf](D:/ProjectDirectory/Neural-IRISDiffuser/docs/TrajSafe_Safety_Baselines_Codex_Kit/experiments/safety_baselines/papers/NeurIPS-2024-constrained-synthesis-with-projected-diffusion-models-Paper-Conference.pdf)
- [NeurIPS-2025-constrained-diffusers-for-safe-planning-and-control-Paper-Conference.pdf](D:/ProjectDirectory/Neural-IRISDiffuser/docs/TrajSafe_Safety_Baselines_Codex_Kit/experiments/safety_baselines/papers/NeurIPS-2025-constrained-diffusers-for-safe-planning-and-control-Paper-Conference.pdf)

|论文|方法|重点核对位置|
|---|---|---|
|SafeDiffuser，ICLR 2025|RoS、ReS、TVS|§3–4、Thm.3.4–3.6、式8–14、Algorithm1、附录C.1；第3页区分离散状态与下层连续控制|
|Constrained Diffusers，NeurIPS 2025|Projected、PD、ALM|§5式10–18；Remark1；§6.1式19及在线控制；附录C第22页的Maze噪声步数与参数|
|Constrained Synthesis with Projected Diffusion Models，NeurIPS 2024|PDM|§4.2式8/Algorithm1；§6.3；**附录C.3第15页的整段线段避障**；附录F第19页T10/M100；附录G训练目标影响|

官方链接/历史核对提交：
- SafeDiffuser：https://github.com/Weixy21/SafeDiffuser ，历史记录commit `e41f38ce289175b4294ab25e6565bfeb7d108f3f`。
- PDM：https://github.com/RAISELab-atUVA/Projected-Diffusion ，历史记录commit `992510f7836a0ffbde0ae09ea80326ca2bc204dd`。
- Constrained Diffusers：https://proceedings.neurips.cc/paper_files/paper/2025/hash/31fb284a0aaaad837d2930a610cd5e50-Abstract-Conference.html 。目前**尚未确认官方代码仓库**，不要虚构。

来源记录：[official_sources.json](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/docs/official_sources.json)。没有保证本地存在完整官方工程。最近只在线核对了部分文件；不要把自己的实现叫作已跑通官方源码。

**PDM来源存在必须解决的差异：** 原文Algorithm1写`γ_t=σ_t²/(2σ_T²)`；官方通用`scoremodel.py`的ALD使用`step_size=eps*(sigma/sigma_min)^2`，更新是`x += .5*step_size*score + sqrt(step_size)*noise`。二者的系数约定必须明确，不能凭效果混用。最近访问的官方main中，`main.py`的trajectories采样分支仍有`NotImplementedError`；这次访问main不代表已经核对了上面历史commit的完整轨迹实现。
- https://github.com/RAISELab-atUVA/Projected-Diffusion/blob/main/projected_diffusion/scoremodel.py
- https://github.com/RAISELab-atUVA/Projected-Diffusion/blob/main/main.py

## 4. 已训练模型：保留并可供接手核查

### 4.1 当前应该使用的新对比基座

架构`map_only_point_diffuser_v1`；直接输出`[B,128,2]`；4,366,146参数；主干1＋末端8；只输入地图、起终点、带噪轨迹点与噪声时间；无骨架、拓扑、安全特征、走廊或反馈。没有`bspline`或`num_controls`模型属性。
- [map_only_points.py](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/models/map_only_points.py)
- [point_data.py](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/models/point_data.py)
- [point_losses.py](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/models/point_losses.py)
- [point_sampler.py](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/adapters/point_sampler.py)
- [train_map_only_points.py](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/scripts/train_map_only_points.py)
- [map_only_points_l1_final8_e100.json](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/configs/map_only_points_l1_final8_e100.json)

训练完成100/100轮，耗时2455.05秒（40分55秒），无记录错误。最佳任务检查点epoch96（0基索引，即第97轮）。
- [best_task.pt](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/results/map_only_points_l1_final8_e100_20261114_000521/ckpt/best_task.pt)
- [best.pt](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/results/map_only_points_l1_final8_e100_20261114_000521/ckpt/best.pt)
- [config_snapshot.json](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/results/map_only_points_l1_final8_e100_20261114_000521/config_snapshot.json)
- [training_summary.json](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/results/map_only_points_l1_final8_e100_20261114_000521/training_summary.json)
- [manifest.json](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/results/map_only_points_l1_final8_e100_20261114_000521/manifest.json)

`best_task.pt` SHA256：`3dbc68df085d26d070b1821758af4614be8cfd7fa52f90897546a7157e856d7f`。严格加载入口：`load_point_baseline`，不会直接接受旧控制点架构检查点。

训练数据加载器只读`curve_gt.npy`（现有128点监督）、`occupancy.npy`、`conditions.npy`，不读取`control_gt`或候选路径标签。训练/采样没有B样条操作；但缓存目标`curve_gt`的原始生成过程应另行核对，不能因此声称它是未加工的CARLA原始轨迹。

**训练配方并非已证实与所有论文兼容：** 沿用16级cosine VP加噪、x0点MSE＋粗预测/平滑/边界辅助损失，AdamW2e-4、batch16、100epochs。不能把它自动当成原始NCSN/DSM或任意Langevin采样均可直接使用的score模型。详见第6节。

### 4.2 完整Ours（不要改）

当前完整Ours为1层主干＋8层末端、100轮版本，保留骨架、安全区域、走廊、ALM与反馈，仍是B样条模型。
- [v4_map16_l1_final8_e100.yaml](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/configs/v4_map16_l1_final8_e100.yaml)
- [best_task.pt](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/results/depth_l1_final8_e100_20261112_190325/ckpt/best_task.pt)
- [planner.py](D:/ProjectDirectory/Neural-IRISDiffuser/src/models/trajsafe/planner.py)
- [sampler.py](D:/ProjectDirectory/Neural-IRISDiffuser/src/diffusion/sampler.py)

### 4.3 已废弃的旧对比基座

旧`map_only_control_diffuser_v1`是48控制点版本，不能用于新的原生点实验：[best_task.pt](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/results/map_only_l1_final8_e100_20261112_212929/ckpt/best_task.pt)。旧`train_map_only.py`、`map_only.py`、`baseline_sampler.py`均保留供溯源，不要误认为当前点模型入口。

## 5. 当前实现的代码导航

|部分|文件|真实状态|
|---|---|---|
|七方法分派与结果记录|[runner.py](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/algorithms/runner.py)|候选实现；不是原文验收通过|
|RoS/ReS/TVS循环|[runner.py](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/algorithms/safediffuser/runner.py)|gamma、sigma、QP、微步积分；仍需原文逐项审查|
|Projected/PD/ALM循环|[runner.py](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/algorithms/constrained_diffusers/runner.py)|Langevin、乘子、slack、rho；当前噪声/训练匹配有问题|
|PDM循环|[runner.py](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/algorithms/pdm/runner.py)|当前按正文gamma、T10/M100；sigma与采样来源尚未验收|
|公式内核|[kernels.py](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/algorithms/kernels.py)|纯公式测试通过不代表全链路正确|
|QP与非凸投影求解|[solvers.py](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/algorithms/solvers.py)|有KKT残差检查、缩放、独立不可行性验证；需要继续审查|
|当前点状态适配|[native_points.py](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/adapters/native_points.py)|输出与优化点相同；**目前project仍逐点，不是PDM原文线段投影**|
|地图距离|[exact_map.py](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/adapters/exact_map.py)|free-cell union距离；分段/最近特征处非光滑|
|折线路径评价|[point_evaluator.py](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/adapters/point_evaluator.py)|检查顶点与连线跨越网格的碰撞；无拟合|
|新小批量入口|[validate_point_methods.py](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/scripts/validate_point_methods.py)|当前只完成20val×1seed×8方法；没有Ours|
|新小批量配置|[seven_native_points_t16.json](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/configs/seven_native_points_t16.json)|记录实际使用参数；不应当作已验收标准配置|
|刚补的原文线段约束|[paper_segment_constraints.py](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/adapters/paper_segment_constraints.py)|圆障碍到整段线段距离、稀疏Jacobian、trust-constr内点法；**尚未接入真实地图/正式runner**|
|线段约束单测|[test_paper_segment_constraints.py](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/tests/test_paper_segment_constraints.py)|3项通过：漏检场景、Jacobian、解析投影解|
|原生点接口单测|[test_native_points.py](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/tests/test_native_points.py)|score换算、QP与投影等局部检查|

**入口警告：** `run.py`/`scripts/campaign.py`是旧控制点实验残留，默认配置仍可指向控制点适配；不能直接拿`--stage S3`当新点模型全量入口。新点小批量使用`validate_point_methods.py`。新的合格全量执行链尚未完成。

## 6. 已确认问题与尚未证实的事项

### P0：旧全量的安全状态与交付曲线不一致（已确认，结果撤回）

旧实现为匹配控制点网络，引入可独立移动的128点状态和额外残差，安全处理后又拟合回48控制点；拟合丢掉安全修正。Projected内部顶点可行1260/1260，最终曲线却碰撞1227/1260；PDM对应1187/1260。点状态到最终曲线平均RMS偏差分别21.83m、14.88m。这是适配错误，不是论文效果。新点模型已去掉这条链路。
- [validity_assessment.md](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/results/seven_direct_full_t16_20261113_111204/validity_assessment.md)
- [validity_assessment.json](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/results/seven_direct_full_t16_20261113_111204/validity_assessment.json)
- [summary.md](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/results/seven_direct_full_t16_20261113_111204/summary.md)

### P0：PDM原文整条线段约束漏实现（已确认，尚未完整修复）

附录C.3要求每条线段与障碍不相交；圆障碍约束是圆心到整条线段的最近距离≥半径。当前`NativePointTask.project`逐点投到自由空间，因此不是原文在该任务中的投影可行集。新`paper_segment_constraints.py`只完成圆几何和内点投影单测：还没有经过审计的占据地图→障碍几何转换，没有连接到PDM循环，也没有真实场景验证。不要声称此问题已经修好。

### P0：采样器与共同基座的匹配未通过（已确认现象，根因待继续分解）

同一val案例、相同NumPy初始种子42、去掉约束后的诊断：
|采样循环|更新数|RMSE|
|---|---:|---:|
|当前DDIM名义循环|16|5.949m|
|Constrained Diffusers的当前Langevin循环|128|19.090m|
|当前PDM-VE循环|1000|34116.744m|

这是单案例诊断，不是总体性能比较，也不证明论文方法本身差；它说明在安全约束之外已经有严重的采样/训练匹配问题。需查噪声日程、score目标和单位、VP/VE变换、原始初始化、步长来源及预算，而不是直接发布高碰撞率。

诊断文件：[native_point_unconstrained_sampler_audit.json](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/docs/native_point_unconstrained_sampler_audit.json)。

当前16级cosine的最低非零噪声，在控制/点坐标的前向加噪标准差约8.715m（不是路径RMSE、不是性能下界）；其VE最高噪声很大。不能在未训练噪声级上凭空外推score，也不能未经明确说明就换成256级重训。用户此前拒绝了因16/256选择而暂停任务的做法；最新点模型仍保持16级，若训练配方必须改变应先给出来源和具体依据。

### P1：SafeDiffuser三种的数值积分/假设未验收

当前点QP为可分问题，有与独立OSQP的局部对照；但解了瞬时QP不等于离散积分后满足约束。val0中RoS顶点侵入约0.2056m；积分步长缩小4倍/16倍后约0.0548m/0.0140m。TVS也随加密下降。说明数值误差不可忽略，不能声称原文连续时间保证成立。地图SDF非光滑、局部陷阱与离散/下层连续执行的关系也要分别审。不能通过改算法或追加兜底假装解决。

证据：[native_point_integration_convergence.json](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/docs/native_point_integration_convergence.json)。

### P1：Constrained Diffusers三种的实验参数/任务机制未验收

论文附录C：umaze为128步规划、64级扩散＋额外64更新，dual η=.025、初始ρ=.05；large为384步规划、256级扩散＋额外200更新，η=1e-3、ρ=2.5e-4，对应不同解析约束。当前是把16个合法训练级重复成64级更新，再加64级；dual/rho沿用旧适配任务参数。它不是原始训练噪声过程，不能只说“64+64所以一致”。

PD/ALM的4粒子、正部期望、固定返回粒子0等选择有出处或任务适配背景，但有限样本与超参数尚未在新点基座上验收。公式内核正确不足以覆盖这些问题。

论文§6.1/在线实验确有DCBF、逆动力学、滚动执行；但原文明确区分离线Maze规划和在线控制任务。不要把“没实现IDM”无差别当作三种离线规划方法全部错误，也不要把它们的在线安全声明转移到离线任意折线路径。

### P1：约束对象与评价对象必须统一

离散点安全不等于线段安全。新评价器同时保留`vertex_collision`、`segment_only_collision`、`collision`。PDM应补原文线段可行集；其他论文必须按各自任务与机制核对，不能为让表好看统一套PDM投影或Ours走廊。仅在评价时加密采样、增加输出点数，都不等于把线段约束加进优化。

## 7. 最近小批量的真实结果（仅错误排查）

[summary.md](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/results/native_points_validation_20261114_011528/summary.md)

[summary.json](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/results/native_points_validation_20261114_011528/summary.json)

[validation_assessment.json](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/results/native_points_validation_20261114_011528/validation_assessment.json)

[results.jsonl](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/results/native_points_validation_20261114_011528/results.jsonl)

20个固定val样本、1种子、8方法，共160次；全部返回原生128点，状态与交付数组逐元素一致，无求解失败。没有运行Ours，也不是420样本全量。

|方法|返回|顶点碰撞|连线路径碰撞|
|---|---:|---:|---:|
|RoS-diffuser|20/20|20/20|20/20|
|ReS-diffuser|20/20|20/20|20/20|
|TVS-diffuser|20/20|20/20|20/20|
|Projected|20/20|0/20|20/20|
|Primal-Dual|20/20|20/20|20/20|
|ALM|20/20|20/20|20/20|
|PDM|20/20|0/20|20/20|
|Diffuser|20/20|13/20|13/20|

**所有七种论文方法仍未通过复现验收。** 本表保留作诊断，不用于论文方法排名；不应通过删失败、放宽标准、选测试集参数或换名字修饰。

## 8. 旧代码哪些能参考，哪些不能当新方法
- [sdf_guidance.py](D:/ProjectDirectory/Neural-IRISDiffuser/src/diffusion/sdf_guidance.py)
- [sdf_field.py](D:/ProjectDirectory/Neural-IRISDiffuser/src/geometry/sdf_field.py)
- [bspline_constraints.py](D:/ProjectDirectory/Neural-IRISDiffuser/src/geometry/bspline_constraints.py)
- [bspline_alm.py](D:/ProjectDirectory/Neural-IRISDiffuser/src/diffusion/bspline_alm.py)

旧SDF修正器的逻辑是`g(Q)=d_safe-d(BQ)`，在干净预测上修正，随后送回DDIM；`q_ref`指当前网络预测，不是搜索通路。可以参考约束与网络输出的关联方式，但当前新点模型已经直接优化轨迹点，不应重新引入B样条。

旧`OccupancySDF`提供的归一化避障方向场，不总是其双线性标量插值的真实导数。直接把它当精确Jacobian用于KKT/CBF证明不合适；只复制标量导数也会遇到障碍内零梯度。真实几何/梯度和算法假设应一并核对。

`legacy_sdf.py`和`validate_legacy_reuse.py`只是误入的旧工程修正器复用诊断。用户明确**没有要求重新全量跑旧方法**。旧方法4场景改善不等于七种新论文算法正确，不要继续此路线。

## 9. 撤回/禁止继续使用的试验

- `physical_trajectory.py`、`trajectory_state.py`：额外点状态/残差再回拟合，不能用于新点模型。
- `curve_control.py`、`paper_task.py`：旧控制点路线，用户已放弃；保留溯源。
- `seven_reused_direction_t16.json`：旧方向场替换真导数试验，已撤回，当前旧任务分派会拒绝该字段。
- 旧罚项修正作为投影初值、额外干净预测选种、可行性罚项预处理等改动已从活动`CurveControlTask`撤回。
- 曾提出额外采样8次基座作为投影初值，自动审批明确拒绝：改变论文初始化和计算流程，用户未授权。该补丁未应用、该采样未执行；不要绕过拒绝重试。
- 不把Ours的搜索骨架/学习到的走廊/ALM/GT给新方法做隐性兜底。
- 历史`seven_S1*`、`seven_frozen_t16_*`和`seven_direct_full_t16_*`不是可继续复用的正式结果。

## 10. 建议接手执行顺序与验收

1. 读原文与官方来源，逐方法填写初始化、训练目标、score单位、更新、噪声日程、参数来源、约束集合、返回值及评价定义。当前审查表：[seven_method_fidelity_audit.md](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/docs/seven_method_fidelity_audit.md)。这是未通过清单，不是认证。
2. 先做原文解析约束/已知Gaussian score的退化和数值对照，再检查真实点基座上的无约束采样。不同采样器不要求逐点等于DDIM，但须证明实现和参数匹配，不能把异常输出带入安全比较。
3. 按PDM附录C.3补全整段约束和真实地图几何转换，使用正确的最近点投影问题及数值求解；保留原循环。不通过简单逐点裁剪代替。
4. 分别完成SafeDiffuser的时间/积分/局部陷阱与CD三种的score/期望/参数/任务约束核对，不强求它们使用同一个不适当的投影器。
5. 固定验证集与参数来源，重新跑小批量；按算法完成状态、端点、有限数值、约束残差、顶点碰撞、线段碰撞、质量、耗时分开报告。碰撞不是禁止继续实验的理由；**遗漏约束、采样不匹配、伪造方法或无效数值结果才是不能发布比较的理由。**
6. 完成上述正确性验收后直接全量420测试样本×3种子；若加Ours和普通Diffusion，共9方法、11,340次。不要重复中规模；不要复用错误实现数据。
7. Ours使用额外地图搜索候选/安全模块，其他基座只用地图和起终点，必须披露信息与预处理差异。历史耗时有CPU/GPU传输和串行开销，不能直接宣称原论文算法慢。

## 11. 数据、环境与可用命令

Python：`E:/CondaEnvData/envs/GGMPC/python.exe`；CUDA PyTorch2.9.1+cu126；RTX4070 12GB；NumPy2.2.6、SciPy1.15.3、OSQP1.1.0、Shapely2.1.2。`cyipopt`和`casadi`未安装，`cvxpy`可用。当前线段原型用SciPy `trust-constr`内点法，不是IPOPT原包。

数据根：`D:/ProjectDirectory/Neural-IRISDiffuser/data/processed/160m-k4p-c48`。train3850/80episodes，val1550/20episodes，test420/15episodes；episode无跨集重叠。**420样本不是420独立episode**。统计应按episode聚类，不能把重复种子当独立场景。分割审计：[split_audit.json](D:/ProjectDirectory/Neural-IRISDiffuser/experiments/safety_baselines/docs/split_audit.json)。

地图约定：occupancy=1是障碍；坐标[-1,1]²；1scene单位=80m；256网格每格0.625m。`k4p`是带边界保护的障碍腐蚀/自由空间扩大，不是障碍膨胀，也不自动代表车辆半径安全裕度。当前公开对比半径为0，不能宣称整车连续安全。

只运行局部验证的命令示例（PowerShell；不启动全量或重训）：
```powershell
Set-Location D:/ProjectDirectory/Neural-IRISDiffuser
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:PYTHONDONTWRITEBYTECODE="1"
& E:/CondaEnvData/envs/GGMPC/python.exe -B -m pytest -q -p no:cacheprovider experiments/safety_baselines/tests/test_native_points.py experiments/safety_baselines/tests/test_point_baseline.py experiments/safety_baselines/tests/test_paper_segment_constraints.py
```

最近局部测试分别有25项点模型/原生点/循环检查通过，以及新增3项线段几何检查通过；存在重叠测试，不能相加称“完整复现验收通过”。不要把pytest绿灯当作论文对比有效性证明。

## 12. 文档可信度

`baseline_source_map.md`、`baseline_reproduction_report.md`、部分README和历史状态说明包含多轮旧设计，**不能直接当当前事实**。优先查本报告、实际代码、检查点配置、原始jsonl及最新审查表。`current_work_state.md`用于历史上下文，可能有旧状态；本次交接之后不代表仍有任务在跑。

当前助手未完成合格的七方法正式比较。交接的主要资产是：可加载的新点基座、原始数据与论文、候选实现、错误证据和少量局部验证；不是一套可直接发表或直接全量复用的已验收系统。
