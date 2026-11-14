# 三篇论文到七个完整实现（2026-11-13）

基座：map_only_control_diffuser_v1，从本项目训练集训练100轮，best_task epoch88。48个三次B样条控制点，1层主干/8层末端；仅地图和起终点条件。完整Ours采用1+8/100轮epoch84。所有新实现位于独立实验目录，核心src只读。

|论文/方法|原文位置|原始采样/变量|完整实现|地图与控制点映射|忠实度与限制|
|---|---|---|---|---|---|
|SafeDiffuser ICLR2025 / RoS-diffuser|Thm3.4 Eq8,13 Alg1|名义扩散速度ν，γ，最小偏差QP|algorithms/safediffuser/runner.py|在前一状态Q上计算h(BQ)与链式梯度；只优化内部46个控制点|task-adapted：名义更新用现有DDIM；共享控制点使CBF约束耦合，QP可能无解|
|同上 / ReS-diffuser|Thm3.5 Eq9,10,14 Alg1|ν,r；w；硬区间γ|同上|每个采样约束有独立r，w=100；最后10个更新w=0；硬区间初值重新计算合法γ|task-adapted：不能删掉最后硬约束；不可行必须失败，无惩罚梯度替代|
|同上 / TVS-diffuser|Thm3.6 Eq11,12,13 Alg1|ν,σ,σ导数|同上|σ(j)=min(h_initial,0)*j/N；σ(N)<=h_initial、σ(0)=0；σ导数沿去噪时间为-min(h_initial,0)/(NΔτ)|task-adapted：使用定理允许的线性时间函数，不冒用附录sigmoid参数|
|Constrained Diffusers NeurIPS2025 / Projected|§5.1 Eq10,11|x + β/2 score + sqrtβ noise，最近点投影|algorithms/constrained_diffusers/runner.py|VP score；16个训练噪声层＋最低噪声层额外16次Langevin，每次投影|task-adapted：附录Maze原生64+64更新映射到当前16层；非凸局部投影无全局最近证书|
|同上 / Primal-Dual|§5.2 Eq12–14，Remark1|x与非负λ交替|同上|每场景4粒子估计E[[g]+]；逐约束λ，不跨场景平均；固定返回粒子0|task-adapted：有限粒子近似且hinge只几乎处处可微，不宣称单轨迹硬安全|
|同上 / ALM|§5.3 Eq15–18|x,s>=0,λ,ρ；ρ增长|同上|每场景4粒子，显式slack、对偶与ρ更新；不调用Ours ALM|task-adapted：同上；η=.001、ρ0=.00025，c=1.05在scene单位下预先声明|
|PDM NeurIPS2024 / PDM|§4.2 Eq7,8 Alg1；附录F p19|T/M两层Langevin+projection；γ=σt²/(2σT²)|algorithms/pdm/runner.py|T=10（从16个合法训练层选10层）；M=100。VE坐标y=xt/sqrtαbar，σ²=(1-αbar)/αbar；score_y=sqrtαbar*score_x|task-adapted：VP到VE坐标的严格score变换，固定端点不随噪声缩放；实际噪声跨度与原实验不同；非凸近似投影可能失败|

## 原文参数与本任务参数

SafeDiffuser附录C.1：原Maze N256、ε1、γ.01、w(j>=10)=100否则0。当前合法噪声层16、Δτ1、ε1；γ不是盲用.01，而用max(0.01m/80, -h_min/(exp(εN)-1), h_max/(exp(εN)+1))以满足初始robust不等式。ReS硬段按其起点重新计算。QP目标为标准二次最小偏差并含||r||²。单位：Q、SDF、γ均scene，1scene=80m。

Constrained Diffusers附录实验Maze使用64步＋额外64步；当前基座只有16个训练层，所以声明为16＋额外16步，而不是假装DDIM等价于Langevin。PD/ALM输入g采用正部，正部期望约束在原文Remark1明确讨论；有限4粒子不提供分布保证。

PDM附录F明确T10/M100；此前M3仅工程短程，结果已标记superseded，不进入正式比较。正文Alg1固定随噪声递减的γ；附录D的Corrector以噪声/score范数比自适应γ，不在本次混用。附录G还指出以其他训练目标得到的模型用于投影可能表现较差，本项目的x0回归基座属于明确的模型任务适配，不能继承其原生实验效果或定理。

## 投影和可行性

ΠC由独立SLSQP求解 min .5||x-z||²，硬固定端点、约束h(BQ)>=0。非凸地图只能报告局部数值结果。首次温启动是独立占据地图A*＋公共B样条拟合，不是Ours骨架、网络先验或GT。温启动仅用于投影求解器；它的计算计入耗时。后续温启动为前一次投影。只有可行残差、求解状态和局部最优性检查均满足才接受；不收敛即失败。对双线性SDF的栅格非光滑点，记录普通或近邻采样Clarke驻点残差；这不是全局最优证书。有限迭代200属于已冻结数值预算。

CBF用OSQP求凸QP并检查原/对偶残差；不允许penalty梯度或Ours终端ALM兜底。非凸占据地图下，128个曲线约束在92维控制点变量上耦合，且双线性SDF存在平坦/不可微点，原定理假设不自动满足；这会产生真实QP不可行，须保留并将其解释为当前任务适配的限制，不能推断原论文算法普遍无效。

## 共同评价与计算预算

独立评价器对9种方法一致：512点基础网格＋近障碍/长段3轮加密；边界/端点/有限数值/超时均判失败。不是连续安全证明。前后违规在两条曲线加密参数的并集上比较，不用不同采样点比较前后值。

随机数由(scene_id,seed,method)的SHA256派生；每任务单场景，与batch顺序无关。外部算法与Diffuser和Ours各有明确随机流，不声称初始噪声逐元素相同。规划计时包含实际读取缓存、模型前向、地图SDF、QP/投影/A*；评价计算不计入规划。Ours离线骨架缓存生成成本不可恢复，明确排除，因此不称为从原始地图实时重建所有预处理的延迟。

原生算法预算允许不同循环数，不强迫16次前向：Safe16、CD32（PD/ALM每次4粒子）、PDM最多1000次score/投影。任务超时：默认10秒，Projected60秒，PDM600秒；用于截断失控求解，不是统一时限结论。统一10秒预算只能另外报告敏感性，不把超时记录删掉。

## 来源核对

SafeDiffuser官方源码核查了p_sample、invariance*在上一状态x处线性化，再x+=ν的时序；官方main默认安全分支为注释，不能直接跑main默认得到安全算法。官方仓库commit/许可见official_sources.json。PDM官方仓库已记录commit；Constrained Diffusers未确认官方代码，逐式以打包PDF为依据。没有复制或宣称完全复用了官方代码。
