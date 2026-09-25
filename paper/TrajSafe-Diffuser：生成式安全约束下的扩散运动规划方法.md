# TrajSafe-Diffuser：生成式安全约束下的扩散运动规划方法

## 1 引言

运动规划是移动机器人、自主智能体和具身系统实现自主行为的基础环节，其目标是在给定环境信息、初始状态与目标状态的条件下，生成满足碰撞约束和运动连续性要求的可执行轨迹。传统运动规划通常采用图搜索、随机采样或轨迹优化等方法。搜索与采样方法能够在显式环境模型上探索可行路径，但在高维状态空间或狭窄通道中往往需要较高的计算开销；优化方法可以直接处理轨迹平滑性和动力学约束，却容易受到初始解与非凸障碍结构的影响。随着任务环境和行为模式日益复杂，仅依赖单一确定性规划过程难以同时兼顾多模态表达、规划效率和安全约束。

扩散模型通过逐步加噪与逆向去噪学习复杂数据分布，在生成质量和多模态建模方面表现出较强能力[1,2]。当轨迹被视为一种结构化序列后，运动规划可以转化为条件生成问题：模型从高斯噪声出发，在环境、起终点或任务目标的条件下逐步恢复完整轨迹。Diffuser 将扩散模型用于轨迹级决策，使采样过程与规划过程在统一概率框架内结合[3]；Decision Diffuser、Diffusion-QL 和 Diffusion Policy 则分别从条件决策、离线强化学习和机器人动作生成等角度验证了扩散模型表达多模态行为的能力[4-6]。与逐时刻回归相比，轨迹级扩散能够联合建模整个规划时域，并允许在推理阶段通过条件、代价或梯度改变生成结果，因此为复杂环境中的运动规划提供了新的技术路径。

然而，从示范数据中学习轨迹分布并不等同于满足安全约束。扩散模型的去噪目标主要刻画数据分布中的统计规律，即使训练轨迹均为可行轨迹，有限数据、分布偏移和逐步采样误差仍可能使生成轨迹穿越障碍或贴近不可行区域。现有研究通常在逆扩散过程中加入碰撞代价、控制屏障函数、投影算子或约束优化器，使当前预测朝可行域移动[9-12]。这类方法说明显式约束能够有效干预生成过程，但其适用性通常依赖一个前提：规划器已经获得可计算的安全函数或可投影的可行集合。对于具有非凸障碍、多个同伦通路和局部窄通道的环境，安全区域并不是一个预先给定的简单凸集。如何从环境中形成与当前任务相匹配、可供扩散模型和优化器共同使用的安全约束，仍是安全扩散规划中的关键问题。

自由空间的拓扑结构与局部凸分解为这一问题提供了几何基础。IRIS 通过障碍分离与最大体积内接椭球的交替优化构造无碰撞凸区域[14]；基于凸集图的方法进一步利用相互连接的凸区域进行全局运动规划[15]；几何约束轨迹优化和凸覆盖方法则表明，连续轨迹能否可靠优化，不仅取决于单个安全区域的大小，还取决于区域之间的连通性以及轨迹参数化与区域约束之间的对应关系[16,17]。但是，直接对整个自由空间进行统一凸分解可能产生大量与当前起终点任务无关的区域，而仅沿一条参考路径构造走廊又可能丢失多种拓扑选择。对于生成式规划器，更合适的安全表示应当同时具备三项性质：能够反映自由空间的候选拓扑，能够转化为连续轨迹的显式约束，并能够随当前生成状态参与后续去噪。

此外，将扩散生成器与安全优化器简单串联仍存在信息割裂。若优化器只在当前去噪步修正轨迹，而去噪网络在下一步无法获知修正方向及约束满足情况，模型可能再次预测出相似的违规轨迹，导致优化器在多个步骤中重复执行较大修正。Constrained Diffusers 已将投影、原始—对偶和增强拉格朗日方法引入受约束逆扩散过程[11]，JM2D 则从联合采样角度研究模型自由生成模块与模型驱动安全模块之间的兼容性[13]。这些工作表明，生成与优化的结合本身并不足以构成新的问题，真正需要解决的是安全约束如何产生、如何进入生成网络，以及已经验证的修正结果如何持续影响后续预测。

针对上述问题，本文提出 TrajSafe-Diffuser，一种生成式安全约束下的扩散运动规划方法。该方法以占据栅格中的自由空间骨架为拓扑先验，为给定起终点构造多条候选通路；随后沿选中通路以固定弧长进度生成局部安全椭圆，并将椭圆与障碍边界分离得到可验证的凸区域序列。相邻区域经过重叠检查和间隙桥接后组成安全走廊，从而把复杂非凸自由空间转换为适用于连续轨迹的局部线性不等式。轨迹采用三次 B 样条控制多边形表示，控制点是扩散过程中的唯一轨迹状态；安全几何通过跨注意力与控制点特征交互，使去噪网络在预测轨迹时能够利用所选拓扑及其局部安全范围。在推理阶段，增强拉格朗日模块依据冻结的安全走廊修正当前干净轨迹预测，并将修正后的控制点、修正量与有效标志编码为历史安全条件，反馈给下一去噪步，由此形成“生成—约束修正—验证—反馈—再生成”的闭环过程。

本文的主要贡献概括如下：

（1）提出一种面向扩散运动规划的生成式安全约束构建方法。该方法以任务相关的自由空间拓扑为候选结构，将局部安全几何预测、凸区域验证和走廊连通检查结合起来，避免将骨架中心或未验证椭圆直接视为安全保证。

（2）提出安全几何与轨迹控制特征的交互机制。方法在 B 样条控制点空间执行扩散生成，并将选中拓扑上的安全查询特征注入控制点特征，使安全表示不仅服务于后处理优化，也直接参与当前轨迹预测。

（3）提出跨去噪步的闭环安全反馈机制。方法将约束优化得到的安全控制点及其相对原始预测的修正量作为下一步去噪的显式条件，使已经验证的安全修正能够持续影响后续生成，并在安全条件无效时退化为普通扩散过程。

## 2 相关工作

扩散模型最初被用于复杂数据分布的生成建模。Ho 等人提出的去噪扩散概率模型通过预定义前向加噪过程和学习逆向去噪过程生成数据[1]；Song 等人提出 DDIM，在保持训练目标的同时构造非马尔可夫采样过程，为减少逆扩散步数提供了基础[2]。这些工作建立了以迭代去噪实现条件生成的基本范式，也使外部条件能够在多个采样阶段持续改变生成结果。

在序列决策与规划领域，Diffuser 将状态—动作轨迹整体作为生成对象，通过迭代去噪获得长时域计划，并利用引导函数或条件补全支持测试时任务调整[3]。Decision Diffuser 将离线决策表述为回报、约束或技能条件下的轨迹生成问题，说明条件扩散可以组合训练阶段出现过的行为属性[4]。Diffusion-QL 采用条件扩散模型表达离线强化学习策略，并通过价值函数项在行为分布附近实现策略改进[5]。Diffusion Policy 则将机器人视觉运动策略建模为条件动作扩散过程，结合滚动时域执行处理多模态动作序列[6]。这些研究从轨迹、策略和动作序列等不同层面表明，扩散模型能够表达传统单峰回归难以覆盖的多模态决策分布。

扩散规划随后逐渐从通用决策拓展到场景条件与机器人运动。SceneDiffuser 将三维场景条件、物理目标和规划任务纳入统一去噪框架，使场景感知生成与测试时优化在迭代采样中结合[7]。DiffuserLite 通过由粗到细的规划细化减少冗余采样计算，说明扩散规划的轨迹表示与采样结构会直接影响决策频率[8]。Motion Planning Diffusion（MPD）学习机器人轨迹分布先验，并在去噪过程中加入任务代价梯度；该方法还使用 B 样条降低轨迹表示维度并保持曲线平滑性[9]。因此，B 样条参数化、代价引导以及生成模型与轨迹优化的组合已经具有明确研究基础。本文采用控制点空间的原因不是将 B 样条本身作为创新，而是使生成变量、连续安全约束和跨步修正共享同一组低维轨迹变量。

生成模型能够表达轨迹先验，但安全规划还要求每条输出满足环境与任务约束。SafeDiffuser 将控制屏障函数和有限时间扩散不变性嵌入去噪过程，以提高扩散规划的安全性[10]。Constrained Diffusers 从受约束采样角度，将投影、原始—对偶和增强拉格朗日算法用于逆扩散，并结合离散控制屏障函数与滚动时域控制处理在线安全约束[11]。CoDiG 使用屏障函数引导去噪采样，在动态障碍条件下研究实时约束满足[12]。这些方法的共同特征是以显式安全函数、约束集合或可计算障碍关系为基础，在采样阶段改变当前轨迹。它们解决了“如何使用约束”的问题，但在复杂栅格环境中，如何由非凸自由空间自动得到与当前起终点和候选通路匹配的连续安全集合，仍需要独立的几何构建过程。

另一类研究关注生成模块与模型驱动优化模块的相容性。JM2D 将二者的结合表述为联合采样问题，通过交互势函数和重要性采样评价多模态生成结果与安全模块之间的兼容程度[13]。这表明生成轨迹与优化结果之间的关系不能简化为单向后处理：当原始预测远离可行域时，大幅投影可能破坏生成先验所表达的轨迹结构；当优化结果没有返回生成网络时，相似的约束违反又可能在后续步骤重复出现。

复杂环境中的显式安全集合通常依赖自由空间的几何表示。IRIS 从无碰撞种子出发，在障碍分离超平面与最大体积内接椭球之间交替优化，得到局部凸多面体[14]。Graphs of Convex Sets 将相互连接的凸区域组织成图，并结合 Bézier 曲线与凸松弛求解障碍环境中的轨迹[15]。GCOPTER 进一步展示了稀疏轨迹表示与几何约束优化的结合方式[16]；凸覆盖优化研究则强调，区域规模、覆盖效率和相邻区域连接共同决定了后续轨迹生成的可行空间[17]。这些几何方法能够提供强于点级碰撞代价的区域约束，但单个种子只能描述局部自由空间，独立构造的一组区域也不必然组成可供连续轨迹通过的走廊。

综上，现有研究已经分别验证了扩散模型的多模态轨迹建模能力、显式约束对逆扩散的引导作用，以及凸区域在连续运动规划中的价值。本文并不重新提出扩散模型、B 样条或增强拉格朗日算法，而是研究三者之间尚未闭合的接口：首先从任务相关的自由空间拓扑生成并验证安全走廊；其次将安全几何作为条件参与控制点去噪；最后将约束修正转化为跨去噪步的历史反馈。由此，安全约束不再只是采样末端的外部检查，而成为从环境构建、网络交互到优化反馈均持续参与的规划变量。

## 3 方法

给定二维占据栅格 $O\in\{0,1\}^{H\times W}$、起点 $s\in\mathbb{R}^2$ 和终点 $g\in\mathbb{R}^2$，本文目标是生成连续轨迹 $\tau:[0,1]\rightarrow\mathbb{R}^2$，使其满足端点条件 $\tau(0)=s$、$\tau(1)=g$，并尽可能保持在由自由空间构造的安全区域内。TrajSafe-Diffuser 不直接对密集轨迹点执行扩散，而以三次 B 样条控制多边形

$$
Q_t=[q_{t,0},q_{t,1},\ldots,q_{t,C-1}]\in\mathbb{R}^{C\times2}
$$

作为第 $t$ 个去噪步的唯一轨迹状态。对应连续曲线为

$$
\tau_t(u)=\sum_{j=0}^{C-1}B_{j,3}(u)q_{t,j},\qquad u\in[0,1],
$$

其中 $B_{j,3}$ 为三次 B 样条基函数。采用夹持节点向量并固定 $q_{t,0}=s$、$q_{t,C-1}=g$ 后，曲线端点可以严格满足任务条件。整体方法由生成式安全约束构建、安全感知扩散轨迹生成和闭环安全反馈三个部分组成。

### 3.1 生成式安全约束构建

占据栅格中的自由空间通常是非凸且多连通的，直接使用全局距离代价难以同时表达不同通路及局部窄通道。本文首先对自由空间执行拓扑保持的细化操作，得到近似位于通行区域中部的骨架，并将骨架像素组织为图结构 $\mathcal{G}=(\mathcal{V},\mathcal{E})$。起终点被连接至其可见的骨架节点，在图上搜索得到 $M$ 条有效候选通路

$$
\Gamma_m:[0,1]\rightarrow\mathbb{R}^2,\qquad m=1,\ldots,M.
$$

候选通路表示不同的局部几何或同伦选择，但骨架只提供安全区域的中心先验，不能直接证明其邻域无碰撞。为此，模型根据当前轨迹控制特征与候选骨架特征计算拓扑概率 $\pi_m$，并选择索引 $m^*$ 对应的稠密骨架曲线 $\Gamma_{m^*}$。安全查询采用固定进度

$$
\alpha_i=\frac{i}{K-1},\qquad c_i=\Gamma_{m^*}(\alpha_i),\qquad i=0,\ldots,K-1,
$$

其中 $c_i$ 是第 $i$ 个局部安全区域的中心。固定进度使安全几何与选中骨架保持稳定对应，避免额外回归进度参数导致区域顺序紊乱。

在每个中心 $c_i$ 处，安全几何分支结合骨架特征和局部场景特征预测椭圆形状

$$
r_i=[\log a_i,\log b_i,\cos 2\theta_i,\sin 2\theta_i],
$$

其中 $a_i\ge b_i>0$ 分别表示长、短半轴，$\theta_i$ 表示方向。对应椭圆写为

$$
\mathcal{E}_i=\left\{x\mid (x-c_i)^\top R_i
\begin{bmatrix}a_i^{-2}&0\\0&b_i^{-2}\end{bmatrix}
R_i^\top(x-c_i)\le1\right\}.
$$

椭圆用于提供局部区域的尺度和方向，但预测椭圆本身仍可能接近障碍，因此不能直接作为最终约束。本文以椭圆为局部度量，从中心向邻近障碍边界构造分离半空间，并将其交集表示为

$$
\mathcal{P}_i=\{x\in\mathbb{R}^2\mid A_i x\le b_i\}.
$$

每个区域都需要通过中心包含性、有界性、有效面积和障碍分离检查。任一基础区域无效时，当前走廊构建失败，而不是以未验证区域继续规划。

连续轨迹还要求相邻区域之间具有足够连接。对相邻多边形 $\mathcal{P}_i$ 和 $\mathcal{P}_{i+1}$，定义归一化重叠率

$$
\rho_i=\frac{\operatorname{area}(\mathcal{P}_i\cap\mathcal{P}_{i+1})}
{\min(\operatorname{area}(\mathcal{P}_i),\operatorname{area}(\mathcal{P}_{i+1}))}.
$$

若 $\rho_i$ 不低于阈值，则两区域直接连接；否则在骨架中间进度 $(\alpha_i+\alpha_{i+1})/2$ 处构造点种子桥接区域，并分别验证其与左右区域的重叠率。无法通过一次桥接闭合的间隙被判为无效，以避免递归增加未经控制的区域。最终得到按进度有序的安全走廊

$$
\mathcal{C}=\{\mathcal{P}_1,\mathcal{P}_2,\ldots,\mathcal{P}_R\},
$$

它既保留候选拓扑所描述的全局通行结构，又以线性不等式提供局部可验证的安全范围。

### 3.2 安全感知扩散轨迹生成

TrajSafe-Diffuser 在控制点空间执行条件扩散。前向过程逐步向真实控制多边形 $Q_0$ 添加高斯噪声：

$$
q(Q_t\mid Q_0)=\mathcal{N}(\sqrt{\bar\alpha_t}Q_0,(1-\bar\alpha_t)I),
$$

逆向网络根据 $Q_t$、时间步 $t$、占据栅格、起终点和候选骨架预测干净控制多边形 $\hat Q_0^{\mathrm{raw}}$。与对密集轨迹点去噪相比，控制点状态减少了生成变量，并使每次安全修正直接作用于连续曲线的参数。

占据栅格首先由场景编码器提取全局场景记忆 $C_G$ 与细粒度几何记忆 $C_E$。控制点经坐标嵌入、序列位置编码和时间条件块得到控制特征 $H_Q$；候选骨架分别编码为 $H_{S,m}$。匹配模块计算每条候选骨架与当前控制序列之间的交互表示 $R_m$，拓扑头据此输出 $\pi_m$。选中拓扑的匹配表示形成路径特征 $H_P$，其骨架查询与局部场景记忆进一步形成安全几何特征 $H_E$。

为了使局部安全范围直接影响轨迹预测，本文采用以控制点为查询、以安全几何为键和值的交叉注意力：

$$
A_{\mathrm{safe}}=\operatorname{softmax}
\left(\frac{(H_PW_Q)(H_EW_K)^\top}{\sqrt d}\right)H_EW_V.
$$

该操作将沿骨架分布的 $K$ 个安全查询聚合到 $C$ 个控制点上，使不同控制点能够按当前轨迹结构选择相关的局部安全信息。随后，将 $H_Q$、$H_P$ 与 $A_{\mathrm{safe}}$ 融合，并结合全局场景记忆预测 $\hat Q_0^{\mathrm{raw}}$。因此，骨架不仅用于选择一条参考路径，安全椭圆也不仅用于采样后的投影；二者共同构成去噪网络内部的几何条件。

网络输出后使用固定边界解码器施加端点条件。设网络自由预测为 $\widetilde Q$，起终点误差为 $e_s=s-\widetilde q_0$、$e_g=g-\widetilde q_{C-1}$，则修正后的控制点为

$$
q_j=\widetilde q_j+w_j^s e_s+w_j^g e_g,
$$

其中 $w^s$ 和 $w^g$ 是分别从起点端和终点端衰减的固定权重，并满足 $w_0^s=1$、$w_{C-1}^g=1$。该解码器不引入可学习参数，能够在不破坏内部控制点相对结构的情况下保证曲线端点精确。密集轨迹仅在网络末端通过 B 样条基矩阵解码，用于碰撞检查、可视化或控制执行，不重新进入扩散状态。

### 3.3 闭环安全反馈机制

安全几何参与网络预测后，有限数据和分布偏移仍可能使 $\hat Q_0^{\mathrm{raw}}$ 违反走廊约束。本文在逆扩散中设置预热、激活和引导三个阶段。预热阶段执行普通去噪，使早期高噪声状态形成基本轨迹结构；激活阶段根据当前拓扑和安全椭圆构造走廊，并在连通验证通过后冻结拓扑及约束；引导阶段在每个后续去噪步对当前干净控制点预测执行连续安全修正。冻结走廊可以避免约束集合随每一步拓扑切换而改变，从而保持跨步修正的一致性。

为了把凸区域约束施加到整条 B 样条曲线上，本文按节点区间将曲线精确转换为三次 Bézier 分段。设第 $l$ 个分段的四个 Bézier 控制点为

$$
Z_l=E_lQ,
$$

其中 $E_l$ 为由 B 样条节点向量确定的提取矩阵。若该分段被分配给走廊区域 $\mathcal{P}_{r(l)}$，则对四个 Bézier 控制点施加

$$
A_{r(l)}z_{l,k}\le b_{r(l)},\qquad k=0,1,2,3.
$$

根据 Bézier 曲线的凸包性质，当四个控制点均位于同一凸区域时，对应连续曲线分段也位于该区域内。由此，安全条件不再只检查有限采样点，而被转换为控制空间中的连续曲线约束。

对当前预测 $Q^{\mathrm{raw}}$，安全修正求解以下形式的问题：

$$
\min_Q\ \frac{1}{2}\|Q-Q^{\mathrm{raw}}\|_F^2
+\lambda_s\mathcal{S}(Q),
\quad\text{s.t.}\quad A_{r(l)}E_lQ\le b_{r(l)},
$$

其中第一项限制修正偏离生成先验的幅度，$\mathcal{S}(Q)$ 用于抑制控制多边形的高阶变化。本文采用增强拉格朗日迭代处理不等式约束，并限制单次修正引起的最大曲线位移。修正结果记为 $Q^{\mathrm{safe}}$，其相对于网络原始预测的变化为

$$
\Delta Q=Q^{\mathrm{safe}}-Q^{\mathrm{raw}}.
$$

只有当走廊有效且修正结果通过约束验证时，$Q^{\mathrm{safe}}$ 才作为当前步的安全干净预测进入 DDIM 更新；走廊构建或验证失败时，该样本退化为未经引导的扩散采样，避免错误约束破坏轨迹。

仅将 $Q^{\mathrm{safe}}$ 用于当前 DDIM 更新，仍不能使下一步网络显式理解优化器进行了何种修正。因此，本文为每个控制点构造历史安全向量

$$
f_j=[q_{j,x}^{\mathrm{safe}},q_{j,y}^{\mathrm{safe}},
\Delta q_{j,x},\Delta q_{j,y},v],
$$

其中 $v\in\{0,1\}$ 表示上一轮安全结果是否有效。反馈编码器将 $f_j$ 映射为 $H_{F,j}$，再通过门控残差注入下一去噪步的控制特征：

$$
G=\sigma(W_g[H_Q;H_F]),\qquad
H_Q'=H_Q+v\,G\odot H_F.
$$

门控系数允许网络按控制点和特征通道选择历史信息。当 $v=0$ 时，反馈项严格为零，网络保持普通扩散行为；当 $v=1$ 时，$Q^{\mathrm{safe}}$ 提供上一轮可行轨迹的位置参考，$\Delta Q$ 则描述原始预测相对可行域的偏差方向与幅度。下一步去噪因而能够同时利用当前噪声状态、环境几何和上一轮安全修正，而不是在每个步骤重新产生彼此独立的违规预测。

通过上述设计，TrajSafe-Diffuser 形成闭环链路：环境拓扑产生候选安全结构，局部几何生成并验证凸安全走廊，安全特征参与控制点预测，连续约束优化修正当前轨迹，修正结果再反馈至后续去噪。该机制将几何可行域、生成先验和优化修正统一到 B 样条控制点空间，同时保留在安全约束不可用时的稳定退化路径。

## 参考文献

[1] HO J, JAIN A, ABBEEL P. Denoising diffusion probabilistic models[C]//Advances in Neural Information Processing Systems. 2020, 33: 6840-6851.

[2] SONG J, MENG C, ERMON S. Denoising diffusion implicit models[C]//International Conference on Learning Representations. 2021.

[3] JANNER M, DU Y, TENENBAUM J, et al. Planning with diffusion for flexible behavior synthesis[C]//Proceedings of the 39th International Conference on Machine Learning. 2022: 9902-9915.

[4] AJAY A, DU Y, GUPTA A, et al. Is conditional generative modeling all you need for decision-making?[C]//International Conference on Learning Representations. 2023.

[5] WANG Z, HUNT J J, ZHOU M. Diffusion policies as an expressive policy class for offline reinforcement learning[C]//International Conference on Learning Representations. 2023.

[6] CHI C, FENG S, DU Y, et al. Diffusion policy: Visuomotor policy learning via action diffusion[C]//Robotics: Science and Systems. 2023.

[7] HUANG S, WANG Z, LI P, et al. Diffusion-based generation, optimization, and planning in 3D scenes[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 16750-16761.

[8] DONG Z, HAO J, YUAN Y, et al. DiffuserLite: Towards real-time diffusion planning[C]//Advances in Neural Information Processing Systems. 2024, 37.

[9] CARVALHO J, LE A T, KICKI P, et al. Motion planning diffusion: Learning and adapting robot motion planning with diffusion models[J]. IEEE Transactions on Robotics, 2025, 41: 4881-4901.

[10] XIAO W, WANG T H, GAN C, et al. SafeDiffuser: Safe planning with diffusion probabilistic models[C]//International Conference on Learning Representations. 2025.

[11] ZHANG J, ZHAO L, PAPACHRISTODOULOU A, et al. Constrained diffusers for safe planning and control[C]//Advances in Neural Information Processing Systems. 2025, 38.

[12] MA H, BODMER S, CARRON A, et al. Constraint-aware diffusion guidance for robotics: Real-time obstacle avoidance for autonomous racing[C]//Proceedings of the 9th Conference on Robot Learning. 2025: 1756-1776.

[13] JUNG W, MISHRA U A, ARACHCHIGE N R, et al. Joint model-based model-free diffusion for planning with constraints[C]//Proceedings of the 9th Conference on Robot Learning. 2025: 4328-4350.

[14] DEITS R, TEDRAKE R. Computing large convex regions of obstacle-free space through semidefinite programming[C]//Algorithmic Foundations of Robotics XI. Cham: Springer, 2015: 109-124.

[15] MARCUCCI T, PETERSEN M, VON WRANGEL D, et al. Motion planning around obstacles with convex optimization[J]. Science Robotics, 2023, 8(84): eadf7843.

[16] WANG Z, ZHOU X, XU C, et al. Geometrically constrained trajectory optimization for multicopters[J]. IEEE Transactions on Robotics, 2022, 38(5): 3259-3278.

[17] WU Y, SPASOJEVIC I, CHAUDHARI P, et al. Towards optimizing a convex cover of collision-free space for trajectory generation[J]. IEEE Robotics and Automation Letters, 2025, 10(5): 4762-4769.
