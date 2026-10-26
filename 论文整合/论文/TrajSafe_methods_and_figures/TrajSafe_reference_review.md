# 参考文献核验与引用位置

核验日期：2026-10-07。以下记录由 Codex 核对公开来源，供作者逐项复核；不代表人工签核。文献事实优先依据正式会议/期刊记录及作者公开版本，引用仅用于其支持的背景或方法，不用于支持本文实验数值。

| 编号 | 对应工作 | 支持内容 | 正文位置 | 可复核来源 |
|---|---|---|---|---|
| [1] | Planning Algorithms | 运动规划问题与搜索、采样和几何表示的背景 | 1 引言 | [来源 1](https://lavalle.pl/planning/) |
| [2] | Sampling-based algorithms for optimal motion planning | 采样式运动规划及其渐近最优算法背景 | 1 引言 | [来源 1](https://arxiv.org/abs/1105.1186)；[来源 2](https://dspace.mit.edu/entities/publication/77dff1d1-ffd6-42f3-a3a5-fef87a964959) |
| [3] | CHOMP: Covariant Hamiltonian optimization for motion planning | 障碍代价与光滑性目标联合用于轨迹优化 | 1 引言 | [来源 1](https://publications.ri.cmu.edu/chomp-covariant-hamiltonian-optimization-for-motion-planning) |
| [4] | Denoising diffusion probabilistic models | 高斯加噪和逐步去噪生成的基础形式 | 1 引言；2.1 扩散轨迹生成与运动规划；3.1 总体框架 | [来源 1](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)；[来源 2](https://arxiv.org/abs/2006.11239) |
| [5] | Planning with diffusion for flexible behavior synthesis | 轨迹去噪、条件补全与代价引导用于规划 | 1 引言；2.1 扩散轨迹生成与运动规划 | [来源 1](https://proceedings.mlr.press/v162/janner22a.html) |
| [6] | Diffusion-based generation, optimization, and planning in 3D scenes | 场景条件生成、物理优化与目标规划的联合框架 | 1 引言；2.1 扩散轨迹生成与运动规划 | [来源 1](https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Diffusion-Based_Generation_Optimization_and_Planning_in_3D_Scenes_CVPR_2023_paper.html)；[来源 2](https://arxiv.org/abs/2301.06015) |
| [7] | Motion planning diffusion: Learning and adapting robot motion planning with diffusion models | 扩散轨迹先验与代价梯度联合使用；扩展版采用 B 样条表示 | 1 引言；2.1 扩散轨迹生成与运动规划 | [来源 1](https://www.dfki.de/web/forschung/projekte-publikationen/publikation/16337)；[来源 2](https://arxiv.org/html/2412.19948v3) |
| [8] | Control barrier function based quadratic programs for safety critical systems | 屏障函数约束与安全集合不变性，不用于证明本文的动态执行安全 | 1 引言；2.2 受约束扩散与安全修正 | [来源 1](https://arxiv.org/abs/1609.06408)；[来源 2](https://doi.org/10.1109/TAC.2016.2638961) |
| [9] | SafeDiffuser: Safe planning with diffusion probabilistic models | 将屏障函数和有限时间扩散不变性融入去噪 | 1 引言；2.2 受约束扩散与安全修正 | [来源 1](https://safediffuser.github.io/safediffuser/)；[来源 2](https://github.com/Weixy21/SafeDiffuser)；[来源 3](https://arxiv.org/abs/2306.00148) |
| [10] | Constrained synthesis with projected diffusion models | 投影/受约束优化用于扩散生成，覆盖凸与非凸约束 | 1 引言；2.2 受约束扩散与安全修正 | [来源 1](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a28af221f2f70be183afc16797a56b91-Abstract-Conference.html) |
| [11] | Simultaneous multi-robot motion planning with projected diffusion models | 多机器人扩散采样结合约束优化，不能视为本文单轨迹基线复现 | 1 引言；2.2 受约束扩散与安全修正 | [来源 1](https://proceedings.mlr.press/v267/liang25e.html) |
| [12] | Computing large convex regions of obstacle-free space through semidefinite programming | 交替障碍分离与内接椭球优化构造无障碍凸区域 | 1 引言；2.3 凸安全区域与连续曲线约束 | [来源 1](https://link.springer.com/chapter/10.1007/978-3-319-16595-0_7)；[来源 2](https://groups.csail.mit.edu/robotics-center/public_papers/Deits14.pdf) |
| [13] | Fast iterative region inflation for computing large 2-D/3-D convex regions of obstacle-free space | 凸区域构造中的质量、效率和种子集合包含要求 | 1 引言；2.3 凸安全区域与连续曲线约束 | [来源 1](https://doi.org/10.1109%2FTRO.2025.3562482)；[来源 2](https://arxiv.org/abs/2403.02977) |
| [14] | Motion planning around obstacles with convex optimization | 凸集图与 Bézier 曲线结合实现连续轨迹优化 | 1 引言；2.3 凸安全区域与连续曲线约束；3.5.1 连续曲线的走廊约束 | [来源 1](https://github.com/RobotLocomotion/gcs-science-robotics)；[来源 2](https://arxiv.org/abs/2205.04422)；[来源 3](https://pubmed.ncbi.nlm.nih.gov/37967206/) |
| [15] | Discrete-guided diffusion for scalable and safe multi-robot motion planning | 离散 MAPF、凸子问题、扩散引导与约束修复的结合 | 1 引言；2.3 凸安全区域与连续曲线约束 | [来源 1](https://ojs.aaai.org/index.php/AAAI/article/view/39512) |
| [16] | Denoising diffusion implicit models | DDIM 采样基础；本文将修正后的干净预测代入状态更新 | 2.1 扩散轨迹生成与运动规划；3.5.3 安全状态更新与历史反馈 | [来源 1](https://arxiv.org/abs/2010.02502) |
| [17] | Analog bits: Generating discrete data using diffusion models with self-conditioning | 此前的干净样本预测可作为后续去噪条件；不用于声称本文安全反馈与其相同 | 2.4 历史条件与后续去噪 | [来源 1](https://arxiv.org/abs/2208.04202) |
| [18] | The NURBS Book | B 样条表示、端点性质与 Bézier 凸包性质 | 3.1 总体框架；3.5.1 连续曲线的走廊约束 | [来源 1](https://link.springer.com/book/10.1007/978-3-642-59223-2) |
| [19] | Scalable diffusion models with transformers | 扩散 Transformer 的条件层归一化调制；不声称本文使用 adaLN-Zero | 3.2 特征提取与融合 | [来源 1](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html)；[来源 2](https://arxiv.org/abs/2212.09748)；[来源 3](https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf) |
| [20] | Attention is all you need | 多头自注意力与交叉注意力基础运算 | 3.2 特征提取与融合 | [来源 1](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) |
| [21] | Numerical Optimization | 罚函数和增广拉格朗日方法的标准数学背景；不用于声称有限次迭代必然可行 | 3.5.2 控制点的增广拉格朗日修正 | [来源 1](https://link.springer.com/book/10.1007/978-0-387-40065-5) |
| [22] | CARLA: An open urban driving simulator | CARLA 仿真平台来源；不用于支持本文数据集划分、处理参数及实验数值 | 4.1.1 数据集与实现设置 | [来源 1](https://proceedings.mlr.press/v78/dosovitskiy17a.html) |

## 核验范围与访问限制

- [6] CVF 直接打开返回 403，元数据由其索引记录与 arXiv 作者记录交叉核对。
- [7] 采用 2025 年正式期刊信息，不把 2024 年预印本日期写成期刊年份。
- [9] OpenReview 需浏览器验证。正式作者顺序为 Wei Xiao、Tsun-Hsuan Wang、Chuang Gan、Ramin Hasani、Mathias Lechner、Daniela Rus；未混用早期 4 作者版本。
- [12] 采用 2015 年正式章节出版年份，区别于 WAFR 2014 会议年份。
- [13] IEEE 索引正文可读取，直接页面需要 JavaScript 验证；采用正式期刊信息。
- [14] 出版商页面不可读取，正式书目信息由 PubMed 原始收录记录核对，方法内容由作者公开版本核对。
- [15] 正式发表日期为 2026-03-14，早于本次核验日期。
- [18] 书目和目录已核验；正文推导由稿件公式独立检查，书籍全文需作者有权访问后复核。
- [19] CVF 直接打开返回 403，正式索引内容与作者 arXiv 记录交叉核对。
- [21] 书目和目录已核验；原书全文中的具体公式需作者有权访问后复核。

书籍全文访问与最终引用格式仍需作者按目标期刊或学校要求复核。所有 22 篇文献均在正文引用；未通过加入写作工具宣传文献或无关综述增加数量。
