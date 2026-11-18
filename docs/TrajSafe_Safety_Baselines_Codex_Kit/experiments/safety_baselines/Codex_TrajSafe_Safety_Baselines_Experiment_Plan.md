# TrajSafe-Diffuser：七种已发表安全扩散方法的复现实验方案（Codex 任务书·独立目录版）

> 目的：在**不改动 Ours 主算法、不重做数据集和主实验**的前提下，把原先缺乏一一对应论文实现的自定义安全基线，替换为 3 篇原论文明确提出的 7 种算法。优先保证算法忠实度、公平性与结果可追溯性；不以“跑出更好的结果”为目标。
>
> Codex 工作原则：**先审查再实现，分阶段提交；禁止一次性大改并直接跑正式实验。** 遇到原论文信息不充分或与现有 DDIM 不兼容的地方，记录问题、提出最小适配方案、标明“论文原生实现 / 任务适配实现 / 改编实现”，不得悄悄简化或冒用原名。
>
> **工作根目录：`experiments/safety_baselines/`**。除明确列出的只读公共组件外，所有外部算法、适配代码、配置、测试、脚本、文档与输出都放在该目录下。**不能将本实验写回 Ours 的 `src/diffusion/`、现有训练脚本或原论文结果目录。**

## 0. 范围、论文与名称（必须严格遵守）

仅研究**二维静态占据栅格地图中的单机器人几何路径生成与障碍安全**，输出 B 样条轨迹；不引入动态障碍、机械臂动力学、多机器人或实物实验；不修改 Ours 训练与已有评估结果。

对比的论文与**原文方法名称**如下（论文标题用于区分；方法名不得擅自加 `CD-` 等前缀）：

| 论文 | 原文方法名称（展示时照原论文使用） | 直接核对的原文位置 |
|---|---|---|
| SafeDiffuser: Safe Planning with Diffusion Probabilistic Models (ICLR 2025) | RoS-diffuser | §3.1 Thm. 3.4、§4 Eq. (13)、Alg. 1 |
| 同上 | ReS-diffuser | §3.2 Thm. 3.5、§4 Eq. (14)、Alg. 1 |
| 同上 | TVS-diffuser | §3.3 Thm. 3.6、§4 Eq. (13)、Alg. 1 |
| Constrained Diffusers for Safe Planning and Control (NeurIPS 2025) | Projected | §5.1 Eqs. (10)–(11) |
| 同上 | Primal-Dual | §5.2 Eqs. (12)–(14) |
| 同上 | ALM | §5.3 Eqs. (15)–(18) |
| Constrained Synthesis with Projected Diffusion Models (NeurIPS 2024) | PDM | §4.2 Eq. (8)、Algorithm 1；附录的 PDM Corrector 需核对差异 |
| Diffusion 基础对照 | Diffuser | 必须明确对应的预训练基座和推理配置 |
| 本文完整方法 | TrajSafe-Diffuser | 当前锁定的 Ours 原实现 |

**禁止**将 `Projected` 与 `PDM` 认为是彼此独立的原创思想：Constrained Diffusers §5.1 明确是在既有投影工作基础上扩展。它们可以作为两个算法实现对比，但必须说明采样循环/投影调用的实际差异，避免重复跑完全同构的代码凑数。

### 原文材料（已随任务书打包）

**七种安全算法对应三篇核心论文**，已放在本任务书相邻的 `papers/` 目录，文件保持上传的 PDF 原名和原始内容，不需要 Codex 再从网络猜测或下载版本：

1. `papers/ICLR-2025-safediffuser-safe-planning-with-diffusion-probabilistic-models-Paper-Conference(1).pdf`
2. `papers/NeurIPS-2025-constrained-diffusers-for-safe-planning-and-control-Paper-Conference.pdf`
3. `papers/NeurIPS-2024-constrained-synthesis-with-projected-diffusion-models-Paper-Conference.pdf`

上述相对路径均以 `experiments/safety_baselines/` 为起点。解压 ZIP 到**现有 TrajSafe-Diffuser 仓库根目录**后，先确认三个文件均存在且 PDF 可读取，再执行 Phase 0。

这些原文是**实现依据**。若有官方代码，记录官方仓库链接、commit、许可、依赖、对应脚本/函数；优先与原文逐式交叉核对。不得把本任务书的概括当作覆盖原文的规范。

### 执行边界与变更隔离

- **专门目录：** `experiments/safety_baselines/`，内部资料与输出自包含；可 `import src...` 读取已有代码，但不复制整个仓库，也不绕过现有数据接口重做数据集。
- **Git 分支：** 建议 `experiment/safety-baselines`。先记录原分支、HEAD 和未提交改动；若工作树含有用户未提交改动，**不能自动 stash/reset/覆盖**，先报告并制定安全隔离方案。只读审查完成后再建立分支或单独 worktree。
- **不可修改的核心资产：** Ours 的 `src/diffusion/sampler.py`、`src/diffusion/bspline_alm.py`、训练代码、已有 checkpoint、正式实验配置和既有结果。确需共享接口时，先提出最小改动清单，须有原有测试及回归对照才可修改。
- **论文 PDF：** 打包供本地研究与复现使用，建议 `papers/*.pdf` 加入忽略规则，避免将 PDF 大文件误提交到 Git；仅提交论文来源/哈希的元数据。
- **结果与日志：** 全部落在本目录 `results/`，每次用唯一 `run_id`，绝不覆盖 Ours 以前的输出；大文件加入 `.gitignore`，正式结果另行备份。
- **范围：** 只增加外部安全扩散算法及统一评价，不对 Ours 做任何“顺便优化”或推断性的改动。

## 1. Phase 0：先审查仓库、本地工作树和既有实验（只读，不修改）

已知公开仓库为 `Hiashekiy/TrajSafe-Diffuser`，其 `v3-skeleton-dynamic` 分支可见以下接口（但**不能断定是用户本地最新工作树**）：

- `src/diffusion/sampler.py`：DDIM / Ours 走廊激活 / 每步 ALM / 历史反馈；`sample()`、`pick_times()`、`dense_validation()`。
- `src/diffusion/bspline_alm.py`：控制点空间 ALM，`bspline_alm_correct()`；**只能作为 Ours 的既有实现，不自动充当 PDM 投影器或 Constrained Diffusers-ALM**。
- `src/geometry/bspline_constraints.py`：B 样条/Bézier 提取和凸走廊约束，包含 `build_constraint_pack()`。
- `src/geometry/bspline.py`：B 样条编解码。
- `src/models/trajsafe/planner.py`：生成器。
- `scripts/eval_campaign_testset.py`、`evaluate.py`：已有评估与统计。

必须先执行并把结果写入 `experiments/safety_baselines/docs/baseline_audit.md`：

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git log -5 --oneline
python -m pytest -q
```

如果 `pytest` 全量耗时大或依赖缺失，应先跑与 sampler/geometry/ALM 相关的现有单测，并记录未运行项目与原因。

审查并确认（禁止凭 README 的旧默认值猜测）：

- 当前**实际**主实验的 `num_controls`、B 样条阶次/端点约束、noise schedule、`DDIM steps`、checkpoint、随机种子；论文当前实验写为 48 个三次 B 样条控制点/16 步 DDIM，而公开分支某些旧配置为 32 控制点；以**本地已完成论文实验所用配置与 checkpoint**为准，冲突必须报告。
- 地图单位/坐标：当前论文 160 m × 160 m、256×256 栅格、0.625 m/格；检查 scene-to-meter、障碍膨胀、安全半径、坐标变换，绝不混淆单位。
- `Diffuser` 的**真实基座**：能否从当前基座模型获得不带学习式安全条件/历史反馈的轨迹输出？不得把完整 Ours 关闭 ALM 后冒充完全无安全信息的 Diffuser。
- 是否存在可靠的**score / ε / x0**转换接口？是否可得到原论文需要的逐步 score？有些论文采用 Langevin / score-based 采样，而 Ours 现用 DDIM；**不得默认采样器等价**。
- 测试集场景 ID 和 split：420 个独立测试场景；确认当前 1260 条轨迹与随机种子对应关系，禁止训练/验证/测试混用。
- 当前 Ours 的失败路径：走廊激活失败、空约束包、ALM 未收敛、数值 NaN、端点误差、地图边界越界；不可把失败样本当作成功样本。

**Phase 0 交付物：** `experiments/safety_baselines/docs/baseline_audit.md`、依赖/版本记录、锁定的 base checkpoint 与配置、`experiments/safety_baselines/docs/baseline_source_map.md`（下节）；不修改 Ours、数据和当前结果。

## 2. Phase 1：建立“论文→代码”忠实度对照矩阵（未完成不得进入实现）

创建 `experiments/safety_baselines/docs/baseline_source_map.md`，至少有这些列：

`论文标题 | 原文方法名 | 核心公式/算法编号 | 输入/输出 | 原始采样器 | 必需优化变量 | 原始约束类型 | 官方代码文件/commit | 适配到控制点空间的映射 | 与论文差异 | 结论（可复现/受限）`

核对每个家族的关键点：

**SafeDiffuser**

- RoS：Eq. (8) + Eq. (13)，鲁棒项 γ，QP 最小化相对名义扩散动态的改动。
- ReS：Eqs. (9)–(10) + Eq. (14)，松弛变量 r、权重 w(j) 随扩散步变化；最后硬约束的阶段必须实现。
- TVS：Eqs. (11)–(12) + Eq. (13)，时间函数 σ(j) 的初末条件；不要和 ReS 的松弛变量混为一谈。
- 同一基础 QP/CBF 框架，但**三种时间条件和优化变量不同**。检查 Eq. (13)/(14) 与算法 1 的每步更新、符号方向、扩散时间索引，不能用“CBF 罚函数梯度”替代原文 QP。

**Constrained Diffusers**

- `Projected`：§5.1，逆扩散 Langevin 更新后调用可行集投影 `Π_C`，不是仅最后投影。
- `Primal-Dual`：§5.2，式 (13)–(14) 对轨迹采样和对偶乘子交替更新；记录约束函数正负号、步长与投影非负正交象限的时机。
- `ALM`：§5.3，式 (15)–(18) 明确存在 slack、乘子和 penalty 更新；它**不是**直接调用 Ours 的 `bspline_alm_correct`。
- 原文还用 DCBF / 逆动力学模型实现闭环控制（§6）；本项目**只复现其用于离线二维安全轨迹生成的受约束采样算法**，不能宣称复现完整在线控制系统；明确列为 task adaptation。
- 原文分布层约束涉及期望 `E_q[g(x)]`，审查以采样平均/批量近似实现的统计含义；不能未经说明直接将期望约束偷偷改成单样本硬约束并声称严格等价。

**PDM**

- §4.2 Algorithm 1：每个扩散时间层有 M 次 score/Langevin 更新，每次之后投影到 C；保留 T/M 两层循环、score/noise 更新、投影与温启动。
- 核对附录 `PDM Corrector Algorithm` 与主文 Algorithm 1 的具体使用场景与 gamma 规则，选择明确版本并报告。
- `PDM` 与 `Constrained Diffusers` 的 `Projected` 若被实现成完全同一个 loop，应中止其中一个并先澄清：这不是合格的七算法复现。

**忠实度标记：** 每种方法打上 `original / task-adapted / modified`，写清唯一必要适配；若改变了原文核心更新方式，标记 `modified`，不得在稿件里暗示“原论文原封不动的算法”。

## 3. Phase 2：统一任务、输入与不依赖 Ours 的安全评价

### 3.1 公共任务

输入统一为 `occupancy grid`、`start`、`goal`、机器人安全半径/障碍膨胀、固定地图坐标规则；输出统一为 `Q [C,2]` 和最终连续曲线 `p(u)`（可在需要时对离散输出做**声明过的**共同后处理）。

- Ours 的骨架搜索/安全几何网络属于 Ours 主流程；**不要将 Ours 预测出的椭圆、凸走廊或训练用 GT 走廊给予外部基线**。
- 外部方法由相同占据地图计算 SDF/可微障碍约束，或使用原方法需要的可行域 `C`；所有方法保持同一地图、障碍膨胀、安全间距、起终点条件。
- 外部方法若确实需要轨迹的初始示范/模型条件，与 Ours 保持数据分割一致；不得使用测试 GT 路径引导基线采样。
- 若基座网络结构依赖骨架输入，Codex 必须在报告中明确“网络级共同先验”的存在；不得宣称各方法拥有完全相同或完全无骨架的输入。优先采用已训练的**无安全条件**的 Diffuser 基座模型。

### 3.2 可微安全函数及控制点映射

在 `experiments/safety_baselines/adapters/constraints.py` 构建与地图一致的可微障碍函数：

- 例如可用 `h(p)=SDF(p)-r_safe`，规定自由空间为 `h>=0`，越界/障碍均为不安全；SDF 在 2D 栅格上采用一致的双线性插值，标记近障碍梯度不稳定或无法定义的点。
- 对 B 样条轨迹 `p(u)=B(u)Q`，必要时使用链式法则 `∇_Q h = B(u)^T ∇_p h`；必须在单测中用有限差分比对自动微分/解析梯度。
- 如果原算法要求约束施加在**扩散状态**而非预测 clean state 上，保持论文对应步骤；不能混淆 `Q_t`、`Q0_pred` 和 `Q_safe`。
- 处理每一条曲线上的采样点、起终点条件及边界；明确哪些方法能提供数学保证，哪些只在离散评价点满足约束。直接使用 512 点碰撞检测**不能声称严格连续安全**。
- 可行集投影的近似求解器、最大迭代、停止容差和失败标准必须独立定义；禁止以 `Ours` 的 `bspline_alm_correct` 偷换为 PDM/Projected 的原文投影算子，除非验证它确实求解相同的最近点投影问题并充分说明。

### 3.3 独立评价器

必须在所有方法最终输出上，使用同一个**与各算法内部优化器无关**的地图碰撞检查。至少包含：

- 到达成功（起终点误差容限固定且提前声明）；
- 轨迹碰撞率（在公共 512 点 + 近障碍自适应加密检测）；
- 规划成功率（路径有效且无碰撞且起终点满足要求且有限时间内返回）；
- 最小安全距离（只有无碰撞成功轨迹计算，明确条件均值）；
- 绕行系数、NRMSE（与共用 GT 的偏差是辅助项，不是通用安全优劣的充分证据）；
- 最大曲率、平均平方曲率；
- 生成前/后公共 SDF 违规最大值与正违规均值（在**同一物理轨迹采样点**计算）；
- 端到端单任务耗时的均值、中位数、P95，包括各方法**实际使用**的 SDF/骨架/走廊预处理、采样、QP/投影/ALM；训练时间单列，不纳入单次推理；
- 各方法内部可行率（例如 Ours 凸走廊约束满足率）单独统计，**不将不同方法内部定义的约束可行率直接当同一个公平指标**；
- 所有无解、超时、QP infeasible、NaN、无法闭合走廊、数值失败必须计入失败率，输出失败原因直方图。

**避免指标混淆：** `0 collision` ≠ `100% continuous constraints feasible` ≠ `100% task success`。

## 4. Phase 3：独立目录与代码隔离（必须遵守）

**所有新代码、配置、实验脚本、文档、测试及结果严格放在 `experiments/safety_baselines/`。** 按如下目录实施。文件名为工程组织用途，论文中的算法展示名称仍完全沿用原文。

```text
TrajSafe-Diffuser/
├── src/                                    # Ours 原实现：原则上只读
├── configs/                                # Ours 原配置：原则上只读
├── scripts/                                # Ours 原脚本：原则上只读
└── experiments/
    └── safety_baselines/
        ├── README.md                       # 使用说明、运行入口、来源索引
        ├── Codex_TrajSafe_Safety_Baselines_Experiment_Plan.md
        ├── run.py                          # S0/S1/S2/S3 的唯一统一入口（待 Codex 实现）
        ├── algorithms/
        │   ├── __init__.py
        │   ├── safediffuser/               # RoS-diffuser / ReS-diffuser / TVS-diffuser
        │   ├── constrained_diffusers/      # Projected / Primal-Dual / ALM
        │   └── pdm/                        # PDM
        ├── adapters/
        │   ├── __init__.py
        │   ├── model_adapter.py            # 安全读取基座与 checkpoint
        │   ├── score_adapter.py            # x0/epsilon/score，核对噪声调度
        │   ├── constraints.py              # 公共地图安全函数、梯度及几何接口
        │   ├── trajectory_adapter.py       # 控制点与离散轨迹数据对接
        │   └── evaluator.py                # 独立于所有算法的统一评价器
        ├── configs/
        │   ├── common.yaml
        │   ├── safediffuser.yaml
        │   ├── constrained_diffusers.yaml
        │   └── pdm.yaml
        ├── scripts/
        │   ├── audit.py
        │   ├── check_algorithms.py
        │   ├── eval_safety_baselines.py
        │   └── summarize_safety_baselines.py
        ├── tests/
        │   ├── test_constraints.py
        │   ├── test_score_adapter.py
        │   ├── test_safediffuser.py
        │   ├── test_constrained_diffusers.py
        │   ├── test_pdm.py
        │   └── test_common_eval.py
        ├── docs/
        │   ├── baseline_audit.md
        │   ├── baseline_source_map.md
        │   ├── baseline_reproduction_report.md
        │   └── baseline_results.md
        ├── papers/                         # 附带的 3 篇核心 PDF，原名保存
        └── results/
            └── <run_id>/
                ├── config_snapshot.json
                ├── manifest.json
                ├── results.jsonl
                ├── aggregate.json
                ├── summary.md
                ├── logs/
                └── figures/
```

**路径约束：**

1. `algorithms/` 各家族算法分别实现，不能把 Ours 的 ALM/投影包装成论文外部方法。每个方法实现共同的 Runner 接口，返回至少 `trajectory`、`status`、`failure_reason`、`wall_time`、`per_step_trace`、`method/source_id/fidelity`。
2. `adapters/` 只处理跨实现的数据对接、坐标转换、score 合法转换、模型读取，不承载 Ours 新算法逻辑。可通过 Python import 复用已有 B 样条编解码、数据集读取及公共几何函数，**不得复制仓库核心模块**；若其行为依赖 Ours 专有走廊则不能混用。
3. `configs/` 中每组算法各有配置；公共数据/评价配置集中在 `common.yaml`。运行时冻结所有最终生效参数至 `results/<run_id>/config_snapshot.json`。
4. `scripts/` 和 `run.py` 必须支持仅执行一组方法或一阶段，并支持断点续跑与 `--dry-run`；不要写多个互相不一致的统计入口。具体 CLI 参数在 Phase 0 审计后制定，不能预设已有命令可用。
5. `tests/` 里有逐式数值测试、输入输出/梯度测试和 Ours 回归测试；从仓库根目录运行时处理好 Python import，不依赖用户手动 `cd` 到奇怪目录。
6. `docs/` 中保存审计报告、逐公式映射、任务适配差异和结果结论，本文最后的“仅审查”指令不得在未获授权时创建算法实现。
7. `results/` 按 `run_id` 分开保存原始数据、失败记录、日志、配置、版本哈希及图表。禁止覆盖已有输出；大文件不进入 Git。
8. 随机数由 `(scene_id, seed, method)` 明确派生且可重现，不依赖 batch 切分顺序；计时须 GPU 预热和 CUDA synchronize，记录硬件及环境版本。
9. 若必须改原项目公共模块：先列出改动前后行为，保留原接口；提供冻结测试数据上同 seed 输出一致性/容差报告，再单独提交。**不得动现有 checkpoint、训练数据及正式实验结果。**
10. 建议每阶段单独中文 commit：`审查基线与论文算法`、`实现约束采样基线`、`实现安全CBF基线`、`实现PDM投影基线`、`统一评估与小规模验证` 等。仅提交新实验相关代码、配置、文档和小型测试，避免包含 PDF 和大型结果。

## 5. Phase 4：逐算法实现与算法级单测（每完成一家族就提交）

### 5.1 首先：Constrained Diffusers（Projected / Primal-Dual / ALM）

逐式复现 §5.1–§5.3：

1. 使用预训练基座提取论文所需 score / 名义逆扩散更新；对 `x0` 预测模型需核对 noise schedule 才能换算 score。若无法合法适配，必须先报告阻塞，不许直接把 DDIM + SDF 罚项叫原文方法。
2. `Projected`：每次指定的逆扩散更新后求 `Π_C`；验证投影后约束残差、最邻近解及求解状态。
3. `Primal-Dual`：按原文式 (13)、(14) 逐步更新轨迹和乘子，记录 `λ` 非负性、约束违规与 multiplier 轨迹。
4. `ALM`：按式 (15)–(18) 更新采样、slack、λ 与 ρ；特别测试 slack 正确性和罚系数更新。
5. 保存论文原文参数与改编参数的对照表。三方法共享地图安全约束函数和 score 适配器，但**各自必须拥有真正不同的核心算法路径**。

**验收单测：** 2D 圆/椭圆避障玩具例、无约束退化例、边界/越界、无可行解、固定种子、公式单步数值对照、轨迹终点硬条件、有限数值性；额外检查 Projected 与 PDM 不能由于共用代码而变成相同调用序列。

### 5.2 然后：SafeDiffuser（RoS-diffuser / ReS-diffuser / TVS-diffuser）

1. 实现共同的名义扩散动态修正变量及最小二乘 QP，校验 `Δτ`、符号与时序。
2. RoS：Eq. (8)/(13) 包括 `γ(N,ε)` 的合法选择。
3. ReS：Eq. (9)/(10)/(14) 包括 `r`、`w(j)` 和末端的硬约束区间；QP 不可行必须报告。
4. TVS：Eq. (11)/(12)/(13) 包括 `σ_k(N)`、`σ_k(0)` 及其时间导数。
5. 验证约束已满足时应尽量不改变名义采样；违反时记录每步 CBF 残差、QP 状态、QP 耗时、局部陷阱表现。

**验收单测：** 无障碍、单椭圆、双障碍重叠/窄通道、局部陷阱、约束时序图、QP 的 KKT/残差、不同分支是否确实满足原文差异。不要宣称离散采样、近似 QP 与离散 SDF 下具有原定理的无条件连续安全保证。

### 5.3 最后：PDM

1. 按 §4.2 Algorithm 1 建立外层噪声时间 `T` 与内层 `M` 次 Langevin + projection，不得简化为“仅每个 DDIM 步投影一次”。
2. 正确求解 `Π_C(z)=argmin ||x-z||², x∈C`，保留 warm start；保留 solver iteration、残差、求解时间。
3. 记录主文 Algorithm 1 与附录 Corrector 的区别（如步长设定），采用明确的复现版本。
4. 非凸地图可行集上投影可能非常慢或者失败；设置经过验证的时间预算和失败判定，禁止无限运行或静默跳过失败样本。

**验收单测：** 凸可行域解析投影、非凸两圆绕障、M 次循环实际调用次数、时间噪声 schedule、独立 score 适配、约束投影 residual、完全相同输入得到相同 seed 结果。

## 6. Phase 5：实验分级与验收关卡

**S0：算法级微型测试**（每个方法必做）

- 2D 解析约束：半空间、圆/椭圆，数值对照原文公式，所有单测通过。
- 单步日志检查：score、约束值、梯度、QP/投影的残差、变量增量和计算时长。
- 没通过不得跑 CARLA 统计。

**S1：20 个覆盖性的 CARLA 场景 × 1 个 seed**（冒烟测试）

- 包含开阔、窄通道、多个绕障分支、曲折道路、起终点临近障碍，**从测试集预先冻结清单**，不得按结果挑选。
- 所有 9 组方法尝试跑完，输出每个案例的成功/失败/耗时与轨迹图；先解决实现错误，再调验证集参数。
- 如果某方法在任务适配上不成立，停止它的正式评估并在报告说明，不许调低安全标准来凑齐 7 方法。

**S2：100 个冻结场景 × 3 个 seeds**（中等规模测试）

- 用验证集选定各自超参数后冻结；S2 不得以结果为由反复改参数。
- 给出成功率、碰撞率、公共违规量、绕行系数、曲率、时间 mean/P95，按场景 paired 统计。
- 检查重复运行确定性、置信区间、超时/失败原因、不同家族之间的计算成本。

**S3：完整正式实验**（通过 S0–S2 才运行）

- 原论文主测试：420 场景 × 3 seeds = 每方法 1260 次推理；**3 seeds 是建议与现有 1260 保持一致，须先审核当前论文的实际采样协议**。
- 每种方法同一场景/seed 键，计时覆盖同一端到端边界；保存逐样本 JSONL 或 Parquet 和完整运行元数据。
- 正式测试数据不得参与调参、改阈值或挑选代表性图；若发现实现 bug，修复后须重跑**受影响的所有方法**并记录版本。
- 对计算特别昂贵的 PDM 可先 S1/S2 单独估算可行性；不能仅挑 PDM 成功的样本做最终对比。

### 统计口径

- 主要结论以**规划成功率 + 轨迹碰撞率 + 公共约束残余 + 端到端时间**为主。
- 轨迹几何质量指标对成功样本报告条件均值，同时给出成功样本数；失败轨迹不可在长度/曲率对比中“消失”。
- 按场景配对做置信区间/差异检验；同场景不同随机种子属于重复测量，不应当不加说明地全部视作独立环境。
- 关键场景图固定选取：1 个开阔、1 个窄通道、1 个多路线、1 个失败案例（失败案例可不需要在论文正文展示，但必须保留审计）。

### 计算预算公平性

至少报告两个视角：

1. **原论文算法忠实设置**：允许各算法采用自身正确的采样步骤/优化迭代，不强行统一为 16 DDIM；记录所有步骤和总耗时；对同一任务比较在不同代价下的实际效果。
2. **统一实际时间预算的敏感性分析**：在尽可能保持算法定义不变的情况下比较相近计算预算下的规划成功率；如果采样器原生机制无法在该预算完成，则报告失败或不适用，绝不可通过改变核心公式冒充原算法。

## 7. 输出格式：论文表格与可复现材料

建议主论文的一张对比表（方法展示名严格照原文，来源由 `Paper` 列或表注标明）：

`Paper | Method | Success % | Collision % | Common SDF violation (m) | Detour | Max curvature | Time mean / P95`

次表或补充材料：`Method | Endpoint error | NRMSE | Clearance | Solver/Projection calls | Failure count | Failure reasons`。

图：同一固定场景的各方法轨迹图（勿全部 9 种挤在一图，可按家族划分），时间–成功率散点或 Pareto 视图（仅在数据显示有意义时制作），论文式图/表均自动由原始测试记录生成。

机器可读交付至少包括：

- `experiments/safety_baselines/docs/baseline_source_map.md`：原文算法到代码的准确映射及偏差；
- `experiments/safety_baselines/docs/baseline_reproduction_report.md`：实现、验证、主要不一致、官方实现对照情况；
- `experiments/safety_baselines/results/<run_id>/{config_snapshot.json,manifest.json,results.jsonl,aggregate.json,summary.md}`；
- `experiments/safety_baselines/scripts/eval_safety_baselines.py`：固定协议正式评估；
- `experiments/safety_baselines/tests/`：原文公式与几何安全单测；
- 对应可复现的 commit SHA/依赖清单/设备/随机种子/失败状态。

每阶段 Codex 提交一个**中文** commit（除非原仓库另有约束），输出变更文件、测试、重要偏差、剩余风险；未经用户确认不要直接提交整个全量计算作业或覆盖已有结果。

## 8. 可交给 Codex 的执行指令（建议逐阶段发送，不要一次性要求做完）

第一条（仅审查）：

> 请阅读 `experiments/safety_baselines/Codex_TrajSafe_Safety_Baselines_Experiment_Plan.md` 和同目录下 `papers/` 中的三篇 PDF 原文。所有新实验资产必须放在 `experiments/safety_baselines/`，原项目 Ours 代码与已有结果只读；工作分支采用 `experiment/safety-baselines`（先审查工作树是否安全再创建）。当前任务是 Phase 0 + Phase 1，**只读审查与文档产出，暂不实现、训练、调参或跑正式实验**。先确认本地 HEAD/config/checkpoint 与论文实验规格的一致性；梳理现有 Diffuser 基座、sampler、B 样条约束、独立评价器；逐式完成 7 算法的 `baseline_source_map.md`，标出原文公式、原始采样器、数学变量、必要任务适配和不能保持忠实复现的点。执行现有相关单测，记录结果。不得擅自起新算法名称，也不要把 Ours ALM 用来冒充外部投影/ALM。

第二条（Phase 2 + Constrained Diffusers）：

> 根据审核通过的映射表，实现放在 `experiments/safety_baselines/` 下的独立公共 SDF/控制点梯度/score 适配/评价代码及 NeurIPS 2025 `Projected`、`Primal-Dual`、`ALM`，逐式对应 Eqs. (10)–(18)。建立公式级单测并跑 S0；不要改动 Ours 采样器和训练代码。提交中文 commit，列明每种方法的忠实度与偏差；不得修改原有 `src/diffusion/` 和正式配置。未通过公式测试不进入 S1。

第三条（SafeDiffuser + PDM）：

> 复现原文 `RoS-diffuser`、`ReS-diffuser`、`TVS-diffuser`（ICLR 2025，Thm.3.4–3.6 / QP13–14 / Alg.1）和 `PDM`（NeurIPS 2024，§4.2 Alg.1），完成 QP/松弛/时间函数/逐步投影的单测；不改变算法名、不偷换为梯度罚项，不用 Ours 的终端 ALM 兜底。独立提交与验收。

第四条（S1→S2→S3）：

> 先在冻结的 20 场景 S1 验证全部方法的同输入、同碰撞评价与失败记录；随后在 100 场景×3 seeds 执行 S2，审计完整时间和失败原因。只有 S0–S2 验收通过，才在 420 场景×3 seeds 执行 S3。保存到 `experiments/safety_baselines/results/<run_id>/`，保留所有逐样本结果与准确的论文算法名称、来源、实现忠实度、commit SHA。严禁调测试集、挑样本或通过静默回退优化成功率。

## 9. 阻塞条件和红线

出现以下情况必须立即报告，不要自行“合理化”：

1. 当前基座无法合法得到原论文采样器需要的 score，而直接套用 DDIM 会改变核心方法；
2. CBF QP 无法在所需约束形式上求解或 ReS/TVS 时间条件被破坏；
3. 非凸投影算法始终失败/超时，或用错误的近似解伪称严格投影；
4. 变更 B 样条控制点数量/归一化/物理单位导致现有训练 checkpoint 不兼容；
5. 使用 Ours 预测走廊、GT 路径或测试集标签给外部基线提供额外信息；
6. 使用两个完全相同的核心算法循环，但分别命名 `Projected` 和 `PDM`；
7. 隐藏失败、改写既有统计结果、把离散无碰撞等同连续安全证明；
8. 原论文完整算法无法在当前任务保持忠实度，却仍以原文算法名无说明地发表结果。

9. 对比算法代码或结果散落在顶层 `src/`、`scripts/`、`configs/`、`outputs/`，破坏本次专属目录的隔离规则；
10. 直接提交 `papers/*.pdf` 或海量正式评估原始结果，未建立归档与追溯机制。

**最终验收标准：** 所有用于主表的对比实现能回指明确原文公式/算法，公式单测通过，任务适配及其局限记录完备，测试协议统一、结果可重跑、失败透明、Ours 现有结果不变。
