from pathlib import Path
import re
import json
import hashlib
import difflib

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'TrajSafe_paper_revised.md'
OUTPUT = ROOT / 'TrajSafe_paper_optimized.md'
original = SOURCE.read_text(encoding='utf-8-sig')
text = original
changes = []

def replace(old, new, reason):
    global text
    if text.count(old) != 1:
        raise ValueError(f'Expected one match: {reason}; found {text.count(old)}')
    text = text.replace(old, new)
    changes.append(reason)

front = r'''## 摘要

扩散模型能够学习多模态轨迹分布，但分布学习本身不能保证生成轨迹满足环境安全约束。在复杂非凸环境中，如何构建与任务匹配的安全表示，并使约束修正持续作用于后续生成，是安全扩散规划的重要问题。本文提出 TrajSafe-Diffuser，一种生成式安全约束下的扩散运动规划方法。首先，根据环境自由空间的候选通路预测局部椭圆尺度与方向，通过障碍分离和区域衔接构建任务相关的凸安全走廊；随后，将局部安全隐特征作为去噪条件，并利用分段 Bézier 曲线的凸包性质，将连续曲线的走廊归属转化为 B 样条控制点的线性约束；最后，在逆扩散过程中采用增广拉格朗日法修正控制点，将修正结果用于采样状态更新，并将历史控制点及修正量反馈至后续去噪，形成几何生成、约束修正与历史反馈相互配合的闭环。在由 CARLA 道路场景构建的二维静态栅格测试集上，420 个样本经 3 个随机种子重复生成，共得到 1260 条轨迹，完整方法未发生碰撞，SDF 可行率达到 100%。消融结果支持安全几何条件、历史反馈和安全状态更新的互补作用；在单种子效率实验中，4 步采样与区域共享的组合配置保持 0/420 碰撞，并将批量折算的平均单轨迹耗时由 84.9 ms 降至 26.8 ms。结果表明，该方法在当前测试协议下能够提高轨迹安全性，并通过减少采样和几何处理开销改善推理效率。

**关键词：**扩散运动规划；生成式安全约束；凸安全走廊；闭环安全反馈；增广拉格朗日法

## 1 引言

运动规划需要在给定环境、初始状态和任务目标的条件下，生成连接起终点并满足安全要求的运动方案[REF:lavalle]。搜索、采样和轨迹优化是实现这一目标的主要途径：采样方法通过探索自由空间寻找可行通路，轨迹优化则根据障碍代价和光滑性目标调整路径[REF:rrtstar,chomp]。复杂非凸环境中的多条绕行通路、连续转弯和局部狭窄区域，使通路选择与连续轨迹形状相互影响，也增加了规划质量与计算开销之间的权衡。

扩散模型通过逐步去噪学习数据分布[REF:ddpm]，为数据驱动的运动规划提供了新的生成方式。Diffuser 将轨迹去噪、任务条件和代价引导结合，使轨迹采样能够承担规划功能[REF:diffuser]；SceneDiffuser 和 Motion Planning Diffusion（MPD）进一步将场景信息或任务代价融入生成过程[REF:scene,mpd]。这些方法能够利用已有示范提供多模态轨迹先验，但学习到可行示范的分布并不等同于每次采样都满足安全约束。环境变化、生成误差及有限采样预算仍可能使轨迹进入障碍区域。

显式约束为安全生成提供了补充。控制屏障函数通过约束系统演化维护安全集合的不变性[REF:cbf]，SafeDiffuser 将相关思想用于扩散去噪过程[REF:safediffuser]；投影扩散与受约束多机器人扩散规划则在采样中引入约束优化[REF:pdm,smd]。这类研究说明，安全要求可以在生成过程中施加，而不仅是在最终输出后检查。对于非凸道路环境，仍需回答两个相互关联的问题：一是如何从环境结构中获得与当前任务匹配、便于连续曲线验证的约束表示；二是如何让修正提供的信息参与后续预测，使生成模型能够利用此前的纠偏结果。

凸区域为连接环境几何与连续轨迹约束提供了基础。IRIS 和 FIRI 通过障碍分离构造局部无障碍凸区域[REF:iris,firi]，凸集图（Graphs of Convex Sets，GCS）进一步结合区域连通关系与曲线优化进行规划[REF:gcs]。近期的 Discrete-Guided Diffusion（DGD）也将离散通路、凸子问题和约束修复用于多机器人扩散规划[REF:dgd]。在这些工作的基础上，本文关注任务相关的局部几何预测如何同时服务于去噪条件和连续安全约束，并通过跨步历史反馈连接几何修正与后续生成。

基于上述考虑，本文提出 TrajSafe-Diffuser。该方法在二维静态占据栅格中选择连接起终点的候选通路，沿通路预测局部椭圆尺度与方向，再结合障碍分离和区域衔接构建连续凸走廊。这里的“生成式安全约束”指根据当前环境和任务自动构建安全表示：学习分支提供局部几何和安全隐特征，几何构造器将其转化为显式半空间约束。安全隐特征参与轨迹去噪；走廊约束通过分段 Bézier 提取作用于 B 样条控制点，并由增广拉格朗日法（Augmented Lagrangian Method，ALM）执行修正。修正控制点既参与下一步采样状态更新，也连同修正量作为历史条件输入后续去噪。由此，环境相关的几何表示、连续曲线修正与学习到的轨迹先验形成闭环。

本文的主要贡献如下：

（1）构建任务相关的生成式安全表示。通过候选通路匹配、局部椭圆预测与确定性凸区域构造，将复杂非凸环境中的一条可行通路组织为有序安全走廊，为轨迹生成提供局部安全条件与显式几何约束。

（2）将安全条件去噪与连续曲线约束相结合。局部安全隐特征通过交叉注意力参与控制点预测；分段 Bézier 曲线的凸包性质将连续曲线的走廊归属转化为控制点线性不等式，使生成引导与几何验证对应同一任务通路。

（3）建立跨步安全修正与历史反馈机制。ALM 修正结果通过采样状态更新和门控历史特征两条路径影响后续去噪，并通过共享参数的两次前向训练学习历史修正的利用方式。CARLA 栅格实验从整体性能、递进消融和推理效率三个方面评价上述机制及其适用边界。

## 2 相关工作

### 2.1 扩散轨迹生成与运动规划

去噪扩散概率模型通过加噪与逆向去噪建立生成过程[REF:ddpm]，DDIM 则提供与扩散训练目标兼容的隐式采样方式[REF:ddim]。在规划领域，Diffuser 将整段轨迹作为生成对象，通过条件补全和代价引导处理任务目标[REF:diffuser]。SceneDiffuser 联合场景条件生成、物理优化和目标规划[REF:scene]，MPD 将学习到的轨迹先验与代价梯度结合，并在其扩展版本中采用 B 样条降低轨迹表示维度[REF:mpd]。本文沿用扩散先验与 B 样条表示，进一步结合任务相关安全几何与连续曲线约束。

### 2.2 受约束扩散与安全修正

控制屏障函数将安全集合的不变性写为可用于优化的约束[REF:cbf]。SafeDiffuser 在去噪过程中引入有限时间扩散不变性和屏障函数约束，以处理安全规划要求[REF:safediffuser]。Projected Diffusion Models 将生成过程表述为受约束优化，覆盖凸约束、非凸约束及物理规律等任务[REF:pdm]；Simultaneous MRMP Diffusion（SMD）进一步将约束优化用于多机器人轨迹生成[REF:smd]。本文在采样中执行安全修正，并进一步将通过接纳检查的修正信息作为后续去噪条件；其作用与直接采用修正结果更新采样状态的作用分别评价。

### 2.3 凸安全区域与连续曲线约束

IRIS 交替执行障碍分离与内接椭球优化，构造局部无障碍凸区域[REF:iris]；FIRI 进一步关注区域质量、计算效率及种子集合的包含要求[REF:firi]。GCS 利用凸区域的图结构与 Bézier 曲线进行避障轨迹优化[REF:gcs]。DGD 将离散多智能体寻路结果与凸子问题结合，并在扩散采样中执行约束修复[REF:dgd]，说明通路结构与受约束生成可以协同使用。本文面向单轨迹静态栅格任务，以候选通路组织安全锚点，由网络预测局部尺度与方向，再构造用于 B 样条连续曲线修正的半空间约束，重点在于局部安全特征、显式曲线约束与跨步历史反馈的结合。

### 2.4 历史条件与后续去噪

自条件扩散将此前的干净样本预测作为后续去噪的附加输入，为利用采样历史提供了一种方式[REF:selfcond]。本文的历史条件来自通过接纳检查的约束修正结果，包括控制点位置、修正量和有效标志；其来源与使用规则均包含安全信息。训练通过两次共享参数的前向构造相邻步骤间的反馈，使网络学习如何利用修正状态和方向。该历史条件与修正结果直接进入 DDIM 的状态更新共同形成闭环，两条路径在第 4.3 节中分别分析。

'''
begin = text.index('## 摘要')
end = text.index('## 3 方法')
text = text[:begin] + front + text[end:]
changes.append('摘要、引言与相关工作围绕几何生成—连续约束—双路径反馈重新衔接；补充并准确限定已有工作')

replace('轨迹采用夹持三次 B 样条表示，控制多边形记为', '轨迹采用夹持三次 B 样条表示[REF:spline]，控制多边形记为', '补充样条表示来源')
replace('设扩散时间步为 \\(t\\)，累计信号系数为 \\(\\bar\\alpha_t\\)。训练时，仅对参考控制多边形', '设扩散时间步为 \\(t\\)，累计信号系数为 \\(\\bar\\alpha_t\\)。采用标准高斯加噪形式[REF:ddpm]，但仅对参考控制多边形', '补充加噪形式来源并说明端点区别')
replace('自适应层归一化（AdaLN）调制各条件注意力子层：', '自适应层归一化（AdaLN）调制各条件注意力子层[REF:dit]：', '补充时间条件调制来源')
replace('每个块依次执行自注意力、地图交叉注意力和前馈变换，并分别保留残差连接：', '每个块依次执行自注意力、地图交叉注意力和前馈变换，并保留残差连接；注意力运算采用标准多头形式[REF:attention]：', '补充注意力来源')
replace('最终融合通过残差相加完成，特征维度仍为 \\(48\\times128\\)。', '最终通过残差相加完成融合。', '压缩重复的特征维度解释')
replace('交叉注意力输出保留查询的控制点轴，因而 \\(U\\) 与 \\(F\\) 均为 \\(48\\times128\\)。融合 MLP 采用层归一化及 \\(128\\rightarrow256\\rightarrow128\\) 的 GELU 变换，对残差融合后的特征进行通道变换。', '交叉注意力保留控制点轴；融合 MLP 采用层归一化及 \\(128\\rightarrow256\\rightarrow128\\) 的 GELU 变换。', '压缩非核心张量说明并保留实现参数')
replace('安全特征 \\(H_{\\mathrm{ell}}\\) 用于条件去噪；椭圆参数用于构造显式约束，二者具有不同的作用。', '安全隐特征 \\(H_{\\mathrm{ell}}\\) 用于条件去噪，椭圆参数为显式约束构造提供局部尺度与方向。预测椭圆本身不作为无障碍证明，连续曲线的安全条件由经过几何检查的凸区域和控制点约束共同定义。', '澄清学习椭圆与显式安全约束的职责')
replace('推理的前三次逆扩散前向用于预热，随后尝试建立走廊。', '推理的前三次逆扩散前向用于预热，随后尝试建立走廊。预热用于在较稳定的预测条件下确定通路及局部几何，走廊建立则为后续修正提供一致的约束。', '解释预热和几何冻结在闭环中的作用')
replace('### 3.5 ALM 修正与反馈\n\n离散轨迹点', '### 3.5 连续安全约束、ALM 修正与反馈\n\n#### 3.5.1 连续曲线的走廊约束\n\n离散轨迹点', '突出连续曲线约束这一核心内容')
replace('本文利用三次 Bézier 曲线的凸包性质，将走廊约束转换为关于原始 B 样条控制点的线性不等式。', '本文利用三次 Bézier 曲线的凸包性质[REF:spline,gcs]，将走廊约束转换为关于原始 B 样条控制点的线性不等式。', '补充凸包性质和连续规划的依据')
replace('该约束给出连续曲线归属的充分条件。对各段汇总后，记约束为', '该约束给出连续曲线归属的充分条件：若各责任区域均位于地图自由空间，且所有段约束满足，则整条曲线位于走廊内。区域重叠用于连接通路，但不单独保证给定样条约束可行；有限次优化的输出仍需检查残差。对各段汇总后，记约束为', '限定连续安全结论的成立条件，区分区域连通与样条可行性')
replace('以 \\(Q^{\\mathrm{ref}}=\\widehat Q_{0,\\mathrm{bdry}}^{(t)}\\) 为参考', '#### 3.5.2 控制点的增广拉格朗日修正\n\n以 \\(Q^{\\mathrm{ref}}=\\widehat Q_{0,\\mathrm{bdry}}^{(t)}\\) 为参考', '分离约束推导与优化求解')
replace('采用不等式增广拉格朗日函数\n', '采用不等式增广拉格朗日函数[REF:optimization]\n', '补充不等式 ALM 的数学依据')
replace('当前修正结果记为 \\(Q_{0,\\mathrm{ALM}}^{(t)}\\)。其作用通过扩散状态更新和历史特征反馈实现。对于下一执行时间步 \\(s<t\\)，确定性 DDIM 使用修正后的干净预测重新计算噪声分量并更新状态：', '#### 3.5.3 安全状态更新与历史反馈\n\n当前修正结果记为 \\(Q_{0,\\mathrm{ALM}}^{(t)}\\)。它通过两条路径作用于后续生成：状态更新直接改变下一步带噪控制点，历史反馈则为网络下一次预测提供修正状态与方向。对于下一执行时间步 \\(s<t\\)，采用确定性 DDIM 更新形式[REF:ddim]，并以修正后的干净预测重新计算噪声分量：', '明确双路径闭环机制并补充 DDIM 来源')
replace('该机制向网络传递此前的预测状态与纠偏方向，使后续去噪能够减少相同位置的重复违规。', '该机制向网络传递此前的修正状态与纠偏方向，为后续预测利用安全修正信息提供条件。其对最终违规的影响由第 4.3 节的反馈消融评价。', '将未经直接测量的重复违规结论改为机制描述')
replace('本文从整体安全性、关键组件贡献和推理效率三个方面评价 TrajSafe-Diffuser。', '本文围绕三个问题评价 TrajSafe-Diffuser：完整闭环能否改善最终轨迹的安全性；安全几何条件、历史反馈和安全状态更新分别发挥何种作用；减少采样与几何构造能否在当前测试中保持安全表现并降低耗时。', '建立贡献与实验问题的对应关系')
replace('实验采用 CARLA 栅格轨迹数据集。', '实验采用由 CARLA 道路场景构建的栅格轨迹数据集。CARLA 是城市驾驶仿真平台[REF:carla]，本文的栅格处理与数据划分按以下协议设置。', '区分 CARLA 平台来源与本文处理后的数据集')
replace('除特别说明外，均采用 16 步 DDIM。', '除特别说明外，均采用 16 步 DDIM；TrajSafe-Diffuser 完整方法的最终指标对应包括第 3.5.2 节末端可行性处理在内的推理输出。', '使最终指标与完整推理流程衔接')
replace('完整方法与其比较反映端到端系统差异。', '完整方法与其比较反映端到端系统差异。这些 SDF 对比项是统一协议下的安全处理器，不等同于第 2.2 节所述完整安全扩散方法的复现。', '就近明确实验处理器与文献方法的关系')
replace('该表只能证明完整系统在当前协议下的端到端优势', '该表支持完整系统在当前协议下的端到端优势', '避免将描述性实验写为证明')
replace('表现为局部曲率变化略有增加。', '表现为所定义的二阶变化指标有所增加。', '避免将二阶差分指标直接等同于几何曲率')
replace('由环境结构生成的椭圆与凸区域特征不仅用于后续优化，也能作为条件信息改善扩散模型的原始轨迹预测。', '局部安全隐特征作为条件信息有助于改善原始轨迹预测；椭圆参数和凸区域则承担显式约束构造的作用。该结果支持安全条件的整体效果，不能区分各几何子模块的独立贡献。', '使消融解释与实际安全隐特征输入保持一致')
replace('该结果说明，历史安全反馈能够向后续去噪步骤传递明确的安全状态和修正方向，使网络主动调整后续预测，而其作用并非来自把 ALM 输出直接作为最终轨迹。', '该结果支持网络利用历史修正状态与方向改善后续预测，其效果与直接采用 ALM 输出更新轨迹相区别。由于递进版本的训练预算并未全部统一，不能将两行之间的性能差异完全归因于反馈模块本身。', '保留反馈证据并就近说明训练预算混杂')
replace('安全轨迹状态更新则负责消除反馈生成仍然残留的随机违规。', '将修正结果用于安全状态更新后，完整流程在该组测试中消除了反馈生成仍然残留的违规。该观察对应最终推理流程，不能进一步分离末端投影的贡献。', '限定零违规归因范围，包含末端处理')
replace('**0.0283**', '0.0283', '纠正消融表平滑度加粗未覆盖基础模型的问题')
replace('**10.48**', '10.48', '纠正消融表 RMSE 最优标记错误')
replace('| **0.0297** |', '| 0.0297 |', '移除主表非全局最优的平滑度加粗')
replace('| **17.46** | **87.06** | **2.233** | **20.919** | **-1.145** |', '| 17.46 | 87.06 | 2.233 | 20.919 | -1.145 |', '统一主表加粗为全表最优，前五种处理器比较由正文说明')
replace('| 15.70 | 0.0274 |', '| 15.70 | **0.0274** |', '标记主表平滑度实际最优值')
replace('| 7.6758 | 16.05 | 0.0275 |', '| 7.6758 | 16.05 | **0.0275** |', '标记消融表平滑度实际最优值')
replace('| 1.3656 | 10.13 | 0.0330 |', '| 1.3656 | **10.13** | 0.0330 |', '标记消融表 RMSE 实际最优值')

conclusion = r'''## 5 结论与展望

本文提出 TrajSafe-Diffuser，将任务相关安全几何生成、安全条件去噪和跨步反馈整合到扩散运动规划中。候选通路提供局部几何锚点，预测的椭圆尺度与方向用于构造有序凸走廊；局部安全隐特征参与控制点预测，分段 Bézier 凸包约束将连续曲线的走廊归属转化为 B 样条控制点的线性不等式。ALM 修正通过 DDIM 状态更新与门控历史条件两条路径影响后续生成，共享参数的两次前向训练则使网络学习利用历史修正信息。

在当前 CARLA 静态栅格测试协议下，完整方法在 1260 条生成轨迹中未发生碰撞，SDF 可行率达到 100%；递进消融支持安全条件、历史反馈与安全状态更新的互补作用。4 步采样与区域共享的组合配置在单种子实验中保持 0/420 碰撞，将批量折算的平均单轨迹耗时由 84.9 ms 降至 26.8 ms，同时平滑度指标有所增加。这些结果支持所提出闭环的安全性与效率表现，但不构成未知环境下的零风险保证。

当前研究限于二维静态栅格和既定障碍扩张设置，未验证车辆动力学可执行性及动态障碍处理；主对比的生成先验和部分消融的训练预算也未完全统一。后续将采用统一训练设置开展等先验比较，进一步分离几何、反馈与末端投影的贡献，并引入车辆动力学、动态障碍及感知和控制误差。效率验证将结合走廊构建批量化、采样与优化预算分配，扩展到单请求延迟和更广泛的场景测试。
'''
text = text[:text.index('## 5 结论与展望')] + conclusion
changes.append('结论对应方法和实验，纳入加速时平滑度变化及有限评价边界')

# Entries and supporting locators are supplied in a separate, reviewable data file.
refs = json.loads((ROOT / 'paper_reference_sources.json').read_text(encoding='utf-8'))
by_key = {r['key']: r for r in refs}
order = []
def cite(m):
    keys = m.group(1).split(',')
    for key in keys:
        if key not in by_key:
            raise ValueError(f'Unknown citation key {key}')
        if key not in order:
            order.append(key)
    return ''.join(f'[{order.index(key)+1}]' for key in keys)
text = re.sub(r'\[REF:([a-z,]+)\]', cite, text)
if set(order) != set(by_key):
    raise ValueError(f'Uncited references {set(by_key)-set(order)}')
text += '\n## 参考文献\n\n'
for number, key in enumerate(order, 1):
    ref = by_key[key]
    ref['number'] = number
    ref['evidence_id'] = f'E{number:03d}'
    ref['checked_by'] = 'Codex'
    ref['checked_on'] = '2026-10-07'
    ref['verification_state'] = 'agent_checked_author_review_pending'
    text += f"[{number}] {ref['reference']}\n\n"
OUTPUT.write_text(text, encoding='utf-8')
(ROOT / 'paper_reference_sources.json').write_text(json.dumps([by_key[k] for k in order], ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
(ROOT / 'TrajSafe_polish_changes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True), text.splitlines(True), fromfile=SOURCE.name, tofile=OUTPUT.name)), encoding='utf-8')

# Verify preservation of all display equations and experimental data rows.
eqs = lambda s: re.findall(r'\\\[.*?\\\]', s, re.S)
normalize_row = lambda s: s.replace('**', '').strip()
rows = lambda s: [normalize_row(x) for x in s.splitlines() if x.startswith('|')]
assert eqs(original) == eqs(text), 'Display equation changed'
assert rows(original) == rows(text), 'Table data changed'
body, bibliography = text.split('## 参考文献\n', 1)
used = [int(x) for x in re.findall(r'\[(\d+)\]', body)]
defined = [int(x) for x in re.findall(r'^\[(\d+)\]', bibliography, re.M)]
assert set(used) == set(defined) == set(range(1, len(order)+1))
assert not re.search(r'\[REF:|CITATION NEEDED|TODO', text)
assert body.count(r'\[') == body.count(r'\]')
assert body.count(r'\(') == body.count(r'\)')
links = re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', body)
missing = [x for x in links if not x.startswith(('http:', 'https:')) and not (ROOT / x).exists()]
headings = re.findall(r'^#{1,4} .+$', text, re.M)
audit = {
    'date': '2026-10-07', 'input': SOURCE.name, 'output': OUTPUT.name,
    'original_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'reference_count': len(order), 'citation_occurrences': len(used),
    'all_references_cited': True, 'display_equations_preserved': len(eqs(text)),
    'table_rows_preserved': len(rows(text)), 'math_delimiters_balanced': True,
    'missing_local_figure_links': missing, 'headings': headings,
    'changes': changes,
    'architecture_confirmation': '2026-10-07: 用户确认已重新验证，现有结果对应正文新架构。',
    'experiment_verification': '仅核对稿件内数值和原稿一致性；当前目录未提供原始实验记录。',
    'human_reference_verification': '待作者复核；来源核验由 Codex 完成，不冒充人工签核。',
    'submission_ready': False,
}
(ROOT / 'TrajSafe_polish_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

# Keep a readable map from each citation occurrence to its evidence and locator.
review = '# 参考文献核验与引用位置\n\n'
review += '核验日期：2026-10-07。以下记录由 Codex 核对公开来源，供作者逐项复核；不代表人工签核。文献事实优先依据正式会议/期刊记录及作者公开版本，引用仅用于其支持的背景或方法，不用于支持本文实验数值。\n\n'
review += '| 编号 | 对应工作 | 支持内容 | 正文位置 | 可复核来源 |\n|---|---|---|---|---|\n'
for key in order:
    r = by_key[key]
    current = ''
    locations = []
    for line in body.splitlines():
        if line.startswith('#'):
            current = line.lstrip('#').strip()
        if f"[{r['number']}]" in line and current not in locations:
            locations.append(current)
    label = r['reference'].split('. ', 1)[1].split('[C]')[0].split('[J]')[0].split('[M]')[0]
    source_links = '；'.join(f'[来源 {i+1}]({url})' for i, url in enumerate(r['sources']))
    review += f"| [{r['number']}] | {label} | {r['supports']} | {'；'.join(locations)} | {source_links} |\n"
review += '\n## 核验范围与访问限制\n\n'
for key in order:
    r = by_key[key]
    if 'access_note' in r:
        review += f"- [{r['number']}] {r['access_note']}\n"
review += '\n书籍全文访问与最终引用格式仍需作者按目标期刊或学校要求复核。所有 22 篇文献均在正文引用；未通过加入写作工具宣传文献或无关综述增加数量。\n'
(ROOT / 'TrajSafe_reference_review.md').write_text(review, encoding='utf-8')
print(json.dumps({k:audit[k] for k in ['reference_count','citation_occurrences','display_equations_preserved','table_rows_preserved','missing_local_figure_links']}, ensure_ascii=False, indent=2))
