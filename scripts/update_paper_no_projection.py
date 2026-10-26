from pathlib import Path
import re, shutil
root=Path('论文/TrajSafe_methods_and_figures')
p=root/'TrajSafe_paper_optimized.md'
history=root/'历史版本';history.mkdir(exist_ok=True)
backup=history/'TrajSafe_paper_optimized_20261009_before_protocol_change.md'
if not backup.exists():shutil.copy2(p,backup)
s=p.read_text(encoding='utf-8-sig'); lines=s.splitlines()
def para(prefix,new):
 global lines
 matches=[i for i,line in enumerate(lines) if line.startswith(prefix)]
 assert len(matches)==1,(prefix,len(matches))
 lines[matches[0]]=new
para('本文取 \\(\\rho=5', '本文取 \\(\\rho=5\\)、\\(\\eta=0.05\\)，每个引导步骤最多执行10次内迭代，并在冻结约束下继承跨步乘子。有限迭代可能保留约束残差，因此修正结果仍需检查。逆扩散结束后，将最后一步控制点结果直接解码为轨迹，并分别评价占据栅格碰撞与连续走廊约束残差。')
para('本文围绕三个问题评价', '本文围绕三个问题评价TrajSafe-Diffuser：完整闭环能否改善最终轨迹的安全性；安全几何条件、历史反馈和安全状态更新分别发挥何种作用；减少采样与几何构造能否降低耗时。测试集包含420个样本；主对比对每个样本使用3个生成随机种子，共生成1260条轨迹。除特别说明外，均采用16步DDIM；完整方法使用逐步ALM修正、安全状态更新与历史反馈，指标对应最后一步输出解码后的轨迹。')
lines=[line for line in lines if not line.startswith('| 局部线性化投影 |')]
para('前五种方法共享', '前三种SDF引导方法与无约束扩散共用基础扩散生成器及相同初始噪声；三种引导方法采用同一SDF实现和10次内迭代预算。TrajSafe-Diffuser使用闭环安全反馈生成器。因此，四个对照项构成同生成先验下的处理器比较，与完整方法的比较反映端到端系统差异。这些SDF对比项是统一协议下的安全处理器，不等同于第2.2节所述完整安全扩散方法的复现。')
para('无约束扩散在1260', '') if any(x.startswith('无约束扩散在1260') for x in lines) else None
para('无约束扩散在 1260', '无约束扩散用于评价生成先验本身的安全性；固定惩罚、原始—对偶和SDF增广拉格朗日引导用于比较不同距离场处理方式。各项结果将在新版基础生成器及统一测试协议下重新测量，指标由逐轨迹记录汇总。')
para('局部线性化投影将', '固定惩罚使用预设代价权重，原始—对偶方法引入约束乘子，SDF增广拉格朗日法进一步加入二次违规罚项。比较重点包括整体碰撞、尾部违规及轨迹质量，具体优劣待复测后分析。')
para('TrajSafe-Diffuser 在 1260', '本文方法先根据任务通路构建凸安全走廊，再在控制点空间处理线性不等式约束。其结果需同时报告占据栅格碰撞、连续约束残差与轨迹质量。由于完整方法与对照项采用不同生成先验，主表评价端到端系统表现，不能将差异全部归因于某一安全模块。')
para('在统一协议下，无约束扩散', '效率实验在相同硬件、批大小和计时边界下比较各方法。本文方法进一步采用采样步数与区域共享步长的二因素设置，分别记录网络前向、几何构造和走廊ALM的耗时。全部时间及轨迹指标待新版配置复测后填写。')
para('仅将采样步数由 16', '采样步数由16降至4用于减少去噪前向次数；区域共享步长由1增至8用于减少基础几何构造数量。单项与组合设置均需同时评价耗时、碰撞、连续可行性及轨迹质量，加速比依据同一轮计时的实测结果计算。')
para('四种 TrajSafe-Diffuser 配置均', '四种配置固定3步预热，4步采样执行1次走廊ALM，16步采样执行13次。不同配置的安全性和质量须分别测量，不能依据标准配置推断加速配置的结果。')
para('加入生成式安全几何后，碰撞率', '第一组递进比较基础扩散与加入安全几何条件的生成器，用于评价安全条件对原始预测的作用。两者应采用相同训练数据、名义轮数预算及验证选择规则，并基于新的测试记录报告差异。')
para('在此基础上加入历史安全反馈后', '第二组在安全条件基础上加入历史反馈。ALM产生修正结果与修正量，二者只用于下一步网络输入；DDIM仍使用原始干净预测。这一行用于评价网络利用历史纠偏信息的效果，具体结果待新配置复测。')
para('完整方法进一步允许 ALM 安全轨迹', '第三组使用同一历史反馈权重和相同ALM参数，仅改变修正结果是否进入DDIM状态更新。该比较区分历史条件作用与安全状态更新作用。完整方法仍需对最后一步结果进行残差与碰撞检查；有限优化预算不自动保证输出完全可行。')
para('上述结果支持三个互补作用', '实验分别评价任务相关几何、历史反馈与安全状态更新，并考察采样步数和区域共享对效率的影响。新版网络及几何构造的正式结果尚待完成；单种子效率测试不能估计跨种子波动，所选场景的可视化也不替代总体定量评价。')
para('在当前 CARLA 静态栅格测试协议下', '本文在CARLA静态栅格数据集上设计整体比较、递进消融、效率和案例实验，评价所提出闭环方法的安全性与轨迹质量。当前方法配置的定量结论将在正式复测后更新；有限样本的零碰撞观察不构成未知环境下的零风险保证。')
for i,line in enumerate(lines):
 if line.startswith('扩散模型能够学习多模态轨迹分布，但分布学习本身'):
  lines[i]=line.split('在由 CARLA')[0]+'本文采用CARLA道路场景构建的二维静态栅格数据，设计整体性能、递进消融、计算效率与轨迹可视化实验，评价上述机制的作用及适用边界。'
 if line.startswith('当前研究限于二维静态栅格'):
  lines[i]=line.replace('进一步分离几何、反馈与末端投影的贡献','进一步分析几何与反馈的贡献')
 if line.startswith('第 1 步仍处于预热阶段，预测轨迹'):
  lines[i]='该案例按当前模型和采样配置重新生成后，展示预热阶段、首次安全激活及后续步骤的原始预测与ALM修正结果；具体变化待新版图像生成后分析。'
 if line.startswith('在这 72 个补充样本中'):
  lines[i]='保留上述72个任务的地图、起终点和距离分组，对当前模型重新生成通路、凸走廊及最终轨迹，并分别检查占据栅格碰撞、完整区域安全和连续走廊约束。展示结论限定于所选4个场景及对应生成种子。'
 if line.startswith('![图4-2'):
  lines[i]='［图4-2待按当前模型与采样配置重新生成；原图保存在历史版本中。］'
 if line.startswith('![图4-3'):
  lines[i]='［图4-3待按当前模型与采样配置重新生成；保留原72个任务及24个展示样本。］'
 if '你可以阅读一下相关的论文和' in line:
  lines[i]=lines[i].replace('你可以阅读一下相关的论文和','')
# Remove stale quantitative cells; preserve each table's protocol and components.
mode=None
for i,line in enumerate(lines):
 if line.startswith('| 方法 | 生成模型配置 |'): mode=('main',2)
 elif line.startswith('| 方法或配置 | DDIM'):mode=('timing',4)
 elif line.startswith('| 配置 | 生成式安全几何 |'):mode=('ablation',4)
 elif mode and not line.startswith('|'):mode=None
 elif mode and line.startswith('|') and not line.startswith('|---'):
  fields=[x.strip() for x in line.strip().strip('|').split('|')]
  # Header was handled by the preceding branches.
  keep=mode[1]; fields[keep:]=['待复测']*(len(fields)-keep)
  lines[i]='| '+' | '.join(fields)+' |'
# Mark manuscript status explicitly without presenting historical results as new.
lines.insert(2,'> 实验更新中：本文已采用修改后的网络与采样配置，第4节数值及案例需按当前协议重新生成。旧结果保存于历史版本，不作为当前配置的实验结论。')
s='\n'.join(lines)+'\n'
assert '末端' not in s and '局部线性化投影' not in s
p.write_text(s,encoding='utf-8')
print('Updated manuscript; historical version preserved at',backup)
