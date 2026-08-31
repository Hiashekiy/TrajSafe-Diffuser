# 基于场景—轨迹交互与椭圆几何监督的安全扩散规划框架

本文方法面向 Maze2D 轨迹扩散规划。二维占据地图和带噪轨迹分别在编码阶段一次性注入空间位置信息：Scene Feature 展开为 Scene Tokens 后加入地图二维位置编码，Trajectory Tokens 在初始化时加入 waypoint 二维位置、轨迹序号和 diffusion timestep 信息。后续 Self-Attention、Point-Scene Cross-Attention 与 Trajectory Decoder 均直接使用已经携带位置信息的隐藏特征，不再重复注入绝对位置或相对位置编码。轨迹上下文特征首先从 Scene Memory 中读取安全相关场景信息并形成共享安全特征；共享安全特征一方面预测 waypoint-level 安全椭圆，另一方面作为 Trajectory Decoder 的轨迹 Query，继续访问同一位置感知 Scene Memory，最终预测 clean trajectory。椭圆标签由占据地图和数据集中的 clean trajectory 离线预生成，并通过参数、区域重叠、障碍碰撞和 waypoint 包含等损失监督共享安全特征。最终预测椭圆与地图进一步用于解析构造凸安全区域。

> **实现说明（与当前代码对齐）**
>
> 本文是原始方法/框架的设计文档。**当前实现**与原文有以下明确差异，各小节均加了“注”：
>
> - **椭圆标签**：由**离线 IRIS MVIE 求解器**（`src/geometry/iris_solver.py`）从占据地图 + clean 轨迹离线生成，**不使用** Neural-IRIS 网络。
> - **数据**：默认把 umaze / medium / large 三张迷宫数据**混合**为一份 train/val/test（`data/processed/mixed`），不再区分单场景/联合。
> - **条件化**：起点/终点条件通过**端点 inpainting**（`src/diffusion/conditioning.py::apply_endpoint_condition`）在训练与采样时重新施加，而非广播加条件。
> - **损失**：实现里**移除了** `L_{traj_ellipse}`（原 §11）；`L_{diff}` 额外含**时间一致性项**（`lambda_var` / `lambda_var_vel`）；`L_{collision}` 用 SDF 采样；训练含 **warmup**（前若干轮只训 `L_{diff}+L_{smooth}`）。
> - **凸安全区域（原 §14）**：当前实现**未包含**下游解析凸区域构造与轨迹修正（`convex_region.py` 已移出），保留为设计可选项。

---

# 1. 问题定义

给定二维迷宫占据地图

$$
M\in\{0,1\}^{H_m\times W_m},
$$

以及长度为 $H$ 的 clean trajectory

$$
x_0=\{x_{0,1},\ldots,x_{0,H}\},
$$

其中 Maze2D 中第 $k$ 个轨迹状态写为

$$
x_{0,k}
=
[a_{x,k},a_{y,k},x_k,y_k,v_{x,k},v_{y,k}]^T
\in\mathbb R^6.
$$

训练时随机采样扩散时间步 $t$ 和高斯噪声

$$
\epsilon\sim\mathcal N(0,I),
$$

按照前向扩散过程构造

$$
\boxed{
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon.
}
$$

模型以

$$
(x_t,t,M)
$$

作为输入，同时预测 clean trajectory 和与 clean trajectory 各 waypoint 对应的安全椭圆：

$$
\boxed{
(\hat x_0,\hat E_t)
=
D_\Theta(x_t,t,M).
}
$$

其中

$$
\hat E_t
=
\{\hat E_{t,1},\ldots,\hat E_{t,H}\}.
$$

方法需要学习两类相互关联的能力：

1. 从带噪轨迹恢复 clean trajectory；
2. 从当前轨迹状态和地图结构恢复每个 clean waypoint 对应的局部安全几何。

因此核心训练关系为

$$
\boxed{
(x_t,t,M)
\rightarrow
\begin{cases}
\hat x_0,\\
\hat E_t.
\end{cases}
}
$$

监督目标分别为

$$
\hat x_0\rightarrow x_0,
\qquad
\hat E_{t,k}\rightarrow E_k^*.
$$

---

# 2. 总体框架

整个网络由六个核心部分构成：

$$
\boxed{
\text{Scene Encoder}
+
\text{Trajectory Token Encoder}
+
\text{Point-Scene Attention}
+
\text{Safety Feature Fusion}
+
\text{Trajectory Decoder}
+
\text{Ellipse Head}
}
$$

空间位置信息只在两类 Token 的初始化阶段注入一次。

地图经过 Scene Encoder 得到二维特征图：

$$
M
\xrightarrow{E_M}
B,
$$

再将其展开为 Scene Tokens，并加入地图二维位置编码：

$$
B
\rightarrow
B^{tok}
\xrightarrow{+\ PE_{map}}
\bar B.
$$

当前带噪轨迹经过 Trajectory Token Encoder，在初始化时同时加入 waypoint 二维位置、轨迹序号和 diffusion timestep：

$$
(x_t,t)
\xrightarrow{E_T}
H_t^0
\xrightarrow{\text{Trajectory Self-Attention}}
F_t^{traj}.
$$

随后，轨迹上下文特征直接以自身作为 Query，从位置感知 Scene Memory $\bar B$ 中读取场景信息：

$$
\boxed{
(F_t^{traj},\bar B)
\xrightarrow{\text{Point-Scene Cross-Attention}}
A_t.
}
$$

将轨迹上下文和场景交互特征拼接并映射得到共享安全特征：

$$
\boxed{
(F_t^{traj},A_t)
\xrightarrow{\text{Concat + MLP}}
S_t.
}
$$

共享安全特征分为两个分支。

椭圆分支：

$$
\boxed{
S_t
\xrightarrow{\text{Ellipse Head}}
\hat E_t.
}
$$

轨迹分支以 $S_t$ 作为 Trajectory Decoder 的初始轨迹 Query，并继续访问位置感知 Scene Memory $\bar B$：

$$
\boxed{
(S_t,\bar B)
\xrightarrow{\text{Trajectory Decoder}}
T_t^L
\xrightarrow{\text{Trajectory Head}}
\hat x_0.
}
$$

其中：

- $B$：Scene Encoder 输出的二维地图内容特征；
- $B^{tok}$：由 $B$ 展开得到的地图内容 Token；
- $\bar B$：在编码阶段加入地图二维位置编码后的 Scene Memory；
- $H_t^0$：在编码阶段一次性加入运动状态、waypoint 二维位置、轨迹序号和 diffusion timestep 的初始轨迹 Token；
- $F_t^{traj}$：经过轨迹 Self-Attention 后的轨迹上下文特征；
- $A_t$：Point-Scene Cross-Attention 得到的场景交互特征；
- $S_t$：融合轨迹上下文与场景交互信息后的共享安全特征；
- $T_t^L$：Trajectory Decoder 输出的最终轨迹生成特征；
- $\hat x_0$：预测 clean trajectory；
- $\hat E_t$：预测 waypoint-level 安全椭圆。

位置信息在初始化后随隐藏特征向后传播，后续 Attention 和 Decoder 不再额外叠加 $PE_{xy}$、$PE_{rel}$ 或其它重复空间位置编码。

整个网络始终保持

$$
\boxed{
\text{第 }k\text{ 个 Token}
\leftrightarrow
\text{第 }k\text{ 个 waypoint}
}
$$

的一一对应关系。

---

# 3. 场景编码

## 3.1 输入地图

Maze2D 地图采用二维占据栅格：

$$
M\in\mathbb R^{1\times H_m\times W_m}.
$$

## 3.2 Scene Encoder

使用二维 CNN 或轻量二维 U-Net Encoder 提取地图内容特征：

$$
\boxed{
B=E_M(M),
}
$$

其中

$$
B\in\mathbb R^{H'\times W'\times C}.
$$

$B$ 中每个空间位置对应地图中的一个固定区域，其内容特征用于表达：

- 墙体与自由空间分布；
- 通道宽度；
- 局部走廊方向；
- 拐角结构；
- 自由空间形状；
- 邻近区域的连通关系。

## 3.3 Scene Token 与一次性地图位置编码

将二维 Scene Feature 展开为

$$
B^{tok}
=
[b_1,\ldots,b_N],
\qquad
N=H'W',
$$

其中

$$
b_j\in\mathbb R^C
$$

表示第 $j$ 个地图位置的内容特征。

设该 Token 对应的二维地图坐标为

$$
m_j=[x_j,y_j]^T.
$$

仅在 Scene Token 初始化阶段加入一次地图二维位置编码：

$$
\boxed{
\bar b_j
=
b_j
+
\phi_m
\left(
PE_{xy}(m_j)
\right).
}
$$

得到位置感知 Scene Memory：

$$
\boxed{
\bar B
=
[\bar b_1,\ldots,\bar b_N]
\in\mathbb R^{N\times C}.
}
$$

$\bar B$ 在后续 Point-Scene Cross-Attention 和 Trajectory Decoder 中重复使用。后续网络层只对 $\bar B$ 做线性投影和 Attention，不再重新计算或叠加地图位置编码。

---

# 4. 轨迹 Token 编码

当前带噪轨迹表示为

$$
x_t\in\mathbb R^{H\times6}.
$$

第 $k$ 个状态为

$$
x_{t,k}
=
[a_{x,k},a_{y,k},x_k,y_k,v_{x,k},v_{y,k}]^T.
$$

为了避免二维位置同时作为普通状态和位置编码重复进入隐藏特征，将其拆分为运动状态和二维空间位置：

$$
\boxed{
d_{t,k}
=
[a_{x,k},a_{y,k},v_{x,k},v_{y,k}]^T,
}
$$

$$
\boxed{
p_{t,k}
=
[x_{t,k},y_{t,k}]^T.
}
$$

初始轨迹 Token 定义为：

$$
\boxed{
h_{t,k}^{0}
=
\phi_d(d_{t,k})
+
\phi_p
\left(
PE_{xy}(p_{t,k})
\right)
+
PE_{traj}(k)
+
E_{diff}(t).
}
$$

其中：

- $\phi_d(d_{t,k})$：运动状态特征；
- $\phi_p(PE_{xy}(p_{t,k}))$：当前 waypoint 的二维空间位置特征；
- $PE_{traj}(k)$：轨迹序号编码；
- $E_{diff}(t)$：扩散时间步嵌入。

二维空间位置仅在该初始化步骤中进入轨迹 Token。后续 Self-Attention、Point-Scene Cross-Attention、共享特征融合和 Trajectory Decoder 都直接使用已经携带空间位置含义的隐藏特征。

整条轨迹表示为

$$
H_t^0
=
[h_{t,1}^{0},\ldots,h_{t,H}^{0}]
\in\mathbb R^{H\times C}.
$$

## 4.1 轨迹内部上下文建模

首先进行 Layer Normalization：

$$
\tilde H_t^0
=
\operatorname{LN}(H_t^0).
$$

随后进行多头自注意力：

$$
Z_t
=
\operatorname{MHSA}(\tilde H_t^0).
$$

对于单个注意力头：

$$
Q=\tilde H_t^0W_Q,
\qquad
K=\tilde H_t^0W_K,
\qquad
V=\tilde H_t^0W_V,
$$

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt d}
\right)V.
$$

对于第 $k$ 个 waypoint：

$$
z_{t,k}
=
\sum_{j=1}^{H}
\alpha_{k,j}v_{t,j}.
$$

通过残差连接得到轨迹上下文特征：

$$
\boxed{
F_t^{traj}
=
H_t^0+Z_t.
}
$$

记

$$
F_t^{traj}
=
[f_{t,1},\ldots,f_{t,H}],
$$

则

$$
\boxed{
f_{t,k}
=
h_{t,k}^{0}
+
z_{t,k}.
}
$$

因此 $f_{t,k}$ 同时包含：

- 当前 waypoint 的运动状态；
- 当前 waypoint 的二维空间位置；
- 轨迹序号；
- 当前扩散阶段；
- 前后 waypoint 的局部运动趋势；
- 整条当前轨迹的上下文信息。

## 4.2 固定 waypoint-level Token 分辨率

整个轨迹网络始终保持

$$
F_t^{traj}\in\mathbb R^{H\times C},
$$

不对轨迹长度维进行下采样。

因此：

$$
\boxed{
p_{t,k}
\leftrightarrow
f_{t,k}
}
$$

始终一一对应。这里 $p_{t,k}$ 只用于说明物理 waypoint 与 Token 的对应关系；其二维位置已经在 $h_{t,k}^0$ 中完成编码，后续网络不会再次将 $p_{t,k}$ 作为额外位置输入。

---

# 5. Point-Scene Cross-Attention

Point-Scene Cross-Attention 用于建立每个轨迹 waypoint 与其周围局部地图之间的直接交互。其核心过程为：

$$
\boxed{
\text{Trajectory Query}
\rightarrow
\text{Local Scene Sampling}
\rightarrow
\text{Cross-Attention}
\rightarrow
\text{Local Scene Interaction Feature}
}
$$

对于第 $k$ 个 waypoint，轨迹 Token 中已经包含当前轨迹点的运动状态、二维位置、轨迹序号以及 diffusion timestep 信息。因此，当前位置 $p_{t,k}$ 在本模块中主要用于确定应该从 Scene Feature 的哪个空间区域采样地图特征，而不再作为新的位置编码重复注入轨迹特征。

---

## 5.1 Query 构造

经过轨迹 Self-Attention 后，第 $k$ 个 waypoint 对应的轨迹上下文特征为

$$
f_{t,k}\in\mathbb R^C.
$$

该特征已经继承初始 Trajectory Token 中的：

- 当前 waypoint 运动状态；
- 当前 waypoint 二维空间位置；
- waypoint 在整条轨迹中的序号；
- diffusion timestep；
- 前后 waypoint 和整条轨迹提供的上下文信息。

因此直接通过线性投影构造 Query：

$$
\boxed{
q_k
=
W_Qf_{t,k}.
}
$$

其中

$$
q_k\in\mathbb R^C.
$$

这里不再额外加入

$$
PE_{xy}(p_{t,k}),
$$

因为二维位置已经在 Trajectory Token 初始化阶段进入 $f_{t,k}$。

---

## 5.2 基于 waypoint 位置的局部地图特征采样

第 $k$ 个 waypoint 当前在二维空间中的位置为

$$
\boxed{
p_{t,k}
=
[x_{t,k},y_{t,k}]^T.
}
$$

Scene Encoder 输出二维地图特征，并在 Scene Token 初始化阶段加入一次地图位置编码，得到位置感知 Scene Memory：

$$
\bar B.
$$

虽然 $p_{t,k}$ 的位置特征已经包含在轨迹 Token 中，但其原始二维坐标仍用于确定当前 waypoint 应该从地图的哪个位置读取局部场景信息。

首先根据 Maze2D 世界坐标与 Scene Feature 网格之间的对应关系，将

$$
p_{t,k}
$$

映射到 Scene Feature 中的对应位置。

以该位置为中心定义局部采样区域：

$$
\boxed{
\mathcal N(p_{t,k}).
}
$$

从位置感知 Scene Memory 中采样得到当前 waypoint 对应的局部 Scene Tokens：

$$
\boxed{
\bar B_k^{loc}
=
\{
\bar b_{k,1},
\bar b_{k,2},
\ldots,
\bar b_{k,N_l}
\}.
}
$$

其中

$$
\bar b_{k,j}\in\mathbb R^C,
$$

$N_l$ 表示局部区域中的 Scene Token 数量。

例如，当 Scene Feature 上使用 $5\times5$ 的局部窗口时：

$$
N_l=25.
$$

因此该过程可以写成：

$$
\boxed{
p_{t,k}
\xrightarrow{\text{Spatial Sampling}}
\bar B_k^{loc}.
}
$$

需要区分的是，这里的 $p_{t,k}$ 仅作为**空间采样坐标**使用，并没有再次通过位置编码加入轨迹隐藏特征。

也就是说：

$$
\boxed{
\text{使用 }p_{t,k}\text{ 定位地图区域}
\neq
\text{重复注入 }PE_{xy}(p_{t,k}).
}
$$

局部 Scene Token 在 Scene Encoder 后已经加入地图位置编码，因此本阶段不再额外构造新的绝对位置编码或 query-dependent 相对位置编码。

---

## 5.3 Key 与 Value 构造

对于第 $k$ 个 waypoint，其局部 Scene Tokens 为

$$
\bar B_k^{loc}
=
\{
\bar b_{k,1},
\ldots,
\bar b_{k,N_l}
\}.
$$

分别通过线性映射得到 Key 和 Value：

$$
\boxed{
K_k
=
W_K\bar B_k^{loc},
}
$$

$$
\boxed{
V_k
=
W_V\bar B_k^{loc}.
}
$$

其中

$$
K_k,V_k
\in
\mathbb R^{N_l\times C}.
$$

Key 用于计算当前轨迹 Query 与局部不同地图位置之间的相关性，而 Value 保存最终需要被聚合的局部场景信息。

由于 $\bar B_k^{loc}$ 已经来自位置感知 Scene Memory，因此这里不再额外加入

$$
PE_{rel}(m_j-p_{t,k})
$$

等新的位置编码。

---

## 5.4 Point-Scene Cross-Attention

第 $k$ 个轨迹 Query

$$
q_k
$$

与当前位置附近的局部 Scene Tokens 执行多头交叉注意力：

$$
\boxed{
a_{t,k}
=
\operatorname{MHCA}
(
q_k,
K_k,
V_k
).
}
$$

对于单个注意力头，首先计算第 $k$ 个轨迹 Query 对第 $j$ 个局部 Scene Token 的注意力权重：

$$
\boxed{
\alpha_{k,j}
=
\frac{
\exp
\left(
q_k^Tk_{k,j}/\sqrt d
\right)
}{
\sum_{l=1}^{N_l}
\exp
\left(
q_k^Tk_{k,l}/\sqrt d
\right)
}.
}
$$

随后对局部 Value 进行加权聚合：

$$
\boxed{
a_{t,k}
=
\sum_{j=1}^{N_l}
\alpha_{k,j}v_{k,j}.
}
$$

其中

$$
a_{t,k}\in\mathbb R^C.
$$

因此，$a_{t,k}$ 表示：

$$
\boxed{
\text{第 }k\text{ 个 waypoint 根据当前轨迹上下文，
从其当前位置附近地图中选择并聚合得到的局部场景信息}.
}
$$

该特征重点描述与当前 waypoint 直接相关的局部安全几何，包括：

- 邻近障碍的位置；
- 当前可通行空间；
- 局部通道宽度；
- 墙体和自由空间边界；
- 局部走廊延伸结构；
- 当前 waypoint 周围的安全空间分布。

整条轨迹对应的场景交互特征为

$$
\boxed{
A_t
=
[a_{t,1},\ldots,a_{t,H}]
\in\mathbb R^{H\times C}.
}
$$

---

## 5.5 Point-Scene Attention 与后续 Trajectory Decoder 的分工

Point-Scene Cross-Attention 主要关注每个 waypoint 周围的**局部安全几何**：

$$
\boxed{
p_{t,k}
\rightarrow
\bar B_k^{loc}
\rightarrow
a_{t,k}.
}
$$

随后，轨迹上下文特征与局部场景交互特征融合：

$$
[f_{t,k};a_{t,k}]
\rightarrow
s_{t,k},
$$

形成共享安全特征

$$
S_t.
$$

共享安全特征用于 Ellipse Head：

$$
S_t
\rightarrow
\text{Ellipse Head}
\rightarrow
\hat E_t.
$$

同时，最终轨迹生成还需要理解更大范围内的地图结构，因此 Trajectory Decoder 不再局限于当前 waypoint 的局部采样区域，而是继续访问完整的位置感知 Scene Memory：

$$
\boxed{
(S_t,\bar B)
\xrightarrow{\text{Trajectory Decoder}}
T_t^L
\xrightarrow{\text{Trajectory Head}}
\hat x_0.
}
$$

因此，两次地图交互具有不同职责：

$$
\boxed{
\text{Point-Scene Attention}
:
\text{局部安全几何提取}
}
$$

以及

$$
\boxed{
\text{Trajectory Decoder}
:
\text{全局地图条件下的最终轨迹生成}.
}
$$

这样既保留了 waypoint 与邻近地图区域之间明确的空间对应关系，又使最终轨迹预测能够利用完整地图中的全局通行结构。



# 6. 轨迹与场景交互特征融合

第 $k$ 个 waypoint 当前具有：

$$
f_{t,k}\in\mathbb R^C,
$$

表示轨迹上下文；

$$
a_{t,k}\in\mathbb R^C,
$$

表示当前轨迹 Query 从地图中筛选并聚合得到的场景交互信息。

采用直接拼接：

$$
\boxed{
h_{t,k}
=
[f_{t,k};a_{t,k}]
\in\mathbb R^{2C}.
}
$$

再通过 Layer Normalization 和轻量 MLP 映射回 $C$ 维：

$$
\boxed{
s_{t,k}
=
\operatorname{MLP}
\left(
\operatorname{LN}(h_{t,k})
\right)
\in\mathbb R^C.
}
$$

定义

$$
\boxed{
s_{t,k}
}
$$

为第 $k$ 个 waypoint 的**共享安全特征（Shared Safety Feature）**。

整条轨迹对应：

$$
\boxed{
S_t
=
[s_{t,1},\ldots,s_{t,H}]
\in\mathbb R^{H\times C}.
}
$$

共享安全特征保持严格的 waypoint 对应：

$$
\boxed{
s_{t,k}
\leftrightarrow
p_{t,k}.
}
$$

$S_t$ 已继承轨迹侧的一次性空间位置编码，并融合了来自位置感知 Scene Memory 的地图信息，因此后续不会再次向 $S_t$ 注入二维位置编码。

$S_t$ 同时承担两个作用：

$$
\boxed{
S_t
\xrightarrow{\text{Ellipse Head}}
\hat E_t,
}
$$

以及作为 Trajectory Decoder 的初始轨迹 Query：

$$
\boxed{
T_t^0=S_t.
}
$$

---

# 7. 面向最终轨迹生成的 Trajectory Decoder

共享安全特征

$$
S_t
=
[s_{t,1},\ldots,s_{t,H}]
\in\mathbb R^{H\times C}
$$

作为 Trajectory Decoder 的初始轨迹 Query：

$$
\boxed{
T_t^0=S_t.
}
$$

第 3.3 节得到的位置感知 Scene Memory

$$
\bar B\in\mathbb R^{N\times C}
$$

在整个 Trajectory Decoder 中固定复用。

Trajectory Decoder 不重新加入 waypoint 二维位置编码，也不重新计算地图相对位置编码。其每一层直接处理已经包含空间信息的 $T_t^l$ 与 $\bar B$。

## 7.1 Trajectory Self-Attention

设第 $l$ 个 Decoder Block 输入：

$$
T_t^l
=
[t_{t,1}^l,\ldots,t_{t,H}^l]
\in\mathbb R^{H\times C}.
$$

首先进行轨迹内部多头自注意力：

$$
\boxed{
\bar T_t^l
=
T_t^l
+
\operatorname{MHSA}
\left(
\operatorname{LN}(T_t^l)
\right).
}
$$

该操作使不同 waypoint 之间继续交换轨迹生成信息。

## 7.2 Scene Cross-Attention

直接由当前 Decoder Token 构造 Query：

$$
\boxed{
Q_t^l
=
W_Q^l\bar T_t^l.
}
$$

位置感知 Scene Memory 构造：

$$
\boxed{
K_B^l
=
W_K^l\bar B,
}
$$

$$
\boxed{
V_B^l
=
W_V^l\bar B.
}
$$

随后执行：

$$
\boxed{
C_t^l
=
\operatorname{MHCA}
\left(
Q_t^l,
K_B^l,
V_B^l
\right).
}
$$

其中

$$
C_t^l
\in
\mathbb R^{H\times C}.
$$

这里每层使用不同的 $W_Q^l,W_K^l,W_V^l$ 对同一位置感知 Scene Memory 进行重新投影，但不会再次注入空间位置编码。

## 7.3 地图信息注入与 FFN

通过残差连接融合 Scene Cross-Attention 输出：

$$
\boxed{
U_t^l
=
\bar T_t^l
+
W_O^lC_t^l.
}
$$

再通过 FFN：

$$
\boxed{
T_t^{l+1}
=
U_t^l
+
\operatorname{FFN}
\left(
\operatorname{LN}(U_t^l)
\right).
}
$$

一个完整 Decoder Block 为：

$$
\boxed{
\text{Trajectory Self-Attention}
\rightarrow
\text{Scene Cross-Attention}(\bar B)
\rightarrow
\text{FFN}.
}
$$

经过 $L$ 个 Decoder Blocks：

$$
\boxed{
T_t^L
=
[t_{t,1}^{L},\ldots,t_{t,H}^{L}]
\in\mathbb R^{H\times C}.
}
$$

整个过程中始终保持：

$$
\boxed{
t_{t,k}^{L}
\leftrightarrow
\text{第 }k\text{ 个 waypoint}.
}
$$

## 7.4 Trajectory Head

最终使用共享 Trajectory Head 将每个 $C$ 维轨迹生成特征映射回 Maze2D 六维 clean trajectory state：

$$
\boxed{
\hat x_{0,k}
=
\operatorname{TrajectoryHead}
(t_{t,k}^{L}).
}
$$

其中

$$
\hat x_{0,k}
=
[
\hat a_{x,k},
\hat a_{y,k},
\hat x_k,
\hat y_k,
\hat v_{x,k},
\hat v_{y,k}
]^T
\in\mathbb R^6.
$$

Trajectory Head 采用轻量 MLP：

$$
\operatorname{TrajectoryHead}(t)
=
W_2\,\sigma(W_1t+b_1)+b_2.
$$

整条预测 clean trajectory：

$$
\boxed{
\hat x_0
=
[\hat x_{0,1},\ldots,\hat x_{0,H}]
\in\mathbb R^{H\times6}.
}
$$

因此：

$$
\boxed{
(S_t,\bar B)
\xrightarrow{\text{Trajectory Decoder}}
T_t^L
\xrightarrow{\text{Trajectory Head}}
\hat x_0.
}
$$

---

# 8. 椭圆几何预测分支

## 8.1 Ellipse Head

第 $k$ 个共享安全特征输入轻量几何头：

$$
\boxed{
s_{t,k}
\xrightarrow{\text{Ellipse Head}}
\hat E_{t,k}.
}
$$

Ellipse Head 输出：

$$
\hat E_{t,k}
=
(
\hat c_{t,k},
\hat r_{1,t,k},
\hat r_{2,t,k},
\hat v_{\theta,t,k}
).
$$

其中：

$$
\hat c_{t,k}
=
[\hat c_{x,t,k},\hat c_{y,t,k}]^T
$$

为预测安全椭圆中心。中心坐标与轨迹位置使用相同的归一化坐标系。

## 8.2 半轴参数化

Ellipse Head 输出两个未约束标量

$$
\rho_{1,t,k},
\qquad
\rho_{2,t,k}.
$$

通过

$$
\bar r_1
=
\operatorname{softplus}(\rho_{1,t,k})+\epsilon,
$$

$$
\bar r_2
=
\operatorname{softplus}(\rho_{2,t,k})+\epsilon
$$

得到正数，并定义

$$
\boxed{
\hat r_{1,t,k}
=
\max(\bar r_1,\bar r_2),
\qquad
\hat r_{2,t,k}
=
\min(\bar r_1,\bar r_2).
}
$$

从而满足

$$
\hat r_{1,t,k}
\ge
\hat r_{2,t,k}>0.
$$

## 8.3 椭圆方向表示

椭圆方向满足

$$
\theta\equiv\theta+\pi.
$$

因此使用双角度向量表示：

$$
\boxed{
\hat v_{\theta,t,k}
=
\begin{bmatrix}
\hat c_{2\theta,t,k}\\
\hat s_{2\theta,t,k}
\end{bmatrix}.
}
$$

归一化：

$$
\hat v_{\theta,t,k}
\leftarrow
\frac{
\hat v_{\theta,t,k}
}{
\|\hat v_{\theta,t,k}\|_2+\epsilon
}.
$$

GT 方向表示为：

$$
v_{\theta,k}^*
=
\begin{bmatrix}
\cos2\theta_k^*\\
\sin2\theta_k^*
\end{bmatrix}.
$$

需要构造椭圆二次型时，可恢复：

$$
\boxed{
\hat\theta_{t,k}
=
\frac12
\operatorname{atan2}
(
\hat s_{2\theta,t,k},
\hat c_{2\theta,t,k}
).
}
$$

---

# 9. 椭圆标签离线构造与混合训练

## 9.1 离线标签生成

对于数据集中的 clean trajectory

$$
\tau_0
=
\{p_1^*,p_2^*,\ldots,p_H^*\},
$$

结合占据地图 $M$，对每个 clean waypoint 离线生成对应安全椭圆：

$$
\boxed{
(M,p_k^*)
\xrightarrow{\text{Geometry Label Generator}}
E_k^*.
}
$$

标签写为：

$$
\boxed{
E_k^*
=
(c_k^*,r_{1,k}^*,r_{2,k}^*,\theta_k^*).
}
$$

因此一条轨迹对应：

$$
E^*
=
\{E_1^*,E_2^*,\ldots,E_H^*\}.
$$

训练数据保存为：

$$
\boxed{
(M,x_0,E^*).
}
$$

若个别 waypoint 无法生成有效安全椭圆，则使用有效性掩码

$$
m_k^{ellipse}\in\{0,1\}
$$

屏蔽对应的椭圆监督项。

## 9.2 Diffusion Query

训练时从 clean trajectory $x_0$ 直接执行标准前向扩散：

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon.
$$

因此

$$
p_{t,k}
$$

就是第 $k$ 个 waypoint 在当前扩散时间步的 noisy Query。

其监督对应关系始终由 waypoint index 保持：

$$
\boxed{
p_{t,k}
\leftrightarrow
p_k^*
\leftrightarrow
E_k^*.
}
$$

## 9.3 联合恢复

模型从

$$
(x_t,t,M)
$$

得到共享安全特征

$$
S_t
=
[s_{t,1},\ldots,s_{t,H}].
$$

椭圆分支直接从 $S_t$ 预测：

$$
\boxed{
S_t
\xrightarrow{\text{Ellipse Head}}
\hat E_t.
}
$$

轨迹分支将 $S_t$ 作为 Trajectory Decoder 的初始 Query，并继续读取编码阶段已经加入地图位置的 Scene Memory $\bar B$：

$$
\boxed{
(S_t,\bar B)
\xrightarrow{\text{Trajectory Decoder}}
T_t^L
\xrightarrow{\text{Trajectory Head}}
\hat x_0.
}
$$

训练监督为：

$$
\boxed{
\hat x_0\rightarrow x_0,
\qquad
\hat E_{t,k}\rightarrow E_k^*.
}
$$

因此同一次前向传播同时执行：

$$
\boxed{
\text{Trajectory Denoising}
+
\text{Safe Geometry Recovery}.
}
$$

两个任务在 $S_t$ 处共享轨迹—场景安全表示；轨迹分支在此基础上进一步通过位置感知 Scene Memory $\bar B$ 获取完整场景信息。轨迹和地图的二维位置均已在编码阶段注入，后续联合恢复过程不再追加新的空间位置编码。

---

# 10. 椭圆多目标几何损失

椭圆监督采用：

$$
\boxed{
L_{ellipse}
=
\lambda_{param}L_{param}
+
\lambda_{iou}L_{iou}
+
\lambda_{ecol}L_{ellipse\_coll}
+
\lambda_{anchor}L_{anchor}.
}
$$

其中四项分别约束椭圆参数、区域重叠、障碍碰撞和 clean waypoint 包含关系。

## 10.1 参数回归损失

定义：

$$
\boxed{
L_{param}
=
\lambda_cL_c
+
\lambda_rL_r
+
\lambda_\theta L_\theta.
}
$$

### 中心损失

$$
\boxed{
L_c
=
\sum_k
m_k^{ellipse}
\operatorname{SmoothL1}
(
\hat c_{t,k},
c_k^*
).
}
$$

### 半轴损失

$$
\boxed{
L_r
=
\sum_k
m_k^{ellipse}
\sum_{i=1}^{2}
(
\hat r_{i,t,k}
-
r_{i,k}^*
)^2.
}
$$

### 方向损失

$$
\boxed{
L_\theta
=
\sum_k
m_k^{ellipse}
\left\|
\hat v_{\theta,t,k}
-
v_{\theta,k}^*
\right\|_2^2.
}
$$

## 10.2 椭圆空间重叠损失

根据预测椭圆构造：

$$
\hat Q_{t,k}
=
R(\hat\theta_{t,k})
\begin{bmatrix}
1/\hat r_{1,t,k}^2&0\\
0&1/\hat r_{2,t,k}^2
\end{bmatrix}
R(\hat\theta_{t,k})^T.
$$

对地图采样位置 $u$ 构造预测 soft mask：

$$
\boxed{
\hat M_{t,k}(u)
=
\sigma
\left(
\tau
\left[
1-
(u-\hat c_{t,k})^T
\hat Q_{t,k}
(u-\hat c_{t,k})
\right]
\right).
}
$$

GT ellipse 预先栅格化为

$$
M_k^*(u).
$$

Soft IoU 为：

$$
\operatorname{IoU}_{soft,k}
=
\frac{
\sum_u
\hat M_{t,k}(u)M_k^*(u)
}{
\sum_u
[
\hat M_{t,k}(u)
+
M_k^*(u)
-
\hat M_{t,k}(u)M_k^*(u)
]
+\epsilon
}.
$$

因此：

$$
\boxed{
L_{iou}
=
\sum_k
m_k^{ellipse}
\left(
1-\operatorname{IoU}_{soft,k}
\right).
}
$$

## 10.3 椭圆障碍碰撞损失

设

$$
O(u)\in\{0,1\}
$$

为障碍掩码，$O(u)=1$ 表示障碍。

定义：

$$
\boxed{
L_{ellipse\_coll}
=
\sum_k
m_k^{ellipse}
\sum_u
\hat M_{t,k}(u)O(u).
}
$$

该项直接抑制预测椭圆与障碍区域重叠。

## 10.4 Clean waypoint 包含损失

计算：

$$
d_{anchor,k}
=
(p_k^*-\hat c_{t,k})^T
\hat Q_{t,k}
(p_k^*-\hat c_{t,k}).
$$

定义：

$$
\boxed{
L_{anchor}
=
\sum_k
m_k^{ellipse}
\max
\left(
0,
d_{anchor,k}-1
\right).
}
$$

该项要求预测椭圆包含与其对应的 clean waypoint。

---

# 11. 轨迹—椭圆几何一致性损失（注：当前实现已移除该损失，见文首实现说明）

对于预测 clean waypoint

$$
\hat p_{0,k},
$$

以及对应 GT ellipse

$$
E_k^*
=
(c_k^*,Q_k^*),
$$

定义：

$$
d_{traj,k}
=
(\hat p_{0,k}-c_k^*)^T
Q_k^*
(\hat p_{0,k}-c_k^*).
$$

使用：

$$
\boxed{
L_{traj\_ellipse}
=
\sum_k
m_k^{ellipse}
\max
\left(
0,
d_{traj,k}-1
\right)^2.
}
$$

该损失直接约束预测 clean waypoint 落在其离线 GT 安全椭圆内，从而把几何标签同时用于轨迹分支，而不是只监督 Ellipse Head。

---

# 12. 轨迹生成损失

## 12.1 Diffusion 重建损失

模型直接预测 clean trajectory：

$$
\boxed{
L_{diff}
=
\mathbb E_{x_0,t,\epsilon}
\left[
\|\hat x_0-x_0\|_2^2
\right].
}
$$

> 注：实现中 `L_{diff}` 为均方误差，并额外叠加**相邻状态时间一致性项**（`lambda_var`：相邻位移方差；`lambda_var_vel`：相邻速度差分方差），见 `src/losses/trajectory_loss.py::l_diff`。

## 12.2 轨迹碰撞损失

根据地图构造 Signed Distance Field：

$$
D(p)>0
\quad
\text{表示自由空间},
$$

$$
D(p)<0
\quad
\text{表示障碍内部}.
$$

对预测位置轨迹

$$
\hat\tau_0
=
\{\hat p_{0,1},\ldots,\hat p_{0,H}\}
$$

定义：

$$
\boxed{
L_{collision}
=
\sum_k
\operatorname{softplus}
\left(
\frac{
m-D(\hat p_{0,k})
}{
\sigma
}
\right).
}
$$

其中 $m$ 为期望安全裕度。

$L_{ellipse\_coll}$ 约束预测椭圆的安全性，而 $L_{collision}$ 直接约束预测轨迹的安全性。

## 12.3 Smoothness Loss

定义二阶差分：

$$
\boxed{
L_{smooth}
=
\sum_{k=2}^{H-1}
\|
\hat p_{0,k+1}
-
2\hat p_{0,k}
+
\hat p_{0,k-1}
\|_2^2.
}
$$

该项用于抑制局部锯齿和不连续弯折。

---

# 13. 总体训练目标

最终训练目标为：

$$
\boxed{
L_{total}
=
L_{diff}
+
\lambda_eL_{ellipse}
+
\lambda_{te}L_{traj\_ellipse}
+
\lambda_{col}L_{collision}
+
\lambda_sL_{smooth}.
}
$$

> 注：当前实现中总体目标**不含** `L_{traj_ellipse}` 项（已移除）；`L_{diff}` 用 `lambda_diff` 加权并含时间一致性项；训练有 **warmup**（前若干轮只训 `L_{diff}+L_{smooth}`）。各子损失权重见 `configs/config.yaml` 的 `loss:` 段。

其中：

$$
L_{ellipse}
=
\lambda_{param}L_{param}
+
\lambda_{iou}L_{iou}
+
\lambda_{ecol}L_{ellipse\_coll}
+
\lambda_{anchor}L_{anchor}.
$$

各项作用为：

- $L_{diff}$：监督 Trajectory Decoder 恢复 clean trajectory；
- $L_{param}$：回归椭圆中心、半轴和方向；
- $L_{iou}$：约束预测椭圆与 GT ellipse 的整体区域一致性；
- $L_{ellipse\_coll}$：抑制预测椭圆覆盖障碍；
- $L_{anchor}$：保证 clean waypoint 位于预测椭圆内；
- $L_{traj\_ellipse}$：约束预测 clean waypoint 位于对应 GT ellipse 内；
- $L_{collision}$：直接约束预测轨迹的障碍安全性；
- $L_{smooth}$：保持轨迹平滑。

监督结构可写为：

$$
\boxed{
\begin{cases}
(S_t,\bar B)
\xrightarrow{\text{Trajectory Decoder}}
T_t^L
\xrightarrow{\text{Trajectory Head}}
\hat x_0
\xrightarrow{
L_{diff},
L_{traj\_ellipse},
L_{collision},
L_{smooth}
}
x_0,\\[3mm]
S_t
\xrightarrow{\text{Ellipse Head}}
\hat E_t
\xrightarrow{
L_{ellipse}
}
E^*.
\end{cases}
}
$$

$S_t$ 是两个任务的共享中间表示，$\bar B$ 是编码阶段一次性加入地图二维位置后的 Scene Memory。后续 Trajectory Decoder 只读取该固定 Scene Memory，不再重复注入空间位置。

---

# 14. 解析凸安全区域（注：当前实现未实现该下游凸区域构造；`convex_region.py` 已移出，属设计可选项）

对于最终预测椭圆

$$
\hat E_{0,k}
=
(
\hat c_{0,k},
\hat r_{1,0,k},
\hat r_{2,0,k},
\hat\theta_{0,k}
),
$$

构造：

$$
\hat Q_{0,k}
=
R(\hat\theta_{0,k})
\begin{bmatrix}
1/\hat r_{1,0,k}^2&0\\
0&1/\hat r_{2,0,k}^2
\end{bmatrix}
R(\hat\theta_{0,k})^T.
$$

结合占据地图，使用解析几何构造分离半空间：

$$
\boxed{
(\hat c_{0,k},\hat Q_{0,k},M)
\xrightarrow{\text{Analytical Safe Region Construction}}
(\hat A_k,\hat b_k).
}
$$

得到凸安全区域：

$$
\boxed{
\hat{\mathcal P}_k
=
\{
p\mid
\hat A_kp\le\hat b_k
\}.
}
$$

网络负责从轨迹和地图中预测安全几何先验，解析模块负责将该先验转换为显式凸安全约束。

推理时仅使用最终扩散结果对应的 $\hat E_0$ 构造凸安全区域，不在每个 reverse diffusion step 内重复执行解析几何。

---

# 15. 训练流程

对于一个训练样本 $(M,x_0,E^*)$：

1. 随机采样扩散时间步 $t$ 和高斯噪声 $\epsilon$；

2. 构造 noisy trajectory：

   $$
   x_t
   =
   \sqrt{\bar\alpha_t}x_0
   +
   \sqrt{1-\bar\alpha_t}\epsilon;
   $$

3. Scene Encoder 提取地图内容特征：

   $$
   M
   \rightarrow
   B;
   $$

4. 将 $B$ 展开为 Scene Tokens，并仅在此处加入一次地图二维位置编码：

   $$
   B
   \rightarrow
   B^{tok}
   \xrightarrow{+\ PE_{map}}
   \bar B;
   $$

5. 将带噪轨迹拆分为运动状态 $d_{t,k}$ 与二维位置 $p_{t,k}$，在 Trajectory Token 初始化时一次性加入运动状态、二维位置、轨迹序号和 diffusion timestep：

   $$
   (x_t,t)
   \rightarrow
   H_t^0
   \rightarrow
   F_t^{traj};
   $$

6. 直接使用 $F_t^{traj}$ 作为轨迹 Query、$\bar B$ 作为 Scene Memory 执行 Point-Scene Cross-Attention：

   $$
   (F_t^{traj},\bar B)
   \rightarrow
   A_t;
   $$

7. 拼接轨迹上下文与场景交互特征：

   $$
   [F_t^{traj};A_t]
   \rightarrow
   S_t;
   $$

8. Ellipse Head 预测：

   $$
   S_t
   \xrightarrow{\text{Ellipse Head}}
   \hat E_t;
   $$

9. Trajectory Decoder 以 $S_t$ 为初始 Query，并持续读取同一个位置感知 Scene Memory $\bar B$：

   $$
   (S_t,\bar B)
   \xrightarrow{\text{Trajectory Decoder}}
   T_t^L;
   $$

10. Trajectory Head 输出：

$$
   T_t^L
   \xrightarrow{\text{Trajectory Head}}
   \hat x_0;
$$

11. 根据 $\hat x_0$、$\hat E_t$、$x_0$、$E^*$ 和地图计算 $L_{total}$，并联合反向传播。

整个训练过程中，二维空间位置只在第 4 步和第 5 步的 Scene Token / Trajectory Token 初始化阶段编码一次。后续 Point-Scene Attention 与 Trajectory Decoder 均使用已携带位置信息的隐藏特征。

---

# 16. 推理流程

给定 Maze2D 地图 $M$、起点和目标点：

## 16.1 地图编码

Scene Encoder 只运行一次：

$$
M
\xrightarrow{\text{Scene Encoder}}
B.
$$

随后将 Scene Feature 展开为 Scene Tokens 并加入一次地图二维位置编码：

$$
\boxed{
B
\rightarrow
B^{tok}
\xrightarrow{+\ PE_{map}}
\bar B.
}
$$

$\bar B$ 在整个 reverse diffusion 过程中作为固定 Scene Memory 重复使用，不在不同扩散时间步或不同 Decoder Block 中重复加入位置编码。

## 16.2 初始化

初始化：

$$
x_T\sim\mathcal N(0,I),
$$

并按照 Maze2D Diffuser 的条件生成方式固定起点和目标条件。

## 16.3 Reverse Diffusion

对于每个 reverse diffusion timestep $t$：

1. 将当前 noisy trajectory 分解为运动状态和二维位置，并在 Trajectory Token 初始化时一次性编码当前 waypoint 二维位置、轨迹序号与 diffusion timestep：

   $$
   (x_t,t)
   \rightarrow
   H_t^0
   \rightarrow
   F_t^{traj};
   $$

2. Point-Scene Cross-Attention：

   $$
   (F_t^{traj},\bar B)
   \rightarrow
   A_t;
   $$

3. 拼接形成共享安全特征：

   $$
   [F_t^{traj};A_t]
   \rightarrow
   S_t;
   $$

4. Ellipse Head 输出当前时间步椭圆：

   $$
   S_t
   \xrightarrow{\text{Ellipse Head}}
   \hat E_t;
   $$

5. Trajectory Decoder 直接使用 $S_t$ 和固定位置感知 Scene Memory $\bar B$：

   $$
   (S_t,\bar B)
   \xrightarrow{\text{Trajectory Decoder}}
   T_t^L;
   $$

6. Trajectory Head 输出当前时间步的 clean trajectory estimate：

   $$
   T_t^L
   \xrightarrow{\text{Trajectory Head}}
   \hat x_0;
   $$

7. DDPM Scheduler 根据

   $$
   (x_t,\hat x_0,t)
   $$

   计算

   $$
   x_{t-1};
   $$

8. 重新施加起点和目标条件。

重复直到 $t=0$。

## 16.4 最终几何安全层

最终得到：

$$
\hat x_0,
\qquad
\hat E_0.
$$

根据 $\hat E_0$ 和地图构造每个 waypoint 的凸安全区域：

$$
(\hat E_0,M)
\rightarrow
\{
(\hat A_k,\hat b_k)
\}_{k=1}^{H}.
$$

随后对最终位置轨迹执行一次受凸安全区域约束的轨迹修正：

$$
\boxed{
\hat x_0
\xrightarrow{\text{Convex Safety Correction}}
x_0^{safe}.
}
$$

解析安全层作用于最终扩散结果，不参与 reverse diffusion 内部循环。

---

# 17. 方法核心关系

离线阶段：

$$
\boxed{
(M,x_0)
\rightarrow
E^*.
}
$$

训练阶段首先执行：

$$
\boxed{
x_0
\xrightarrow{\text{Forward Diffusion}}
x_t.
}
$$

编码阶段分别一次性注入轨迹和地图二维位置信息：

$$
\boxed{
(x_t,t)
\rightarrow
H_t^0,
\qquad
M
\rightarrow
B
\rightarrow
\bar B.
}
$$

随后构造共享安全特征：

$$
\boxed{
(F_t^{traj},\bar B)
\rightarrow
A_t,
}
$$

$$
\boxed{
(F_t^{traj},A_t)
\rightarrow
S_t.
}
$$

两个分支为：

$$
\boxed{
S_t
\xrightarrow{\text{Ellipse Head}}
\hat E_t,
}
$$

以及

$$
\boxed{
(S_t,\bar B)
\xrightarrow{\text{Trajectory Decoder}}
T_t^L
\xrightarrow{\text{Trajectory Head}}
\hat x_0.
}
$$

监督：

$$
\boxed{
(\hat x_0,\hat E_t)
\rightarrow
(x_0,E^*).
}
$$

推理阶段每个 reverse diffusion timestep 中：

$$
\boxed{
x_t
\rightarrow
S_t,
}
$$

$$
\boxed{
(S_t,\bar B)
\rightarrow
\hat x_0
\rightarrow
x_{t-1},
}
$$

同时：

$$
\boxed{
S_t
\rightarrow
\hat E_t.
}
$$

最终：

$$
\boxed{
(\hat x_0,\hat E_0,M)
\rightarrow
x_0^{safe}.
}
$$

整个方法可以概括为：

$$
\boxed{
\text{One-Time Position-Aware Encoding}
\rightarrow
\text{Trajectory-Scene Safety Interaction}
\rightarrow
S_t
\rightarrow
\begin{cases}
\text{Ellipse Geometry Recovery},\\
\text{Scene-Conditioned Trajectory Decoding}
\end{cases}
\rightarrow
\text{Final Convex Safety Correction}.
}
$$

其中轨迹位置和地图位置只在各自 Token 初始化时编码一次，之后随隐藏特征在整个网络中传播。

---

# 18. 与 Planning-Oriented 思想的关系

该结构采用规划导向的共享表征思路：地图场景特征通过 trajectory Query 与规划特征直接交互，并由最终轨迹生成损失和安全几何损失共同优化。

在本方法中：

- occupancy map 经过 Scene Encoder 得到地图内容特征 $B$，并在 Scene Token 初始化时加入一次二维地图位置编码形成 $\bar B$；
- noisy trajectory 在 Trajectory Token 初始化时一次性加入 waypoint 二维位置、轨迹序号和 diffusion timestep；
- Point-Scene Cross-Attention 直接使用已经携带位置信息的轨迹特征和 Scene Memory，不重新注入绝对位置或相对位置；
- Shared Safety Feature $S_t$ 汇聚轨迹上下文和安全场景信息；
- Ellipse Head 通过几何监督约束 $S_t$ 学习可解释的自由空间几何；
- Trajectory Decoder 将 $S_t$ 作为轨迹 Query，并持续以同一个位置感知 Scene Memory $\bar B$ 为 Key/Value，通过多层 Scene Cross-Attention 获取面向最终轨迹生成的地图信息；
- diffusion trajectory loss 直接优化 Trajectory Decoder 的 clean trajectory prediction。

因此：

$$
\boxed{
\text{Geometry Supervision}
\rightarrow
S_t
\xrightarrow[\text{Position-Aware Scene Memory }\bar B]
{\text{Trajectory Decoder}}
\text{Diffusion Planning}.
}
$$

这里多层 Decoder 会重复访问 $\bar B$，但不会重复加入位置编码；位置只在编码阶段进入一次，之后作为隐藏特征的一部分持续传播。

---

# 19. 消融实验

消融实验设置如下：

| Model | Point-Scene Attention | Ellipse Supervision | Scene-Conditioned Trajectory Decoder | Traj-Ellipse Loss | Trajectory Collision | Final Convex Safety |
|---|---:|---:|---:|---:|---:|---:|
| Base Diffuser | × | × | × | × | × | × |
| SceneAttn | ✓ | × | × | × | × | × |
| SceneAttn + Ellipse | ✓ | ✓ | × | × | × | × |
| + Trajectory Decoder | ✓ | ✓ | ✓ | × | × | × |
| + Traj-Ellipse | ✓ | ✓ | ✓ | ✓ | × | × |
| + Collision | ✓ | ✓ | ✓ | ✓ | ✓ | × |
| Full | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

其中 “Scene-Conditioned Trajectory Decoder” 用于验证：在已经获得共享安全特征 $S_t$ 后，让最终轨迹分支继续通过 Cross-Attention 访问地图特征 $B$，是否优于直接使用 $S_t$ 预测 clean trajectory。

主要指标：

- Success Rate；
- Collision Rate；
- Collision Points；
- Minimum Clearance；
- Goal Error；
- Path Length；
- Smoothness；
- Ellipse Center Error；
- Radius Error；
- Axis Direction Error；
- Ellipse IoU；
- Ellipse Collision Rate；
- Convex Region Validity；
- Runtime。

---

# 参考思想

Hu, Y., Yang, J., Chen, L., et al. *Planning-Oriented Autonomous Driving (UniAD)*, CVPR 2023.
