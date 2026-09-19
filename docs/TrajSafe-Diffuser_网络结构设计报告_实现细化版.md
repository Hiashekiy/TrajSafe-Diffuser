# TrajSafe-Diffuser 网络结构设计报告

## 1. 总体任务

唯一参与 diffusion 的状态是轨迹：

$$
\boxed{
P_t=
[p_1^t,\ldots,p_H^t]
\in\mathbb R^{B\times H\times2}
}
$$

其中：

$$
p_i^t=(x_i^t,y_i^t)
$$

表示 diffusion timestep $t$ 下第 $i$ 个 noisy waypoint。

主要维度：

$$
H=128,\qquad D=128.
$$

单个 reverse timestep 内的总体流程为：

$$
\boxed{
P_t
\rightarrow
H^{traj}
\rightarrow
\{R_m\}_{m=1}^{M}
\rightarrow
m
\rightarrow
H^{prog}
\rightarrow
s
\rightarrow
c
\rightarrow
H^{ell}
\rightarrow
H^{clean}
\rightarrow
\hat P_0^t
}
$$

同时：

$$
H^{traj}
\rightarrow
\tilde P_0^t
$$

产生辅助 coarse clean trajectory。

最终：

$$
\boxed{
P_t,\hat P_0^t
\rightarrow
DDIM
\rightarrow
P_{t-1}
}
$$

每一个 reverse timestep 都重新完成上述过程。

---


## 1.1 基线实现超参数

为避免实现阶段自行补全网络结构，本报告把第一版 baseline 固定为下表。除非专门做 ablation，否则 DSH 实现时按照这些值构建；改变层数、head 数、FFN 宽度、归一化位置或额外增加 token / attention，均视为**网络结构修改**，而不是普通工程实现差异。

| 项目 | baseline |
|---|---:|
| trajectory waypoint 数 $H$ | 128 |
| latent dimension $D$ | 128 |
| attention heads $N_h$ | 4 |
| single-head dimension $d_h=D/N_h$ | 32 |
| Transformer FFN width $D_{ff}$ | 512 |
| head hidden width $D_{head}$ | 256 |
| Trajectory Backbone blocks $N_T$ | 8 |
| Skeleton Transformer blocks $N_S$ | 2 |
| Final Denoiser blocks $N_F$ | 3 |
| dropout | 0.0 |
| occupancy input resolution | $256\times256$ |
| Global Memory resolution | $16\times16$ |
| Fine Geometry decode resolution | $64\times64$ |
| Fine Geometry Memory resolution | $32\times32$ |
| coordinate PE scale | 128 |
| geometry bias $\sigma_{geo}$ | 0.25 |
| geometry bias clip $b_{clip}$ | 8.0 |
| progress positive-gap $\epsilon$ | $10^{-4}$ |

所有 attention 均满足：

$$
D=N_h d_h=4\times32=128.
$$

除 Skeleton Transformer 外，凡是与 diffusion state 直接交互的 Transformer block 都通过 $h_t$ 使用 AdaLN。Skeleton geometry 是静态条件，因此 Skeleton Transformer 使用普通 LayerNorm，不注入 timestep。

---

# 2. 主要符号

| 符号 | 含义 |
|---|---|
| $P_t$ | 当前 noisy trajectory |
| $H^{traj}$ | Trajectory Backbone latent |
| $\tilde P_0^t$ | coarse clean trajectory |
| $S_m$ | 第 $m$ 条 Skeleton 的固定长度网络输入 |
| $H_m^S$ | 第 $m$ 条 Skeleton 的结构特征 |
| $\Gamma_m$ | 第 $m$ 条完整 Skeleton Curve |
| $R_m$ | trajectory 与 candidate $m$ 的共享匹配特征 |
| $\pi_m$ | candidate $m$ 的 topology probability |
| $m^*$ | 训练阶段 GT-best candidate |
| $m_{sel}$ | 推理阶段预测 candidate |
| $H^{prog}$ | Progress Head feature |
| $s_i$ | waypoint $i$ 的 Skeleton progress |
| $c_i$ | 根据 $s_i$ 在 Skeleton Curve 上解码的 center |
| $C_G$ | Global Map Memory |
| $C_E$ | Fine Geometry Memory |
| $H^{ell}$ | center 附近局部自由空间 feature |
| $c_i^*$ | 第 $i$ 个 waypoint 的 GT ellipse center |
| $shape_i^*$ | GT ellipse shape：$[\log a_i^*,\log b_i^*,\cos2\theta_i^*,\sin2\theta_i^*]$ |
| $M_i^*$ | GT ellipse soft mask |
| $h_t$ | diffusion timestep feature |
| $H^{clean}$ | Final Denoiser feature |
| $\hat P_0^t$ | 最终 clean trajectory |

---

# 3. 编码方式

网络中严格区分：

$$
\text{二维空间坐标编码}
,\qquad
\text{序列 index 编码}
,\qquad
\text{diffusion timestep 编码}.
$$

## 3.1 二维坐标编码

所有二维坐标统一采用：

$$
\boxed{
\Phi_{xy}(x,y)
}
$$

坐标归一化为：

$$
x,y\in[-1,1].
$$

沿用稳定版空间尺度：

$$
\boxed{
\lambda_{xy}=128
}
$$

因此：

$$
\Phi_{xy}(x,y)
=
[
\sin(128xf_k),
\cos(128xf_k),
\sin(128yf_k),
\cos(128yf_k)
]_k
$$

其中：

$$
f_k=10000^{-k/K}.
$$

如果坐标自身就是独立输入：

$$
\boxed{
p
\rightarrow
\Phi_{xy}(p)
\rightarrow
MLP
}
$$

例如 trajectory point、Skeleton point、ellipse center。

如果已经存在 CNN feature：

$$
\boxed{
F+\Phi_{xy}(p)
}
$$

直接叠加，不增加额外位置 MLP。

---

## 3.2 序列位置

Trajectory：

$$
\boxed{
PE_{1D}(i),
\quad i=0,\ldots,H-1
}
$$

Skeleton：

$$
\boxed{
PE_{1D}(j),
\quad j=0,\ldots,L-1
}
$$

均直接使用整数 index。

---

## 3.3 Diffusion timestep

原始 timestep：

$$
t\in\{0,\ldots,T-1\}
$$

先经过 sinusoidal embedding，再：

$$
\boxed{
h_t=
Linear
\rightarrow
SiLU
\rightarrow
Linear
}
$$

得到：

$$
h_t\in\mathbb R^D.
$$

$h_t$ 通过 AdaLN 调制 Transformer：

$$
\boxed{
AdaLN(X,h_t)
=
(1+\gamma(h_t))LN(X)+\beta(h_t).
}
$$

---


## 3.4 三种编码模块的精确实现

### SpatialPE

`SpatialPE` 只实例化一套并在 trajectory、Skeleton、map grid、ellipse center 之间共享权重。其内部结构为：

```text
(x,y)
  │
  ├─ sinusoidal 2D features [D]
  │
  └─ Linear(D,D)
        │
        ▼
     Φ_xy(x,y)
```

这里的 `Linear(D,D)` 属于 $\Phi_{xy}$ 自身的一部分。对于 map feature，不允许再额外接位置 MLP；对于独立坐标输入，则在 $\Phi_{xy}$ 后使用各自独立的 `CoordMLP`。

统一定义独立坐标的投影结构：

$$
\boxed{
CoordMLP(x)=Linear(D,D_{head})\rightarrow SiLU\rightarrow Linear(D_{head},D)
}
$$

其中 $D_{head}=256$。`MLP_T`、`MLP_S`、`MLP_C` 结构相同但**参数不共享**。

### IndexPE

`PE_1D` 是无参数 sinusoidal encoding，输入必须是整数 index，不做归一化、不乘 128、不再经过额外 MLP。

### TimeEmbedding 与 AdaLN

时间编码：

$$
TimeEmbedding(t)=SinPE(t)\rightarrow Linear(D,D)\rightarrow SiLU\rightarrow Linear(D,D).
$$

AdaLN 精确实现为：

$$
\hat X=LN(X),
$$

$$
[\gamma_t,\beta_t]=Linear(h_t)\in\mathbb R^{2D},
$$

$$
\boxed{
AdaLN(X,h_t)=(1+\gamma_t)\odot\hat X+\beta_t.
}
$$

其中 LayerNorm 设置 `elementwise_affine=False`；产生 $\gamma_t,\beta_t$ 的线性层零初始化，使网络初始时 AdaLN 等价于普通 LayerNorm。每个 Transformer sublayer 使用**独立**的 AdaLN 参数，不共享 modulation layer。

---

# 4. Scene Encoder

Occupancy map：

$$
O\in\mathbb R^{B\times1\times R\times R}
$$

经过 SceneCNN：

$$
\boxed{
O
\rightarrow
F_G,F_E
}
$$

其中：

$$
F_G\in\mathbb R^{B\times N_G\times D}
$$

用于全局环境信息；

$$
F_E\in\mathbb R^{B\times N_E\times D}
$$

用于精细局部几何信息。

例如：

$$
N_G=16\times16,
\qquad
N_E=32\times32.
$$

---


## 4.1 SceneCNN 的精确结构

SceneCNN 第一版固定采用 encoder + fine-geometry decoder。每个基础卷积块均为：

$$
\boxed{
Conv3\times3\rightarrow GroupNorm(1,C)\rightarrow SiLU.
}
$$

Encoder 的空间尺寸与通道变化为：

| stage | 操作 | 输出尺寸 |
|---|---|---|
| stem | $1\rightarrow32$，stride 1 | $B\times32\times256\times256$ |
| down1 | $32\rightarrow64$ + stride-2 conv | $B\times64\times128\times128$ |
| down2 | $64\rightarrow96$ + stride-2 conv | $B\times96\times64\times64$ |
| down3 | $96\rightarrow128$ + stride-2 conv | $B\times128\times32\times32$ |
| down4 | $128\rightarrow128$ + stride-2 conv | $B\times128\times16\times16$ |
| bottleneck | Conv3×3 + GN + SiLU | $B\times128\times16\times16$ |
| global projection | Conv1×1, $128\rightarrow D$ | $B\times D\times16\times16$ |

Fine Geometry decoder 复用 encoder 的 $32\times32$ 与 $64\times64$ skip feature：

```text
Global 16×16×D
    │ bilinear ×2
    ├─ concat encoder F32
    ▼
Conv3×3 → GN → SiLU → Conv3×3 → GN → SiLU
    │                    32×32×D
    │ bilinear ×2
    ├─ concat encoder F64
    ▼
Conv3×3 → GN → SiLU → Conv3×3 → GN → SiLU
    │                    64×64×D
    ▼
Conv3×3 stride=2 → GN → SiLU
    │
    ▼
Fine Geometry Feature 32×32×D
```

因此 SceneCNN 返回两个张量：

$$
F_G^{map}\in\mathbb R^{B\times D\times16\times16},
$$

$$
F_E^{map}\in\mathbb R^{B\times D\times32\times32}.
$$

展平后才得到报告中使用的 $F_G\in\mathbb R^{B\times256\times D}$ 和 $F_E\in\mathbb R^{B\times1024\times D}$。

第一版 Global Memory **只包含 map tokens**。起点、终点通过 noisy trajectory 的 hard endpoint conditioning 进入网络，不额外构造 start token / goal token，也不增加 role embedding。

---

# 5. Global Map Memory

global grid 第 $j$ 个中心：

$$
q_j^G=(x_j^G,y_j^G).
$$

定义：

$$
\boxed{
C_{G,j}
=
F_{G,j}
+
\Phi_{xy}(q_j^G)
}
$$

所以：

$$
\boxed{
C_G
\in
\mathbb R^{B\times N_G\times D}.
}
$$

---

# 6. Fine Geometry Memory

fine grid 第 $j$ 个中心：

$$
q_j^E=(x_j^E,y_j^E).
$$

定义：

$$
\boxed{
C_{E,j}
=
F_{E,j}
+
\Phi_{xy}(q_j^E)
}
$$

因此：

$$
\boxed{
C_E
\in
\mathbb R^{B\times N_E\times D}.
}
$$

---


## 6.1 Map token 构造顺序

Global 与 Fine Geometry 两套 memory 均严格采用同一流程：

```text
CNN feature map
   │ flatten spatial dimension
   ▼
[B,N,D]
   │
   ├─ scene_grid_centres() → [N,2]
   ├─ shared SpatialPE → [N,D]
   ▼
feature + position
```

即：

$$
C_G=Flatten(F_G^{map})+\Phi_{xy}(Q_G),
$$

$$
C_E=Flatten(F_E^{map})+\Phi_{xy}(Q_E).
$$

这里没有额外 `LayerNorm`、没有额外 positional MLP、没有 token type embedding。$C_G$ 和 $C_E$ 在同一个 reverse timestep 内计算一次，随后被所有相关模块复用。

---

# 7. Trajectory Encoder

对于：

$$
p_i^t=(x_i^t,y_i^t)
$$

首先：

$$
\Phi_{xy}(p_i^t)
$$

再经过 trajectory input projection：

$$
MLP_T.
$$

加入 waypoint index：

$$
\boxed{
T_i^0
=
MLP_T(\Phi_{xy}(p_i^t))
+
PE_{1D}(i)
}
$$

得到：

$$
\boxed{
T^0
\in
\mathbb R^{B\times H\times D}.
}
$$

---


## 7.1 Trajectory input projection 的精确结构

`MLP_T` 使用第 3.4 节定义的 `CoordMLP`：

```text
p_i^t [2]
  │
  ▼
shared SpatialPE
  │ [D]
  ▼
Linear(D,256)
  │
 SiLU
  │
Linear(256,D)
  │
  + PE_1D(i)
  ▼
T_i^0 [D]
```

$MLP_T$ 对所有 waypoint 共享参数，但与 `MLP_S`、`MLP_C` 不共享参数。输入层不增加 trajectory type embedding。$T^0$ 后不单独做 LayerNorm，因为第一个 Trajectory Block 的每个 sublayer 已经采用 pre-norm AdaLN。

---

# 8. Trajectory Backbone

Trajectory Backbone 每层：

$$
\boxed{
SelfAttention
\rightarrow
GlobalMapCrossAttention
\rightarrow
FFN
}
$$

Global Map Cross-Attention：

$$
Q=T,
\qquad
K=V=C_G.
$$

各层使用：

$$
h_t
$$

进行 AdaLN conditioning。

最终：

$$
\boxed{
H^{traj}
=
[h_1^{traj},\ldots,h_H^{traj}]
\in
\mathbb R^{B\times H\times D}.
}
$$

---


## 8.1 单个 Trajectory Block 的精确计算

设第 $r$ 个 block 输入为 $X^{(r)}\in\mathbb R^{B\times H\times D}$。每个 block 固定包含三个 residual sublayer：

$$
X_1=X^{(r)}+SA_r\left(AdaLN_{r,1}(X^{(r)},h_t)\right),
$$

$$
X_2=X_1+CA_r\left(AdaLN_{r,2}(X_1,h_t),C_G\right),
$$

$$
\boxed{
X^{(r+1)}=X_2+FFN_r\left(AdaLN_{r,3}(X_2,h_t)\right).
}
$$

Self-Attention 使用 $N_h=4$ 个 head，Q/K/V 均由当前 trajectory tokens 线性投影得到：

$$
[Q,K,V]=Linear_{qkv}(X),
$$

并保留 waypoint 相对距离 bias：

$$
B_{ij}^{traj,h}=b_h^{traj}(|i-j|),
$$

$$
Score_{ij}^{h}=\frac{Q_i^h(K_j^h)^T}{\sqrt{d_h}}+B_{ij}^{traj,h}.
$$

这里 $b_h^{traj}(\cdot)$ 是每个 head 独立的可学习表，长度为 $H$。它只表达 waypoint index 间距，不使用坐标距离。

Global Map Cross-Attention 使用：

$$
Q=Linear_Q(X),\qquad [K,V]=Linear_{KV}(C_G),
$$

不额外添加 spatial bias，因为 map position 已经包含在 $C_G$ 中。

FFN 固定为：

$$
\boxed{
Linear(D,512)\rightarrow GELU\rightarrow Linear(512,D)\rightarrow Dropout(0)
}
$$

Trajectory Backbone 共堆叠 $N_T=8$ 个上述 block，各 block 参数互不共享。最终 $H^{traj}=X^{(8)}$。

---

# 9. Coarse Clean Trajectory

定义共享输出头：

$$
\boxed{
Head_P:\mathbb R^D\rightarrow\mathbb R^2.
}
$$

得到：

$$
\boxed{
\tilde P_0^t
=
Head_P(H^{traj})
}
$$

并 hard overwrite：

$$
\tilde p_1^0=p_s,
\qquad
\tilde p_H^0=p_g.
$$

该 branch 只作为：

$$
L_{coarse}
$$

的辅助监督。

后面的 Skeleton matching **直接使用 $H^{traj}$**，不重新编码 $\tilde P_0^t$。

---


## 9.1 Shared trajectory head

`Head_P` 固定为一个线性层：

$$
\boxed{Head_P=Linear(D,2).}
$$

Coarse 与 Final trajectory **必须调用同一个 `Head_P` 实例**，不是两个结构相同但参数独立的 head。`Head_P` 输出后再执行 endpoint overwrite；overwrite 不应写进 head 内部。

---

# 10. Skeleton 表示

每条 candidate 同时有两种表示。

## 10.1 网络 Skeleton

固定重采样：

$$
\boxed{
S_m=
[s_{m,1},\ldots,s_{m,L}]
}
$$

其中：

$$
s_{m,j}\in\mathbb R^2.
$$

例如：

$$
L=128.
$$

它负责进入神经网络。

## 10.2 Skeleton Curve

完整有序几何曲线：

$$
\boxed{
\Gamma_m=
[q_{m,1},\ldots,q_{m,N_m}]
}
$$

其中：

$$
q_{m,k}\in\mathbb R^2.
$$

$\Gamma_m$ 不进入神经网络，只用于：

$$
\boxed{
s_i
\rightarrow
c_i.
}
$$

即根据 progress 按弧长解码 center。

---

# 11. Skeleton Encoder

对于 Skeleton point：

$$
s_{m,j}
$$

构造：

$$
\boxed{
S_{m,j}^0
=
MLP_S(\Phi_{xy}(s_{m,j}))
+
PE_{1D}(j)
}
$$

因此：

$$
S_m^0
\in
\mathbb R^{B\times L\times D}.
$$

再经过 $1\sim2$ 层轻量 Skeleton Transformer：

$$
\boxed{
S_m^0
\rightarrow
SkeletonTransformer
\rightarrow
H_m^S
}
$$

每层：

$$
SelfAttention
\rightarrow
FFN.
$$

输出：

$$
\boxed{
H_m^S
=
[h_{m,1}^S,\ldots,h_{m,L}^S]
\in
\mathbb R^{B\times L\times D}.
}
$$

它负责把：

$$
\text{逐点坐标信息}
$$

变成：

$$
\text{具有整条 Skeleton 上下文的结构特征}.
$$

---


## 11.1 Skeleton Encoder 的精确实现

Skeleton 网络输入只使用二维坐标 $s_{m,j}=(x,y)$；不再额外拼接 tangent、人工 progress、路径长度等手工特征。

`MLP_S`：

$$
Linear(D,256)\rightarrow SiLU\rightarrow Linear(256,D).
$$

输入 token：

$$
S_{m,j}^0=LN_{in}\left(MLP_S(\Phi_{xy}(s_{m,j}))+PE_{1D}(j)\right).
$$

随后使用 $N_S=2$ 个标准 pre-norm Transformer Encoder layer。对第 $r$ 层：

$$
Y=S+MHA_r(LN_{r,1}(S)),
$$

$$
\boxed{
S'=Y+FFN_r(LN_{r,2}(Y)).
}
$$

其中 MHA 为 4-head self-attention；FFN 为：

$$
Linear(D,512)\rightarrow GELU\rightarrow Linear(512,D).
$$

第二层之后再做一次：

$$
\boxed{H_m^S=LN_{out}(S^{(2)}).}
$$

Skeleton Transformer 不输入 $h_t$、不访问 $C_G/C_E$、不与 trajectory 做 attention。它的唯一职责是把静态 Skeleton 坐标序列编码成上下文化 path tokens。

实现时可将 $[B,M,L,D]$ reshape 成 $[BM,L,D]$ 一次性跑 2 层 Transformer，再 reshape 回 $[B,M,L,D]$；这只改变计算方式，不改变网络定义。

---

# 12. Trajectory–Skeleton Shared Matching Block

这里是本版最重要的修改。

对于每条 Skeleton candidate $m$，**只做一次 trajectory–Skeleton Cross-Attention**。

输入：

$$
H^{traj}
\in
\mathbb R^{B\times H\times D}
$$

和：

$$
H_m^S
\in
\mathbb R^{B\times L\times D}.
$$

首先：

$$
\boxed{
A_m
=
CrossAttention
\left(
Q=AdaLN(H^{traj},h_t),
K=H_m^S,
V=H_m^S
\right)
}
$$

得到：

$$
A_m
\in
\mathbb R^{B\times H\times D}.
$$

其中：

$$
a_{m,i}
$$

表示第 $i$ 个 trajectory waypoint 从 candidate $m$ 中读取到的 Skeleton structural information。

残差：

$$
\boxed{
U_m
=
H^{traj}
+
A_m
}
$$

再经过 FFN：

$$
\boxed{
R_m
=
U_m
+
FFN_{match}(LN(U_m))
}
$$

所以：

$$
\boxed{
R_m
\in
\mathbb R^{B\times H\times D}.
}
$$

$R_m$ 是后面两个任务共享的核心 feature：

$$
\boxed{
R_m
=
Trajectory
+
Skeleton_m
+
waypoint\ correspondence.
}
$$

Topology 和 Progress 都从这个 $R_m$ 出发，不再做第二次 Cross-Attention。

---


## 12.1 Shared Matching Block 的精确实现

对每个 candidate $m$，只存在一套 trajectory-to-Skeleton Cross-Attention：

$$
\tilde H^{traj}=AdaLN_{match}(H^{traj},h_t),
$$

$$
Q=Linear_Q(\tilde H^{traj}),
$$

$$
[K_m,V_m]=Linear_{KV}(H_m^S).
$$

按 4 个 head 计算：

$$
A_m=Concat_h\left(Softmax\left(\frac{Q^h(K_m^h)^T}{\sqrt{d_h}}\right)V_m^h\right)W_O.
$$

这里**不添加**额外 Chamfer feature、不把 coarse trajectory 坐标拼进 matching latent，也不对 selected Skeleton 再运行第二个 Cross-Attention。

残差后：

$$
U_m=H^{traj}+A_m.
$$

`FFN_match` 为：

$$
Linear(D,512)\rightarrow GELU\rightarrow Linear(512,D).
$$

最终：

$$
\boxed{
R_m=U_m+FFN_{match}(LN_{match}(U_m)).
}
$$

`LN_match` 为普通 LayerNorm；FFN 不再额外接 AdaLN，因为 timestep 信息已经通过 query 端的 `AdaLN_match` 进入共享匹配特征。

所有 candidate 共享同一个 MatchBlock 参数，即 $m=1,\ldots,M$ 并不是 $M$ 套独立 Cross-Attention。实现时推荐将 Skeleton candidate 维展开并广播 $H^{traj}$，批量得到：

$$
R\in\mathbb R^{B\times M\times H\times D}.
$$

---

# 13. Topology Head

Topology 是 candidate-level 判断，因此先对 $R_m$ 沿 waypoint 维度聚合：

$$
\boxed{
z_m
=
\frac1H
\sum_{i=1}^{H}
R_{m,i}
}
$$

得到：

$$
z_m
\in
\mathbb R^{B\times D}.
$$

再：

$$
\boxed{
l_m
=
MLP_{score}(z_m)
}
$$

全部 candidate 得到：

$$
[l_1,\ldots,l_M].
$$

进行：

$$
\boxed{
\pi_m
=
\frac{\exp(l_m)}
{\sum_{k\in\mathcal V}\exp(l_k)}
}
$$

其中：

$$
\mathcal V
$$

表示有效 candidate 集合。

预测：

$$
\boxed{
m_{sel}
=
\arg\max_m\pi_m.
}
$$

因此 Topology branch 为：

$$
\boxed{
R_m
\rightarrow
MeanPool
\rightarrow
MLP_{score}
\rightarrow
\pi_m.
}
$$

---


## 13.1 Topology Head 的精确实现

先对 waypoint 维度做无参数 mean pooling：

$$
z_m=MeanPool_H(R_m)\in\mathbb R^{B\times D}.
$$

`MLP_score` 固定为 pointwise candidate scorer：

$$
\boxed{
LN(D)\rightarrow Linear(D,256)\rightarrow SiLU\rightarrow Linear(256,256)\rightarrow SiLU\rightarrow Linear(256,1).
}
$$

因此每条 candidate 只产生一个 logit $l_m$。对无效 candidate，在 softmax **之前**执行：

$$
l_m=-\infty.
$$

再得到：

$$
\pi=Softmax(l,dim=M).
$$

若一个 batch row 完全没有有效 candidate，则该 row 不参与 $L_{topo}$，并在最终输出处直接退化为 coarse trajectory；实现不能让全 $-\infty$ softmax 产生 NaN 后继续传播。

---

# 14. Progress Head

这里直接复用已经计算好的：

$$
R_m.
$$

训练阶段根据 GT trajectory 与所有 Skeleton 的 nDTW 得到：

$$
\boxed{
m^*
=
\arg\min_m nDTW(P_0,S_m)
}
$$

并使用：

$$
\boxed{
R^{use}
=
R_{m^*}
\qquad
\text{training}
}
$$

推理阶段则使用：

$$
\boxed{
R^{use}
=
R_{m_{sel}}
\qquad
\text{inference}.
}
$$

这避免训练初期 topology 选错导致 Progress Head 一直学习错误 Skeleton。这个训练/推理区分也是最新方案中的明确设计。

然后仅做一个 task-specific projection：

$$
\boxed{
H^{prog}
=
MLP_{prog}(R^{use})
}
$$

得到：

$$
\boxed{
H^{prog}
=
[h_1^{prog},\ldots,h_H^{prog}]
\in
\mathbb R^{B\times H\times D}.
}
$$

这里不再存在：

$$
CrossAttention(H^{traj},H^S,H^S).
$$

---


## 14.1 Progress feature 与输出头的精确结构

训练时取 $R^{use}=R_{m^*}$，推理时取 $R^{use}=R_{m_{sel}}$。之后不再访问 $H_m^S$。

任务投影 `MLP_prog`：

$$
\boxed{
MLP_{prog}=LN(D)\rightarrow Linear(D,256)\rightarrow SiLU\rightarrow Linear(256,D).
}
$$

逐 waypoint 得到：

$$
H^{prog}=MLP_{prog}(R^{use}).
$$

`Head_prog` 对每个 waypoint 共享参数：

$$
\boxed{
Head_{prog}=Linear(D,256)\rightarrow SiLU\rightarrow Linear(256,1).
}
$$

得到 $u\in\mathbb R^{B\times H}$。只使用前 $H-1$ 个 logit 构造 gap：

$$
\tilde u=u_{:,1:H-1},
$$

最后一个 token 的 scalar 输出不参与 gap；也可以在实现中直接仅对前 $H-1$ 个 feature 调用 `Head_prog`，两者数值定义等价。第一版取 $\epsilon=10^{-4}$。

这里的 `MLP_prog` 和 `Head_prog` 都是 token-wise MLP，不做 temporal convolution、RNN 或额外 self-attention。waypoint 之间的上下文已经包含在 $R^{use}$ 中。

---

# 15. 单调 Progress

Progress Head 对 $H-1$ 个 interval 输出：

$$
u_i
=
Head_{prog}(h_i^{prog}),
\qquad i=1,\ldots,H-1.
$$

正值化：

$$
\boxed{
w_i
=
softplus(u_i)+\epsilon
}
$$

归一化：

$$
\boxed{
\Delta s_i
=
\frac{w_i}
{\sum_{j=1}^{H-1}w_j}
}
$$

设：

$$
s_1=0
$$

然后：

$$
\boxed{
s_i
=
\sum_{j=1}^{i-1}\Delta s_j,
\qquad i=2,\ldots,H.
}
$$

因此结构上保证：

$$
\boxed{
0=s_1<s_2<\cdots<s_H=1.
}
$$

---

# 16. Skeleton Curve 解码中心

与 $R^{use}$ 对应的 Skeleton Curve 也同步选择。

训练：

$$
\boxed{
\Gamma=\Gamma_{m^*}
}
$$

推理：

$$
\boxed{
\Gamma=\Gamma_{m_{sel}}.
}
$$

设：

$$
\Gamma=[q_1,\ldots,q_N].
$$

计算累计弧长：

$$
\ell_1=0
$$

$$
\ell_k
=
\sum_{r=2}^{k}
\|q_r-q_{r-1}\|.
$$

总长：

$$
L_\Gamma=\ell_N.
$$

根据：

$$
s_i
$$

得到目标弧长：

$$
d_i=s_iL_\Gamma.
$$

沿 curve 插值：

$$
\boxed{
c_i
=
\Gamma(s_i)
}
$$

其中：

$$
c_i=(c_{x,i},c_{y,i}).
$$

所以：

$$
\boxed{
H^{prog}
\rightarrow
s_i
\rightarrow
\Gamma(s_i)
\rightarrow
c_i.
}
$$

---

# 17. Ellipse Geometry Query

对每个 center：

$$
c_i
$$

首先进行独立坐标编码：

$$
\boxed{
e_i^c
=
MLP_C(\Phi_{xy}(c_i))
}
$$

和 progress feature 融合：

$$
\boxed{
q_i^E
=
h_i^{prog}
+
e_i^c.
}
$$

因此：

$$
Q^E
=
[q_1^E,\ldots,q_H^E]
\in
\mathbb R^{B\times H\times D}.
$$

---


## 17.1 Center embedding 的精确实现

`MLP_C` 与其它独立坐标投影结构一致但参数独立：

$$
MLP_C=Linear(D,256)\rightarrow SiLU\rightarrow Linear(256,D).
$$

完整 query 构造顺序固定为：

```text
center c_i [2]
   │
shared SpatialPE
   │ [D]
MLP_C
   │ [D]
   └──── + h_i^prog
             │
             ▼
            q_i^E
```

不将 $c_i$ 与 $h_i^{prog}$ 先 concat 后再投影；也不把 $H^{traj}$ 再次直接拼入 Ellipse Query。trajectory 信息已经通过 $R^{use}\rightarrow H^{prog}$ 进入该分支。

---

# 18. Center-biased Fine Geometry Attention

Ellipse branch 不是普通全图查询，而是围绕：

$$
c_i
$$

对 Fine Geometry Memory 进行空间加权。

Fine Geometry token $j$ 的坐标：

$$
q_j^E.
$$

距离：

$$
\boxed{
d_{ij}^2
=
\|c_i-q_j^E\|_2^2.
}
$$

构造 Gaussian spatial bias：

$$
\boxed{
B_{ij}^{geo}
=
Clamp
\left(
-
\bar\alpha_t
\frac{
\|c_i-q_j^E\|^2
}{
2\sigma_{geo}^2
},
-b_{clip},
0
\right).
}
$$

Attention score：

$$
\boxed{
Score_{ij}
=
\frac{
Q_i^E(K_j^E)^T
}{
\sqrt{d_h}
}
+
B_{ij}^{geo}.
}
$$

然后：

$$
\alpha_{ij}
=
Softmax_j(Score_{ij})
$$

并：

$$
a_i^E
=
\sum_j
\alpha_{ij}V_j^E.
$$

因此：

$$
\boxed{
A^E
=
CenterBiasedCrossAttention
(
AdaLN(Q^E,h_t),
C_E,
B^{geo}
)
}
$$

输出：

$$
A^E
\in
\mathbb R^{B\times H\times D}.
$$

残差得到：

$$
\boxed{
H^{ell}
=
Q^E+A^E
}
$$

其中：

$$
H^{ell}
=
[h_1^{ell},\ldots,h_H^{ell}].
$$

它表示：

> 以 Skeleton center 为中心、经过空间距离加权得到的局部自由空间特征。

---


## 18.1 Center-biased Attention 的精确实现

令：

$$
\tilde Q^E=AdaLN_{geo}(Q^E,h_t).
$$

attention projection 为：

$$
Q=Linear_Q(\tilde Q^E),
$$

$$
[K,V]=Linear_{KV}(C_E).
$$

$C_E$ 含 $32\times32=1024$ 个 fine geometry tokens。4 个 head 共用同一张空间 bias，bias 在 head 维广播：

$$
B^{geo}\in\mathbb R^{B\times1\times H\times1024}.
$$

代码中 diffusion 强度必须使用：

$$
\boxed{\bar\alpha_t=(\sqrt{\bar\alpha_t})^2=ab^2,}
$$

即：

$$
B_{ij}^{geo}
=Clamp\left(-ab_t^2\frac{\|c_i-q_j^E\|^2}{2\sigma_{geo}^2},-8,0\right).
$$

然后：

$$
A^E=Concat_h\left(Softmax\left(\frac{Q^h(K^h)^T}{\sqrt{d_h}}+B^{geo}\right)V^h\right)W_O,
$$

$$
\boxed{H^{ell}=Q^E+A^E.}
$$

这一局部几何模块第一版**到此结束**：不再追加第二次 geometry attention，也不在 $H^{ell}$ 后额外增加 Transformer FFN。后续非线性变换由 Ellipse Shape Head 完成。若 fine geometry memory 不可用，定义 $A^E=0$，因此 $H^{ell}=Q^E$，保证 forward 可退化执行。

---

# 19. Ellipse Shape Head

这一部分恢复稳定版 V2 的 structural mapping。

对于：

$$
h_i^{ell}
$$

Head 输出：

$$
\boxed{
[l_{1,i},l_{2,i},u_i,v_i]
=
Head_E(h_i^{ell})
}
$$

前两个是 raw log-axis value。

定义：

$$
\boxed{
\log a_i
=
\max(l_{1,i},l_{2,i})
}
$$

$$
\boxed{
\log b_i
=
\min(l_{1,i},l_{2,i})
}
$$

所以：

$$
\boxed{
\log a_i\ge\log b_i.
}
$$

物理半轴：

$$
\boxed{
a_i=e^{\log a_i}
}
$$

$$
\boxed{
b_i=e^{\log b_i}
}
$$

因此：

$$
\boxed{
a_i\ge b_i>0.
}
$$

这与稳定版 `raw_to_shape4()` 的 `maximum/minimum` 逻辑一致。

方向首先归一化：

$$
r_i
=
\sqrt{u_i^2+v_i^2+\epsilon}
$$

$$
\hat u_i=\frac{u_i}{r_i},
\qquad
\hat v_i=\frac{v_i}{r_i}.
$$

定义：

$$
\boxed{
\hat u_i=\cos2\theta_i,
\qquad
\hat v_i=\sin2\theta_i
}
$$

所以：

$$
\boxed{
\theta_i
=
\frac12
atan2(\hat v_i,\hat u_i).
}
$$

若：

$$
u_i=v_i=0
$$

则 fallback：

$$
(\hat u_i,\hat v_i)=(1,0).
$$

最终 shape：

$$
\boxed{
shape_i=
[
\log a_i,
\log b_i,
\cos2\theta_i,
\sin2\theta_i
].
}
$$

物理椭圆：

$$
\boxed{
E_i=
(c_i,a_i,b_i,\theta_i).
}
$$

其中 center：

$$
\boxed{
c_i=\Gamma(s_i)
}
$$

始终由 Skeleton geometry 决定，不由 Ellipse Head 回归。

---


## 19.1 Ellipse Shape Head 的精确网络结构

首先对 $H^{ell}$ 做一次 timestep-conditioned normalization：

$$
\tilde H^{ell}=AdaLN_{ell}(H^{ell},h_t).
$$

随后使用逐 waypoint MLP：

$$
\boxed{
Head_E:
D\rightarrow256\rightarrow256\rightarrow4
}
$$

具体为：

```text
AdaLN(H_ell,h_t)
      │
Linear(D,256)
      │
    SiLU
      │
Linear(256,256)
      │
    SiLU
      │
Linear(256,4)
      │
[l1,l2,u,v]
```

四个 raw output 不使用 sigmoid/tanh。轴参数只做排序：

$$
\log a=\max(l_1,l_2),\qquad \log b=\min(l_1,l_2).
$$

shape supervision 使用未裁剪的 $\log a,\log b$。只有在将其转换成物理半轴用于 rasterization / safety calculation 时，为防止 `exp` 数值溢出允许执行：

$$
a=\exp(Clamp(\log a,-8,8)),\qquad b=\exp(Clamp(\log b,-8,8)).
$$

该 clamp 是数值保护，不改变 `shape4` 参数定义。

方向输出先判断：

$$
u^2+v^2>10^{-12}.
$$

正常时归一化为单位向量；退化的 $(0,0)$ 使用常量 $(1,0)$ 替代，再计算 $\theta=\frac12 atan2(v,u)$。该 fallback 必须采用 `where` 类无 NaN 分支实现，不能直接对 $(0,0)$ 调 `atan2` 后再事后替换。

Ellipse Head **没有 center 输出层**。其唯一网络输出就是 4 维 shape raw parameters。

---

# 20. Ellipse 标签生成

Ellipse Head 采用参数监督，因此训练数据中为每个 waypoint 提供完整 ellipse target：

$$
\boxed{
E_i^*
=
(c_i^*,shape_i^*)
}
$$

其中：

$$
\boxed{
shape_i^*
=
[
\log a_i^*,
\log b_i^*,
\cos2\theta_i^*,
\sin2\theta_i^*
].
}
$$

标签生成分成 **GT center 生成** 和 **GT shape 生成** 两部分。标签只在训练数据预处理阶段生成，不参与推理。

## 20.1 GT Skeleton 与 GT progress

对于一条 GT clean trajectory：

$$
P_0=[p_1^0,\ldots,p_H^0],
$$

首先在所有 candidate Skeleton 中选择与 GT trajectory 最匹配的一条：

$$
\boxed{
m^*
=
\arg\min_m nDTW(P_0,S_m).
}
$$

对应完整 Skeleton Curve：

$$
\Gamma^*=\Gamma_{m^*}.
$$

将每个 GT waypoint $p_i^0$ 投影到 $\Gamma^*$ 的各线段上，取最近投影点对应的归一化弧长，得到原始 progress：

$$
\tilde s_i^*\in[0,1].
$$

对 $\{\tilde s_i^*\}$ 做单调投影，并固定：

$$
\boxed{
s_1^*=0,
\qquad
s_H^*=1,
\qquad
s_{i+1}^*\ge s_i^*.
}
$$

得到最终 GT progress：

$$
S^*=[s_1^*,\ldots,s_H^*].
$$

## 20.2 GT ellipse center

center 标签直接由 GT progress 在 GT Skeleton Curve 上解码：

$$
\boxed{
c_i^*
=
\Gamma^*(s_i^*)
}
$$

因此：

$$
\boxed{
C^*
=
[c_1^*,\ldots,c_H^*]
\in
\mathbb R^{H\times2}.
}
$$

这与网络预测 center 的定义完全一致：

$$
c_i=\Gamma(s_i).
$$

因此 center supervision 不再监督一个额外的 center head，而是直接监督 Progress Head 产生的几何中心。

## 20.3 Skeleton-centered ellipse shape lookup table

因为所有 ellipse center 都位于 Skeleton Curve 上，shape label 不需要对每条 trajectory 重复求解。对每张地图的 Skeleton dense point：

$$
q_k\in Skeleton
$$

离线生成一次局部安全椭圆：

$$
\boxed{
E_k^{map}
=
(q_k,a_k^*,b_k^*,\theta_k^*)
}
$$

并缓存：

$$
\boxed{
Y_k
=
[
\log a_k^*,
\log b_k^*,
\cos2\theta_k^*,
\sin2\theta_k^*
].
}
$$

shape label 由 occupancy map 直接生成，不依赖 GT trajectory。具体步骤如下。

首先对 occupancy map 做与安全 margin 一致的障碍膨胀，得到安全占据图 $O_{safe}$。以 $q_k$ 为固定中心，在：

$$
\theta_j=\frac{j\pi}{N_\theta},
\qquad
j=0,\ldots,N_\theta-1
$$

上枚举 candidate orientation。

对每个 $\theta_j$，沿主轴正反方向和其垂直方向进行 grid ray traversal，得到到最近障碍或地图边界的距离：

$$
d_{j,+},\quad d_{j,-},\quad
 d_{j,\perp+},\quad d_{j,\perp-}.
$$

构造初始两个半轴：

$$
\boxed{
a_j^{(0)}
=
\min(d_{j,+},d_{j,-},R_{local})
}
$$

$$
\boxed{
b_j^{(0)}
=
\min(d_{j,\perp+},d_{j,\perp-},R_{local})
}
$$

其中 $R_{local}$ 限制 ellipse 只表示 center 周围的局部安全空间。

随后对完整 ellipse boundary / interior 在 $O_{safe}$ 上进行碰撞检查。若候选 ellipse 发生碰撞，则统一缩放：

$$
a_j=\lambda_j a_j^{(0)},
\qquad
b_j=\lambda_j b_j^{(0)},
\qquad
0<\lambda_j\le1,
$$

并通过二分搜索得到该 orientation 下最大的安全 $\lambda_j$。

最后在所有 candidate orientation 中选择面积最大的安全 ellipse：

$$
\boxed{
j^*
=
\arg\max_j a_jb_j.
}
$$

令：

$$
\boxed{
a_k^*=\max(a_{j^*},b_{j^*}),
\qquad
b_k^*=\min(a_{j^*},b_{j^*})
}
$$

并相应调整 $\theta_k^*$，从而保证：

$$
\boxed{a_k^*\ge b_k^*>0.}
$$

最终每张地图保存：

$$
\boxed{
ShapeTable(q_k)
=
[
\log a_k^*,
\log b_k^*,
\cos2\theta_k^*,
\sin2\theta_k^*
]
}
$$

以及有效标记：

$$
Valid(q_k)\in\{0,1\}.
$$

## 20.4 为每个训练 waypoint 读取 shape label

对于 GT center：

$$
c_i^*=\Gamma^*(s_i^*),
$$

在同一条 Skeleton Curve 上找到距离 $c_i^*$ 最近的 dense Skeleton point：

$$
\boxed{
k_i^*
=
\arg\min_k\|q_k-c_i^*\|_2.
}
$$

直接读取：

$$
\boxed{
shape_i^*=ShapeTable(q_{k_i^*}).
}
$$

第一版使用 nearest lookup，不对 $\theta$ 做额外插值，避免角度周期问题。

于是每个 waypoint 最终拥有完整参数标签：

$$
\boxed{
E_i^*
=
[
c_{x,i}^*,c_{y,i}^*,
\log a_i^*,\log b_i^*,
\cos2\theta_i^*,\sin2\theta_i^*
].
}
$$

## 20.5 GT ellipse mask

为了复用稳定版 soft-IoU supervision，根据：

$$
(c_i^*,a_i^*,b_i^*,\theta_i^*)
$$

使用与训练 loss 完全相同的 soft ellipse rasterizer 生成：

$$
\boxed{
M_i^*
\in
[0,1]^{R_e\times R_e}.
}
$$

训练数据最终额外保存：

$$
\boxed{
C^*,\qquad Shape^*,\qquad M^*,\qquad ShapeValid.
}
$$

其中 `ShapeValid` 用于屏蔽无法得到可靠安全椭圆的少数位置。

---


## 20.6 标签生成的 baseline 实现参数

为避免离线标签脚本出现不同几何定义，第一版固定采用以下默认值，后续可作为数据侧 ablation 调整：

| 参数 | baseline |
|---|---:|
| orientation 数 $N_\theta$ | 36 |
| orientation 范围 | $[0,\pi)$ |
| local radius 上限 $R_{local}$ | 0.25 scene unit |
| occupancy dilation | 1 cell |
| ellipse boundary check points | 128 |
| scale binary-search iterations | 12 |
| minimum valid semi-axis | $10^{-3}$ |
| GT mask resolution $R_e$ | 64 |
| soft-mask temperature | 10 |

离线生成时，`ShapeTable`、`Valid`、`C*`、`Shape*`、`M*` 的坐标系必须与训练网络一致，统一使用 scene-normalized $[-1,1]^2$。任何 world/obs frame 数据在写入训练数据前必须完成转换，训练阶段不再临时做坐标系猜测。

---

# 21. 最终 Feature Fusion

每个 waypoint 具有：

$$
h_i^{traj}
$$

当前 trajectory latent；

$$
h_i^{prog}
$$

融合 trajectory 与 selected Skeleton 后得到的 correspondence feature；

$$
h_i^{ell}
$$

局部安全空间 feature。

拼接：

$$
\boxed{
Z_i
=
[
h_i^{traj};
h_i^{prog};
h_i^{ell}
]
\in
\mathbb R^{3D}.
}
$$

经过：

$$
\boxed{
F_i
=
MLP_{fuse}(Z_i)
}
$$

得到：

$$
F
\in
\mathbb R^{B\times H\times D}.
$$

---


## 21.1 Fusion MLP 的精确结构

对三个 waypoint-level latent 直接沿 channel 维拼接：

$$
Z=Concat(H^{traj},H^{prog},H^{ell})\in\mathbb R^{B\times H\times3D}.
$$

`MLP_fuse` 固定为：

$$
\boxed{
LayerNorm(3D)\rightarrow Linear(3D,2D)\rightarrow GELU\rightarrow Linear(2D,D).
}
$$

在 baseline 中：

```text
384 → 256 → 128
```

该模块只做 channel fusion，不做 self-attention，不重新查询 Skeleton，也不重新查询 Fine Geometry Memory。输出 $F$ 的 waypoint 数保持 $H=128$ 不变。

---

# 22. Final Clean-Trajectory Denoiser

$F$ 经过 $N_F$ 层：

$$
\boxed{
SelfAttention
\rightarrow
GlobalMapCrossAttention
\rightarrow
FFN.
}
$$

Global map：

$$
Q=F,
\qquad
K=V=C_G.
$$

每层使用：

$$
h_t
$$

进行 AdaLN conditioning。

输出：

$$
\boxed{
H^{clean}
=
[h_1^{clean},\ldots,h_H^{clean}]
\in
\mathbb R^{B\times H\times D}.
}
$$

---


## 22.1 Final Denoiser 的精确结构

Final Denoiser 共 $N_F=3$ 个 block，结构与 Trajectory Backbone 的 `TrajBlock` 相同，但参数**完全独立**。第 $r$ 层：

$$
Y_1=Y+SA_r^{final}(AdaLN_{r,1}^{final}(Y,h_t)),
$$

$$
Y_2=Y_1+CA_r^{final}(AdaLN_{r,2}^{final}(Y_1,h_t),C_G),
$$

$$
\boxed{
Y'=Y_2+FFN_r^{final}(AdaLN_{r,3}^{final}(Y_2,h_t)).
}
$$

Self-Attention 同样使用 4 heads 和独立的 learnable horizon-distance bias；Global Cross-Attention 仍只访问 $C_G$；FFN 为 $128\rightarrow512\rightarrow128$，激活为 GELU。

输入：

$$
Y^{(0)}=F.
$$

输出：

$$
\boxed{H^{clean}=Y^{(3)}.}
$$

Final Denoiser 不再显式接收 $C_E$ 或 Skeleton tokens，因为这两类信息已经通过 $H^{prog}$、$H^{ell}$ 注入 $F$。

---

# 23. Final Clean Trajectory

和 coarse branch 共享：

$$
Head_P.
$$

所以：

$$
\boxed{
\tilde P_0^t
=
Head_P(H^{traj})
}
$$

以及：

$$
\boxed{
\hat P_0^t
=
Head_P(H^{clean}).
}
$$

最终：

$$
\hat p_1^0=p_s,
\qquad
\hat p_H^0=p_g.
$$

只有：

$$
\boxed{
\hat P_0^t
}
$$

进入 DDIM。

---


## 23.1 Final decode 与 endpoint conditioning

Final decode 与 coarse decode 使用同一参数：

$$
\hat P_0^t=Head_P(H^{clean}),\qquad \tilde P_0^t=Head_P(H^{traj}).
$$

随后分别执行同一个 endpoint overwrite 函数：

```text
p[:,0]  = start
p[:,-1] = goal
```

起终点硬约束发生在 head 输出之后，因此 trajectory loss 只计算内部 waypoint $2\ldots H-1$。DDIM 也只接收 hard-overwrite 后的 $\hat P_0^t$。

---

# 24. Reverse Diffusion

最终单步链路为：

$$
\boxed{
P_t
\rightarrow
H^{traj}
\rightarrow
\{R_m\}
\rightarrow
R^{use}
\rightarrow
H^{prog}
\rightarrow
s
\rightarrow
c
\rightarrow
H^{ell}
\rightarrow
H^{clean}
\rightarrow
\hat P_0^t
\rightarrow
P_{t-1}.
}
$$

其中：

$$
R^{use}
=
\begin{cases}
R_{m^*}, & training\\
R_{m_{sel}}, & inference
\end{cases}
$$

对应 Skeleton Curve：

$$
\Gamma^{use}
=
\begin{cases}
\Gamma_{m^*}, & training\\
\Gamma_{m_{sel}}, & inference.
\end{cases}
$$

---

# 25. Loss

当前版本不再使用 $L_{gap}$、$L_{area}$、$L_{ratio}$、$L_{inside}$ 这类间接正则来塑造 ellipse。Ellipse 已恢复完整参数标签，因此直接复用稳定版的参数回归 + soft IoU + safety 监督。

## 25.1 Final trajectory loss

最终 clean trajectory：

$$
\boxed{
L_{traj}
=
MSE(
\hat P_{0,2:H-1},
P_{0,2:H-1}
)
}
$$

起点和终点已经 hard overwrite，因此不参与该项回归。

## 25.2 Coarse trajectory loss

辅助 coarse prediction：

$$
\boxed{
L_{coarse}
=
MSE(
\tilde P_{0,2:H-1},
P_{0,2:H-1}
)
}
$$

该项直接监督 $H^{traj}$ 具有 clean trajectory 表达能力。

## 25.3 Trajectory smoothness

保留稳定版 trajectory smoothness：

$$
\boxed{
L_{smooth}
=
\lambda_{acc}L_{acc}
+
\lambda_{jerk}L_{jerk}.
}
$$

其中 $L_{acc}$ 约束二阶差分，$L_{jerk}$ 约束三阶差分，用于抑制高频折线和局部抖动。

## 25.4 Topology loss

GT topology：

$$
\boxed{
m^*
=
\arg\min_m nDTW(P_0,S_m).
}
$$

Selector supervision：

$$
\boxed{
L_{topo}
=
CE(\pi,m^*).
}
$$

## 25.5 Ellipse parameter regression

网络预测 center：

$$
c_i=\Gamma(s_i),
$$

GT center：

$$
c_i^*=\Gamma^*(s_i^*).
$$

center 参数监督定义为：

$$
\boxed{
L_{center}
=
\frac1H\sum_{i=1}^{H}
\|c_i-c_i^*\|_2^2.
}
$$

由于 $c_i$ 由 progress 生成，因此：

$$
\boxed{
L_{center}
\rightarrow
c_i
\rightarrow
s_i
\rightarrow
ProgressHead
}
$$

直接形成 Progress Head 的主要几何监督，不再额外使用原来的 $L_{align}$。

Ellipse Shape Head 输出：

$$
shape_i
=
[
\log a_i,
\log b_i,
\cos2\theta_i,
\sin2\theta_i
].
$$

对应 GT：

$$
shape_i^*
=
[
\log a_i^*,
\log b_i^*,
\cos2\theta_i^*,
\sin2\theta_i^*
].
$$

shape 参数监督：

$$
\boxed{
L_{shape}
=
\frac1{N_{valid}}
\sum_{i\in\mathcal I_{valid}}
\|shape_i-shape_i^*\|_2^2.
}
$$

其中 $\mathcal I_{valid}$ 为具有有效 ellipse shape label 的 waypoint 集合。

沿用稳定版 2 个 center 参数 + 4 个 shape 参数的平均方式：

$$
\boxed{
L_E
=
\frac{2L_{center}+4L_{shape}}{6}.
}
$$

因此 $L_E$ 同时监督：

$$
\boxed{
(c_x,c_y,\log a,\log b,\cos2\theta,\sin2\theta).
}
$$

## 25.6 Ellipse soft-IoU loss

根据预测：

$$
(c_i,a_i,b_i,\theta_i)
$$

使用 soft ellipse rasterizer 得到预测 mask：

$$
\hat M_i.
$$

离线标签提供：

$$
M_i^*.
$$

采用稳定版 fuzzy soft-IoU：

$$
\boxed{
L_{iou}
=
\frac1{N_{valid}}
\sum_{i\in\mathcal I_{valid}}
\left(
1-
\frac{
\sum_x\min(\hat M_i(x),M_i^*(x))+\epsilon
}{
\sum_x\max(\hat M_i(x),M_i^*(x))+\epsilon
}
\right).
}
$$

$L_E$ 负责参数空间的直接回归，$L_{iou}$ 负责监督完整 ellipse 几何形状，两者互补。

## 25.7 Ellipse safety loss

保留稳定版整椭圆 occupancy safety loss。

对第 $i$ 个 ellipse，定义 unsafe fraction：

$$
u_i
=
1-
\frac{
\text{predicted ellipse soft area inside free space}
}{
\text{complete theoretical ellipse soft area}
}.
$$

然后：

$$
\boxed{
L_{safe}
=
Mean(u)
+
\lambda_{CVaR}CVaR(u).
}
$$

其中 CVaR 关注一条 trajectory 中最危险的一部分 ellipse anchor，防止平均值掩盖局部严重碰撞。

## 25.8 Overall loss

最终训练目标为：

$$
\boxed{
\begin{aligned}
L={}&
\lambda_{traj}L_{traj}
+\lambda_{coarse}L_{coarse}
+\lambda_{smooth}L_{smooth}\\
&+\lambda_{topo}L_{topo}
+\lambda_E L_E
+\lambda_{iou}L_{iou}
+\lambda_{safe}L_{safe}.
\end{aligned}
}
$$

其中 ellipse 部分完整写为：

$$
\boxed{
L_{ellipse}
=
\lambda_E
\frac{2L_{center}+4L_{shape}}{6}
+
\lambda_{iou}L_{iou}
+
\lambda_{safe}L_{safe}.
}
$$

Ellipse 参数监督恢复以后，不再需要依靠 $L_{area}$ 与 $L_{ratio}$ 互相拉扯来间接决定椭圆大小，也不再使用 $L_{gap}$ 强行平滑 progress。

初始实验可直接以稳定版 ellipse 权重作为 baseline：

$$
\lambda_E=1.0,
\qquad
\lambda_{iou}=0.25,
\qquad
\lambda_{safe}=0.1,
$$

其余主任务权重继续按当前配置做单独消融和调整。

---

# 26. 最终结构图

```text
训练标签（offline）：

GT trajectory P0 + Candidate Skeletons
        │
        ├─ nDTW → m* → Γ*
        ├─ GT waypoint 投影到 Γ* → s* → c*=Γ*(s*)
        └─ Skeleton dense point + occupancy → safe ellipse ShapeTable
                                      │
                                      └─ lookup(c*) → shape* → GT ellipse mask


推理/训练共同网络：

Occupancy Map
      │
      ▼
   SceneCNN
   ├──────────── Global F_G ── + Φxy(grid) ──→ C_G
   │
   └──────────── Fine F_E ──── + Φxy(grid) ──→ C_E


P_t
 │
 ├─ Φxy(p_i)
 ├─ MLP_T
 └─ + PE_1D(i)
        │
        ▼
 Trajectory Encoder
        │
        ▼
 Trajectory Backbone
   ↑             ↑
  C_G           h_t
        │
        ▼
      H_traj
      │     \
      │      └── Head_P ──→ coarse P0 ──→ L_coarse
      │
      ▼


Candidate Skeleton S_m
 │
 ├─ Φxy
 ├─ MLP_S
 └─ + PE_1D(j)
        │
        ▼
 Skeleton Transformer
        │
        ▼
      H_m^S


对每个 candidate m：

H_traj ───────────────┐
   │                   │
   ▼                   │
CrossAttention ← H_m^S│
   │                   │
   ▼                   │
  A_m                  │
   └──── + H_traj ─────┘
            │
            ▼
           U_m
            │
      LN → FFN_match
            │
       residual +
            │
            ▼
           R_m
         /     \
        /       \
       ▼         ▼
 MeanPool     保存 R_m
    │
MLP_score
    │
   l_m


所有 l_m
    │
Masked Softmax
    │
    ▼
m_selected


训练：
m* ───────┐
           ▼
        R_m*

推理：
m_selected ─┐
            ▼
        R_selected

             │
             ▼
          MLP_prog
             │
             ▼
           H_prog
             │
             ▼
          Head_prog
             │
             ▼
       positive gaps
             │
             ▼
         progress s
             │
             ▼
     Skeleton Curve Γ
             │
             ▼
         c_i = Γ(s_i)
             │
       Φxy(c_i) → MLP_C
             │
          + H_prog
             │
             ▼
            Q_E
             │
     center-based bias
             │
             ▼
Center-biased CrossAttention ← C_E
             │
             ▼
           H_ell
             │
             ├── Ellipse Head
             │      │
             │      ▼
             │ [l1,l2,u,v]
             │      │
             │      ├─ max/min → log a, log b
             │      └─ normalize → cos2θ,sin2θ
             │      │
             │      ├────────────→ shape* ──→ L_shape
             │      └────────────→ GT mask ─→ L_iou
             │
             ├── c_i ────────────→ c_i* ────→ L_center
             │
             └── predicted ellipse + occupancy ─────→ L_safe
             │
             ▼

H_traj
H_prog
H_ell
   │
   ▼
 concat
   │
   ▼
MLP_fuse
   │
   ▼
   F
   │
   ▼
Final Denoiser
 ↑          ↑
C_G        h_t
   │
   ▼
H_clean
   │
   ▼
shared Head_P
   │
   ▼
P0_hat
   │
   ▼
 DDIM
   │
   ▼
P_{t-1}
```

现在 Skeleton 部分已经真正简化成：

$$
\boxed{
\text{一次共享 Trajectory-Skeleton MatchBlock}
\rightarrow
\begin{cases}
Topology\ Head\\
Progress\ Head
\end{cases}
}
$$

而 Ellipse 参数恢复成稳定版：

$$
\boxed{
[l_1,l_2,u,v]
\rightarrow
[\log a,\log b,\cos2\theta,\sin2\theta].
}
$$

本版同时补全了训练监督：离线生成与 Skeleton center 一致的完整 ellipse 参数标签，训练阶段直接恢复稳定版的 $L_E+L_{iou}+L_{safe}$。其中 $L_{center}$ 直接监督 $c_i=\Gamma(s_i)$，因此 Progress Head 不再依赖额外的 $L_{align}$ 或 $L_{gap}$。

---

# 27. DSH 实现契约：模块接口与张量形状

这一节用于约束实际代码实现。只要接口和内部结构满足本节，就与本报告网络一致；如果额外增加 token、attention、head、手工特征或改变归一化位置，应单独记录为 architecture change。

## 27.1 模块接口

| 模块 | 输入 | 输出 |
|---|---|---|
| `SceneCNN` | `occ [B,1,256,256]` | `global_map [B,256,D]`, `geo_mem [B,1024,D]` |
| `TrajectoryEncoder` | `p_t [B,H,2]` | `T0 [B,H,D]` |
| `TrajectoryBackbone` | `T0, C_G, h_t` | `H_traj [B,H,D]` |
| `Head_P` | `[B,H,D]` | `[B,H,2]` |
| `SkeletonEncoder` | `candidate_xy [B,M,L,2]` | `H_S [B,M,L,D]` |
| `MatchBlock` | `H_traj, H_S, h_t` | `R [B,M,H,D]` |
| `TopologyHead` | `R, candidate_mask` | `logits/pi [B,M]` |
| `ProgressHead` | `R_use [B,H,D]` | `H_prog [B,H,D]`, `s [B,H]` |
| `CurveDecoder` | `Gamma_use, s` | `center [B,H,2]` |
| `EllipseGeometry` | `H_prog, center, C_E, h_t, ab` | `H_ell [B,H,D]` |
| `EllipseShapeHead` | `H_ell,h_t` | `shape4 [B,H,4]`, `a,b,theta` |
| `FusionMLP` | `H_traj,H_prog,H_ell` | `F [B,H,D]` |
| `FinalDenoiser` | `F,C_G,h_t` | `H_clean [B,H,D]` |
| shared `Head_P` | `H_clean` | `P0_hat [B,H,2]` |

## 27.2 单步 forward 的标准伪代码

```python
# scene
C_G, C_E = scene_encoder(occ)
h_t = time_embedding(t)

# trajectory backbone
T0 = MLP_T(SpatialPE(p_t)) + IndexPE(plan_idx)
H_traj = trajectory_backbone(T0, C_G, h_t)
coarse = hard_endpoints(Head_P(H_traj), start, goal)

# skeleton static encoder
H_S = skeleton_encoder(candidate_xy)              # [B,M,L,D]

# one shared trajectory-skeleton matching
R = match_block(H_traj, H_S, h_t)                 # [B,M,H,D]
pi = topology_head(R, candidate_mask)              # [B,M]

if training:
    idx = topology_gt_m_star
else:
    idx = pi.argmax(-1)

R_use = gather_candidate(R, idx)                   # [B,H,D]
Gamma_use = gather_candidate(candidate_curve, idx)

# progress -> geometric centre
H_prog = MLP_prog(R_use)
s = monotone_progress(Head_prog(H_prog))
center = curve_arclength_interpolate(Gamma_use, s)

# local geometry -> ellipse shape
center_feat = MLP_C(SpatialPE(center))
Q_E = H_prog + center_feat
H_ell = center_biased_geo_attention(Q_E, C_E, center, h_t, ab)
shape4 = ellipse_shape_head(H_ell, h_t)

# final trajectory
F = MLP_fuse(concat(H_traj, H_prog, H_ell, dim=-1))
H_clean = final_denoiser(F, C_G, h_t)
P0_hat = hard_endpoints(Head_P(H_clean), start, goal)
```

## 27.3 必须保持一致的结构约束

| 项目 | 本报告定义 |
|---|---|
| diffusion state | 只有 trajectory $P_t$ |
| Skeleton–trajectory fusion | 每个 candidate 只做一次 Match Cross-Attention |
| Progress | 直接从 selected $R_m$ 预测，不再第二次 Cross-Attention |
| Skeleton Curve | 纯几何数据，只用于 $s\rightarrow c$ |
| Ellipse center | $c=\Gamma(s)$，没有 center head |
| Ellipse shape | 4 维 $[\log a,\log b,\cos2\theta,\sin2\theta]$ |
| Fine geometry | 只在 center-biased Cross-Attention 中查询一次 |
| Fusion | concat 三个 latent 后用 MLP 投影到 $D$ |
| Final Denoiser | 3 个独立 TrajBlock |
| trajectory output head | coarse / final 共享同一个 `Linear(D,2)` |
| start / goal conditioning | hard endpoints，不额外加入 role token |
| type embedding | baseline 不使用 trajectory / ellipse type embedding |

## 27.4 建议的代码级 shape assertions

实现中建议在 debug / unit test 模式加入：

```python
assert H_traj.shape == (B, H, D)
assert H_S.shape == (B, M, L, D)
assert R.shape == (B, M, H, D)
assert pi.shape == (B, M)
assert H_prog.shape == (B, H, D)
assert s.shape == (B, H)
assert center.shape == (B, H, 2)
assert H_ell.shape == (B, H, D)
assert shape4.shape == (B, H, 4)
assert F.shape == (B, H, D)
assert H_clean.shape == (B, H, D)
assert P0_hat.shape == (B, H, 2)
```

并检查 progress 的结构约束：

```python
assert allclose(s[:, 0], 0)
assert allclose(s[:, -1], 1)
assert (s[:, 1:] > s[:, :-1]).all()
```

以及 ellipse 参数约束：

```python
assert (log_a >= log_b).all()
assert isfinite(shape4).all()
```

这组 assertions 的目标不是训练时长期保留，而是让 DSH 在重构过程中第一时间发现 tensor routing、candidate gather、progress 长度或 ellipse parameterization 是否偏离报告。
