# 椭圆中心递推修正与 ALM 安全引导完整实现方案

可以，按你刚才确定的思路，完整实现就固定成这一条链：

```math
\boxed{ \text{Joint Diffusion} \rightarrow \text{椭圆中心递推修正} \rightarrow \text{最终凸区域序列} \rightarrow \text{Trajectory ALM} \rightarrow \text{DDIM} }
```

不引入 CBF，不把椭圆中心当轨迹优化，不新增训练损失。核心就是：**第一个中心锁定起点；后面只有出现不可靠中心时，才投影到前一个最终接受的安全凸区域。**

你当前代码里 `EllipseRegionBuilder` 是直接用

```math
c_k=p_k+\Delta c_k
```

一次性并行构造所有凸区域，而 sampler 随后直接把这些区域交给 ALM。

这个结构要改成下面这样。

---

# 一、最终数学定义

模型在某个 reverse step `t` 输出：

```math
\hat P_0 = \{\hat p_0,\ldots,\hat p_{H-1}\}
```

以及：

```math
\hat E_0 = \{\hat e_0,\ldots,\hat e_{H-1}\}.
```

当前 E6：

```math
\hat e_k= [ \Delta\hat c_{x,k}, \Delta\hat c_{y,k}, \log a_k, \log b_k, \cos2\theta_k, \sin2\theta_k ].
```

所以原始物理中心：

```math
\boxed{ \hat c_k = \hat p_k+\Delta\hat c_k }
```

形状部分统一记为：

```math
\xi_k= (a_k,b_k,\theta_k).
```

我们最终要得到：

```math
\boxed{ c_0^*,c_1^*,\ldots,c_{H-1}^* }
```

和：

```math
\boxed{ \mathcal C_0,\mathcal C_1,\ldots,\mathcal C_{H-1}. }
```

---

# 二、第一个椭圆：强制中心等于起点

定义：

```math
\boxed{ c_0^*=s }
```

其中：

```math
s=p_{\text{start}}.
```

因为你的 trajectory endpoint 本来就是 hard condition：

```math
\hat p_0=s,
```

因此第一个椭圆新的 offset 直接变成：

```math
\boxed{ \Delta c_0^* = c_0^*-\hat p_0 =0. }
```

但是：

```math
a_0,b_0,\theta_0
```

全部保留模型预测。

于是：

```math
E_0^* = [ 0,0, \log a_0,\log b_0, \cos2\theta_0,\sin2\theta_0 ].
```

然后用：

```math
(c_0^*,a_0,b_0,\theta_0,M)
```

构造第一个区域：

```math
\boxed{ \mathcal C_0 = \{x:A_0x\le b_0\}. }
```

这就是整条递推链的安全锚点。

---

# 三、后续中心什么时候需要修？

对于：

```math
k=1,\ldots,H-1
```

先得到 raw center：

```math
\hat c_k.
```

不要因为它不在：

```math
\mathcal C_{k-1}
```

里就修改它。

否则会把正常预测也强行拖进前一区域，容易使整条 ellipse sequence 越来越保守。

真正的接受条件定为：

```math
\boxed{ \operatorname{Free}(\hat c_k)=1 \quad\land\quad \operatorname{ValidRegion}(\hat c_k,\xi_k)=1 }
```

也就是：

1. 中心不是障碍点；
2. 用它生成的凸区域本身有效。

如果两个都满足：

```math
\boxed{ c_k^*=\hat c_k }
```

并直接接受模型原预测区域：

```math
\boxed{ \mathcal C_k=\hat{\mathcal C}_k. }
```

不做任何修正。

---

# 四、中心不安全时，投影到前一个最终安全区域

如果：

```math
\operatorname{Free}(\hat c_k)=0
```

或者这个中心构造区域失败，那么：

```math
\hat c_k
```

不能再作为 seed。

直接求：

```math
\boxed{ c_k^* = \Pi_{\mathcal C_{k-1}}(\hat c_k) }
```

即：

```math
\boxed{ c_k^* = \arg\min_c \frac12 \|c-\hat c_k\|_2^2 }
```

subject to：

```math
\boxed{ A_{k-1}c\le b_{k-1}. }
```

这里就采用**标准欧氏投影**。

不要再引入：

- ellipse-weighted distance；
- ALM；
- CBF；
- 新 loss；
- 人工 correction weight。

中心修复就是一个非常明确的：

```math
\boxed{\text{2D convex projection}}
```

问题。

---

# 五、为什么这里不需要再额外加大的 projection margin

你现在构造区域时本身已经有：

```math
m=0.02
```

的 safety margin。当前实现里 obstacle plane 是：

```math
b=n^To-m.
```

所以：

```math
\mathcal C_{k-1}
```

本身已经是缩过的区域。

再额外做：

```math
A c\le b-\epsilon
```

并加一个明显的几何 margin，会重复收缩。

因此我建议：

```math
\boxed{ \text{直接投影到 }\mathcal C_{k-1} }
```

只保留：

```math
10^{-6}\sim10^{-5}
```

级别的数值 tolerance，不再引入新的 safety hyperparameter。

---

# 六、投影本身不用引入 QP 库

因为这是二维：

```math
c\in\mathbb R^2,
```

而且你的区域最多大约：

```math
M=20
```

个 half-spaces。

可以直接写一个**解析二维 convex-polytope projection**。

对于：

```math
\mathcal C=\{x:Ax\le b\}.
```

给定：

```math
y=\hat c_k.
```

首先如果：

```math
Ay\le b,
```

那么：

```math
\Pi_{\mathcal C}(y)=y.
```

否则 projection 最终一定位于：

- 某条 polygon edge；
- 或某个 vertex。

对于第 `j` 个面：

```math
a_j^Tx=b_j
```

其正交投影：

```math
\boxed{ q_j = y- \frac{a_j^Ty-b_j} {\|a_j\|^2} a_j }
```

如果：

```math
Aq_j\le b,
```

它就是一个合法 candidate。

再计算任意两个非平行边界：

```math
a_i^Tx=b_i, \qquad a_j^Tx=b_j
```

的交点：

```math
q_{ij} = \begin{bmatrix} a_i^T\\ a_j^T \end{bmatrix}^{-1} \begin{bmatrix} b_i\\ b_j \end{bmatrix}.
```

如果：

```math
Aq_{ij}\le b,
```

它也是 candidate。

最后：

```math
\boxed{ c_k^* = \arg\min_{q\in\mathcal Q} \|q-\hat c_k\|^2. }
```

这样：

- 不需要 `scipy.optimize`；
- 不需要 `cvxpy`；
- 不需要 `qpth`；
- 可以纯 Torch；
- 可以直接跑 GPU；
- `M=20` 时计算量非常小。

建议单独写：

```python
project_point_to_polytope_2d(...)
```

---

# 七、投影以后重新生成当前椭圆的凸区域

得到：

```math
c_k^*
```

以后，ellipse shape 不修改：

```math
a_k^*=a_k, \qquad b_k^*=b_k, \qquad \theta_k^*=\theta_k.
```

于是：

```math
\boxed{ E_k^* = (c_k^*,a_k,b_k,\theta_k). }
```

然后重新调用 Neural-IRIS-style builder：

```math
\boxed{ \mathcal C_k = \operatorname{BuildRegion} (c_k^*,a_k,b_k,\theta_k,M). }
```

因为：

```math
c_k^*\in\mathcal C_{k-1}
```

且：

```math
\mathcal C_{k-1}
```

已经是安全区域，所以：

```math
\boxed{ c_k^*\in\mathcal F. }
```

于是这个新 region 不再从一个墙内 seed 开始。

---

# 八、而且修正时自动保证相邻区域有公共点

因为：

```math
c_k^*\in\mathcal C_{k-1},
```

新的凸区域构造又要求：

```math
c_k^*\in\mathcal C_k.
```

因此：

```math
\boxed{ c_k^* \in \mathcal C_{k-1} \cap \mathcal C_k. }
```

也就是：

```math
\boxed{ \mathcal C_{k-1}\cap\mathcal C_k\neq\varnothing. }
```

所以只要发生 center repair，这一步顺便就把：

> “前一个区域和当前区域完全断开”

的问题解决了。

---

# 九、修正中心后新凸区域仍构造失败：保留失败状态，但继续使用最近有效安全区域传播

假设第 `k` 个原始椭圆中心不可靠，已经通过前一个可用于传播的安全区域完成投影：

```math
c_k^*=\Pi_{\mathcal R_{k-1}}(\hat c_k),
```

其中：

```math
\mathcal R_{k-1}
```

表示**截至第 `k-1` 个位置最近一次成功构造并验证通过的安全凸区域**。

因此投影之后必然有：

```math
c_k^*\in\mathcal R_{k-1}.
```

由于：

```math
\mathcal R_{k-1}\subseteq\mathcal F,
```

所以：

```math
\boxed{c_k^*\in\mathcal F.}
```

接下来仍然使用第 `k` 个椭圆自身预测的形状参数：

```math
\xi_k=(a_k,b_k,\theta_k)
```

以及修正后的中心：

```math
c_k^*
```

尝试构造当前椭圆自己的凸区域：

```math
\mathcal C_k=
\operatorname{BuildRegion}(c_k^*,\xi_k).
```

正常情况下，如果：

```math
\operatorname{Valid}(\mathcal C_k)=1,
```

则当前区域构造成功，并令：

```math
\boxed{\mathcal R_k=\mathcal C_k.}
```

这样从下一位置开始，新的中心修正就使用最新生成成功的区域：

```math
c_{k+1}^*
=
\Pi_{\mathcal R_k}(\hat c_{k+1}).
```

---

但是，即使中心已经修正到安全位置，新的凸区域仍可能因为以下原因构造失败：

- 当前 ellipse shape 预测异常；
- `max_faces` 上限不足；
- 局部障碍结构过于复杂；
- 数值退化；
- region validity check 未通过。

此时必须明确区分两个概念。

### 1. 当前椭圆真正生成的区域 `\mathcal C_k`

如果构造失败，则：

```math
\boxed{\operatorname{Valid}(\mathcal C_k)=0.}
```

也就是说：

> 第 `k` 个椭圆没有成功得到自己的有效凸区域。

此时**不能写成**

```math
\mathcal C_k=\mathcal C_{k-1},
```

因为这会把“第 `k` 个区域构造失败”和“第 `k` 个区域等于前一区域”混为一谈。

当前区域应明确保留为：

```text
C_k = invalid
```

并且：

```math
\mathcal C_k
```

不交给 trajectory ALM 使用。

---

### 2. 用于后续中心投影的传播区域 `\mathcal R_k`

虽然：

```math
\mathcal C_k
```

构造失败，但是：

```math
c_k^*\in\mathcal R_{k-1}
```

已经保证第 `k` 个中心是安全的。

因此没有必要终止整个中心修正序列。

直接令：

```math
\boxed{\mathcal R_k=\mathcal R_{k-1}.}
```

也就是说：

> 第 `k` 个椭圆自己的凸区域构造失败时，不伪造一个新的 `C_k`；只是继续保留“最近一个已经验证成功的安全凸区域”，作为下一位置中心投影的约束集合。

于是下一步：

```math
c_{k+1}^*
=
\Pi_{\mathcal R_k}(\hat c_{k+1})
=
\Pi_{\mathcal R_{k-1}}(\hat c_{k+1}).
```

整个递推不会因为一个区域构造失败而中断。

---

因此传播区域统一定义为：

```math
\boxed{
\mathcal R_k=
\begin{cases}
\mathcal C_k, &
\operatorname{Valid}(\mathcal C_k)=1,\\[4pt]
\mathcal R_{k-1}, &
\operatorname{Valid}(\mathcal C_k)=0.
\end{cases}
}
```

其中：

- `\mathcal C_k`：第 `k` 个椭圆**真正生成的凸区域**，供 trajectory ALM 使用；
- `\mathcal R_k`：截至第 `k` 个位置的**最近有效安全区域**，只负责后续 ellipse center projection。

例如：

```text
k=8:
C8 成功
R8 = C8

k=9:
C9 成功
R9 = C9

k=10:
中心先投影到 R9 内
C10 构造失败
C10 = invalid
R10 = R9 = C9

k=11:
用 R10（即 C9）修正 c11
如果 C11 构造成功：
R11 = C11
```

所以这里真正的规则是：

```math
\boxed{
\text{当前区域失败：} \mathcal C_k\text{ 保持 invalid，}
\quad
\text{传播区域继续沿用：} \mathcal R_k=\mathcal R_{k-1}.
}
```

这样既不会中断后续中心修正，也不会把旧区域冒充成第 `k` 个椭圆的新区域。

---
# 十、第一个区域如果构造失败

由于：

```math
c_0^*=s
```

是已知自由点。

但 builder 仍可能因为 face cap / 数值原因返回 invalid。

不能让整个 sequence 从第 0 个就挂掉。

这里设计一个**唯一的特殊 fallback**：

以 start 所在的 free occupancy cell 为基础，构造一个 cell interior box。

例如地图：

```math
256\times256
```

cell size：

```math
\Delta=\frac2{256}.
```

取 start 所在的 free cell：

```math
[x_{\min},x_{\max}] \times [y_{\min},y_{\max}]
```

然后构造：

```math
\boxed{ \mathcal C_0^{fallback} = [x_{\min},x_{\max}] \times [y_{\min},y_{\max}] }
```

对应四个 half-spaces。

它虽然很小，但：

```math
\boxed{ s\in\mathcal C_0^{fallback} }
```

并且在 occupancy-grid 语义下是安全的。

如果连 start 所在 cell 都不是 free：

```math
\boxed{\text{该 sample 的 guidance 直接判 invalid}}
```

不能人为创造安全区域。

---

# 十一、当前 `EllipseRegionBuilder` 怎么重构

你现在 `__call__()` 里面直接：

```python
center_all = p0 + clean_e[..., :2]
```

然后后面的 region construction 全绑定在这个 `center_all` 上。

建议拆成三层。

### 1. 解码 ellipse

```python
decode_ellipse_shape(e0)
```

返回：

```text
axes
theta
Q
```

---

### 2. 显式中心建域

新增：

```python
build_from_centers(
    centers,
    e0,
)
```

其中：

```text
centers: [B,H,2]
e0:      [B,H,6]
```

只使用 `e0[...,2:]` 的 shape 信息。

即：

```math
\boxed{ (c_k,\xi_k) \rightarrow (A_k,b_k,mask_k,valid_k). }
```

这样 region builder 不再自己决定中心是谁。

---

### 3. 保留旧接口

原来的：

```python
__call__(p0, e0)
```

可以保留为：

```python
centers = p0 + e0[..., :2]
return self.build_from_centers(centers, e0)
```

这样其他代码不会全部被打断。

---

# 十二、再新增中心修正模块

建议文件：

```text
src/geometry/ellipse_center_repair.py
```

核心类：

```python
class SafeEllipseCenterRepair:
```

输入：

```text
p0       [B,H,2]
e0       [B,H,6]
start    [B,2]
builder
```

输出：

```text
centers       [B,H,2]
e0_repaired   [B,H,6]

A             [B,H,M,2]
b             [B,H,M]
face_mask     [B,H,M]
valid         [B,H]

# C_k：每个位置真正生成的区域，由 A/b/mask/valid 表示
# R_k：顺序处理时最近一个有效安全区域，用于后续中心投影
propagation_region

stats
```

所以这个模块一次完成：

```math
\boxed{ \text{中心修正} + \text{最终凸区域构造} }
```

而不是 sampler 自己到处写 if。

---

# 十三、具体算法伪代码

逻辑就固定成这样：

```python
raw_centers = p0 + e0[..., :2]

# ---------- k = 0 ----------
center[:, 0] = start

C0 = build_region(
    center[:, 0],
    shape=e0[:, 0]
)

if C0.valid:
    R = C0
else:
    # 第一个位置没有更早的有效区域可继承，因此使用唯一的起点 fallback。
    R = free_start_cell_box(start)
    C0 = invalid_region()

store_region(k=0, region=C0)


# ---------- k = 1 ... H-1 ----------
for k in range(1, H):

    raw_c = raw_centers[:, k]

    raw_free = point_is_free(raw_c)

    raw_region = build_region(
        raw_c,
        shape=e0[:, k]
    )

    raw_ok = raw_free & raw_region.valid

    # Case A: 当前预测本身可靠，完全保留网络输出。
    if raw_ok:
        center[:, k] = raw_c
        Ck = raw_region

    # Case B: 当前中心或当前区域不可靠。
    else:
        projected = project_to_polytope(
            raw_c,
            R                      # 最近一个有效安全区域
        )

        center[:, k] = projected

        repaired_region = build_region(
            projected,
            shape=e0[:, k]
        )

        if repaired_region.valid:
            Ck = repaired_region
        else:
            # 当前 C_k 明确保持 invalid，不用旧区域冒充当前区域。
            Ck = invalid_region()

    store_region(k=k, region=Ck)

    # 只有当前区域真正构造成功时，才更新传播区域。
    if Ck.valid:
        R = Ck
    # 否则 R 保持不变，相当于 R_k = R_{k-1}。
```

实际上 batch 维需要用 mask 分支，而不是 Python `if` 对整个 batch，但算法就是这个逻辑。

---

# 十四、free 判断直接复用现有 occupancy 体系

你现在已经有：

```python
waypoint_needs_guidance(...)
```

内部通过 `grid_sample` 查询 dilated occupancy。

把它抽象成：

```python
points_are_free(points)
```

即可。

例如：

```math
free(c) = \neg \left[ occ_{\rm dilated}(c)>\tau \right]
```

并且：

```math
|c_x|\le1,\quad |c_y|\le1.
```

中心修正建议直接复用当前：

```yaml
guidance_dilation_cells: 1
guidance_occupancy_threshold: 1e-3
```

不再引入另一套 center threshold。

---

# 十五、效率上不要真的一个个重建所有 raw region

逻辑上是顺序的，但实现不需要把 GPU 性能扔掉。

你当前 builder 能一次并行处理所有：

```math
B\times H
```

ellipse。

所以推荐：

### 一次性先算 raw candidate

```math
\hat c_{0:H-1}
```

并行构造：

```math
\hat{\mathcal C}_{0:H-1}.
```

同时并行得到：

```math
raw\_free_{0:H-1}.
```

然后顺序遍历 `k`。

如果：

```math
raw\_free_k\land raw\_valid_k
```

直接拿预先算好的：

```math
\hat{\mathcal C}_k.
```

只有：

```math
\boxed{\text{need\_repair=True}}
```

的 ellipse 才：

1. projection；
2. 单独重新 build region。

所以绝大多数正常 ellipse 不增加额外建域成本。

---

# 十六、sampler 中最终顺序

你当前 sampler 是：

```math
\hat P_0,\hat E_0 \rightarrow \text{Build Regions} \rightarrow \text{ALM} \rightarrow \text{修 E offset} \rightarrow DDIM.
```

修改以后固定成：

```math
\boxed{ \hat P_0,\hat E_0 }
```

↓

### ① endpoint inpainting

```math
\hat p_0=s,\qquad \hat p_{H-1}=g.
```

↓

### ② ellipse-center repair

```math
(\hat P_0,\hat E_0,M,s) \rightarrow (C^*,E_0^*,\{\mathcal C_k\})
```

↓

### ③ trajectory ALM

```math
\hat P_0 \xrightarrow{\{\mathcal C_k\}} \tilde P_0
```

↓

### ④ 重新编码 ellipse offset

↓

### ⑤ DDIM

---

# 十七、最重要的一点：中心修好以后，ALM 不能再把它带跑

中心修复以后得到物理中心：

```math
c_k^*.
```

此时 ellipse E6：

```math
\Delta c_k^* = c_k^*-\hat p_k.
```

然后 trajectory ALM 会：

```math
\hat p_k \rightarrow \tilde p_k.
```

这时候不要再沿用当前 sampler 这种“相对 raw center 修 offset”的间接写法。当前代码现在是：

```python
x0_e[..., :2] += raw_p - x0_p
```

来保持原 physical center。

有了 center repair 后应该直接改成更明确的：

```math
\boxed{ \Delta c_k^{final} = c_k^* - \tilde p_k. }
```

代码逻辑：

```python
x0_e = repaired_e.clone()
x0_e[..., :2] = repaired_centers - x0_p_guided
```

因此恒有：

```math
\boxed{ \tilde p_k+\Delta c_k^{final}=c_k^*. }
```

Trajectory ALM 无论怎么修轨迹，都不会破坏已经验证过的 physical ellipse center。

---

# 十八、修正后的 ellipse 要不要写回 diffusion？

```math
\boxed{\text{要。}}
```

这是我建议直接固定下来的选择。

DDIM 中 ellipse 原来使用：

```math
\hat E_0.
```

现在使用：

```math
\boxed{ E_0^{final}. }
```

也就是：

```math
\hat\epsilon_E = \frac{ E_t-\sqrt{\bar\alpha_t}E_0^{final} }{ \sqrt{1-\bar\alpha_t} }
```

然后：

```math
E_{t-1} = \sqrt{\bar\alpha_{t-1}}E_0^{final} + \sqrt{1-\bar\alpha_{t-1}}\hat\epsilon_E.
```

原因是：

> 既然我们已经判断 raw center 是错误的，就没有理由一边用 corrected center 建安全域，一边让 ellipse diffusion latent 继续朝错误中心演化。

这不是“破坏 diffusion”，而就是 ellipse guidance。

而且只改：

```math
\Delta c_x,\Delta c_y
```

四个 shape 维度：

```math
\log a,\log b,\cos2\theta,\sin2\theta
```

仍完全由 diffusion model 决定。

---

# 十九、什么时候开始中心修正

和 ALM 完全统一：

```math
\boxed{ t\le7 }
```

才启用。

也就是当前：

```yaml
start_t: 7
```

。当前 sampler 也是最后 8 个 reverse levels 才运行 ALM。

高噪声：

```math
t=15,\ldots,8
```

不碰 ellipse prediction。

低噪声：

```math
t=7,\ldots,0
```

执行：

```math
\boxed{ \text{Center Repair} \rightarrow \text{ALM}. }
```

这样不需要再增加新的 `center_start_t`。

---

# 二十、配置不需要再增加一堆参数

我建议只加：

```yaml
alm:
  center_repair: true
```

其余全部复用现有参数：

- `start_t`
- `safety_margin`
- `guidance_dilation_cells`
- `guidance_occupancy_threshold`
- `max_faces`
- `obstacle_window_half`

投影内部只有：

```text
feasibility_tol = 1e-6
parallel_tol = 1e-8
```

这种纯数值常量，不作为实验超参数。

这样不会再多出一堆需要调的东西。

---

# 二十一、必须增加的统计量

每个 reverse step 输出：

```math
\boxed{ center\_raw\_unsafe\_rate }
```

原预测中心有多少在 obstacle / margin 内。

```math
\boxed{ center\_repair\_rate }
```

实际有多少中心被投影。

```math
\boxed{ center\_projection\_mean }
```

```math
\boxed{ center\_projection\_max }
```

修正距离：

```math
\|c_k^*-\hat c_k\|.
```

还有：

```math
\boxed{ center\_post\_unsafe\_rate }
```

这个最终应该接近：

```math
\boxed{0}
```

以及：

```math
\boxed{ region\_build\_failure\_rate }
```

表示当前 `\mathcal C_k` 实际构造失败的比例。

另外增加：

```math
\boxed{ propagation\_reuse\_rate }
```

表示由于当前 `\mathcal C_k` 无效，传播区域继续采用：

```math
\mathcal R_k=\mathcal R_{k-1}
```

的比例。

注意这里复用的是**后续中心投影所使用的传播区域 `\mathcal R_k`**，不是把旧区域写成当前的 `\mathcal C_k`。

最后再比较：

```math
region\_valid\_rate^{raw}
```

和：

```math
region\_valid\_rate^{repaired}.
```

这两个是你最需要看的。

---

# 二十二、可视化也一次做好

你现在已有 convex-region overlay。

再增加：

- raw ellipse center：红色叉号；
- repaired center：青色实心点；
- 被修过的中心：
  ```math
  \hat c_k\rightarrow c_k^*
  ```
  画一条细箭头；
- 最终 convex region；
- trajectory。

这样你一眼就能看出：

```text
墙里的红叉
       ↘
        ● 修正后的中心
      [新的安全凸区域]
```

而不是再靠日志猜修正有没有发生。

---

# 二十三、单元测试一次覆盖完整逻辑

我建议直接写下面 7 个，不要后面发现一个洞再加一个。

1. **首中心锚定**

```math
c_0^*=start.
```

2. **正常中心不修改**

如果 raw center free 且 raw region valid：

```math
c_k^*=\hat c_k
```

必须 exact equality。

3. **墙内中心被投影**

构造简单 box：

```math
\mathcal C_{k-1}=[-0.5,0.5]^2
```

raw：

```math
\hat c_k=(0.8,0.1)
```

应得到：

```math
c_k^*=(0.5,0.1).
```

4. **投影点一定满足前一区域**

```math
A_{k-1}c_k^*\le b_{k-1}.
```

5. **连续多个坏中心不会断**

连续 3～5 个墙内 center，确保：

```math
C_{k-1}\rightarrow c_k^*\rightarrow C_k
```

可以递推。

6. **新 region 构造失败时保持失败状态并复用传播区域**

验证当前区域：

```math
\operatorname{Valid}(\mathcal C_k)=0
```

同时传播区域满足：

```math
\boxed{\mathcal R_k=\mathcal R_{k-1}.}
```

然后第 `k+1` 个中心仍然可以正常执行：

```math
c_{k+1}^*=\Pi_{\mathcal R_k}(\hat c_{k+1}).
```

必须确保代码不会把：

```math
\mathcal R_{k-1}
```

伪装成当前第 `k` 个椭圆生成的：

```math
\mathcal C_k.
```

7. **ALM 后 physical center 不变**

必须满足：

```math
\boxed{ \tilde p_k+\Delta c_k^{final} = c_k^* }
```

误差例如：

```math
<10^{-6}.
```

---

# 二十四、最终代码结构

建议最后形成：

```text
src/
├── geometry/
│   ├── convex_corridor.py
│   ├── ellipse_center_repair.py      # 新增
│   └── polytope_projection.py        # 新增
│
├── diffusion/
│   ├── alm_guidance.py
│   └── sampler_v1.py
```

职责明确：

### `polytope_projection.py`

只负责：

```math
\boxed{ y\rightarrow\Pi_{\{Ax\le b\}}(y) }
```

---

### `ellipse_center_repair.py`

负责：

```math
\boxed{ \hat E \rightarrow E^* + \{\mathcal C_k\} }
```

即：

- 首中心=start；
- raw center safety check；
- projection；
- region rebuild；
- `\mathcal C_k` 有效性管理；
- 最近有效传播区域 `\mathcal R_k` 的维护；
- stats。

---

### `convex_corridor.py`

只负责：

```math
\boxed{ (c,Q,M)\rightarrow(A,b) }
```

不负责 center repair。

---

### `alm_guidance.py`

完全不负责 ellipse。

只负责：

```math
\boxed{ P\rightarrow\tilde P }
```

---

### `sampler_v1.py`

只负责把四个阶段串起来。

---

# 二十五、最后整个算法就变成这一张流程

```math
(P_t,E_t)
```

```math
\downarrow f_\theta
```

```math
(\hat P_0,\hat E_0)
```

```math
\downarrow
```

```math
\boxed{ c_0^*=start }
```

```math
\downarrow
```

```math
\boxed{ \hat c_k \overset{\text{安全}}{\longrightarrow} \hat c_k \qquad \hat c_k \overset{\text{不安全}}{\longrightarrow} \Pi_{\mathcal C_{k-1}}(\hat c_k) }
```

```math
\downarrow
```

```math
\boxed{ E_0^* + \{\mathcal C_k\} }
```

```math
\downarrow
```

```math
\boxed{ \text{Trajectory ALM} }
```

```math
\downarrow
```

```math
\tilde P_0
```

```math
\downarrow
```

重新编码：

```math
\boxed{ \Delta c_k^{final}=c_k^*-\tilde p_k }
```

```math
\downarrow
```

```math
\boxed{ (\tilde P_0,E_0^{final}) }
```

```math
\downarrow DDIM
```

```math
\boxed{ (P_{t-1},E_{t-1}) }
```

这套方案的关键点就三个：

```math
\boxed{ \textbf{第一个安全 seed 是 start} }
```

```math
\boxed{ \textbf{错误 center 只向前一个已验证凸域做最近点投影} }
```

```math
\boxed{ \textbf{修正后的 center 写回 ellipse diffusion，并且 trajectory ALM 不再改变它的物理位置} }
```

这样中心错误、凸区域错误、ALM 使用错误区域这三个问题是在同一条链上一起解决的，而不是各自打补丁。