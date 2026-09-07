# 椭圆条件轨迹扩散

**User:** Anonymous  
**Created:** 9/4/2026 0:03:02  
**Updated:** 9/8/2026 0:09:07  
**Exported:** 9/8/2026 0:09:47  
**Link:** [https://chatgpt.com/c/6a999a34-b1f0-83ea-b4eb-5b050565df03](https://chatgpt.com/c/6a999a34-b1f0-83ea-b4eb-5b050565df03)  

## Response:
2026/9/8 00:03:42

> **规划实现路径**
>
> 我正在区分现有代码与新增模块：当前采样器使用 DDIM，预测位置还需叠加椭圆偏移；仓库没有现成凸区域或 ALM。建议新增走廊构建、约束投影、采样器迭代修正，并添加单元测试。
>
> **规划了实现路径**
>
> 
>
> **固定走廊与对偶**
>
> 我正在权衡三种策略：每步重建并重置对偶变量、固定走廊但保留对偶变量，或稳定匹配约束面。首版建议在较晚阶段冻结走廊，随后持续更新对偶压力，便于真正体现 ALM。
>
> **固定走廊与对偶分析完成**
>
> 
>
> **检查并评估安全椭圆采样及扩散训练代码**
>
> 
>
> 思考了 3m 14s

可以。结合你**现在 GitHub 上的代码结构**，我建议第一版 ALM 不碰训练、不加 DCBF，也不改网络，直接作为一个**推理阶段的 constrained reverse sampling 模块**插入现有 `x0-prediction + DDIM` sampler。

最核心的实施思路先压缩成一句：

$$
\boxed{
(P_t,E_t)
\xrightarrow{\text{model}}
(\hat P_0,\hat E_0)
\xrightarrow{\text{构造凸安全走廊}}
(A,b)
\xrightarrow{\text{ALM}}
\tilde P_0
\xrightarrow{\text{现有 DDIM}}
P_{t-1}
}
$$

也就是说：

$$
\boxed{\text{ALM 修正的是每一步模型预测出来的 clean trajectory }\hat P_0}
$$

而**不是直接去修 noisy $P_t$**。

这和你当前代码是最匹配的。

---

# 一、先明确现在的代码是什么状态

当前 `sampler_v1.py` 是标准的 **x0-prediction deterministic DDIM**。

模型每一步输出：

$$
\hat P_0,\qquad \hat E_0
$$

然后从：

$$
P_t
$$

反推出当前预测噪声：

$$
\hat\epsilon_P
=
\frac{
P_t-\sqrt{\bar\alpha_t}\hat P_0
}{
\sqrt{1-\bar\alpha_t}
}
$$

再得到：

$$
P_{t-1}
=
\sqrt{\bar\alpha_{t-1}}\hat P_0
+
\sqrt{1-\bar\alpha_{t-1}}\hat\epsilon_P.
$$

最后一步：

$$
t=0
$$

直接：

$$
P_0=\hat P_0.
$$

当前一共只有 16 个 diffusion steps，并且起点终点在每次 reverse update 后都会硬覆盖为真实 start / goal。

所以最自然的改法不是推翻 sampler，而只是把：

```python
x0_p
```

替换成：

```python
x0_p_alm
```

然后下面原来的 DDIM 全部保持不动。

---

# 二、第一版 ALM 不需要重新训练

当前模型已经训练出了：

$$
f_\theta(P_t,E_t,M,s,g,t)
=
(\hat P_0,\hat E_0)
$$

，而且当前 E6 仍然是：

$$
E=
[
\Delta c_x,\Delta c_y,\log a,\log b,\cos2\theta,\sin2\theta
]
$$

其中：

$$
c=p+\Delta c.
$$

Dataset 和 evaluate 都还是按照这个定义解码 ellipse。 

所以第一阶段：

$$
\boxed{
\text{checkpoint 完全冻结}
}
$$

不需要：

- 重新训练 ALM；
- 改 loss；
- 改 transformer；
- 改 ellipse head；
- 改 diffusion schedule。

我们只修改 inference。

---

# 三、ALM 的约束不要再做 point-wise，而做 segment-wise

这是这次实现最关键的地方。

不能仅仅定义：

$$
A_kp_k\le b_k.
$$

我们前面已经分析了：这只能保证 waypoint 安全。

假设第 $k$ 段轨迹：

$$
S_k=[p_k,p_{k+1}]
$$

对应一个经过 occupancy 验证的安全凸域：

$$
\boxed{
\mathcal C_k
=
\{x:A_kx\le b_k\}
\subseteq\mathcal F
}
$$

那么我们要求：

$$
\boxed{
A_kp_k\le b_k
}
$$

同时：

$$
\boxed{
A_kp_{k+1}\le b_k.
}
$$

这样，因为 $\mathcal C_k$ 是凸集：

$$
p_k,p_{k+1}\in\mathcal C_k
$$

就直接有：

$$
\boxed{
[p_k,p_{k+1}]
\subseteq\mathcal C_k
\subseteq\mathcal F.
}
$$

因此 ALM 不应该约束 128 个独立 waypoint，而应该约束：

$$
\boxed{127\text{ 个 trajectory segments}}
$$

---

# 四、首先规定 convex corridor 模块的输出接口

建议新建：

```text
src/geometry/convex_corridor.py
```

第一版 ALM 不关心你内部到底如何由 ellipse 构造 convex region。

它只要求得到：

```python
A      # [B, H-1, M, 2]
b      # [B, H-1, M]
mask   # [B, H-1, M]
valid  # [B, H-1]
```

其中：

$$
H=128
$$

所以：

$$
H-1=127.
$$

`M` 是最多保留多少个 half-space，例如：

$$
M=8
$$

或者：

$$
M=12.
$$

不足的部分通过 `mask` 屏蔽。

---

# 五、这里必须统一 half-space 的尺度

每个安全边界：

$$
a_{k,j}^{T}p\le b_{k,j}
$$

构造完成后，一定进行：

$$
\boxed{
a_{k,j}
\leftarrow
\frac{a_{k,j}}{\|a_{k,j}\|_2}
}
$$

同时：

$$
\boxed{
b_{k,j}
\leftarrow
\frac{b_{k,j}}{\|a_{k,j}^{old}\|_2}.
}
$$

也就是：

$$
\|a_{k,j}\|=1.
$$

为什么这一步非常重要？

因为 ALM 使用：

$$
g=a^Tp-b.
$$

如果一个 constraint 写成：

$$
a^Tp-b\le0
$$

另外一个完全一样的约束乘 100：

$$
100a^Tp-100b\le0,
$$

几何意义完全一样，

但：

$$
g
$$

和：

$$
\nabla g
$$

却放大了 100 倍。

那么：

$$
\lambda,\rho,\eta
$$

全部失去统一意义。

所以必须先让：

$$
\boxed{\|a\|_2=1}
$$

这样：

$$
a^Tp-b
$$

大致具有“scene distance”的意义。

你的整个 scene 坐标本来就统一到了：

$$
[-1,1]^2
$$

而且 SDF 也是以 scene units 储存，所以这套尺度是统一的。

---

# 六、不要给每一个 face 一个长期 $\lambda$

理论上可以定义：

$$
g_{k,j}^{L}
=
a_{k,j}^Tp_k-b_{k,j}
$$

以及：

$$
g_{k,j}^{R}
=
a_{k,j}^Tp_{k+1}-b_{k,j}.
$$

然后给每一个约束一个：

$$
\lambda_{k,j}^{L},
\qquad
\lambda_{k,j}^{R}.
$$

但我**不建议你当前代码第一版这么做**。

原因不是算力。

而是：

$$
\boxed{
A_k^{(t)},b_k^{(t)}
}
$$

是根据每一步：

$$
\hat E_0^{(t)}
$$

重新构造出来的。

第 $t$ 步的第 3 个 face 和下一步第 3 个 face：

$$
\boxed{\text{未必还是同一个障碍边界。}}
$$

那么：

$$
\lambda_{k,3}^{(t)}
$$

继承给下一步：

$$
\lambda_{k,3}^{(t-1)}
$$

物理意义就乱了。

---

# 七、我建议每个 segment 只维护一个 ALM constraint

这是我认为最适合你当前动态凸域的方式。

先计算第 $k$ 个 segment 所有 face violation。

左端点：

$$
v_{k,j}^{L}
=
a_{k,j}^{T}p_k-b_{k,j}.
$$

右端点：

$$
v_{k,j}^{R}
=
a_{k,j}^{T}p_{k+1}-b_{k,j}.
$$

如果要求全部安全，就是：

$$
\max_{j}
\{
v_{k,j}^{L},
v_{k,j}^{R}
\}
\le0.
$$

但是 `max` 在切换 active face 时不光滑。

所以我建议使用一个**保守 smooth maximum**：

$$
\boxed{
g_k(P)
=
\tau_g
\log
\left(
\sum_j
e^{v_{k,j}^{L}/\tau_g}
+
\sum_j
e^{v_{k,j}^{R}/\tau_g}
\right)
}
$$

其中：

$$
\tau_g>0.
$$

---

# 八、这个 smooth-max 有一个特别好的性质

因为：

$$
\operatorname{LSE}(v)
\ge
\max(v),
$$

所以：

$$
g_k(P)\le0
$$

必然推出：

$$
\max(v)\le0.
$$

进而：

$$
v_{k,j}^{L}\le0,
\qquad
v_{k,j}^{R}\le0,
\quad\forall j.
$$

也就是：

$$
p_k,p_{k+1}\in C_k.
$$

所以：

$$
\boxed{
g_k(P)\le0
}
$$

是一个**比原始 half-space constraints 稍微保守，但是足够保证 segment safety 的光滑约束**。

这样整个轨迹只需要：

$$
\boxed{
127\text{ 个 constraint values}
}
$$

而：

$$
\lambda
$$

只需要：

```python
lam.shape == [B, 127]
```

这就干净很多。

即使下一 diffusion step 的 half-spaces 变了：

$$
A_k^{(t)}
\rightarrow
A_k^{(t-1)},
$$

$\lambda_k$ 的语义始终还是：

> 第 $k$ 段轨迹长期存在多大的安全约束压力。

不会出现 face identity 错位。

---

# 九、ALM 模块建议新建一个文件

新建：

```text
src/diffusion/alm_guidance.py
```

它不需要引用模型。

它只负责：

```python
P
A
b
mask
lambda
rho
```

然后返回：

```python
P_corrected
lambda_new
metrics
```

核心接口可以设计成：

```python
def alm_correct(
    p0,
    A,
    b,
    face_mask,
    lam,
    rho,
    step_size,
    smooth_tau,
    inner_steps,
):
    ...
```

这样 ALM 和网络完全解耦。

---

# 十、ALM 每次计算什么？

对于每个 segment：

$$
g_k(P)\le0.
$$

已有：

$$
\lambda_k\ge0.
$$

首先计算 slack：

$$
\boxed{
s_k
=
\left[
-g_k-\frac{\lambda_k}{\rho}
\right]_+
}
$$

然后 residual：

$$
\boxed{
r_k=g_k+s_k.
}
$$

然后约束力度：

$$
\boxed{
\omega_k
=
\lambda_k+\rho r_k.
}
$$

于是 augmented-Lagrangian 对 trajectory 的梯度：

$$
\boxed{
\nabla_P L_{\rm ALM}
=
\sum_k
\omega_k
\nabla_Pg_k.
}
$$

接着做：

$$
\boxed{
P
\leftarrow
P-\eta\nabla_PL_{\rm ALM}.
}
$$

这就是一次 primal correction。

Constrained Diffusers 原论文对应的核心就是：

$$
\nabla\log p
-
(\lambda+\rho r)^T\nabla g.
$$

ALM 相比 PD 多出来的正是：

$$
\rho r
$$

这个即时 violation pressure。`Zhang 等 - 2025 - Constrained Diffusers for Safe Planning and Control.pdf`

---

# 十一、这里甚至不需要 autograd

这个非常适合你当前 `sampler_v1.py`。

现在 sampler 整个循环都包在：

```python
with torch.no_grad():
```

里面。

如果我们用 analytic gradient，就不用破坏它。

因为：

$$
g_k
=
\tau\log\sum_l e^{v_l/\tau}.
$$

所以对 $v_l$：

$$
\frac{\partial g_k}{\partial v_l}
=
\frac{e^{v_l/\tau}}
{\sum_m e^{v_m/\tau}}
=
w_l.
$$

也就是 softmax。

因此：

$$
\boxed{
w_{k,j}^{L}
=
\operatorname{softmax}
\left(
\frac{v_k}{\tau}
\right)_{L,j}
}
$$

$$
\boxed{
w_{k,j}^{R}
=
\operatorname{softmax}
\left(
\frac{v_k}{\tau}
\right)_{R,j}.
}
$$

那么：

$$
\nabla_{p_k}g_k
=
\sum_j
w_{k,j}^{L}a_{k,j}
$$

$$
\nabla_{p_{k+1}}g_k
=
\sum_j
w_{k,j}^{R}a_{k,j}.
$$

全是矩阵运算。

不用：

```python
requires_grad=True
```

不用：

```python
backward()
```

不用让梯度穿过模型。

---

# 十二、代码逻辑大概就是这样

假设：

```python
p_left = p[:, :-1]       # [B,127,2]
p_right = p[:, 1:]       # [B,127,2]
```

那么：

```python
v_left = (A * p_left[:, :, None, :]).sum(-1) - b
v_right = (A * p_right[:, :, None, :]).sum(-1) - b
```

得到：

```text
[B,127,M]
```

然后：

```python
v = torch.cat([v_left, v_right], dim=-1)
```

得到：

```text
[B,127,2M]
```

无效 face：

```python
v = v.masked_fill(~constraint_mask, -1e6)
```

然后：

```python
g = tau * torch.logsumexp(v / tau, dim=-1)
```

得到：

```text
[B,127]
```

---

# 十三、然后计算 ALM coefficient

```python
s = torch.relu(-g - lam / rho)
r = g + s
omega = lam + rho * r
```

也就是：

$$
s_k
=
[-g_k-\lambda_k/\rho]_+
$$

$$
r_k=g_k+s_k
$$

$$
\omega_k=\lambda_k+\rho r_k.
$$

---

# 十四、然后求 smooth-max 的法向

```python
weights = torch.softmax(v / tau, dim=-1)

w_left = weights[..., :M]
w_right = weights[..., M:]
```

然后：

```python
grad_left = (w_left[..., None] * A).sum(dim=2)
grad_right = (w_right[..., None] * A).sum(dim=2)
```

都是：

```text
[B,127,2]
```

---

# 十五、把 127 个 segment gradient 累加到 128 个 waypoint

初始化：

```python
grad_p = torch.zeros_like(p)
```

然后：

```python
grad_p[:, :-1] += omega[..., None] * grad_left
grad_p[:, 1:]  += omega[..., None] * grad_right
```

为什么内部 waypoint 会收到两个梯度？

因为：

$$
p_k
$$

既属于：

$$
C_{k-1}
$$

又属于：

$$
C_k.
$$

所以它自然同时受到：

$$
\text{前一段 corridor}
+
\text{后一段 corridor}
$$

的约束。

这正是我们想要的。

---

# 十六、不要让 ALM 修改 start / goal

当前代码已经把：

$$
p_0=s,
\qquad
p_{H-1}=g
$$

硬约束。

因此 ALM 里：

```python
grad_p[:, 0] = 0
grad_p[:, -1] = 0
```

然后：

```python
p = p - eta * grad_p
```

再调用：

```python
p = endpoints(p)
```

保证不出问题。

这也意味着 corridor builder 必须保证：

$$
s\in C_0
$$

以及：

$$
g\in C_{126}.
$$

否则第一段或者最后一段先天 infeasible。

---

# 十七、一个 diffusion step 里建议做 2～3 次 ALM inner update

模型 forward 一次非常贵。

但 ALM correction 只是几百个矩阵运算，非常便宜。

所以：

```python
for _ in range(inner_steps):
    compute g
    compute slack
    compute gradient
    update p
    recompute residual
    update lambda
```

第一版建议：

```yaml
inner_steps: 2
```

如果约束下降不够：

```yaml
inner_steps: 3
```

不建议一开始就 10 次。

---

# 十八、$\lambda$ 更新放在 primal correction 以后

我建议工程上用标准一点的顺序：

首先：

$$
P^{(l)}
\rightarrow P^{(l+1)}
$$

然后重新计算新的：

$$
g^{new}.
$$

再：

$$
s^{new}
=
[-g^{new}-\lambda/\rho]_+
$$

$$
r^{new}
=
g^{new}+s^{new}
$$

最后：

$$
\boxed{
\lambda
\leftarrow
\lambda+\rho r^{new}.
}
$$

为了防止浮点误差：

```python
lam = torch.clamp(lam + rho * r_new, min=0.0)
```

理论上 slack update 已经能保持：

$$
\lambda\ge0
$$

但代码里 clamp 一下比较稳。

---

# 十九、$\rho$ 不要在 16 步里无脑指数爆炸

论文使用：

$$
\rho_{t-1}=c\rho_t,
\quad c>1.
$$

并在：

$$
\rho\rightarrow\infty
$$

下讨论收敛。`Zhang 等 - 2025 - Constrained Diffusers for Safe Planning and Control.pdf`

但是你的 diffusion：

$$
T=16.
$$

如果：

$$
c=2
$$

连续 16 步：

$$
\rho
$$

会爆得非常快。

所以建议实际实现：

$$
\boxed{
\rho
\leftarrow
\min(c\rho,\rho_{\max})
}
$$

而且只在**每一个 diffusion step 结束以后增加一次**，

不要每个 inner ALM iteration 都增加。

---

# 二十、第一版不要从 t=15 开始 ALM

这一点我非常建议。

当前 diffusion：

$$
t=15\rightarrow0.
$$

早期：

$$
P_t,E_t
$$

噪声很大。

虽然网络预测的是：

$$
\hat P_0,\hat E_0,
$$

但高噪阶段 geometry prediction 的可靠性肯定不如后期。

所以第一版：

$$
\boxed{
t=15,\ldots,8:
\text{原始 DDIM}
}
$$

$$
\boxed{
t=7,\ldots,0:
\text{ALM-guided DDIM}
}
$$

也就是最后：

$$
8
$$

步开启。

后面直接 ablation：

$$
4,\quad8,\quad12,\quad16
$$

个 constrained reverse steps。

---

# 二十一、真正插入 `sampler_v1.py` 的位置

当前是：

```python
out = model(...)
x0_p, x0_e = out["x0_p"], out["x0_e"]

if t == 0:
    p, e = x0_p, x0_e
else:
    eps_p = ...
    p = ...
```

我们要改成：

```python
out = model(...)

x0_p = out["x0_p"]
x0_e = out["x0_e"]

x0_p = endpoints(x0_p)

if alm_active(t):
    A, b, mask, valid = build_segment_corridors(
        x0_p,
        x0_e,
        map_tensor,
    )

    x0_p, lam, alm_stats = alm_correct(
        x0_p,
        A,
        b,
        mask,
        lam,
        rho,
        ...
    )

    x0_p = endpoints(x0_p)
```

然后：

```python
eps_p = (p - sa_t * x0_p) / s1_t
p = sa_prev * x0_p + s1_prev * eps_p
```

完全不动。

---

# 二十二、为什么 ALM 应该修改 `x0_p`，而不是 `p_t`

因为你的安全凸域来自：

$$
\hat E_0
$$

对应的是：

$$
\boxed{\text{clean physical geometry}}
$$

而：

$$
P_t
$$

在高噪时根本不是物理轨迹。

如果约束：

$$
A_kp_k^t\le b_k
$$

相当于强迫 noisy latent：

$$
P_t
$$

也必须待在真实走廊里。

这没有必要，也会严重破坏 diffusion。

所以：

$$
\boxed{
A,b
\text{ 约束 }
\hat P_0^{(t)}
}
$$

才是与你现在 x0-prediction 模型语义一致的。

---

# 二十三、当前 ellipse 怎么构造 convex region？

你当前物理 ellipse center 必须使用：

$$
\boxed{
c_k
=
\hat p_k+
\widehat{\Delta c}_k
}
$$

因为当前 E6 没有改成绝对 center。

所以 corridor builder 开始应该先：

$$
c
=
x0_p+x0_e[:,:,:2]
$$

$$
a=e^{\log a}
$$

$$
b=e^{\log b}
$$

$$
\theta=
\frac12\operatorname{atan2}
(\sin2\theta,\cos2\theta).
$$

训练阶段的 safety loss 对 axes 已经采用：

$$
\log a,\log b\in[-6,0.7]
$$

的数值 clamp，所以构造 corridor 时第一版也可以采用一致范围做数值保护。

但是：

$$
\boxed{
\text{Ellipse 只负责提供 seed / metric / obstacle ordering}
}
$$

最终：

$$
A,b
$$

必须通过 occupancy 做验证。

不要直接把 ellipse 本身当安全约束。

---

# 二十四、ALM 修正时不要重新构造 region

假设当前一个 reverse step：

$$
\hat P_0
\rightarrow A,b.
$$

然后 ALM inner loop：

$$
P^{(0)}
\rightarrow
P^{(1)}
\rightarrow
P^{(2)}.
$$

在这个 inner loop 里面：

$$
\boxed{
A,b\text{ 固定}
}
$$

不要每移动一次 waypoint，就重新：

$$
P\rightarrow E\rightarrow A,b.
$$

否则会出现：

> 轨迹往外跑，安全区域也跟着跑。

约束就失去意义。

因此一个 diffusion step 应该：

```text
model forward
      ↓
freeze x0_e / corridor
      ↓
A,b
      ↓
ALM iteration 1
      ↓
ALM iteration 2
      ↓
得到 x0_p_alm
```

然后进入下一 diffusion step，才允许重新预测：

$$
\hat E_0
$$

重新构造新的 corridor。

---

# 二十五、$\lambda$ 要不要跨 diffusion step 保留？

我的建议：

$$
\boxed{\text{保留。}}
$$

因为我们已经没有使用 face-wise：

$$
\lambda_{k,j}
$$

而是使用：

$$
\boxed{
\lambda_k
}
$$

对应：

> 第 $k$ 段轨迹的 worst/smooth-max corridor violation。

所以即使：

$$
A_k^{(t)}
$$

下一步略微变化，

$\lambda_k$ 的语义仍然一致。

初始化：

```python
lam = torch.zeros(B, H - 1, device=device)
```

直到：

$$
t=0
$$

结束。

---

# 二十六、建议配置文件这样加

现在 `config_v1.yaml` 的 diffusion 是：

$$
T=16,\quad x0\text{-prediction}.
$$

建议独立增加：

```yaml
alm:
  enabled: true

  # only apply to t <= 7, i.e. final 8 reverse steps
  start_t: 7

  inner_steps: 2

  # smooth max over segment half-space violations
  smooth_tau: 0.03

  # primal update in scene [-1,1] coordinates
  step_size: 0.03

  rho_init: 1.0
  rho_growth: 1.3
  rho_max: 20.0

  # safety / numerical controls
  max_grad_norm: 1.0
  max_correction_per_step: 0.10

  # corridor
  max_faces: 8
  safety_margin: 0.02
```

这里数值我把它们当作：

$$
\boxed{\text{第一轮实验起点，而不是最终论文参数}}
$$

真正需要 ablation 的是：

$$
\eta,\rho_0,c,\tau_g,\text{ALM steps}.
$$

---

# 二十七、为什么需要 `max_correction_per_step`

ALM 不是 hard projection。

如果：

$$
\rho r
$$

突然很大，

可能：

$$
\Delta p_k
$$

一次移动很远。

这会破坏 diffusion prior。

所以建议：

$$
\Delta P
=
-\eta\nabla L_A
$$

之后限制：

$$
\boxed{
\|\Delta p_k\|_2\le\delta_{\max}
}
$$

例如第一版：

$$
\delta_{\max}=0.1
$$

scene units。

注意这不是 safety guarantee。

只是数值稳定措施。

---

# 二十八、建议再加 gradient norm clip

比如：

```python
norm = grad_p.norm(dim=-1, keepdim=True)
scale = torch.clamp(max_grad_norm / (norm + 1e-8), max=1.0)
grad_p = grad_p * scale
```

然后再：

```python
delta = -eta * grad_p
```

否则几个相邻 corridor 同时压一个 waypoint 时：

$$
p_k
$$

可能受到多个很大的法向力。

---

# 二十九、第一版 `rho` 更新流程

假设从：

$$
t=7
$$

开始。

初始化：

$$
\lambda=0,
\qquad
\rho=1.
$$

那么：

```text
t=7:
rho = 1.00
ALM × 2
↓

t=6:
rho = 1.30
ALM × 2
↓

t=5:
rho = 1.69
↓

t=4:
rho = 2.20
...
```

到后期 constraint pressure 自然变强。

这其实非常符合 diffusion：

$$
\boxed{
\text{早期更多相信生成模型}
\rightarrow
\text{后期更多强调约束}
}
$$

---

# 三十、但 `rho` 不等于最终 safety weight

ALM 实际安全力：

$$
\boxed{
\omega_k
=
\lambda_k+\rho r_k.
}
$$

所以不是：

$$
\rho\text{ 越大就无脑越推。}
$$

如果一个 segment 已经足够安全，

slack 会吸收安全裕量，并使：

$$
\lambda
$$

逐渐降低甚至回到：

$$
0.
$$

因此安全 segment 不应该长期受到强制移动。

---

# 三十一、评价代码需要增加哪些指标

当前 `evaluate.py` 已经有：

- `collision_frac`
- `clearance_p05`
- `traj_smoothness`
- ellipse boundary collision
- endpoint error

这些可以直接保留。

ALM 新增：

### 1. Corridor violation

$$
\boxed{
V_{\max}
=
\max_{k,j}
[
a_{k,j}^Tp-b_{k,j}
]_+
}
$$

---

### 2. Segment feasible rate

满足：

$$
p_k,p_{k+1}\in C_k
$$

的 segment 比例。

---

### 3. ALM correction magnitude

$$
\boxed{
D_{\rm alm}
=
\frac1H
\sum_k
\|
p_k^{alm}-p_k^{raw}
\|_2
}
$$

这个特别重要。

因为你希望证明：

> ALM 用很小的 trajectory deviation 换来了更低 collision。

---

### 4. 最大 correction

$$
D_{\max}
=
\max_k
\|p_k^{alm}-p_k^{raw}\|.
$$

---

### 5. Active constraint ratio

$$
\frac{
|\{k:g_k>0\}|
}{
H-1
}.
$$

可以看出来到底有多少 segment 真正需要安全修正。

---

# 三十二、实验一定使用同一个 seed 对比

直接比较：

### Baseline

```text
Joint Diffusion
```

### 方法

```text
Joint Diffusion + ALM
```

固定：

- checkpoint；
- start / goal；
- initial Gaussian noise；
- DDIM steps；
- seed。

这样你就可以画：

```text
灰色：raw diffusion
绿色：ALM trajectory
蓝色：segment convex corridors
黑色：obstacles
```

这个图非常有说服力。

---

# 三十三、第一轮实验不要追求 0 collision

这点要明确。

因为：

$$
\boxed{\text{ALM 本身不是 hard guarantee}}
$$

第一阶段主要回答：

### 问题 1

ALM 能不能显著降低：

$$
\text{corridor violation}
$$

？

### 问题 2

ALM 能不能降低：

$$
\text{collision rate}
$$

？

### 问题 3

是否只产生很小：

$$
\text{trajectory deviation}
$$

？

### 问题 4

smoothness 有没有明显恶化？

如果答案是：

$$
\boxed{
\text{violation 大幅下降}
+
\text{collision 大幅下降}
+
\text{deviation 很小}
}
$$

那么 ALM guidance 这条路就成立。

之后再讨论需不需要：

- SafeDiffuser-style finite-time constraint；
- final projection；
- hard feasibility closure。

---

# 三十四、我建议代码最终形成这四层

```text
src/
├── geometry/
│   ├── scene_frame.py
│   └── convex_corridor.py       # 新增
│
├── diffusion/
│   ├── schedule.py
│   ├── sampler_v1.py
│   └── alm_guidance.py          # 新增
│
└── models/
    └── joint/
        └── ...
```

其中：

### `JointPlanner`

完全不改。

### `convex_corridor.py`

负责：

$$
(\hat P_0,\hat E_0,M)
\rightarrow
(A,b)
$$

### `alm_guidance.py`

负责：

$$
(\hat P_0,A,b,\lambda,\rho)
\rightarrow
\tilde P_0.
$$

### `sampler_v1.py`

只负责把三部分串起来。

这个工程职责很干净。

---

# 三十五、整个最终伪代码

你现在的 sampler 可以概念上改成：

```python
p = randn(...)
e = randn(...)

lam = zeros(B, H - 1)
rho = rho_init

for t in reversed(range(T)):

    # ==========================
    # 1. existing model
    # ==========================
    out = model(p, e, map_tensor, cond, tb, ab)

    p0_raw = endpoints(out["x0_p"])
    e0_raw = out["x0_e"]

    p0 = p0_raw

    # ==========================
    # 2. ALM guidance
    # ==========================
    if alm_enabled and t <= alm_start_t:

        # ellipse -> verified segment corridors
        A, b, mask, valid = build_segment_corridors(
            p0_raw,
            e0_raw,
            map_tensor,
        )

        # A,b fixed during inner loop
        p0, lam, stats = alm_correct(
            p0,
            A,
            b,
            mask,
            lam,
            rho,
            step_size=eta,
            smooth_tau=tau,
            inner_steps=K,
        )

        p0 = endpoints(p0)

        rho = min(rho * rho_growth, rho_max)

    # ==========================
    # 3. existing DDIM
    # ==========================
    if t == 0:
        p = p0
        e = e0_raw

    else:
        eps_p = (p - sa_t * p0) / s1_t
        eps_e = (e - sa_t * e0_raw) / s1_t

        p = sa_prev * p0 + s1_prev * eps_p
        e = sa_prev * e0_raw + s1_prev * eps_e

    p = endpoints(p)
```

最值得注意的就是这一条：

$$
\boxed{
\text{我们不改 DDIM 方程，只把 }\hat P_0
\text{ 替换成 ALM 修正后的 }\tilde P_0.
}
$$

---

# 三十六、所以第一版 ALM 的完整数据链就是

$$
P_t,E_t
$$

$$
\downarrow
$$

$$
f_\theta
$$

$$
\downarrow
$$

$$
\boxed{
\hat P_0^{(t)},\hat E_0^{(t)}
}
$$

$$
\downarrow
$$

根据：

$$
\hat E_0^{(t)}+M
$$

得到：

$$
\boxed{
C_k^{(t)}
=
\{x:A_k^{(t)}x\le b_k^{(t)}\}
}
$$

$$
\downarrow
$$

segment constraint：

$$
\boxed{
g_k(\hat P_0^{(t)})\le0
}
$$

$$
\downarrow
$$

ALM：

$$
s_k=[-g_k-\lambda_k/\rho]_+
$$

$$
r_k=g_k+s_k
$$

$$
\omega_k=\lambda_k+\rho r_k
$$

$$
\boxed{
\tilde P_0^{(t)}
=
\hat P_0^{(t)}
-
\eta
\sum_k
\omega_k\nabla_Pg_k
}
$$

$$
\downarrow
$$

更新：

$$
\lambda_k
\leftarrow
[\lambda_k+\rho r_k]_+
$$

$$
\rho\leftarrow
\min(c\rho,\rho_{\max})
$$

$$
\downarrow
$$

原 DDIM：

$$
\boxed{
\tilde P_0^{(t)}
\rightarrow
P_{t-1}
}
$$

这就是我认为**最适合你当前代码的 ALM V1 实施方案**。

而且第一版非常干净：**不改模型、不重新训练、不加 DCBF、不用 SDF gradient、不做 hard projection。**唯一真正需要先完成的是 `convex_corridor.py`--把现在预测出来的 ellipse 稳定转换成每一段轨迹对应的、经过 occupancy 验证的 $A_k,b_k$。这个完成以后，ALM 本身的代码其实不会很复杂。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)