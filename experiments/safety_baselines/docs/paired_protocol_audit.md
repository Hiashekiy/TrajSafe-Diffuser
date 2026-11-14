# 配对协议与随机种子审计（Phase B3）

## 1. 四类随机量，各自的作用域

|随机量|实现|是否与方法有关|用途|
|---|---|---|---|
|场景种子|`initial_seed(scene_id, seed)` = sha256(`'{scene}|{seed}|initial'`)|**无关**|生成唯一一条初始噪声，八种方法共用|
|初始折线|`initial_noise(scene,seed,128)` 后把首末点设成 `cond`|**无关**|同上；`NativePointTask.initial_override`、`sample_points(init=)`、`run_method(init_noise=)`|
|多粒子附加粒子|`np.random.default_rng(initial_seed(scene,seed))`|**无关**|粒子 1..P−1；保证 Primal-Dual 与 ALM 拿到**完全相同**的 4 粒子集合|
|逐级采样噪声|`derived_seed(scene,seed,method)` = sha256(`'{scene}|{seed}|{method}'`)|有关|各算法自己的 Langevin/SGLD 噪声，必须独立（不同算法随机变量数量不同）|
|算法内部优化种子|求解器全部确定性（OSQP/SLSQP/trust-constr 无随机）|—|—|

要点：**“所有方法共享初始状态”与“每个方法有自己的采样噪声”是两件事**，不能强行让不同算法使用相同数量的
随机变量（PDM 是 10×20 步 SGLD，CD 是 256+64 步，SafeDiffuser 是 256 级微步积分）。

## 2. B1：PD/ALM 的四粒子（本轮修复的真 bug）

修复前：`NativePointTask.initial` 只在 `particles==1` 时使用 `initial_override`，因此
`particles=4` 的 Primal-Dual / ALM 完全忽略公共初始噪声，且两者因 `derived_seed` 含方法名而**各自不同**。

修复后（`adapters/native_points.py: initial`）：

```
particle 0      = initial_override（八方法共享的初始折线）
particle 1..P-1 = default_rng(initial_seed(scene,seed)).normal(...)     # 与方法无关
```

回归测试 `tests/test_phase_b_protocol.py`：

- `test_b1_pd_and_alm_share_the_same_four_particle_initial_state`：两者初始化数组**逐元素相等**，粒子 0 等于共享噪声，
  粒子 1 与粒子 0 不同（不是复制粘贴伪造的 4 粒子）；
- `test_b1_particle_set_is_reproducible_and_method_independent`：换 rng 不影响结果，换场景种子会改变结果；
- `test_b1_single_particle_methods_still_get_the_shared_noise`。

## 3. B2：val / test 索引分离

修复前：`--split test` 仍然读 `cfg['validation_indices']`（那是 1550 个 val 样本的下标，用到 420 个 test 样本上是错的/越界的）。

修复后：`scripts/validate_point_methods.py: resolve_indices(cfg, split, spec, dataset_size)`

- `val` → `cfg['validation_indices']`；`test` → `cfg['test_indices']`（冻结 20 个，0…419 等距）；
- `all` → `range(dataset_size)`；显式列表按逗号解析；
- 越界 → `IndexError`；重复 → `ValueError`；
- manifest 里记录 `split`、`indices`、`scene_ids`。

实测（`sample_id.npy`）：冻结 val 场景 id {61001, 69032, …}、冻结 test 场景 id {1001, 2003, …}，
**交集为空**；整个 val 与 test 划分的 id 交集也是 0（与 `docs/split_audit.json` 一致）。

## 4. 仍未处理

- `best.pt` 是按 val 损失选的，正式表应在 test 上出；本轮只把 `--split test` 修好，**没有**跑正式 test 表。
- 多粒子方法是否该把 4 个粒子的期望换成更多粒子（原论文未规定），留给 Phase D5 的敏感性实验。
