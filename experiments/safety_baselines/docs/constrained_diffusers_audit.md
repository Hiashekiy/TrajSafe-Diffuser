# Constrained Diffusers 审计（Phase D）

## D1 256 级完整采样

`cd_levels = 256 = 基座级数`，逐级访问（`level_ladder(256,256,64)` ⇒ 256 级 + 最低级 64 次额外更新）。
eq.(6) 的 `beta_t` 是"该级自己的 β"，只有在逐级访问时才严格成立，所以主配置用 `cd_beta="level"`。
`"interval"` 模式（`beta_eff = 1 - alphabar_t/alphabar_next`）仍实现并测试，用于跳采场景。

## D2 高斯平稳性（本轮更正一处错误结论）

单步递推 `x_{k+1} = (1 - β/(2v)) x_k + sqrt(β) z`：

```
稳定条件      |1 - β/(2v)| < 1        ⇔  β < 4v
平稳方差      V∞ = v / (1 - β/(4v)) = β / (1 - (1-β/(2v))^2)
```

`tests/test_phase_d_cd.py` 用 20 万步 Monte Carlo 在 4 组 `(v,β)` 上验证（含 MC 有效样本数修正的容差），
并验证等价的几何级数形式。

**更正**：本仓库此前称"`interval` 模式在高噪声段会让平稳方差发散"——**错**。实际计算 256 级日程：

|模式|max β/(4v)|V∞/v 上界|
|---|---|---|
|`level`|0.250|1.333|
|`interval`（64-of-256 跳采）|0.250|1.333|

两者最坏情形相同（都由最高噪声级 β→0.999 主导），**是 33% 的方差偏置，不是发散**。发散需要 β ≥ 4v，
本日程不满足。相关文档（`DSH_FIX_REPORT_20261114.md §4d`、`comparison_methods_implementation.md §2.3`）已同步更正。

选择 `level` + 不跳采的理由不是"interval 会炸"，而是 eq.(6) 的 β 本来就该是逐级自己的 β。

## D3 乘子单位缩放

`g'=s·g`、`J'=s·J` 下，要保持 x 更新与乘子更新都等价，必须：

```
λ' = λ/s        η' = η/s²        ρ' = ρ/s²
```

逐项推导：
- 漂移项 `λᵀJ`：`λ'ᵀJ' = (λ/s)(sJ) = λᵀJ` ✓
- PD 对偶：`λ'_{new} = λ' + η'E[g'] = λ/s + η'·s·E[g]`，要求等于 `λ_new/s = λ/s + ηE[g]/s` ⇒ `η'=η/s²` ✓
- ALM：`coeff'J' = (λ' + ρ'·s·residual)s = λ + ρ's²residual` ⇒ `ρ'=ρ/s²` ✓，且 `slack'=s·slack` 自洽

配置由 `scripts/make_aligned_configs.py` 从论文 Appendix C "large" 值反算：`dual_eta=1e-3/16384`、`rho=2.5e-4/16384`，
测试 `test_config_dual_parameters_are_the_paper_values_over_scale_squared` 与 `test_d3_multiplier_units_follow_the_squared_rule` 双向固定。

## D4 单次随机噪声

a17a667 已修：每次更新只抽一次噪声，同时用于 `last_nominal`（记录）与实际更新。
回归测试 `test_pd_and_alm_use_one_noise_draw_for_record_and_update`：构造零修正 toy（约束不激活 ⇒ 修正项为 0），
断言"更新后的状态 == 记录的名义状态"，逐元素相等。修复前两者因噪声不同必然不等。

每条记录现在含 `trajectory_before`（名义状态）与 `trajectory`（修正后状态），可直接算修正幅度。

## D5 粒子/超参敏感性（未完成，明确标记）

原计划测 `particles ∈ {1,2,4}` 与 `rho / dual_eta` 的合理区间。**本轮未做**，原因与现状：
- 修完 D3 后乘子量级整体缩小 16384 倍，`λ` 的实际取值需要先看一段时间序列再定搜索区间；
- 4 粒子的 `E[g]` 估计噪声大，测 `particles=1` 会退化为"单样本乘子"，与论文的期望口径不同，需要单独说明；
- 因此这一项在 `phase_acceptance_report.md` 的 G2 里标为**未完成**。

## 结论

CD 的三种更新式与论文 eq (10)(13)(14)(16)(17)(18) 一致；本轮修掉的是**日程口径**（D1）、**乘子单位**（D3）、
**噪声记录**（D4）。它们的合力量级 < 15%（见 `DSH_FIX_REPORT_20261114.md §5.4` 的前后对比），
说明 CD 的表现主要由基座与"顶点 vs 整段"的约束对象决定，而不是这些系数。
