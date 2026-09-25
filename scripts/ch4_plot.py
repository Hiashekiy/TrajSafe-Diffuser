#!/usr/bin/env python
"""第4章 4.4「安全约束引导过程分析」配图脚本（paper/TrajSafe-Diffuser）.

WHY 这个脚本只画图、不重算指标
------------------------------
``scripts/eval_chapter4.py`` 已经把每个「方法 × seed」的逐步过程序列写进了结果
JSON 的 ``per_step``（由 ``aggregate_steps`` 生成，字段名见那里的 ``STEP_SERIES``）。
论文 4.4 的图和 4.2 的表必须与评测输出**逐字对齐**，所以这里绝不在绘图脚本里
重算任何指标：本脚本只做三件事

1. 读 ``outputs/ch4/**/<method>__s<seed>.json``，把 ``per_step`` 里 scene 单位的量
   乘以**该 JSON 自己的** ``scene_to_meter``（不硬编码 80）换成米再画；
   ``null``（预热步 / 该方法不产生该量）保留为断点，**不画 0**；
2. 用模块级唯一的「方法顺序 / 颜色 / marker」字典作图，保证 4.4 的几张图
   颜色、图例顺序完全一致；
3. 把 4.2 主表渲染成 PNG（列 = 方法 | 碰撞数/总数 | 碰撞率% | SDF可行率% |
   SDF平均违规量(m) | SDF-P95违规量(m) | 安全点比例% | RMSE(m) | 平滑度(m) | 时间(ms)）。

四个子命令
----------
``curves``      4.4.1/4.4.2 主曲线：修正前违规量 / 修正后违规量 / 安全修正距离
``smoothness``  4.4.2 之 2：平滑度（safe 实线 vs raw 虚线）+ 逐步可行率
``cases``       4.4.3 中间轨迹：逐反向步的「原始预测 / 安全修正 / 最终输出」叠在
                占用栅格上
``table``       4.2 主表 PNG

``cases`` 为什么要重跑生成、以及怎么跑
-------------------------------------
4.4.3 要的是**同一条样本在反向过程中的中间轨迹**，而 ``eval_chapter4.py`` 的结果
JSON 只保存**逐 step 的聚合统计**，不保存单样本的曲线；占用栅格、start/goal 也
不在 JSON 里。所以 ``cases`` 必须现跑一次生成，且调用形状**逐参数照抄**
``scripts/eval_chapter4.py::main``：

    dataset = CarlaSplineDataset(split, processed_root, geometry_points=1280,
                                 indices=<要画的样本下标>)
    collate = make_collate(dataset)
    batch   = collate([dataset[local_index]])          # 局部下标
    schedule = NoiseSchedule(diffusion.timesteps, beta_schedule, beta_start, beta_end)
    model, _, load_report = load_model(cfg, ckpt, device=device, verbose=False)
    out = sample(model, schedule,
                 batch["cond"], batch["occupancy"],
                 batch["candidate_xy"], batch["candidate_mask"],
                 batch["candidate_geometry"], batch["candidate_geometry_lengths"],
                 device=device, steps=steps, seed=seed,
                 seed_offset=<该样本在 split 里的全局下标>,   # 与 campaign 同一份噪声
                 return_trace=True,
                 alm_config=alm_cfg, corridor_config=corridor_cfg,
                 guidance=guidance, sdf_config=sdf_cfg)

``seed_offset`` 用**全局样本下标**是公平性要求（指南 4.1.4）：同一 ``--seed`` 下
同一样本在所有方法里拿到完全相同的初始噪声，所以图里的中间轨迹可以直接对比。
但要注意 ``sample()`` 的噪声是**一次** ``torch.randn(B, C, 2)``（B = 该次批大小），
而 campaign 按 ``--chunk`` 分批、每批传 ``seed_offset=批起点``，所以「单样本 +
``seed_offset=index``」只在 ``index`` 恰好是 chunk 起点时才与该轮评测逐位一致。
要**完全复现**某一行，用 ``--exact --chunk <campaign 的 chunk>``：它整段重放那个
chunk（同批大小、同 device、同 ``seed_offset=批起点``）再取目标行。
（实测：只跑前 ``index+1`` 行是不够的 —— CPU 的 ``randn`` 前缀不变，但 cuRAND 不是，
B=3 与 B=16 同一 seed 下第 2 行就不同，所以必须整段重放。）

``out["trace"]`` 是**每个已执行反向步一个 dict**（长度 = steps），关键字段
（已核对 ``src/diffusion/sampler.py`` 中 ``trace.append({...})`` 的真实键名）：
``t`` / ``s`` / ``q0_raw`` / ``q0_safe`` / ``p_raw`` / ``p_safe`` / ``final`` /
``guided`` / ``guard_violation_before`` / ``guard_violation_after`` /
``raw_violation`` / ``alm_correction`` / ``curve_smoothness_raw`` /
``curve_smoothness_safe`` / ``smooth_raw`` / ``smooth_safe`` / ``alm_stats``。

注意一个**实测到的 sampler 语义**：``trace[i]["final"]`` 与 ``trace[i]["p_safe"]``
是同一个张量（都来自 ``decode_controls(q0_used)``），它并不是采样器最终返回的那条
轨迹 —— 真正的最终输出（含循环结束后的末端硬投影）在 ``out["p"]``。所以本脚本把
``out["p"]`` 画成「最终输出」曲线，并在运行时校验两个键确实相同，若将来 sampler
把 ``final`` 改成真·最终输出，这里的校验会失败并打印提示。

坐标约定（与 repo 完全一致，已对着 tests/test_bspline_carla.py::test_8_carla_y_axis
核对）：canonical occupancy（``occupancy.npy``，即 dataset 读到的那个）**行号随 +y
增大**，即 row 0 <=> y=-1。因此

    col = (x + 1) * 256 / 2 - 0.5        row = (y + 1) * 256 / 2 - 0.5
    imshow(occ, origin="lower", extent=[-1, 1, -1, 1])

三者等价（extent 把像素中心放在 (x=-1+cell/2 ...) 上），本脚本用 extent 直接以 scene
坐标画，点就不用再做像素换算。占用值为 1 = 障碍（``sampler._free_mask`` 判 free 的
条件是 occupancy <= 0.5），所以 ``cmap="gray_r"`` -> 障碍黑、可通行白。

用法
----
    python scripts/ch4_plot.py curves --runs outputs/ch4/smoke/*__s0.json \
        --out outputs/ch4/figures/process_curves.png
    python scripts/ch4_plot.py smoothness --runs outputs/ch4/smoke/*__s0.json \
        --out outputs/ch4/figures/smoothness.png
    python scripts/ch4_plot.py cases \
        --config configs/config_160k4p_c48_oneshot.yaml \
        --ckpt outputs/oneshot_k4p_c48/ckpt/best_task.pt \
        --guidance alm --samples 0,1,2 --out-dir outputs/ch4/figures/cases
    python scripts/ch4_plot.py table --runs outputs/ch4/smoke/*__s0.json \
        --out outputs/ch4/figures/main_table.png

无显示环境，固定 ``matplotlib.use("Agg")``。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")          # 必须在 pyplot 之前：本机没有 display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ===========================================================================
# 1. 全局样式：方法顺序 / 颜色 / marker（四张图共用同一份，改这里就全改）
# ===========================================================================

#: 论文 4.2 主表的方法行序；未列出的方法排在后面（按出现顺序）
METHOD_ORDER = [
    "Diffusion",
    "Penalty-guided",
    "Projected",
    "Primal-Dual",
    "SDF-ALM",
    "Full (A-C48+ALM+FB+Proj)",
]

#: 本文完整方法的 JSON ``method`` 值（4.4 的阴影区间默认选它）
FULL_METHOD = "Full (A-C48+ALM+FB+Proj)"

METHOD_STYLE = {
    "Diffusion": {"color": "#7f7f7f", "marker": "o",
                  "label": "Diffusion (无条件)"},
    "Penalty-guided": {"color": "#1f77b4", "marker": "s",
                       "label": "Penalty-guided"},
    "Projected": {"color": "#2ca02c", "marker": "^", "label": "Projected"},
    "Primal-Dual": {"color": "#9467bd", "marker": "v",
                    "label": "Primal-Dual"},
    "SDF-ALM": {"color": "#ff7f0e", "marker": "D", "label": "SDF-ALM"},
    FULL_METHOD: {"color": "#d62728", "marker": "*",
                  "label": "Full (A-C48+ALM+FB+Proj) 本文"},
}

#: 未知方法（例如临时跑的消融 tag）的确定性回退配色，保证每次跑同一名字同色
_FALLBACK_COLORS = ["#8c564b", "#17becf", "#e377c2", "#bcbd22", "#1f77b4",
                    "#2ca02c", "#ff7f0e", "#d62728", "#7f7f7f", "#9467bd"]
_FALLBACK_MARKERS = ["P", "X", "<", ">", "h", "p", "8", "H", "+", "x"]

#: 4.4 的三条过程曲线（与 eval_chapter4.STEP_SERIES 的键名一致）
PROCESS_SERIES = ("raw_violation", "safe_violation", "correction",
                  "smooth_raw", "smooth_safe")

#: 4.4.3 中间轨迹图的 5 个反向步（1-based 的「已执行反向步骤」）
CASE_STEPS = (1, 4, 8, 12, 16)


def method_style(method: str, fallback_index: int = 0) -> dict:
    """返回 ``{"color","marker","label"}``；未登记的方法走确定性回退。"""
    spec = METHOD_STYLE.get(method)
    if spec is not None:
        return dict(spec)
    return {
        "color": _FALLBACK_COLORS[fallback_index % len(_FALLBACK_COLORS)],
        "marker": _FALLBACK_MARKERS[fallback_index % len(_FALLBACK_MARKERS)],
        "label": method,
    }


def _method_rank(method: str, fallback_index: int) -> tuple:
    """图例排序键：先按 METHOD_ORDER，再按调用顺序（保证图例顺序稳定）。"""
    if method in METHOD_ORDER:
        return (METHOD_ORDER.index(method), fallback_index)
    return (len(METHOD_ORDER), fallback_index)


def is_full_method(run: dict) -> bool:
    """是不是论文的完整方法（走廊 ALM，含 feedback / 末端硬投影）。"""
    name = str(run.get("method") or "")
    if "Full" in name or "完整" in name:
        return True
    return str(run.get("guidance") or "").lower() == "alm"


# ===========================================================================
# 2. 中文字体：先探测可用字形，缺字形就整体退回英文标签（绝不输出方框）
# ===========================================================================

_CJK_FONT_CANDIDATES = (
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
    "Source Han Sans CN", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
    "PingFang SC", "Heiti SC", "Hiragino Sans GB", "DengXian",
    "Arial Unicode MS", "SimSun",
)

#: 图中真正会出现的汉字（用于验证候选字体确实含这些字形，防止 tofu 方框）
_PROBE_CHARS = ("第章方法安全约束引导过程分析已执行反向步骤修正前后违规量距离平滑度"
                "可行率碰撞总数时间完整预热原始预测终输出起点终点点集曲线分位区间"
                "所选均值阴影无处理注序列缺失未做任何外部位姿语料库")

#: 实际选中的 CJK 字体名；None = 没有可用 CJK 字体，全部标签退回英文
CJK_FONT: str | None = None


def _font_has_glyphs(font_path: str, chars: str) -> bool:
    """该字体文件是否覆盖 ``chars`` 中的每一个字符（False = 会画成方框）。"""
    try:
        from matplotlib.ft2font import FT2Font
        face = FT2Font(font_path)
        return all(face.get_char_index(ord(ch)) != 0 for ch in set(chars))
    except Exception:                                   # pragma: no cover
        return False


def configure_fonts(verbose: bool = True) -> str | None:
    """挑一个**真的含中文字形**的系统字体；没有就退回英文标签。"""
    global CJK_FONT
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_FONT_CANDIDATES:
        if name not in available:
            continue
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=name),
                fallback_to_default=False)
        except Exception:                               # pragma: no cover
            continue
        if not _font_has_glyphs(path, _PROBE_CHARS):
            if verbose:
                print("[ch4_plot] 字体 %s 缺部分中文字形，跳过" % name)
            continue
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False      # 负号不要画成方框
        CJK_FONT = name
        if verbose:
            print("[ch4_plot] CJK 字体: %s (%s) -> 中文标签启用" % (name, path))
        return name
    plt.rcParams["axes.unicode_minus"] = False
    if verbose:
        print("[ch4_plot] 警告: 系统里没有覆盖所需汉字的中文字体，"
              "全部标签退回英文（图中不会出现方框）。")
    return None


def L(zh: str, en: str) -> str:
    """标签选择器：有 CJK 字体用中文，否则用英文（中文原文仍保留在源码里）。"""
    return zh if CJK_FONT else en


# ===========================================================================
# 3. 读 JSON / 校验 / 单位换算
# ===========================================================================

_REQUIRED_HEADER_KEYS = ("method", "guidance", "scene_to_meter", "per_step",
                         "aggregate")


def load_run(path: str) -> dict:
    """读一个 eval_chapter4 结果 JSON，并校验 4.4/4.2 需要的最小字段集。"""
    if not os.path.isfile(path):
        raise SystemExit("[ch4_plot] 找不到输入 JSON: %s" % path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise SystemExit("[ch4_plot] 读取 %s 失败: %s" % (path, exc))
    missing = [key for key in _REQUIRED_HEADER_KEYS if key not in payload]
    if missing:
        raise SystemExit("[ch4_plot] %s 缺少必需字段 %s；"
                         "请用 scripts/eval_chapter4.py 重新生成"
                         % (path, ", ".join(missing)))
    step = payload["per_step"]
    if not step.get("step_index"):
        raise SystemExit("[ch4_plot] %s 的 per_step.step_index 为空；"
                         "该结果没有 4.4 过程序列，无法作图" % path)
    for name in PROCESS_SERIES + ("feasible_rate",):
        if name not in step:
            raise SystemExit("[ch4_plot] %s 的 per_step 缺少序列 %r"
                             % (path, name))
        for key in ("mean", "p25", "p75"):
            if key not in step[name]:
                raise SystemExit("[ch4_plot] %s 的 per_step[%r] 缺少 %r"
                                 % (path, name, key))
    n_steps = len(step["step_index"])
    for name in PROCESS_SERIES + ("feasible_rate", "lambda_max", "inner_steps"):
        if name in step and len(step[name]["mean"]) != n_steps:
            raise SystemExit(
                "[ch4_plot] %s: per_step[%r].mean 长度 %d != step_index 长度 %d"
                % (path, name, len(step[name]["mean"]), n_steps))
    payload["_path"] = path
    payload["_label"] = os.path.basename(path)
    return payload


def scene_to_meter(payload: dict) -> float:
    value = payload.get("scene_to_meter")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise SystemExit("[ch4_plot] %s 的 scene_to_meter 不是数字: %r"
                         % (payload.get("_path"), value))
    if value <= 0.0:
        raise SystemExit("[ch4_plot] %s 的 scene_to_meter <= 0: %r"
                         % (payload.get("_path"), value))
    return value


def series_m(payload: dict, name: str, key: str = "mean",
             to_metre: bool = True) -> np.ndarray:
    """取 per_step 的一条序列，``None`` -> NaN（画成断点），并按需换算成米。"""
    scale = scene_to_meter(payload) if to_metre else 1.0
    raw = payload["per_step"][name][key]
    return np.asarray([np.nan if v is None else float(v) * scale for v in raw],
                      dtype=float)


def step_index(payload: dict) -> np.ndarray:
    return np.asarray([int(v) for v in payload["per_step"]["step_index"]],
                      dtype=float)


def warmup_steps(payload: dict) -> int:
    try:
        return int(payload.get("warmup_reverse_steps") or 0)
    except (TypeError, ValueError):
        return 0


def _all_nan(values: np.ndarray) -> bool:
    return not bool(np.isfinite(values).any())


def select_band_run(runs: list) -> dict:
    """阴影区间（25%–75% 分位）用哪个 run：优先本文完整方法，否则第一个。"""
    for run in runs:
        if is_full_method(run):
            return run
    return runs[0]


def _unique_labels(runs: list) -> list:
    """同一 method 出现多次（多 seed）时给重复项加 #2/#3 后缀。"""
    seen: dict = {}
    out = []
    for index, run in enumerate(runs):
        base = method_style(run["method"], index)["label"]
        seen[base] = seen.get(base, 0) + 1
        out.append(base if seen[base] == 1 else "%s #%d" % (base, seen[base]))
    return out


def _null_note(runs: list, names) -> str | None:
    """哪些 run 的哪条序列全为 null —— 写进图脚注，避免读者以为是画错了。"""
    parts = []
    for run in runs:
        missing = [name for name in names if _all_nan(series_m(run, name))]
        if missing:
            parts.append("%s: %s" % (run["method"], ", ".join(missing)))
    if not parts:
        return None
    return L("注：以下序列在该 run 中全为 null（未做该处理 / 预热外无活动样本），"
             "曲线留空不画 0 —— ",
             "note: series are null in these runs (no such processing / no "
             "active sample) and are left as gaps, never drawn as 0 -- ") \
        + " | ".join(parts)


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def report_written(path: str) -> None:
    size = os.path.getsize(path) if os.path.exists(path) else -1
    print("[ch4_plot] wrote %s (%.1f KiB)" % (path, size / 1024.0))


def _pad_ylim(ax, floor: float = 0.02) -> None:
    """全 0 / 常量序列会让 matplotlib 给出退化 ylim，这里兜一下底。"""
    lo, hi = ax.get_ylim()
    if not (np.isfinite(lo) and np.isfinite(hi)):
        ax.set_ylim(-1.0, 1.0)
        return
    if hi - lo < 1e-9:
        pad = max(abs(lo) * 0.1, floor)
        ax.set_ylim(lo - pad, hi + pad)


def _warmup_marker(ax, warmup: int, annotate: bool = False) -> None:
    if warmup <= 0:
        return
    ax.axvline(warmup + 0.5, color="0.45", ls=":", lw=1.2, zorder=1)
    if annotate:
        ax.annotate(
            L("预热 %d 步（不做约束处理）" % warmup,
              "warm-up %d steps (no processing)" % warmup),
            xy=(warmup + 0.5, 0.98), xycoords=("data", "axes fraction"),
            xytext=(4, -4), textcoords="offset points", fontsize=8,
            color="0.35", va="top", ha="left")


def _empty_panel_note(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="0.45", wrap=True)


# ===========================================================================
# 4. 子命令 curves（4.4.1 / 4.4.2 主图）
# ===========================================================================

def cmd_curves(args) -> int:
    runs = [load_run(path) for path in args.runs]
    band_run = select_band_run(runs)
    labels = _unique_labels(runs)
    band_style = method_style(band_run["method"], runs.index(band_run))
    band_label = band_style["label"]

    panels = [
        ("raw_violation", ("修正前违规量 (m)", "violation before correction (m)"),
         True),
        ("safe_violation", ("修正后违规量 (m)", "violation after correction (m)"),
         True),
        ("correction", ("安全修正距离 (m)", "safety correction distance (m)"),
         True),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(10.4, 10.6), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.665, top=0.90, bottom=0.13,
                        hspace=0.14)

    # 图例条目先收集，最后按 METHOD_ORDER 排序，保证每张图的图例顺序一致
    legend_entries, legend_seen = [], set()
    band_patch = None

    for panel_index, (ax, (name, (zh_ylabel, en_ylabel), to_metre)) in \
            enumerate(zip(axes, panels)):
        drew_any = False
        for index, run in enumerate(runs):
            style = method_style(run["method"], index)
            x = step_index(run)
            y = series_m(run, name, "mean", to_metre)
            if _all_nan(y):
                continue
            line, = ax.plot(x, y, color=style["color"], marker=style["marker"],
                            ms=4.6, lw=1.7, zorder=3, label=labels[index])
            drew_any = True
            if panel_index == 0 and labels[index] not in legend_seen:
                legend_seen.add(labels[index])
                legend_entries.append((_method_rank(run["method"], index),
                                       line, labels[index]))
        # 25%–75% 分位阴影，只给「所选方法」
        lo = series_m(band_run, name, "p25", to_metre)
        hi = series_m(band_run, name, "p75", to_metre)
        if not (_all_nan(lo) or _all_nan(hi)):
            ax.fill_between(step_index(band_run), lo, hi,
                            color=band_style["color"], alpha=0.18, lw=0,
                            zorder=2)
            if panel_index == 0:
                band_patch = Patch(facecolor=band_style["color"], alpha=0.18,
                                   lw=0)
        if not drew_any:
            _empty_panel_note(
                ax, L("该序列在所有 run 中均为 null（无可画数据）",
                      "series is null in every run (nothing to plot)"))
        ylabel = L(zh_ylabel, en_ylabel)
        if CJK_FONT:
            ylabel = "%s\n%s" % (ylabel, L("均值", "mean"))
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.grid(alpha=0.25, lw=0.6, zorder=0)
        _warmup_marker(ax, warmup_steps(runs[0]), annotate=(panel_index == 0))
        _pad_ylim(ax)

    legend_entries.sort(key=lambda item: item[0])
    legend_handles = [entry[1] for entry in legend_entries]
    legend_labels = [entry[2] for entry in legend_entries]
    if band_patch is not None:
        legend_handles.append(band_patch)
        # 图例里只写「分位区间」，具体是哪个方法写在标题里（否则标签太长被裁）
        legend_labels.append(L("25%–75% 分位区间", "p25-p75 band"))

    axes[-1].set_xlabel(L("已执行反向步骤", "executed reverse step"), fontsize=11.5)
    axes[-1].set_xticks(step_index(runs[0]))
    axes[-1].set_xlim(0.4, float(step_index(runs[0]).max()) + 0.6)

    fig.suptitle(
        L("图 4-7  安全约束引导过程曲线（均值；阴影 = %s 的 25%%–75%% 分位）"
          % band_label,
          "Fig 4-7  process curves (mean; shaded = p25-p75 of %s)"
          % band_label),
        fontsize=12.5)
    fig.text(0.10, 0.945,
             L("每条曲线 = 一个 run（方法 × seed），纵轴已按 scene_to_meter 换成米",
               "one line per run (method x seed); axis converted to metres via "
               "scene_to_meter"),
             fontsize=9, color="0.3")
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="center left",
                   bbox_to_anchor=(0.675, 0.55), fontsize=8.8, frameon=True,
                   title=L("方法", "method"))
    note = _null_note(runs, PROCESS_SERIES)
    if note:
        fig.text(0.01, 0.012, note, fontsize=7.6, color="0.3", wrap=True)

    ensure_parent(args.out)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    report_written(args.out)

    print("[ch4_plot] curves: %d run(s) | 阴影方法=%s | warmup=%d | "
          "scene_to_meter=%s"
          % (len(runs), band_run["method"], warmup_steps(runs[0]),
             scene_to_meter(runs[0])))
    for run in runs:
        counts = []
        for name in PROCESS_SERIES:
            n_valid = int(np.isfinite(series_m(run, name)).sum())
            counts.append("%s=%d/%d" % (name, n_valid,
                                        len(run["per_step"]["step_index"])))
        print("    %-32s %s" % (run["method"], " ".join(counts)))
    return 0


# ===========================================================================
# 5. 子命令 smoothness（4.4.2 之 2）
# ===========================================================================

def cmd_smoothness(args) -> int:
    runs = [load_run(path) for path in args.runs]
    band_run = select_band_run(runs)
    index = runs.index(band_run)
    style = method_style(band_run["method"], index)
    label = style["label"]
    labels = _unique_labels(runs)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    fig.subplots_adjust(left=0.07, right=0.78, top=0.83, bottom=0.14,
                        wspace=0.26)
    ax_raw, ax_feas = axes

    x = step_index(band_run)
    raw = series_m(band_run, "smooth_raw", "mean", True)
    safe = series_m(band_run, "smooth_safe", "mean", True)
    drew = False
    if not _all_nan(raw):
        ax_raw.plot(x, raw, color=style["color"], ls="--", marker="o", ms=4.2,
                    lw=1.6, label=L("修正前 smooth_raw", "raw smooth_raw"))
        drew = True
    if not _all_nan(safe):
        ax_raw.plot(x, safe, color=style["color"], ls="-", marker="s", ms=4.2,
                    lw=1.9, label=L("修正后 smooth_safe", "safe smooth_safe"))
        drew = True
    if not drew:
        _empty_panel_note(ax_raw, L("所选方法的平滑度序列全为 null",
                                    "smoothness series is null for the "
                                    "selected method"))
    ax_raw.set_title(L("平滑度：修正前 vs 修正后（%s）" % label,
                       "smoothness: raw vs safe (%s)" % label), fontsize=11)
    ax_raw.set_xlabel(L("已执行反向步骤", "executed reverse step"))
    ax_raw.set_ylabel(L("平均二阶差分 (m)", "mean 2nd difference (m)"))
    ax_raw.grid(alpha=0.25, lw=0.6)
    ax_raw.set_xticks(x)
    _warmup_marker(ax_raw, warmup_steps(band_run), annotate=True)
    _pad_ylim(ax_raw, floor=1e-3)
    ax_raw.legend(fontsize=9, loc="best")

    entries = []
    for i, run in enumerate(runs):
        st = method_style(run["method"], i)
        y = series_m(run, "feasible_rate", "mean", False)
        if _all_nan(y):
            continue
        line, = ax_feas.plot(step_index(run), y, color=st["color"],
                             marker=st["marker"], ms=4.4, lw=1.6,
                             label=labels[i])
        entries.append((_method_rank(run["method"], i), line, labels[i]))
    # 与 curves 用同一个 METHOD_ORDER 排序，图例顺序跨图一致
    entries.sort(key=lambda item: item[0])
    handles = [entry[1] for entry in entries]
    legend_labels = [entry[2] for entry in entries]
    ax_feas.set_title(L("逐步可行率 feasible_rate", "per-step feasible rate"),
                      fontsize=11)
    ax_feas.set_xlabel(L("已执行反向步骤", "executed reverse step"))
    ax_feas.set_ylabel(L("可行样本比例", "feasible sample fraction"))
    ax_feas.set_ylim(-0.05, 1.05)
    ax_feas.set_yticks(np.linspace(0.0, 1.0, 6))
    ax_feas.set_xticks(step_index(runs[0]))
    ax_feas.grid(alpha=0.25, lw=0.6)
    _warmup_marker(ax_feas, warmup_steps(runs[0]))
    if not handles:
        _empty_panel_note(ax_feas, L("所有 run 的 feasible_rate 均为 null",
                                     "feasible_rate is null in every run"))

    fig.suptitle(L("图 4-8  平滑度与逐步可行率（%s）" % label,
                   "Fig 4-8  smoothness and per-step feasible rate (%s)"
                   % label), fontsize=13)
    if handles:
        fig.legend(handles, legend_labels, loc="center left",
                   bbox_to_anchor=(0.79, 0.5), fontsize=9, frameon=True,
                   title=L("方法", "method"))
    note = _null_note(runs, ("smooth_raw", "smooth_safe", "feasible_rate"))
    if note:
        fig.text(0.01, 0.012, note, fontsize=7.6, color="0.3", wrap=True)

    ensure_parent(args.out)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    report_written(args.out)
    print("[ch4_plot] smoothness: 选定方法=%s | gap(smooth_raw)=%d "
          "gap(smooth_safe)=%d"
          % (band_run["method"], int(_all_nan(raw)), int(_all_nan(safe))))
    return 0


# ===========================================================================
# 6. 子命令 cases（4.4.3 中间轨迹，需要重跑生成）
# ===========================================================================

def _occupancy_free_rate(occ_row: np.ndarray, points: np.ndarray,
                         res: int = 256) -> float:
    """按 repo 约定（row 随 +y 增大）查点是否落在 free 像素（<=0.5）上。"""
    cell = 2.0 / float(res)
    iy = np.floor((points[:, 1] + 1.0) / cell).astype(int)
    ix = np.floor((points[:, 0] + 1.0) / cell).astype(int)
    inside = (iy >= 0) & (iy < res) & (ix >= 0) & (ix < res)
    if not inside.all():
        return float("nan")
    return float((occ_row[iy, ix] <= 0.5).mean())


def _to_float(value, index: int = 0) -> float | None:
    """trace 里的值是 tensor / numpy / None，统一成第 index 个样本的 float。"""
    if value is None:
        return None
    try:
        array = np.asarray(value.detach().cpu() if hasattr(value, "detach")
                           else value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return float(array[index]) if array.size > index else None


def _case_violations(step: dict, scale: float, row: int = 0) -> dict:
    """一个反向步的「修正前/修正后违规量 + 修正距离」，单位米（可能为 None）。"""
    before = step.get("guard_violation_before")
    after = step.get("guard_violation_after")
    correction = step.get("guard_correction")
    alm = step.get("alm_stats") or {}
    if before is None:
        before = step.get("raw_violation")
    if after is None:
        after = alm.get("max_violation_after")
    if correction is None:
        correction = step.get("alm_correction")

    def _m(value):
        value = _to_float(value, row)
        return None if value is None else value * scale
    return {"before": _m(before), "after": _m(after), "corr": _m(correction)}


def _fmt_m(value, fmt: str = "%.3f") -> str:
    return "n/a" if value is None else fmt % value


def _run_one_generation(cfg_path: str, ckpt: str, guidance: str, split: str,
                        steps: int, seed: int, device: str | None,
                        processed: str | None, indices: list,
                        alm_set, sdf_set, feedback, final_project,
                        allow_partial_load: bool, chunk: int = 32,
                        exact: bool = False, campaign_samples: int = 0,
                        verbose: bool = True) -> list:
    """按 eval_chapter4.main 的调用形状，给每个全局样本下标跑一次生成。

    返回 ``[(global_index, batch, result, scene_to_meter, row), ...]``，``row`` 是
    该样本在这一次 ``sample()`` 批里的行号。

    噪声与 campaign 的对齐（重要，已实测）
    -------------------------------------
    ``sample()`` 里初始噪声是**一次** ``torch.randn(B, C, 2)``（B = 该次批大小），
    而 ``eval_chapter4`` 按 ``--chunk`` 分批、每批传 ``seed_offset=批起点``，所以
    只有**同样的批**才能复现同样的噪声：

    * 默认（``exact=False``）单样本一批 + ``seed_offset=index``：快，但只有
      ``index`` 恰好是 campaign chunk 起点时噪声才与该轮评测一致（其余情况是
      另一份合法采样，仍可用于看图）；
    * ``exact=True``：整段重放 campaign 的那一个 chunk（``[chunk_start,
      chunk_start + chunk)``，同一批大小 / 同一 device / 同一个
      ``seed_offset=chunk_start``），再取目标行。**不能**只跑前 ``index+1`` 行来
      省算力：CPU 的 ``randn`` 是前缀不变的，但 cuRAND 不是（实测 B=3 与 B=16
      同一 seed 下第 2 行不同），所以只有整段重放才在 GPU 上逐位一致。
    """
    import torch

    from src.datasets.carla_spline_dataset import (CarlaSplineDataset,
                                                   make_collate)
    from src.diffusion.sampler import sample
    from src.diffusion.schedule import NoiseSchedule
    from src.utils.checkpoint import load_model
    from src.utils.config import load_config

    cfg = load_config(cfg_path)
    if processed:
        cfg.setdefault("data", {})["processed_root"] = processed
    processed_root = cfg["data"]["processed_root"]
    scene_to_meter_value = float(cfg["data"].get("scene_to_meter", 80.0))

    alm_cfg = dict(cfg.get("alm") or {})
    corridor_cfg = dict(cfg.get("corridor") or {})
    _apply_overrides(alm_cfg, alm_set, "alm-set")
    if feedback is not None:
        # 与 eval_chapter4 一致：4.3.1 的 feedback 开关语义
        alm_cfg["force_feedback_valid_zero"] = (feedback == "off")
    if final_project is not None:
        alm_cfg["final_project"] = (final_project == "on")
    sdf_cfg: dict = {}
    _apply_overrides(sdf_cfg, sdf_set, "sdf-set")
    alm_cfg["guidance"] = guidance

    dev = torch.device(device or ("cuda" if torch.cuda.is_available()
                                  else "cpu"))
    dcfg = cfg["diffusion"]
    schedule = NoiseSchedule(
        int(dcfg.get("timesteps", 16)),
        beta_schedule=dcfg.get("beta_schedule", "squaredcos_cap_v2"),
        beta_start=float(dcfg.get("beta_start", 1e-4)),
        beta_end=float(dcfg.get("beta_end", 0.02)))
    model, _, load_report = load_model(cfg, ckpt, device=dev, verbose=False)
    if load_report is not None and not allow_partial_load:
        missing = list(load_report.get("missing") or [])
        adapted = list(load_report.get("adapted") or [])
        unexpected = list(load_report.get("unexpected") or [])
        if missing or adapted or unexpected:
            raise SystemExit(
                "[ch4_plot] checkpoint/config 不匹配 | config=%s | missing=%d "
                "%s | shape-adapted=%d %s | unexpected=%d %s | "
                "（确认无误再传 --allow-partial-load）"
                % (cfg_path, len(missing), missing[:6], len(adapted),
                   adapted[:4], len(unexpected), unexpected[:6]))

    # 不带 limit / indices：dataset[i] 就是 split 的第 i 行，与 campaign 同一映射
    dataset = CarlaSplineDataset(split, processed_root, geometry_points=1280)
    collate = make_collate(dataset)
    total = len(dataset)
    if campaign_samples and campaign_samples > 0:
        # campaign 用了 --samples N 时，最后一个 chunk 会比 chunk 短
        total = min(total, int(campaign_samples))
    for index in indices:
        if index >= total:
            raise SystemExit(
                "[ch4_plot] --samples %d 超界: split=%s 只有 %d 个样本"
                "%s；campaign 截断请用 --campaign-samples"
                % (index, split, total,
                   "（你传了 --campaign-samples %d）" % campaign_samples
                   if campaign_samples else ""))

    chunk = max(1, int(chunk))
    jobs: dict = {}
    for index in indices:
        if exact:
            start = (index // chunk) * chunk
            jobs.setdefault((start, min(start + chunk, total)), []).append(index)
        else:
            jobs.setdefault((index, index + 1), []).append(index)
    if verbose:
        print("[ch4_plot] cases: split=%s | %d sample(s) %s | device=%s | "
              "guidance=%s | steps=%d | seed=%d | exact=%s (chunk=%d)"
              % (split, len(indices), indices, dev, guidance, steps, seed,
                 exact, chunk))
        if exact:
            print("[ch4_plot] --exact: 整段重放 %d 个 campaign chunk %s "
                  "（每个 chunk 一次 sample()，与评测同批同 seed_offset）"
                  % (len(jobs), sorted(jobs)))
        elif any(index % chunk for index in indices):
            print("[ch4_plot] 注意: 未开 --exact，单样本批次的 seed_offset=index，"
                  "只有 index 落在 campaign chunk 起点时才与该轮评测噪声逐位一致；"
                  "要完全复现请加 --exact --chunk <campaign 的 chunk>")

    out_list = []
    for (start, end), targets in jobs.items():
        batch = collate([dataset[row] for row in range(start, end)])
        with torch.no_grad():
            result = sample(
                model, schedule,
                batch["cond"].to(dev), batch["occupancy"].to(dev),
                batch["candidate_xy"].to(dev),
                batch["candidate_mask"].to(dev),
                batch["candidate_geometry"].to(dev),
                batch["candidate_geometry_lengths"].to(dev),
                device=str(dev), steps=steps, seed=seed,
                seed_offset=start,       # 与 campaign 同一份噪声（见 docstring）
                return_trace=True,
                alm_config=alm_cfg, corridor_config=corridor_cfg,
                guidance=guidance, sdf_config=sdf_cfg)
        for index in targets:
            out_list.append((index, batch, result, scene_to_meter_value,
                             index - start))
    out_list.sort(key=lambda item: indices.index(item[0]))
    return out_list


def _draw_case_figure(global_index: int, batch: dict, result: dict,
                      scale: float, guidance: str, cfg_path: str, ckpt: str,
                      seed: int, out_path: str, row: int = 0) -> dict:
    """一个样本一张图：5 个反向步并排，占用栅格 + 3 条曲线 + 起终点。

    ``row`` 是目标样本在 ``batch`` / ``result`` 里的行号（``--exact`` 会为了
    对齐 campaign 噪声而多跑几行，只画最后一行）。
    """
    import torch

    trace = result.get("trace") or []
    if not trace:
        raise SystemExit("[ch4_plot] 该次生成没有 trace（需要 return_trace=True）")
    occ = np.asarray(batch["occupancy"][row, 0].detach().cpu(), dtype=float)
    cond = np.asarray(batch["cond"][row].detach().cpu(), dtype=float)
    final_curve = np.asarray(result["p"][row].detach().cpu(), dtype=float)

    # sampler 的 trace["final"] 目前就是 trace["p_safe"] 的别名（见模块 docstring）
    alias = all(
        step.get("final") is not None and step.get("p_safe") is not None
        and bool(torch.allclose(step["final"], step["p_safe"]))
        for step in trace)
    if not alias:
        print("[ch4_plot] 提示: sampler 的 trace['final'] 已不再等于 "
              "trace['p_safe']，请确认本图「最终输出」的定义")

    show = [step for step in CASE_STEPS if 1 <= step <= len(trace)]
    dropped = [step for step in CASE_STEPS if step > len(trace)]
    if dropped:
        print("[ch4_plot] 警告: trace 只有 %d 步，跳过后面的第 %s 步"
              % (len(trace), dropped))

    n_show = len(show)
    fig, axes = plt.subplots(1, n_show, figsize=(4.15 * n_show, 5.4))
    fig.subplots_adjust(left=0.035, right=0.99, top=0.775, bottom=0.085,
                        wspace=0.10)
    if n_show == 1:
        axes = [axes]

    handles = [
        Line2D([], [], color="#1f77b4", ls="--", lw=1.8,
               label=L("原始预测（修正前）", "raw prediction (before)")),
        Line2D([], [], color="#d62728", ls="-", lw=1.9,
               label=L("安全修正后", "after safety correction")),
        Line2D([], [], color="#2ca02c", ls="-.", lw=1.9,
               label=L("最终输出（含末端硬投影）", "final output")),
        Line2D([], [], color="none", marker="o", ms=9, mfc="#22c55e",
               mec="black", label=L("起点", "start")),
        Line2D([], [], color="none", marker="*", ms=14, mfc="#eab308",
               mec="black", label=L("终点", "goal")),
    ]

    summary = []
    for ax, step_no in zip(axes, show):
        step = trace[step_no - 1]
        ax.imshow(occ, origin="lower", extent=(-1.0, 1.0, -1.0, 1.0),
                  cmap="gray_r", vmin=0.0, vmax=1.0,
                  interpolation="nearest", zorder=0)
        raw = np.asarray(step["p_raw"][row].detach().cpu(), dtype=float)
        safe = np.asarray(step["p_safe"][row].detach().cpu(), dtype=float)
        ax.plot(raw[:, 0], raw[:, 1], color="#1f77b4", ls="--", lw=1.8,
                zorder=4)
        ax.plot(safe[:, 0], safe[:, 1], color="#d62728", ls="-", lw=1.9,
                zorder=5)
        ax.plot(final_curve[:, 0], final_curve[:, 1], color="#2ca02c",
                ls="-.", lw=1.9, zorder=3)
        ax.plot(cond[0, 0], cond[0, 1], marker="o", ms=9, mfc="#22c55e",
                mec="black", mew=1.1, ls="none", zorder=6)
        ax.plot(cond[1, 0], cond[1, 1], marker="*", ms=14, mfc="#eab308",
                mec="black", mew=1.1, ls="none", zorder=6)

        info = _case_violations(step, scale, row)
        # 注意 ``guided`` 只反映走廊 ALM 的激活状态，四个外部 SDF 处理器走的是
        # guard_* 通道（guided 恒 False），所以这里用「该步是否真的产生了统计量」
        # 判断有没有做安全处理。
        active = (step.get("guard_violation_before") is not None
                  or step.get("guard_violation_after") is not None
                  or step.get("alm_stats") is not None
                  or step.get("raw_violation") is not None)
        sub = L("反向第%d步 (t=%d)" % (step_no, int(step["t"])),
                "reverse step %d (t=%d)" % (step_no, int(step["t"])))
        if not active:
            sub += "\n" + L("该步未做安全处理", "no safety processing")
        else:
            sub += "\n" + L("违规 %.3f→%.3f m | 修正 %.3f m",
                            "viol %.3f->%.3f m | corr %.3f m") % (
                info["before"] if info["before"] is not None else float("nan"),
                info["after"] if info["after"] is not None else float("nan"),
                info["corr"] if info["corr"] is not None else float("nan"))
            status = step.get("guard_status")
            if isinstance(status, list) and len(status) > row and status[row]:
                sub += "\n%s" % status[row]
        ax.set_title(sub, fontsize=9.5)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("0.6")
        if ax is axes[0]:
            ax.legend(handles=handles, fontsize=8.4, loc="upper left",
                      framealpha=0.92, borderpad=0.4)
        summary.append((step_no, int(step["t"]), info))

    # 起点/终点/GT 是否落在 free 像素上（与 test_8_carla_y_axis 同一约定）
    free_start = _occupancy_free_rate(occ, cond[0][None, :])
    free_goal = _occupancy_free_rate(occ, cond[1][None, :])
    free_gt = _occupancy_free_rate(occ, np.asarray(batch["pos"][row]))
    if not (free_start == 1.0 and free_goal == 1.0):
        print("[ch4_plot] 警告: sample %d 的起点/终点不在 free 像素上 "
              "(start=%.2f goal=%.2f) -- 坐标约定可能有问题"
              % (global_index, free_start, free_goal))

    fig.suptitle(
        L("图 4-9  样本 %d 的中间轨迹（guidance=%s，%d 步反向去噪）"
          % (global_index, guidance, len(trace)),
          "Fig 4-9  intermediate trajectories of sample %d (guidance=%s, %d "
          "reverse steps)" % (global_index, guidance, len(trace))),
        fontsize=12, y=0.985)
    fig.text(0.035, 0.915,
             "sample_id=%s | %s | seed=%d | %s | scene_to_meter=%.1f "
             "(1 scene = %.1f m) | 起终点 free rate %.2f/%.2f | GT free rate %.2f"
             % (int(batch["sample_id"][row]), os.path.basename(cfg_path), seed,
                os.path.basename(ckpt), scale, scale, free_start, free_goal,
                free_gt),
             fontsize=8.2, color="0.3")
    fig.text(0.035, 0.014,
             L("背景 = 占用栅格（origin=lower, extent=[-1,1,-1,1]，行号随 +y 增大；"
               "黑 = 障碍 occupancy>0.5，白 = 可通行）；曲线为 scene 坐标下的 128 点解码",
               "background = occupancy grid (origin=lower, extent=[-1,1,-1,1], "
               "row index grows with +y; black = obstacle); curves are the "
               "128-point decode in scene coordinates"),
             fontsize=7.8, color="0.3")

    ensure_parent(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    report_written(out_path)
    return {"free_start": free_start, "free_goal": free_goal,
            "free_gt": free_gt, "steps": summary, "alias": alias}


def cmd_cases(args) -> int:
    indices = []
    for token in str(args.samples).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            indices.append(int(token))
        except ValueError:
            raise SystemExit("[ch4_plot] --samples 需要逗号分隔的整数下标，"
                             "收到 %r" % args.samples)
    if not indices:
        raise SystemExit("[ch4_plot] --samples 为空（例如 --samples 0,1,2）")
    if any(i < 0 for i in indices):
        raise SystemExit("[ch4_plot] --samples 不能含负数: %s" % indices)
    if len(set(indices)) != len(indices):
        keep, seen = [], set()
        for i in indices:
            if i not in seen:
                seen.add(i)
                keep.append(i)
        print("[ch4_plot] --samples 有重复，去重后: %s" % keep)
        indices = keep

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    results = _run_one_generation(
        args.config, args.ckpt, args.guidance, args.split, args.steps,
        args.seed, args.device, args.processed, indices, args.alm_set,
        args.sdf_set, args.feedback, args.final_project,
        args.allow_partial_load, chunk=args.chunk, exact=args.exact,
        campaign_samples=args.campaign_samples)

    written = []
    for global_index, batch, result, scale, row in results:
        out_path = os.path.join(out_dir, "case_%03d_%s.png"
                                % (global_index, args.guidance))
        report = _draw_case_figure(global_index, batch, result, scale,
                                   args.guidance, args.config, args.ckpt,
                                   args.seed, out_path, row=row)
        written.append(out_path)
        print("[ch4_plot]   sample %3d: 起终点 free=%.2f/%.2f GT free=%.2f | "
              "trace.final==trace.p_safe: %s"
              % (global_index, report["free_start"], report["free_goal"],
                 report["free_gt"], report["alias"]))
        for step_no, timestep, info in report["steps"]:
            print("      反向第%-2d步 (t=%-2d): 违规 %s -> %s m | 修正 %s m"
                  % (step_no, timestep, _fmt_m(info["before"]),
                     _fmt_m(info["after"]), _fmt_m(info["corr"])))
    print("[ch4_plot] cases: 写出 %d 张图到 %s" % (len(written), out_dir))
    return 0


# ===========================================================================
# 7. 子命令 table（4.2 主表）
# ===========================================================================

def _num(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if (out != out) else out          # NaN -> None


def _fmt_or_dash(value, scale: float = 1.0, fmt: str = "%.4f") -> str:
    number = _num(value)
    return "n/a" if number is None else fmt % (number * scale)


def _table_columns() -> list:
    """4.2 主表的 10 列定义（全部取值来自 JSON，best 方向用于加粗）。"""
    return [
        {"label": L("方法", "method"),
         "text": lambda a, name: name, "value": None, "best": None,
         "align": "left"},
        {"label": L("碰撞数/总数", "collisions/total"),
         "text": lambda a, name: "%d/%d" % (int(a["collisions"]), int(a["n"])),
         "value": lambda a: float(a["collisions"]), "best": "low",
         "align": "center"},
        {"label": L("碰撞率%", "collision rate %"),
         "text": lambda a, name: _fmt_or_dash(a.get("collision_rate"), 100.0,
                                              "%.2f"),
         "value": lambda a: _num(a.get("collision_rate")), "best": "low",
         "align": "center"},
        {"label": L("SDF可行率%", "SDF feasible %"),
         "text": lambda a, name: _fmt_or_dash(a.get("sdf_feasible_rate"),
                                              100.0, "%.2f"),
         "value": lambda a: _num(a.get("sdf_feasible_rate")), "best": "high",
         "align": "center"},
        {"label": L("SDF平均违规量(m)", "SDF mean viol (m)"),
         "text": lambda a, name: _fmt_or_dash(a.get("sdf_mean_violation_m"),
                                              1.0, "%.4f"),
         "value": lambda a: _num(a.get("sdf_mean_violation_m")), "best": "low",
         "align": "center"},
        {"label": L("SDF-P95违规量(m)", "SDF P95 viol (m)"),
         "text": lambda a, name: _fmt_or_dash(a.get("sdf_p95_violation_m"),
                                              1.0, "%.4f"),
         "value": lambda a: _num(a.get("sdf_p95_violation_m")), "best": "low",
         "align": "center"},
        {"label": L("安全点比例%", "safe-point %"),
         "text": lambda a, name: _fmt_or_dash(a.get("sdf_safe_point_rate"),
                                              100.0, "%.2f"),
         "value": lambda a: _num(a.get("sdf_safe_point_rate")), "best": "high",
         "align": "center"},
        {"label": L("RMSE(m)", "RMSE (m)"),
         "text": lambda a, name: _fmt_or_dash(a.get("rmse_m"), 1.0, "%.2f"),
         "value": lambda a: _num(a.get("rmse_m")), "best": "low",
         "align": "center"},
        {"label": L("平滑度(m)", "smoothness (m)"),
         "text": lambda a, name: _fmt_or_dash(a.get("smoothness_m"), 1.0,
                                              "%.3f"),
         "value": lambda a: _num(a.get("smoothness_m")), "best": "low",
         "align": "center"},
        {"label": L("时间(ms)", "time (ms)"),
         "text": lambda a, name: _fmt_or_dash(a.get("runtime_ms_mean"), 1.0,
                                              "%.1f"),
         "value": lambda a: _num(a.get("runtime_ms_mean")), "best": "low",
         "align": "center"},
    ]


def _best_flags(rows: list, columns: list) -> list:
    """每列的最优值集合（用于加粗）；全列为 None 的列不加粗。"""
    flags = [[False] * len(columns) for _ in rows]
    for col_index, column in enumerate(columns):
        if column["value"] is None or column["best"] is None:
            continue
        values = [column["value"](row["aggregate"]) for row in rows]
        present = [v for v in values if v is not None]
        if not present:
            continue
        best = min(present) if column["best"] == "low" else max(present)
        for row_index, value in enumerate(values):
            if value is not None and abs(value - best) <= 1e-12:
                flags[row_index][col_index] = True
    return flags


def cmd_table(args) -> int:
    runs = [load_run(path) for path in args.runs]
    columns = _table_columns()
    names = []
    seen: dict = {}
    for index, run in enumerate(runs):
        base = run["method"]
        seen[base] = seen.get(base, 0) + 1
        names.append(base if seen[base] == 1 else "%s #%d" % (base, seen[base]))

    rows_text = []
    for run, name in zip(runs, names):
        agg = run["aggregate"]
        rows_text.append([column["text"](agg, name) for column in columns])
    flags = _best_flags(runs, columns)

    header = [column["label"] for column in columns]
    n_data = len(rows_text)
    n_rows = n_data + 1
    fig = plt.figure(figsize=(16.5, 1.75 + 0.52 * n_rows))
    ax = fig.add_axes([0.02, 0.08, 0.96, 0.76])
    ax.axis("off")

    col_widths = [0.22] + [(1.0 - 0.22) / (len(columns) - 1)] * (len(columns) - 1)
    table = ax.table(cellText=[header] + rows_text, colWidths=col_widths,
                     cellLoc="center", loc="center", bbox=[0.0, 0.0, 1.0, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.7)
        cell.set_edgecolor("0.75")
        cell.PAD = 0.045
        text = cell.get_text()
        if row == 0:
            text.set_fontweight("bold")
            cell.set_facecolor("#e8e8e8")
        elif flags[row - 1][col]:
            text.set_fontweight("bold")
            cell.set_facecolor("#fdf3d8")
        if col == 0:
            text.set_ha("left")

    protocol = runs[0]
    fig.suptitle(
        L("表 4-2  第4章主表：安全性与轨迹质量（数据直接取自 eval_chapter4 结果 JSON）",
          "Table 4-2  main results (read verbatim from the eval JSON)"),
        fontsize=13.5, y=0.975)
    fig.text(0.02, 0.905,
             "split=%s | n=%d/method | steps=%d | seed=%d | warmup=%d | "
             "dense_points=%s | scene_to_meter=%s | ckpt=%s"
             % (protocol.get("split"), protocol.get("n_samples"),
                protocol.get("steps"), protocol.get("seed"),
                protocol.get("warmup_reverse_steps"),
                protocol.get("dense_points"), protocol.get("scene_to_meter"),
                os.path.basename(str(protocol.get("checkpoint") or ""))),
             fontsize=9, color="0.3")

    footnote = L(
        "加粗 = 该列最优（碰撞数/碰撞率/违规量/RMSE/平滑度/时间 越低越好；"
        "SDF可行率/安全点比例 越高越好）。"
        "「时间(ms)」= runtime_ms_mean（eval_chapter4 排除前 3 个计时预热 chunk "
        "后的每轨迹墙钟时间），整轮都在预热内时该列为 n/a。",
        "bold = best in column (lower is better for collisions/rate/violation/"
        "RMSE/smoothness/time; higher for feasible/safe-point rates). "
        "'time (ms)' = runtime_ms_mean; n/a when every chunk was timing warm-up.")
    fig.text(0.02, 0.018, footnote, fontsize=8.2, color="0.3")

    # 最后一行是本文完整方法时，在它上方加一条分隔线
    last_name = names[-1] if names else ""
    separator = n_data >= 2 and ("Full" in last_name or "完整" in last_name)
    if separator:
        fig.canvas.draw()
        ax.axhline(1.0 / float(n_rows), xmin=0.0, xmax=1.0, color="black",
                   lw=1.8, zorder=5)

    ensure_parent(args.out)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    report_written(args.out)

    if separator:
        print("[ch4_plot] table: 在最后一行（%s）上方加了分隔线" % last_name)
    print("[ch4_plot] table 内容（加粗列以 * 标出）:")
    for row, name in zip(rows_text, names):
        print("    " + " | ".join(row))
    missing_time = [name for run, name in zip(runs, names)
                    if _num(run["aggregate"].get("runtime_ms_mean")) is None]
    if missing_time:
        print("[ch4_plot] 警告: 以下 run 的 runtime_ms_mean 为 null（计时 chunk "
              "数没超过预热），时间列显示 n/a: %s" % ", ".join(missing_time))
    return 0


# ===========================================================================
# 8. CLI
# ===========================================================================

def _cast(value: str):
    low = value.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def _apply_overrides(cfg: dict, items, label: str) -> None:
    """``--alm-set KEY=VALUE`` / ``--sdf-set KEY=VALUE``，与 eval_chapter4 同语义。"""
    for item in items or []:
        if "=" not in item:
            raise SystemExit("[ch4_plot] --%s 需要 KEY=VALUE，收到 %r"
                             % (label, item))
        key, _, value = item.partition("=")
        cfg[key.strip()] = _cast(value)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="第4章 4.4 配图 + 4.2 主表 PNG（纯读 eval_chapter4 的 JSON）")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("curves", help="4.4.1/4.4.2 过程曲线（3 面板）")
    p.add_argument("--runs", nargs="+", required=True, metavar="JSON")
    p.add_argument("--out", default=os.path.join(ROOT, "outputs", "ch4",
                                                 "figures", "process_curves.png"))
    p.set_defaults(func=cmd_curves)

    p = sub.add_parser("smoothness", help="4.4.2 平滑度 + 逐步可行率")
    p.add_argument("--runs", nargs="+", required=True, metavar="JSON")
    p.add_argument("--out", default=os.path.join(ROOT, "outputs", "ch4",
                                                 "figures", "smoothness.png"))
    p.set_defaults(func=cmd_smoothness)

    p = sub.add_parser("cases", help="4.4.3 中间轨迹（需要重跑一次生成）")
    p.add_argument("--config", default="configs/config_160k4p_c48_oneshot.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--guidance", default="alm",
                   choices=["none", "penalty", "projection", "primal_dual",
                            "alm_sdf", "alm"])
    p.add_argument("--samples", required=True,
                   help="逗号分隔的测试样本下标，例如 0,1,2")
    p.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "ch4",
                                                     "figures", "cases"))
    p.add_argument("--split", default="test")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=32,
                   help="campaign 评测时的 chunk（--exact 用它定位并重放整段 chunk）")
    p.add_argument("--campaign-samples", type=int, default=0,
                   help="campaign 评测时的 --samples N（0=整个 split）；用于确定"
                        "最后一个 chunk 的长度，保证 --exact 逐位复现")
    p.add_argument("--exact", action="store_true",
                   help="整段重放 campaign 的那个 chunk，使目标行的初始噪声与"
                        "该轮评测逐位一致（代价 = 一个 chunk 的算力）")
    p.add_argument("--device", default=None)
    p.add_argument("--processed", default=None)
    p.add_argument("--feedback", default=None, choices=["on", "off"])
    p.add_argument("--final-project", default=None, choices=["on", "off"])
    p.add_argument("--alm-set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--sdf-set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--allow-partial-load", action="store_true")
    p.set_defaults(func=cmd_cases)

    p = sub.add_parser("table", help="4.2 主表 PNG")
    p.add_argument("--runs", nargs="+", required=True, metavar="JSON")
    p.add_argument("--out", default=os.path.join(ROOT, "outputs", "ch4",
                                                 "figures", "main_table.png"))
    p.set_defaults(func=cmd_table)
    return ap


def main(argv=None) -> int:
    # Windows 控制台默认 cp936，中文日志会变乱码；显式按 UTF-8 输出
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                   # pragma: no cover
        pass
    args = build_parser().parse_args(argv)
    configure_fonts(verbose=True)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
