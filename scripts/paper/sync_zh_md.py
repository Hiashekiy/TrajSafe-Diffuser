#!/usr/bin/env python
"""从中文原稿生成同步版 `TrajSafe_paper_optimized_synced.md`。

**只读原文件**，把 Word / 英文 RA-L 版里已经修好、但没回写到 md 的几处差异补上：

1. 图 3-4：原稿引用的 `fig3-4a_denoising.png` / `fig3-4b_denoising_block.png`
   两个文件已被删除（源图合并为 `fig3-4_denoising.png`），改为单张合并图 +
   合并图题，正文里的 "图 3-4(a)/(b)" 相应改成 "图 3-4 上半部分/下半部分"。
2. 图 3-3(b)：精简版把「局部几何预测如图 3-3(b) 所示」删掉了，补回 3.3 节引言句。
   （图 3-3(a) 原稿本来就有引用，无需改动。）
3. 图 3-5：精简版把「如图 3-5 所示」删掉了，补回。

用法：
    python scripts/paper/sync_zh_md.py
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "论文" / "TrajSafe_methods_and_figures" / "TrajSafe_paper_optimized.md"
DST = ROOT / "论文" / "TrajSafe_methods_and_figures" / "TrajSafe_paper_optimized_synced.md"

FIG = ("D:/ProjectDirectory/Neural-IRISDiffuser/论文/TrajSafe_methods_and_figures"
       "/figures/png")

HEADER = """<!--
    本文件由 scripts/paper/sync_zh_md.py 从 TrajSafe_paper_optimized.md 自动生成，
    请勿手工编辑——改原稿后重新运行脚本即可。原文件未被修改。

    与原稿的差异（均为排版/引用层面的同步，未改动任何技术内容、数据或结论）：
    1. 图 3-4：fig3-4a/fig3-4b 两个图文件已被删除，合并为单张 fig3-4_denoising.png；
       两条图题合并为一条，正文 "图 3-4(a)/(b)" 改为 "图 3-4 上半部分/下半部分"。
    2. 图 3-3(b)：补回 3.3 节引言中的图引用（原稿 3.3 节无引言句）。
    3. 图 3-5：在 ALM 求解句前补回 "如图 3-5 所示"。
-->

"""

# (旧, 新, 期望匹配次数)
RULES: list[tuple[str, str, int]] = [
    # ---- 1. 图 3-4 (a) -> 合并图 + 合并图题 ----
    (
        f"![安全特征融合与干净控制点预测]({FIG}/fig3-4a_denoising.png)\n"
        "\n"
        "**图 3-4(a) 安全条件融合与控制点预测。** 控制特征通过交叉注意力读取局部安全隐特征，"
        "融合特征经三层条件去噪模块、坐标预测头和端点补偿得到满足起终点位置约束的控制点。\n",
        "![安全特征融合与干净控制点预测（上）与单层条件去噪模块（下）]"
        f"({FIG}/fig3-4_denoising.png)\n"
        "\n"
        "**图 3-4 安全条件融合与控制点预测（上）与单层条件去噪模块（下）。** "
        "上半部分：控制特征通过交叉注意力读取局部安全隐特征，融合特征经三层条件去噪模块、"
        "坐标预测头和端点补偿得到满足起终点位置约束的控制点。"
        "下半部分：单层条件去噪模块通过 AdaLN 注入扩散时间条件，依次进行控制点自注意力、"
        "全局地图交叉注意力和前馈变换，三个子层分别采用残差连接。\n",
        1,
    ),
    # ---- 1b. 删掉图 3-4(b) 的图 + 图题 ----
    (
        f"\n![单层条件去噪模块]({FIG}/fig3-4b_denoising_block.png)\n"
        "\n"
        "**图 3-4(b) 单层条件去噪模块。** 通过 AdaLN 注入扩散时间条件，"
        "依次进行控制点自注意力、全局地图交叉注意力和前馈变换，三个子层分别采用残差连接。\n",
        "",
        1,
    ),
    # ---- 1c. 正文里的 (a)/(b) 引用 ----
    ("如图 3-4(a) 所示，$\\widetilde H_{\\mathrm{ctrl}}$ 作为查询",
     "如图 3-4 上半部分所示，$\\widetilde H_{\\mathrm{ctrl}}$ 作为查询", 1),
    ("单层去噪模块如图 3-4(b) 所示",
     "单层去噪模块如图 3-4 下半部分所示", 1),
    # ---- 2. 图 3-3(b)：补回 3.3 节引言 ----
    (
        "### 3.3 安全特征与约束生成\n\n**（1）通路匹配与选择**",
        "### 3.3 安全特征与约束生成\n\n"
        "本节中，通路匹配与局部几何预测分别如图 3-3(a) 和图 3-3(b) 所示。\n\n"
        "**（1）通路匹配与选择**",
        1,
    ),
    # ---- 3. 图 3-5：补回正文引用 ----
    (
        "以 $Q^{\\mathrm{ref}}=\\widehat Q_{0,\\mathrm{bdry}}^{(t)}$ 为参考，采用 ALM 求解",
        "如图 3-5 所示，以 $Q^{\\mathrm{ref}}=\\widehat Q_{0,\\mathrm{bdry}}^{(t)}$ "
        "为参考，采用 ALM 求解",
        1,
    ),
]


def main() -> int:
    src = SRC.read_text(encoding="utf-8")
    out = src
    for i, (old, new, expect) in enumerate(RULES, 1):
        n = out.count(old)
        if n != expect:
            print(f"规则 {i} 匹配 {n} 次，期望 {expect} 次 —— 原稿可能已改动，中止")
            return 1
        out = out.replace(old, new)
        print(f"规则 {i} ok")
    out = HEADER + out

    if "fig3-4a_denoising" in out or "fig3-4b_denoising_block" in out:
        print("仍有失效图引用，中止")
        return 1

    DST.write_text(out, encoding="utf-8")
    print(f"\n原文件（未改动）: {SRC}")
    print(f"已写出          : {DST}")
    print(f"行数 {len(src.splitlines())} -> {len(out.splitlines())}")

    diff = list(difflib.unified_diff(
        src.splitlines(), out.splitlines(), "original", "synced",
        lineterm="", n=1))
    print(f"\n差异 {len([d for d in diff if d[:1] in '+-' and d[:3] not in ('+++','---')])} 行：")
    for d in diff[2:]:
        print("   " + d[:150])
    return 0


if __name__ == "__main__":
    sys.exit(main())
