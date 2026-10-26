#!/usr/bin/env python
"""把中文稿的源图同步进 LaTeX 工程（按 LaTeX 侧的文件名）。

`论文/.../figures/png/` 下的图用的是中文学位论文的编号（fig3-1_…），
而 `paper_ral_english/main.tex` 里用的是 IEEE 连续编号（fig1_…），
两边文件名不同，所以每次源图更新后都要跑一次本脚本。

只同步内容有变化的文件。纯 ASCII 输出，方便被 .bat 调用。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "论文" / "TrajSafe_methods_and_figures" / "figures" / "png"
DST = ROOT / "paper_ral_english" / "figures"

# (源文件名, LaTeX 工程里的文件名) —— 顺序与论文里 Fig. 1..7 一致
MAP: list[tuple[str, str]] = [
    ("fig3-1_framework.png",              "fig1_framework.png"),
    ("fig3-2_feature_fusion.png",         "fig2_feature_fusion.png"),
    ("fig3-3a_route_selection.png",       "fig3a_route_selection.png"),
    ("fig3-3b_safety_geometry.png",       "fig3b_safety_geometry.png"),
    ("fig3-4_denoising.png",              "fig4_denoising.png"),
    ("fig3-5_alm.jpg",                    "fig5_alm_constraint.jpg"),
    ("case_293_custom_endpoints_alm.png", "fig6_diffusion_step_correction.png"),
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    missing, updated, same = [], [], []
    for src_name, dst_name in MAP:
        s, d = SRC / src_name, DST / dst_name
        if not s.exists():
            missing.append(src_name)
            print(f"  MISSING  {src_name}")
            continue
        if d.exists() and _sha(s) == _sha(d):
            same.append(dst_name)
            print(f"  same     {dst_name}")
            continue
        d.write_bytes(s.read_bytes())
        updated.append(dst_name)
        print(f"  updated  {dst_name}   <- {src_name}")

    print(f"figure sync: {len(updated)} updated, {len(same)} unchanged, "
          f"{len(missing)} missing")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
