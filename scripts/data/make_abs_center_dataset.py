"""Create an ABSOLUTE-centre dataset from the existing offset one.

The physical ellipse centres are unchanged (offset ``e6[:2] = c - p``), so the
absolute version is simply ``e6_abs[:2] = e6_off[:2] + p``.  positions /
conditions / maze_id / maps / GT masks are reused unchanged — the offline GT
masks are rasterized from the *physical* centre, which does not change.  Only
``ellipses6.npy`` is rewritten.

Usage:
  python scripts/data/make_abs_center_dataset.py \
      --src data/processed_scene_v1 --dst data/processed_scene_v1_abscenter
"""
import argparse
import os
import shutil

import numpy as np

SPLITS = ("train", "val", "test")
COPIES = ("positions.npy", "conditions.npy", "maze_id.npy", "ellipse_masks64_u8.h5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed_scene_v1")
    ap.add_argument("--dst", default="data/processed_scene_v1_abscenter")
    args = ap.parse_args()
    src, dst = args.src, args.dst
    os.makedirs(dst, exist_ok=True)

    if os.path.isdir(os.path.join(src, "maps")) and not os.path.isdir(os.path.join(dst, "maps")):
        shutil.copytree(os.path.join(src, "maps"), os.path.join(dst, "maps"))
    if os.path.exists(os.path.join(src, "meta.json")):
        shutil.copy2(os.path.join(src, "meta.json"), os.path.join(dst, "meta.json"))

    for split in SPLITS:
        sd = os.path.join(src, split)
        dd = os.path.join(dst, split)
        if not os.path.isdir(sd):
            continue
        os.makedirs(dd, exist_ok=True)
        pos = np.load(os.path.join(sd, "positions.npy"))
        e6 = np.load(os.path.join(sd, "ellipses6.npy"))
        assert pos.shape[:2] == e6.shape[:2], f"{split}: pos/e6 shape mismatch"
        e6_abs = e6.copy()
        e6_abs[..., :2] = e6_abs[..., :2] + pos          # offset -> absolute centre
        np.save(os.path.join(dd, "ellipses6.npy"), e6_abs)
        # Reuse everything else unchanged (physical centres identical).
        for name in COPIES:
            p = os.path.join(sd, name)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(dd, name))
        # Sanity: abs_center - p must reproduce the original offset.
        assert np.allclose(e6_abs[0, :, :2] - pos[0], e6[0, :, :2]), f"{split}: transform mismatch"
        print(f"[{split}] n={len(pos)} e6_abs[..., :2] = offset + position (sanity OK)")
    print("DONE absolute-centre dataset at", dst)


if __name__ == "__main__":
    main()
