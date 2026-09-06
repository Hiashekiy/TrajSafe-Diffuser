"""Kaggle "2d-path-planning-dataset" (dcaffo/2dpathplanningdataset) downloader + preview.

What it does
------------
1. Downloads a handful of ``.pt`` samples of the Kaggle dataset
   (map [100,100] occupancy, start/goal [2], D*-Lite GT path [n,2]).
   NOTE: kaggle CLI 1.7.4.5 builds a raw-slash per-file URL that the server
   rejects with 404; the fix used here URL-encodes the whole file path
   (slashes -> %%2F), which the server redirects to storage.
2. Loads samples despite the dataset author's missing ``dataset`` python
   namespace (samples pickle classes from ``dataset.map_sample``): a meta-path
   finder fabricates those modules/classes (see _kaggle_ns_fix/__init__.py).
3. Prints a per-sample summary and renders a map + start/goal + GT path grid.

Usage
-----
  python scripts/data/kaggle_preview.py                      # 6 test samples
  python scripts/data/kaggle_preview.py --split test --n 8
  python scripts/data/kaggle_preview.py --no-download --figure path.png
"""

import argparse
import base64
import json
import os
import sys
import urllib.parse

import requests
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data"))
from _kaggle_ns_fix import ensure_ns_fix  # noqa: E402

DATASET = "dcaffo/2dpathplanningdataset"
API_BASE = "https://www.kaggle.com/api/v1/datasets"
DEFAULT_SAMPLES = {
    "test": [
        "00059811-3c7e-4d56-8126-fc76231941b8",
        "0006484e-0fd9-42e1-923d-60687e2a46cb",
        "0007753b-248b-432a-a164-33a5796f3511",
        "00087d8f-17a1-4609-8bea-3524ec61471a",
        "000bab3b-aa0f-4b03-ad6b-612f76c1e33e",
        "000c39ac-378e-4455-92c9-a156fbaec3ad",
    ],
}


def _credential_headers():
    cred_path = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    with open(cred_path, "r", encoding="utf-8") as fh:
        cred = json.load(fh)
    token = base64.b64encode(f"{cred['username']}:{cred['key']}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def download_sample(split, uuid, out_dir, headers):
    """Download a single .pt file; return local path."""
    file_name = f"map_dataset/{split}/{uuid}.pt"
    # Whole-path URL encoding is required (raw slashes -> 404 server-side).
    url = f"{API_BASE}/download/{DATASET}/{urllib.parse.quote(file_name, safe='')}"
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, f"{uuid}.pt")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    resp = requests.get(url, headers=headers, allow_redirects=True, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as fh:
        fh.write(resp.content)
    return dest


def load_sample(path):
    """Load a .pt sample into a plain dict (map/start/goal/path tensors)."""
    ensure_ns_fix()
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not all(hasattr(obj, k) for k in ("map", "start", "goal", "path")):
        raise TypeError(f"unexpected sample object: {type(obj)} {getattr(obj, '__dict__', {})}")
    return {k: getattr(obj, k).float() for k in ("map", "start", "goal", "path")}


def summarize(sample):
    m, start, goal, path = sample["map"], sample["start"], sample["goal"], sample["path"]
    return {
        "map": [int(x) for x in m.shape],
        "free_cells": float((m == 0).float().mean()),
        "start": [int(v) for v in start.tolist()],
        "goal": [int(v) for v in goal.tolist()],
        "path_len": int(path.shape[0]),
        "path_endpoint_ok": bool(torch.equal(path[0].long(), start.long())
                                 and torch.equal(path[-1].long(), goal.long())),
    }


def plot_preview(samples, out_png, cols=3):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(samples)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    axes = axes.reshape(-1) if n > 1 else [axes]
    for ax, (name, sample) in zip(axes, samples):
        m = sample["map"].numpy()
        path = sample["path"].numpy()
        ax.imshow(m, cmap="gray_r", origin="upper", vmin=0, vmax=1)
        ax.plot(path[:, 1], path[:, 0], color="tab:red", lw=1.4, alpha=0.95)
        ax.scatter(*path[0][::-1], marker="o", s=26, color="tab:green", zorder=5,
                   label="start")
        ax.scatter(*path[-1][::-1], marker="*", s=120, color="tab:orange", zorder=5,
                   label="goal")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    if n > 1:
        axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "test", "validation"])
    ap.add_argument("--n", type=int, default=6, help="how many samples to preview")
    ap.add_argument("--out-dir", default=os.path.join("data", "kaggle_2dpathplanning", "samples"))
    ap.add_argument("--figure", default=os.path.join("outputs", "kaggle_2dpath_preview.png"))
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    split_dir = os.path.join(args.out_dir, args.split)
    headers = _credential_headers()
    samples = []
    rows = []
    for i in range(args.n):
        uuid = DEFAULT_SAMPLES.get(args.split, DEFAULT_SAMPLES["test"])[i % len(DEFAULT_SAMPLES["test"])]
        path = os.path.join(split_dir, f"{uuid}.pt") if args.no_download else \
            download_sample(args.split, uuid, split_dir, headers)
        sample = load_sample(path)
        samples.append((f"{args.split[:4]}_{uuid[:8]}", sample))
        rows.append([uuid[:8]] + list(summarize(sample).values()))

    print(f"downloaded/loaded {len(samples)} samples from split '{args.split}' -> {split_dir}")
    print("uuid      map      free%   start      goal       path_len  endpoint_ok")
    for row in rows:
        print("  ".join(str(x) for x in row))
    plot_preview(samples, args.figure)
    print("figure saved:", os.path.abspath(args.figure))


if __name__ == "__main__":
    main()
