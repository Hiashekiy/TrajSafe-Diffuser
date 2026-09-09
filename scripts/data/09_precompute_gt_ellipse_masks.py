"""Precompute quantized 64x64 soft masks for every V1 GT ellipse.

The masks are stored as uint8 HDF5 datasets with per-sample LZF-compressed
chunks. During training uint8 values are divided by 255, avoiding repeated GT
ellipse rasterization while keeping storage and host/GPU transfer manageable.

Usage:
  python scripts/data/09_precompute_gt_ellipse_masks.py \
      --config configs/config_v1.yaml
"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.geometry.ellipse_utils import physical_ellipse_center


@torch.no_grad()
def rasterize_batch(p, e, raster_res, tau, anchor_chunk, absolute=False):
    """Return quantized soft masks [B,H,R,R] on CPU."""
    center = physical_ellipse_center(p, e, absolute)
    a = torch.exp(torch.clamp(e[..., 2], -6.0, 0.7))
    b = torch.exp(torch.clamp(e[..., 3], -6.0, 0.7))
    theta = 0.5 * torch.atan2(e[..., 5], e[..., 4])

    coord = ((torch.arange(raster_res, device=p.device, dtype=p.dtype) + 0.5)
             * (2.0 / raster_res) - 1.0)
    gy, gx = torch.meshgrid(coord, coord, indexing="ij")
    gx = gx[None, None]
    gy = gy[None, None]

    result = torch.empty(
        (p.shape[0], p.shape[1], raster_res, raster_res),
        dtype=torch.uint8,
        device="cpu",
    )
    for start in range(0, p.shape[1], anchor_chunk):
        end = min(start + anchor_chunk, p.shape[1])
        dx = gx - center[:, start:end, 0, None, None]
        dy = gy - center[:, start:end, 1, None, None]
        ct = torch.cos(theta[:, start:end])[:, :, None, None]
        st = torch.sin(theta[:, start:end])[:, :, None, None]
        xr = ct * dx + st * dy
        yr = -st * dx + ct * dy
        q = ((xr / a[:, start:end, None, None]).square()
             + (yr / b[:, start:end, None, None]).square())
        mask = torch.sigmoid(tau * (1.0 - q))
        result[:, start:end] = mask.mul(255.0).round().to(torch.uint8).cpu()
    return result.numpy()


def process_split(split_dir, raster_res, tau, batch_size, anchor_chunk, device):
    pos = np.load(os.path.join(split_dir, "positions.npy"), mmap_mode="r")
    e6 = np.load(os.path.join(split_dir, "ellipses6.npy"), mmap_mode="r")
    if pos.shape[:2] != e6.shape[:2]:
        raise ValueError(f"Position/e6 shape mismatch: {pos.shape} vs {e6.shape}")

    output = os.path.join(split_dir, f"ellipse_masks{raster_res}_u8.h5")
    temp = output + ".tmp"
    if os.path.exists(temp):
        os.remove(temp)

    n, horizon = pos.shape[:2]
    with h5py.File(temp, "w") as f:
        masks = f.create_dataset(
            "masks",
            shape=(n, horizon, raster_res, raster_res),
            dtype=np.uint8,
            chunks=(1, horizon, raster_res, raster_res),
            compression="lzf",
            shuffle=True,
        )
        masks.attrs["raster_res"] = raster_res
        masks.attrs["tau"] = tau
        masks.attrs["quantization"] = "round(mask * 255)"
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            # torch.tensor intentionally copies the read-only memmap slices.
            p = torch.tensor(np.asarray(pos[start:end]), device=device)
            e = torch.tensor(np.asarray(e6[start:end]), device=device)
            masks[start:end] = rasterize_batch(
                p, e, raster_res, tau, anchor_chunk,
                absolute=absolute_center,
            )
            if end == n or end % max(batch_size * 20, 1) == 0:
                print(f"[{os.path.basename(split_dir)}] {end}/{n}", flush=True)
    os.replace(temp, output)
    size_gb = os.path.getsize(output) / (1024 ** 3)
    print(f"saved {output} ({size_gb:.2f} GiB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_v1.yaml")
    ap.add_argument("--res", type=int, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--anchor-chunk", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--absolute-center", action="store_true",
                    help="treat e6[:2] as absolute centre rather than offset c-p")
    args = ap.parse_args()
    absolute_center = bool(args.absolute_center)

    cfg = load_config(args.config)
    loss_cfg = cfg["loss"]
    raster_res = args.res or int(loss_cfg.get("ellipse_safe_res", 64))
    tau = args.tau or float(loss_cfg.get("ellipse_mask_tau", 10.0))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = cfg["data"]["base"]
    print(f"device={device} res={raster_res} tau={tau}", flush=True)
    for split in ("train", "val", "test"):
        process_split(
            os.path.join(base, split), raster_res, tau,
            args.batch_size, args.anchor_chunk, device,
        )


if __name__ == "__main__":
    main()
