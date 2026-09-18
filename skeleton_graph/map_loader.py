"""Map loading and binarisation.

Everything downstream of this module speaks exactly one language:

    occupancy : ``uint8`` ndarray ``(H, W)`` with ``0 = obstacle`` and
                ``255 = free``.

The pipeline thins *free space*, so this loader guarantees that ``255`` marks
precisely the region that will be handed to the thinning step.

Supported inputs
----------------
``.npy``
    Project convention (``data/processed_scene_v1/maps/*.npy``):
    ``1 = obstacle``, ``0 = free`` (float32, effectively binary).
``.png`` / ``.jpg`` / ``.jpeg`` / ``.bmp`` / ``.tif`` / ``.tiff`` / ``.webp``
    Read as 8-bit greyscale.  By default bright = free and dark = obstacle,
    which matches the task statement ("white = free, black = obstacle").
    Polarity can be auto-detected or forced.

Polarity
--------
``invert=None`` (auto)
    ``.npy``    -> the documented ``1 = obstacle`` convention.
    image       -> whichever grey level dominates the image border is taken
                   as the obstacle.  A 2-D map is almost always framed by an
                   obstacle border, so this is a reliable heuristic; the
                   decision is always logged so it can be overridden.
``invert=False``
    Use the natural convention above (no swap).
``invert=True``
    Swap free <-> obstacle.
"""

from __future__ import annotations

import os

import numpy as np

# ---------------------------------------------------------------------------
# public value constants
# ---------------------------------------------------------------------------

OBSTACLE = 0
FREE = 255

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def load_map(
    path: str,
    invert: bool | None = None,
    threshold: int = 127,
    verbose: bool = True,
) -> np.ndarray:
    """Load ``path`` and return a ``uint8`` occupancy map (0 = obstacle, 255 = free).

    Parameters
    ----------
    path
        ``.npy`` or image file.
    invert
        ``None`` to auto-detect polarity, otherwise force the decision.
    threshold
        Grey level separating obstacle from free for non-binary inputs.
    verbose
        Print the resolved polarity decision.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"map not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        raw = np.load(path, allow_pickle=False)
        occupancy, info = _binarize_array(raw, invert, threshold)
    elif ext in _IMAGE_EXTS:
        raw = _read_image_gray(path)
        occupancy, info = _binarize_image(raw, invert, threshold)
    else:
        raise ValueError(
            f"unsupported map extension {ext!r} for {path!r}; "
            f"expected .npy or one of {_IMAGE_EXTS}"
        )

    info["path"] = os.path.abspath(path)
    info.update(map_stats(occupancy))
    if verbose:
        print(
            f"[map] {os.path.basename(path)} {occupancy.shape[1]}x{occupancy.shape[0]} "
            f"source={info['source']} invert={info['invert']} "
            f"free={info['free_pixels']} obstacle={info['obstacle_pixels']}"
        )
    return occupancy


def free_mask(occupancy: np.ndarray) -> np.ndarray:
    """Boolean mask of free space (``True`` where thinning will run)."""
    return np.asarray(occupancy) > (FREE // 2)


def map_stats(occupancy: np.ndarray) -> dict:
    """Shape / area statistics of an occupancy map."""
    occ = np.asarray(occupancy)
    free = int(free_mask(occ).sum())
    total = int(occ.size)
    return {
        "height": int(occ.shape[0]),
        "width": int(occ.shape[1]),
        "total_pixels": total,
        "free_pixels": free,
        "obstacle_pixels": total - free,
        "free_ratio": round(free / total, 6) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _read_image_gray(path: str) -> np.ndarray:
    """Read any supported image as a 2-D ``uint8`` greyscale array."""
    from PIL import Image

    with Image.open(path) as img:
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        return np.asarray(img.convert("L"), dtype=np.uint8)


def _binarize_array(
    raw: np.ndarray, invert: bool | None, threshold: int
) -> tuple[np.ndarray, dict]:
    """Binarise a numeric array (``.npy``) into the occupancy convention."""
    arr = np.asarray(raw)
    if arr.ndim == 3:
        # (H, W, C) -> take the first channel; a 1-channel map is the norm.
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D map, got shape {arr.shape}")

    if not np.all(np.isfinite(arr)):
        raise ValueError("map contains NaN/Inf values")

    is_bool_like = arr.dtype == np.bool_ or np.isin(
        np.unique(arr), (0, 1)
    ).all()
    if is_bool_like:
        # Project convention for data/processed_scene_v1/maps/*.npy:
        # 1 = obstacle, 0 = free.
        obstacle = arr.astype(bool)
        natural_invert = False
        source = "npy-binary(1=obstacle)"
    else:
        obstacle = arr > threshold
        natural_invert = False
        source = f"npy-threshold(>{threshold}=obstacle)"

    do_invert = natural_invert if invert is None else bool(invert)
    return _pack(obstacle, do_invert), {
        "source": source,
        "invert": do_invert,
        "invert_mode": "auto" if invert is None else "forced",
    }


def _binarize_image(
    gray: np.ndarray, invert: bool | None, threshold: int
) -> tuple[np.ndarray, dict]:
    """Binarise a greyscale image; bright = free unless told otherwise."""
    bright = gray > threshold  # candidate free space

    if invert is None:
        # The outer frame of a 2-D map is (almost) always obstacle, so the
        # class that dominates the border is taken as the obstacle.
        border = np.concatenate(
            [bright[0, :], bright[-1, :], bright[:, 0], bright[:, -1]]
        )
        border_bright = float(border.mean())
        do_invert = border_bright > 0.5
        invert_mode = f"auto(border_bright={border_bright:.3f})"
    else:
        do_invert = bool(invert)
        invert_mode = "forced"

    return _pack(~bright, do_invert), {
        "source": f"image-threshold(>{threshold}=free)",
        "invert": do_invert,
        "invert_mode": invert_mode,
    }


def _pack(obstacle: np.ndarray, do_invert: bool) -> np.ndarray:
    """Turn an obstacle mask into a ``0/255`` occupancy map."""
    if do_invert:
        obstacle = ~obstacle
    occupancy = np.where(obstacle, OBSTACLE, FREE)
    return occupancy.astype(np.uint8)
