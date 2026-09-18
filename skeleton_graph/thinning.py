"""Guo-Hall thinning of the free space.

The thinning step takes the *free space* (never the obstacle region) and
reduces it to a one-pixel-wide skeleton::

    M_free  ->  S

Two backends are provided:

``cv2``
    ``cv2.ximgproc.thinning(..., THINNING_GUOHALL)`` -- used automatically
    when ``opencv-contrib-python`` (i.e. the ``ximgproc`` module) is
    importable.
``numpy``
    A faithful, vectorised implementation of Guo & Hall (1989), "Parallel
    thinning with two sub-iteration algorithms", CVGIP 33(1).  Identical
    conditions, no extra dependency.  This is the fallback when
    ``ximgproc`` is missing, which is the case for the plain
    ``opencv-python`` wheel.

Neighbour indexing follows the paper::

    p9 p2 p3
    p8 p1 p4
    p7 p6 p5

Each iteration has two sub-iterations; in sub-iteration 1 a foreground
pixel ``p1`` is marked for deletion when

    (a) 2 <= B(p1) <= 6                   (B = number of nonzero neighbours)
    (b) A(p1) == 1                        (0->1 transitions in p2..p9,p2)
    (c) p2*p4*p6 == 0 and p4*p6*p8 == 0

and in sub-iteration 2 when (a) and (b) hold and

    (c') p2*p4*p8 == 0 and p2*p6*p8 == 0

Marked pixels are removed *simultaneously* at the end of each sub-iteration.
"""

from __future__ import annotations

import numpy as np

from skeleton_graph.topology import neighbour_offsets, topology_preserved

#: Hard stop so a pathological input can never spin forever.
MAX_SUBITERATIONS = 400


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def guo_hall_thinning(
    occupancy: np.ndarray,
    backend: str = "auto",
    verbose: bool = True,
) -> np.ndarray:
    """Reduce the *free space* of ``occupancy`` to a one-pixel skeleton.

    Parameters
    ----------
    occupancy
        ``uint8`` ``(H, W)`` map with ``0 = obstacle`` and ``255 = free``,
        as produced by :func:`skeleton_graph.map_loader.load_map`.
    backend
        ``"auto"`` | ``"cv2"`` | ``"numpy"``.

    Returns
    -------
    ``uint8`` ``(H, W)`` array with ``1`` on the skeleton and ``0`` elsewhere.
    """
    foreground = np.asarray(occupancy) > (255 // 2)
    if not foreground.any():
        raise ValueError("free space is empty; nothing to thin")

    resolved = resolve_backend(backend)
    if resolved == "cv2":
        skeleton = _guo_hall_cv2(foreground)
    else:
        skeleton = _guo_hall_numpy(foreground)

    if not skeleton.any():
        raise RuntimeError(
            f"thinning backend {resolved!r} produced an empty skeleton"
        )

    skeleton = skeleton.astype(np.uint8, copy=False)
    if verbose:
        print(
            f"[thinning] backend={resolved} "
            f"skeleton_pixels={int(skeleton.sum())}"
        )
    return skeleton


def resolve_backend(requested: str) -> str:
    """Resolve ``"auto"`` into a concrete backend name."""
    if requested == "cv2":
        if not cv2_ximgproc_available():
            raise RuntimeError(
                "thinning_backend='cv2' requested but cv2.ximgproc is not "
                "importable. Install opencv-contrib-python, or use "
                "thinning_backend='numpy'."
            )
        return "cv2"
    if requested == "auto":
        return "cv2" if cv2_ximgproc_available() else "numpy"
    return "numpy"


def cv2_ximgproc_available() -> bool:
    """``True`` when ``cv2.ximgproc.thinning`` can be used."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning")


# ---------------------------------------------------------------------------
# degree / diagnostics
# ---------------------------------------------------------------------------


def compute_degree(skeleton: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """8-neighbour degree ``d(p) = sum_{q in N(p)} S(q)`` for every pixel.

    The returned array has the same shape as ``skeleton``; only entries where
    the skeleton is set are meaningful.
    """
    skel = np.asarray(skeleton) > 0
    h, w = skel.shape
    padded = np.pad(skel, 1, mode="constant")
    degree = np.zeros((h, w), dtype=np.uint8)
    for dy, dx in neighbour_offsets(connectivity):
        degree += padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
    return degree


def classify_pixels(skeleton: np.ndarray, connectivity: int = 8) -> dict:
    """Split skeleton pixels into endpoints / chain pixels / junctions."""
    skel = np.asarray(skeleton) > 0
    degree = compute_degree(skel, connectivity)
    values = degree[skel]
    return {
        "endpoints": skel & (degree == 1),
        "chain": skel & (degree == 2),
        "junctions": skel & (degree >= 3),
        "isolated": skel & (degree == 0),
        "degree_values": values,
    }


def skeleton_diagnostics(skeleton: np.ndarray, connectivity: int = 8) -> dict:
    """Pixel-level sanity report for a skeleton."""
    skel = np.asarray(skeleton) > 0
    parts = classify_pixels(skel, connectivity)

    # A fully 2x2 block of skeleton pixels means the skeleton is not one
    # pixel wide at that spot.
    p = skel.astype(np.uint8)
    blocks = (
        p[:-1, :-1].astype(np.int16)
        + p[:-1, 1:].astype(np.int16)
        + p[1:, :-1].astype(np.int16)
        + p[1:, 1:].astype(np.int16)
    )
    values = parts.pop("degree_values")
    return {
        "pixels": int(skel.sum()),
        "endpoints": int(parts["endpoints"].sum()),
        "chain": int(parts["chain"].sum()),
        "junctions": int(parts["junctions"].sum()),
        "isolated": int(parts["isolated"].sum()),
        "thick_2x2_blocks": int((blocks == 4).sum()),
        "max_degree": int(values.max()) if values.size else 0,
    }


def validate_thinning(foreground: np.ndarray, skeleton: np.ndarray) -> dict:
    """Check that thinning preserved the topology of the free space.

    Thinning must keep the number of connected components and the number of
    enclosed holes, otherwise the corridor structure (and with it every
    cycle) has been damaged.
    """
    ok, details = topology_preserved(
        np.asarray(foreground) > 0, np.asarray(skeleton) > 0
    )
    details["ok"] = ok
    return details


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------


def _guo_hall_numpy(foreground: np.ndarray) -> np.ndarray:
    """Vectorised Guo-Hall thinning (pure numpy)."""
    img = np.ascontiguousarray(foreground.astype(np.uint8))

    for _ in range(MAX_SUBITERATIONS):
        changed = False
        for subiteration in (1, 2):
            n2, n3, n4, n5, n6, n7, n8, n9 = _neighbours(img)

            b = n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9

            # A(p1): number of 0 -> 1 transitions in the cyclic sequence
            # p2, p3, ..., p9, p2.
            sequence = (n2, n3, n4, n5, n6, n7, n8, n9)
            a = np.zeros(img.shape, dtype=np.uint8)
            for k in range(8):
                cur = sequence[k]
                nxt = sequence[(k + 1) % 8]
                a += ((cur == 0) & (nxt == 1)).astype(np.uint8)

            if subiteration == 1:
                cond = (n2 * n4 * n6 == 0) & (n4 * n6 * n8 == 0)
            else:
                cond = (n2 * n4 * n8 == 0) & (n2 * n6 * n8 == 0)

            marked = (img == 1) & (b >= 2) & (b <= 6) & (a == 1) & cond
            if not marked.any():
                continue
            changed = True
            img[marked] = 0

        if not changed:
            break

    return img


def _neighbours(img: np.ndarray):
    """Return the eight Guo-Hall neighbours ``p2..p9`` as ``(H, W)`` arrays."""
    h, w = img.shape
    padded = np.pad(img, 1, mode="constant")
    n2 = padded[0:h, 1 : w + 1]        # north
    n3 = padded[0:h, 2 : w + 2]        # north-east
    n4 = padded[1 : h + 1, 2 : w + 2]  # east
    n5 = padded[2 : h + 2, 2 : w + 2]  # south-east
    n6 = padded[2 : h + 2, 1 : w + 1]  # south
    n7 = padded[2 : h + 2, 0:w]        # south-west
    n8 = padded[1 : h + 1, 0:w]        # west
    n9 = padded[0:h, 0:w]              # north-west
    return n2, n3, n4, n5, n6, n7, n8, n9


def _guo_hall_cv2(foreground: np.ndarray) -> np.ndarray:
    """Guo-Hall thinning via ``cv2.ximgproc``."""
    import cv2

    source = np.ascontiguousarray(foreground.astype(np.uint8) * 255)
    thinned = cv2.ximgproc.thinning(
        source, thinningType=cv2.ximgproc.THINNING_GUOHALL
    )
    return (thinned > 0).astype(np.uint8)
