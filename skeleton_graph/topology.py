"""Connected-component / hole helpers used for topology checks.

Thinning and pruning must not change the topology of the free space: the
number of free-space components and the number of enclosed holes must stay
the same.  These helpers are what the acceptance checks compare against.

Convention: foreground (free space / skeleton) uses **8-connectivity**, its
complement uses **4-connectivity**.  That pairing is the only one that makes
"a one-pixel-wide diagonal line" a single connected object without also
merging the two sides of that line.
"""

from __future__ import annotations

from collections import deque

import numpy as np

_STRUCT8 = np.ones((3, 3), dtype=np.int32)
_STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)

_NEIGHBOURS_8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)
_NEIGHBOURS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def neighbour_offsets(connectivity: int = 8) -> tuple[tuple[int, int], ...]:
    """``(dy, dx)`` offsets of the 4- or 8-neighbourhood."""
    if connectivity == 8:
        return _NEIGHBOURS_8
    if connectivity == 4:
        return _NEIGHBOURS_4
    raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")


def label_components(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Label connected components of ``mask``; returns ``(labels, n)``.

    ``labels`` is ``int32`` with ``0`` as background and ``1..n`` the
    component ids.  Uses ``scipy.ndimage`` when available, otherwise a small
    BFS fallback so the package still runs on a bare numpy install.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32), 0

    try:  # pragma: no cover - exercised implicitly when scipy is present
        from scipy import ndimage

        structure = _STRUCT8 if connectivity == 8 else _STRUCT4
        labels, n = ndimage.label(mask, structure=structure)
        return labels.astype(np.int32, copy=False), int(n)
    except ImportError:
        return _label_components_bfs(mask, connectivity)


def count_components(mask: np.ndarray, connectivity: int = 8) -> int:
    """Number of connected components of ``mask``."""
    return label_components(mask, connectivity)[1]


def count_holes(mask: np.ndarray) -> int:
    """Number of holes enclosed by ``mask``.

    Computed as the number of 4-connected background components that do not
    touch the image border.
    """
    mask = np.asarray(mask, dtype=bool)
    background = ~mask
    labels, n = label_components(background, connectivity=4)
    if n == 0:
        return 0

    border_labels = set()
    border_labels.update(np.unique(labels[0, :]).tolist())
    border_labels.update(np.unique(labels[-1, :]).tolist())
    border_labels.update(np.unique(labels[:, 0]).tolist())
    border_labels.update(np.unique(labels[:, -1]).tolist())
    border_labels.discard(0)
    return n - len(border_labels)


def topology_signature(mask: np.ndarray) -> dict:
    """``{components, holes}`` of a binary mask."""
    return {
        "components": count_components(mask, connectivity=8),
        "holes": count_holes(mask),
    }


def topology_preserved(
    reference: np.ndarray, candidate: np.ndarray
) -> tuple[bool, dict]:
    """Compare component/hole counts of two masks.

    Returns ``(ok, details)`` where ``details`` records both signatures and
    the per-field differences.
    """
    ref = topology_signature(reference)
    cand = topology_signature(candidate)
    diff = {k: cand[k] - ref[k] for k in ref}
    return all(v == 0 for v in diff.values()), {
        "reference": ref,
        "candidate": cand,
        "delta": diff,
    }


def component_sizes(mask: np.ndarray, connectivity: int = 8) -> list[int]:
    """Sorted-descending list of component sizes."""
    labels, n = label_components(mask, connectivity)
    if n == 0:
        return []
    return sorted(np.bincount(labels.ravel())[1:].tolist(), reverse=True)


# ---------------------------------------------------------------------------
# fallback
# ---------------------------------------------------------------------------


def _label_components_bfs(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    offsets = _NEIGHBOURS_8 if connectivity == 8 else _NEIGHBOURS_4
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current = 0
    for sy in range(h):
        row = mask[sy]
        for sx in range(w):
            if not row[sx] or labels[sy, sx]:
                continue
            current += 1
            labels[sy, sx] = current
            queue = deque([(sy, sx)])
            while queue:
                y, x = queue.popleft()
                for dy, dx in offsets:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = current
                        queue.append((ny, nx))
    return labels, current
