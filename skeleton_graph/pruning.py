"""Spur (short-branch) pruning of a skeleton.

Raw thinning leaves short hairs wherever the free space is locally wide::

    ----------------
           |
           |        <- hair, no structural meaning
           |

This module removes them, and *only* them.

Rules that are deliberately enforced
------------------------------------
* Only ``endpoint -> junction`` branches are candidates.  A branch that ends
  at another endpoint (i.e. an isolated path component) is never removed,
  because that would delete real free-space structure and break connectivity.
* A branch connecting two junctions is never removed, however short it is.
  Short corridors are real corridors.
* The junction pixel itself is never removed, only the chain pixels between
  the endpoint and the junction.
* Removing an endpoint-rooted tree cannot disconnect anything and cannot
  destroy a cycle, so pruning is topology safe by construction.

Pruning is iterative: after a hair is removed the junction it was attached to
may drop to degree 2, which can expose a shorter hair behind it.  Each round
recomputes degrees; the loop stops as soon as a round removes nothing.

Pixel indexing inside this module uses ``(y, x)`` == ``(row, column)`` for
direct numpy indexing.  The ``(x, y)`` output convention only starts at the
graph/visualisation boundary.
"""

from __future__ import annotations

import math

import numpy as np

from skeleton_graph.thinning import compute_degree
from skeleton_graph.topology import neighbour_offsets

#: Safety stop; the loop is monotone (pixels only disappear) so this is
#: never reached in practice.
MAX_ROUNDS = 64


def prune_skeleton(
    skeleton: np.ndarray,
    min_branch_length: float = 10.0,
    connectivity: int = 8,
    max_prune_fraction: float = 0.5,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Remove endpoint-rooted branches shorter than ``min_branch_length``.

    Parameters
    ----------
    skeleton
        ``uint8`` ``(H, W)`` array, ``1`` on the skeleton.
    min_branch_length
        Euclidean length threshold in pixels (axis step = 1, diagonal =
        ``sqrt(2)``).  A branch is removed when its length is strictly less
        than this value.
    max_prune_fraction
        Safety budget: pruning never removes more than this fraction of the
        original skeleton pixels.  Shortest branches are removed first, so a
        mis-set threshold degrades gracefully instead of erasing the whole
        skeleton.  Set to ``1.0`` to disable the guard.

    Returns
    -------
    ``(pruned_skeleton, info)`` where ``info`` reports how many branches and
    pixels were removed and lists every branch length that was measured, so
    the threshold can be tuned without re-running by hand.
    """
    skel = np.asarray(skeleton) > 0
    pixels_before = int(skel.sum())

    branches = trace_endpoint_branches(skel, connectivity)
    lengths_before = sorted(b["length"] for b in branches["spurs"])
    candidate_spurs = sorted(
        (
            {
                "length": round(b["length"], 4),
                "endpoint": [int(b["start"][1]), int(b["start"][0])],
                "junction": [int(b["end"][1]), int(b["end"][0])],
                "pixels": len(b["path"]),
            }
            for b in branches["spurs"]
        ),
        key=lambda b: b["length"],
    )

    pruned = skel.copy()
    removed_branches: list[dict] = []
    rounds = 0
    budget_exhausted = False
    budget = max_prune_fraction * pixels_before

    if min_branch_length > 0 and pixels_before:
        for rounds in range(1, MAX_ROUNDS + 1):
            found = trace_endpoint_branches(pruned, connectivity)
            doomed = sorted(
                (b for b in found["spurs"] if b["length"] < min_branch_length),
                key=lambda b: b["length"],
            )
            if not doomed:
                rounds -= 1
                break

            removed_so_far = pixels_before - int(pruned.sum())
            for branch in doomed:
                if removed_so_far + len(branch["path"]) > budget:
                    budget_exhausted = True
                    break
                for y, x in branch["path"]:
                    pruned[y, x] = False
                removed_so_far += len(branch["path"])
                removed_branches.append(
                    {
                        "length": round(branch["length"], 4),
                        "endpoint": [int(branch["start"][1]), int(branch["start"][0])],
                        "pixels": len(branch["path"]),
                        "junction": [int(branch["end"][1]), int(branch["end"][0])],
                    }
                )
            if budget_exhausted:
                break
        else:  # pragma: no cover - defensive
            raise RuntimeError("pruning did not converge")

    pixels_after = int(pruned.sum())
    info = {
        "enabled": True,
        "min_branch_length": float(min_branch_length),
        "max_prune_fraction": float(max_prune_fraction),
        "budget_exhausted": budget_exhausted,
        "rounds": rounds,
        "pixels_before": pixels_before,
        "pixels_after": pixels_after,
        "pixels_removed": pixels_before - pixels_after,
        "branches_removed": len(removed_branches),
        "removed_branches": removed_branches,
        "candidate_spur_lengths_before": [round(v, 4) for v in lengths_before],
        "candidate_spurs_before": candidate_spurs,
        "isolated_paths_kept": len(branches["isolated_paths"]),
    }

    if verbose:
        print(
            f"[pruning] min_branch_length={min_branch_length} "
            f"rounds={rounds} branches_removed={len(removed_branches)} "
            f"pixels {pixels_before}->{pixels_after}"
            + ("  [budget exhausted]" if budget_exhausted else "")
        )
    return pruned.astype(np.uint8, copy=False), info


def no_pruning(skeleton: np.ndarray) -> tuple[np.ndarray, dict]:
    """Pass-through used when pruning is disabled, with the same info shape."""
    skel = np.asarray(skeleton) > 0
    count = int(skel.sum())
    return skel.astype(np.uint8, copy=False), {
        "enabled": False,
        "min_branch_length": None,
        "max_prune_fraction": None,
        "budget_exhausted": False,
        "rounds": 0,
        "pixels_before": count,
        "pixels_after": count,
        "pixels_removed": 0,
        "branches_removed": 0,
        "removed_branches": [],
        "candidate_spur_lengths_before": [],
        "candidate_spurs_before": [],
        "isolated_paths_kept": 0,
    }


# ---------------------------------------------------------------------------
# branch tracing
# ---------------------------------------------------------------------------


def trace_endpoint_branches(skeleton: np.ndarray, connectivity: int = 8) -> dict:
    """Trace every branch that starts at an endpoint.

    Returns ``{"spurs": [...], "isolated_paths": [...]}``.

    * ``spurs`` -- branches that terminate at a junction.  These are the only
      pruning candidates.  Each entry carries ``start``, ``end``, ``path``
      (chain pixels, *excluding* the junction pixel) and ``length``, where
      the length includes the final step into the junction.
    * ``isolated_paths`` -- branches whose far end is another endpoint, i.e.
      a free-space component with no junction at all.  Never pruned.

    All coordinates are returned in the *unpadded* image frame.
    """
    skel = np.asarray(skeleton) > 0
    # One ring of background padding removes every bounds check from the
    # walk: skeleton pixels at the image border can be traced safely.
    padded = np.pad(skel, 1, mode="constant")
    degree = compute_degree(padded, connectivity)
    offsets = neighbour_offsets(connectivity)

    endpoint_pixels = np.argwhere(padded & (degree == 1))
    spurs: list[dict] = []
    isolated_paths: list[dict] = []

    for py, px in endpoint_pixels:
        start = (int(py), int(px))
        path: list[tuple[int, int]] = [start]
        previous: tuple[int, int] | None = None
        current = start

        while True:
            y, x = current
            nxt_candidates = [
                (y + dy, x + dx)
                for dy, dx in offsets
                if padded[y + dy, x + dx] and (y + dy, x + dx) != previous
            ]
            if len(nxt_candidates) != 1:
                break
            nxt = nxt_candidates[0]
            nxt_degree = int(degree[nxt])

            if nxt_degree >= 3:
                record = {
                    "start": start,
                    "end": nxt,
                    "path": path,
                    "length": polyline_length(path + [nxt]),
                }
                spurs.append(record)
                break
            if nxt_degree <= 1:
                isolated_paths.append(
                    {
                        "start": start,
                        "end": nxt,
                        "path": path,
                        "length": polyline_length(path + [nxt]),
                    }
                )
                break

            path.append(nxt)
            previous, current = current, nxt

    return {
        "spurs": [_unpad_branch(b) for b in spurs],
        "isolated_paths": [_unpad_branch(b) for b in isolated_paths],
    }


def _unpad_branch(branch: dict) -> dict:
    """Shift a traced branch from padded to original image coordinates."""
    return {
        "start": (branch["start"][0] - 1, branch["start"][1] - 1),
        "end": (branch["end"][0] - 1, branch["end"][1] - 1),
        "path": [(y - 1, x - 1) for y, x in branch["path"]],
        "length": branch["length"],
    }


def polyline_length(points) -> float:
    """Euclidean length of a pixel polyline (1 for axis steps, sqrt(2) for diagonals)."""
    total = 0.0
    for (y0, x0), (y1, x1) in zip(points, points[1:]):
        dy, dx = y1 - y0, x1 - x0
        total += math.hypot(dy, dx)
    return total
