"""Occupancy grid with the few primitives the rest of the library needs."""

from __future__ import annotations

from collections import deque

import numpy as np

# (dr, dc) order is fixed everywhere: up, down, left, right
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


class Grid:
    """A binary occupancy grid. ``0``/``False`` is free, anything else is an obstacle."""

    def __init__(self, occupancy, free_value=0):
        occ = np.asarray(occupancy)
        if occ.ndim != 2:
            raise ValueError(f"occupancy must be 2-D, got shape {occ.shape}")
        self.occupancy = (occ != free_value).astype(np.int8)
        self.free = self.occupancy == 0
        self.height, self.width = self.occupancy.shape
        self.size = self.height * self.width
        self._build_neighbours()

    # ------------------------------------------------------------------ #
    def _build_neighbours(self) -> None:
        """Neighbour table (size, 4) with -1 for missing or blocked neighbours."""
        flat = np.arange(self.size, dtype=np.int64).reshape(self.height, self.width)
        nbr = np.full((self.size, 4), -1, dtype=np.int64)
        for k, (dr, dc) in enumerate(MOVES):
            src = flat[max(0, -dr):self.height - max(0, dr), max(0, -dc):self.width - max(0, dc)]
            dst = flat[max(0, dr):self.height + min(0, dr), max(0, dc):self.width + min(0, dc)]
            ok = self.free[max(0, -dr):self.height - max(0, dr), max(0, -dc):self.width - max(0, dc)] & \
                 self.free[max(0, dr):self.height + min(0, dr), max(0, dc):self.width + min(0, dc)]
            nbr[src[ok], k] = dst[ok]
        self._nbr = nbr

    # ------------------------------------------------------------------ #
    def idx(self, r: int, c: int) -> int:
        return int(r) * self.width + int(c)

    def rc(self, i: int) -> tuple[int, int]:
        return divmod(int(i), self.width)

    def is_free(self, r: int, c: int) -> bool:
        return bool(self.free[r, c])

    def neighbors(self, i: int) -> np.ndarray:
        """Free 4-neighbours of flat index ``i``."""
        row = self._nbr[int(i)]
        return row[row >= 0]

    # ------------------------------------------------------------------ #
    def bfs_distance(self, seeds) -> np.ndarray:
        """Unit-cost geodesic distance from ``seeds`` to every cell (-1 = unreachable)."""
        seeds = np.asarray(seeds, dtype=np.int64).reshape(-1)
        dist = np.full(self.size, -1, dtype=np.int32)
        queue = deque()
        for s in seeds:
            s = int(s)
            if 0 <= s < self.size and self.free.flat[s] and dist[s] < 0:
                dist[s] = 0
                queue.append(s)
        nbr = self._nbr
        while queue:
            v = queue.popleft()
            nd = dist[v] + 1
            for u in nbr[v]:
                if u >= 0 and dist[u] < 0:
                    dist[u] = nd
                    queue.append(u)
        return dist.reshape(self.height, self.width)

    def free_fraction(self) -> float:
        return float(self.free.mean())

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"Grid({self.height}x{self.width}, free={self.free_fraction():.2f})"
