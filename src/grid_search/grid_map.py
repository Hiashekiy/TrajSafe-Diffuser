"""Occupancy semantics, obstacle inflation and the safe grid.

Coordinate conventions are the PROJECT ones (``src/datasets/carla_spline_dataset``
and ``evaluate.py``):

    cell index (col, row)  <->  array ``occupancy[row, col]``
    world_to_cell(p) = (p + 1) / 2 * res - 0.5
    cell_to_world(i) = (i + 0.5) * 2 / res - 1.0

so ``scene_to_meter`` is metres per SCENE unit (the crop is ``[-1, 1]`` in scene
units, i.e. ``2 * scene_to_meter`` metres wide) and

    cell_m = 2 * scene_to_meter / res.

``occupancy`` is the canonical (already ``flipud``-ed) bitmap produced by
``scripts/data/carla/*``: row 0 <-> ``scene_y = -1``; 0 = free, 1 = obstacle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["GridSpec", "OccupancyGrid", "SafeGrid", "inflate_obstacles",
           "build_safe_grid", "clearance_field"]


@dataclass(frozen=True)
class GridSpec:
    """Grid <-> scene-coordinate mapping (project convention)."""

    height: int
    width: int
    cell_m: float
    scene_to_meter: float = None   # metres per scene unit; derived if omitted

    def __post_init__(self):
        if self.height <= 0 or self.width <= 0 or self.cell_m <= 0:
            raise ValueError("GridSpec needs positive height/width/cell_m")
        if self.scene_to_meter is None:
            object.__setattr__(self, "scene_to_meter",
                               float(self.cell_m) * max(self.height, self.width) / 2.0)

    # -- scene <-> cell -----------------------------------------------------
    @classmethod
    def from_resolution(cls, height, width, res, scene_to_meter):
        """``res`` = square pixel resolution used by the project pipeline."""
        del res
        cell_m = 2.0 * float(scene_to_meter) / float(max(height, width))
        return cls(int(height), int(width), cell_m, float(scene_to_meter))

    def world_to_cell(self, points) -> np.ndarray:
        """scene coords [... , 2] -> continuous cell coords (col, row)."""
        p = np.asarray(points, dtype=np.float64)
        cells = np.empty_like(p)
        cells[..., 0] = (p[..., 0] + 1.0) / 2.0 * self.width - 0.5
        cells[..., 1] = (p[..., 1] + 1.0) / 2.0 * self.height - 0.5
        return cells

    def cell_to_world(self, cells) -> np.ndarray:
        c = np.asarray(cells, dtype=np.float64)
        p = np.empty_like(c)
        p[..., 0] = (c[..., 0] + 0.5) * 2.0 / self.width - 1.0
        p[..., 1] = (c[..., 1] + 0.5) * 2.0 / self.height - 1.0
        return p

    # -- integer indexing ---------------------------------------------------
    @property
    def n_cells(self) -> int:
        return int(self.height) * int(self.width)

    def rc_to_index(self, rc) -> np.ndarray:
        rc = np.asarray(rc, dtype=np.int64)
        return rc[..., 0] * self.width + rc[..., 1]

    def index_to_rc(self, idx) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        return np.stack([idx // self.width, idx % self.width], axis=-1)

    def cell_to_index(self, cells) -> np.ndarray:
        """Continuous (col, row) cell coords -> nearest integer cell INDEX.

        Note the axis swap: ``rc_to_index`` works in (row, col) order.
        """
        rc = np.rint(np.asarray(cells, dtype=np.float64)[..., ::-1]).astype(np.int64)
        return self.rc_to_index(rc)

    def index_to_cell(self, idx) -> np.ndarray:
        return self.index_to_rc(idx).astype(np.float64)

    def in_bounds_index(self, idx) -> np.ndarray:
        rc = self.index_to_rc(idx)
        return ((rc[..., 0] >= 0) & (rc[..., 0] < self.height)
                & (rc[..., 1] >= 0) & (rc[..., 1] < self.width))


@dataclass
class OccupancyGrid:
    """Semantic view of a raw occupancy array."""

    occupancy: np.ndarray            # [H, W] raw values
    spec: GridSpec
    free_values: tuple = (0,)
    obstacle_values: tuple = (1,)
    unknown_as_obstacle: bool = True

    def __post_init__(self):
        occ = np.asarray(self.occupancy)
        if occ.ndim != 2:
            raise ValueError("occupancy must be [H, W], got %s" % (occ.shape,))
        if (occ.shape[0], occ.shape[1]) != (self.spec.height, self.spec.width):
            raise ValueError("occupancy shape %s != GridSpec %s"
                             % (occ.shape, (self.spec.height, self.spec.width)))
        self.values = np.unique(occ)
        self.free = np.isin(occ, list(self.free_values))
        self.occupied = np.isin(occ, list(self.obstacle_values))
        self.unknown = ~self.free & ~self.occupied
        if not self.unknown_as_obstacle:
            # only explicitly known obstacles block the grid
            self.free = self.free | self.unknown

    @property
    def blocked(self) -> np.ndarray:
        return ~self.free


def clearance_field(free: np.ndarray, cell_m: float) -> np.ndarray:
    """Euclidean distance (metres) from every FREE cell to the nearest blocked cell.

    Blocked cells themselves get 0.  Uses ``scipy.ndimage.distance_transform_edt``
    on the free mask -- the same library the data pipeline uses for erosion -- so
    the geometry is identical to the project's own morphological operations.
    """
    from scipy.ndimage import distance_transform_edt

    free = np.asarray(free, dtype=bool)
    out = np.zeros(free.shape, dtype=np.float64)
    if free.any() and (~free).any():
        out[free] = distance_transform_edt(free)[free] * float(cell_m)
    elif free.any():
        out[free] = np.inf
    return out


def inflate_obstacles(free: np.ndarray, radius_cells: float,
                      cell_m: float = 1.0, return_distance: bool = False):
    """Circular (Euclidean) obstacle inflation.

    A cell stays free iff its centre is farther than ``radius_cells`` from every
    blocked cell CENTRE.  ``radius_cells <= 0`` returns the input unchanged.

    With ``return_distance`` the raw clearance field (metres, distance to the
    nearest blocked cell) is returned as well: it is the same EDT call
    ``path_centering`` would otherwise repeat, so callers can share it.
    """
    r = float(radius_cells)
    if r <= 0.0:
        f = np.array(free, dtype=bool, copy=True)
        return (f, clearance_field(f, cell_m)) if return_distance else f
    dist_cells = clearance_field(free, 1.0)     # metres==cells when cell_m=1
    safe = dist_cells > r
    safe &= np.asarray(free, dtype=bool)
    if return_distance:
        return safe, dist_cells * float(cell_m)
    return safe


@dataclass
class SafeGrid:
    """The inflated collision model the search actually runs on."""

    free: np.ndarray                 # [H, W] bool: enterable cells
    spec: GridSpec
    clearance_m: np.ndarray          # [H, W] float: distance of the RAW map
    inflation_cells: float
    robot_radius_m: float
    safety_margin_m: float
    notes: list = field(default_factory=list)
    raw_free: np.ndarray = None
    raw_clearance_m: np.ndarray = None   # distance of the RAW map, reused by centring

    @property
    def n_free(self) -> int:
        return int(self.free.sum())

    def is_free_index(self, idx) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        ok = self.spec.in_bounds_index(idx)
        out = np.zeros(idx.shape, dtype=bool)
        rc = self.spec.index_to_rc(idx[ok])
        out[ok] = self.free[rc[:, 0], rc[:, 1]]
        return out

    def clearance_at_index(self, idx) -> np.ndarray:
        flat = self.clearance_m.reshape(-1)
        idx = np.asarray(idx, dtype=np.int64)
        out = np.zeros(idx.shape, dtype=np.float64)
        ok = self.spec.in_bounds_index(idx)
        out[ok] = flat[idx[ok]]
        return out


def build_safe_grid(occupancy, spec: GridSpec, robot_radius_m: float = 0.8,
                    safety_margin_m: float = 0.3, free_values=(0,),
                    unknown_as_obstacle: bool = True,
                    already_inflated_cells: float = 0.0) -> SafeGrid:
    """Parse the occupancy semantics and build the inflated safe grid.

    ``already_inflated_cells`` documents obstacle widening that the map itself
    already carries (e.g. the processed CARLA crop had its obstacles eroded by 4
    cells to widen the driving channels -- see
    ``scripts/data/carla_full/05_erode_occupancy.py``).  It is subtracted from
    the required inflation so the same physical clearance is not demanded twice,
    and it is recorded in ``notes`` for the report.
    """
    grid = OccupancyGrid(occupancy, spec, free_values=free_values,
                         unknown_as_obstacle=unknown_as_obstacle)
    raw_free = grid.free
    r_inflate_m = float(robot_radius_m) + float(safety_margin_m)
    r_cells = r_inflate_m / float(spec.cell_m)
    r_eff = max(0.0, r_cells - float(already_inflated_cells))
    free, raw_clearance = inflate_obstacles(raw_free, r_eff, spec.cell_m,
                                            return_distance=True)
    notes = []
    if grid.unknown.any():
        notes.append("%d unknown cells treated as obstacle"
                     % int(grid.unknown.sum()))
    notes.append("raw values %s, free=%s" % (grid.values.tolist(),
                                             list(free_values)))
    if already_inflated_cells:
        notes.append("map already widened by %.1f cells; additional inflation "
                     "%.2f cells (r=%.2f m, cell=%.3f m)"
                     % (already_inflated_cells, r_eff, r_inflate_m, spec.cell_m))
    else:
        notes.append("inflation %.2f cells (r=%.2f m, cell=%.3f m)"
                     % (r_eff, r_inflate_m, spec.cell_m))
    return SafeGrid(free=free, spec=spec,
                    clearance_m=clearance_field(free, spec.cell_m),
                    inflation_cells=r_eff, robot_radius_m=float(robot_radius_m),
                    safety_margin_m=float(safety_margin_m), notes=notes,
                    raw_free=raw_free, raw_clearance_m=raw_clearance)
