"""Central configuration for the stage-1 skeleton -> graph pipeline.

Conventions fixed for the whole package
---------------------------------------
occupancy map
    ``uint8`` array of shape ``(H, W)``.

    =====  =========
    value  meaning
    =====  =========
    0      obstacle
    255    free space
    =====  =========

    Thinning is applied to **free space**, never to the obstacle region.

skeleton
    ``uint8`` array of shape ``(H, W)``.

    =====  =========
    value  meaning
    =====  =========
    0      background
    1      skeleton pixel
    =====  =========

coordinates
    ``(x, y) = (image column, image row)``.

    ``x`` is the *column* index and ``y`` is the *row* index, i.e. the same
    order matplotlib/numpy use for ``imshow``.  ``(row, col)`` ordering is
    never used in this package; every array that stores coordinates stores
    them as ``[x, y]``.

All tunables live here, no magic numbers are inlined in the algorithm
modules.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, fields

#: Maze names shipped under ``data/processed_scene_v1/maps``.
MAZE_NAMES = ("umaze", "medium", "large")

#: Repo root (``.../Neural-IRISDiffuser``), derived from this file's location.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Default location of the stage-1 data.
DEFAULT_MAPS_DIR = os.path.join(REPO_ROOT, "data", "processed_scene_v1", "maps")

#: Default output directory.
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "skeleton_graph", "outputs")

VALID_THINNING_BACKENDS = ("auto", "cv2", "numpy")
VALID_GRAPH_CONTAINERS = ("auto", "graph", "multigraph")


@dataclass
class Config:
    """Runtime configuration of one pipeline run."""

    # ------------------------------------------------------------------ input
    map_path: str = ""
    #: ``None`` = auto-detect polarity, ``True``/``False`` = force.
    invert_input: bool | None = None
    #: Threshold used to binarise non-binary inputs (PNG/JPG or float maps).
    binarize_threshold: int = 127

    # ----------------------------------------------------------------- output
    output_dir: str = DEFAULT_OUTPUT_DIR

    # --------------------------------------------------------------- thinning
    #: ``"auto"`` uses ``cv2.ximgproc.thinning`` when available and falls back
    #: to the bundled pure-numpy Guo-Hall implementation otherwise.
    thinning_backend: str = "auto"

    # ---------------------------------------------------------------- pruning
    enable_pruning: bool = True
    #: Spur branches shorter than this (in pixels, euclidean length) are
    #: removed.  Only endpoint -> junction branches are ever removed.
    min_branch_length: float = 10.0
    #: Safety budget: pruning never removes more than this fraction of the
    #: skeleton, so a mis-set ``min_branch_length`` degrades gracefully
    #: instead of erasing the structure.  ``1.0`` disables the guard.
    max_prune_fraction: float = 0.5

    # --------------------------------------------------------- graph building
    #: Neighbourhood used for skeleton pixel degree: 8 (default) or 4.
    connectivity: int = 8
    #: Number of auxiliary nodes injected into a "pure cycle" skeleton
    #: component (a component in which every pixel has degree 2).  1 = the
    #: cycle becomes a self-loop, 2 = the cycle becomes a 2-node loop.
    pure_cycle_aux_nodes: int = 2
    #: ``"auto"`` -> ``networkx.Graph`` unless parallel edges are required by
    #: the extracted topology, in which case ``networkx.MultiGraph`` is used
    #: so that no cycle is silently collapsed.
    graph_container: str = "auto"
    #: Guo-Hall leaves a short diagonal ladder of ``d >= 3`` pixels at every
    #: 90 degree bend.  Merging that ladder into one node would leave a
    #: "junction" with exactly two edges, i.e. a fake node at a plain corner.
    #: When enabled, such nodes are dissolved back into a single edge.
    dissolve_degree2_junctions: bool = True

    # ---------------------------------------------------------- visualisation
    dpi: int = 150
    show_node_id: bool = True
    node_id_fontsize: int = 7
    edge_linewidth: float = 1.8

    # -------------------------------------------------------------------- misc
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.thinning_backend not in VALID_THINNING_BACKENDS:
            raise ValueError(
                f"thinning_backend must be one of {VALID_THINNING_BACKENDS}, "
                f"got {self.thinning_backend!r}"
            )
        if self.graph_container not in VALID_GRAPH_CONTAINERS:
            raise ValueError(
                f"graph_container must be one of {VALID_GRAPH_CONTAINERS}, "
                f"got {self.graph_container!r}"
            )
        if self.connectivity not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8, got {self.connectivity}")
        if self.pure_cycle_aux_nodes not in (1, 2):
            raise ValueError(
                "pure_cycle_aux_nodes must be 1 or 2, "
                f"got {self.pure_cycle_aux_nodes}"
            )
        if self.min_branch_length < 0:
            raise ValueError("min_branch_length must be >= 0")
        if not 0.0 < self.max_prune_fraction <= 1.0:
            raise ValueError("max_prune_fraction must be in (0, 1]")

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot (written next to every result)."""
        return asdict(self)


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for ``main.py`` (kept here so config and CLI never drift)."""
    parser = argparse.ArgumentParser(
        prog="skeleton_graph",
        description=(
            "Occupancy map -> Guo-Hall skeleton -> pruned skeleton -> "
            "node/edge graph -> visualisation (stage 1, no planning)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--map",
        dest="maps",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "occupancy map to process (.npy or PNG/JPG). Repeatable. "
            "Defaults to the three mazes in data/processed_scene_v1/maps."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="root output directory; one sub-directory per map",
    )
    parser.add_argument(
        "--min-branch-length",
        type=float,
        default=10.0,
        help="spur branches shorter than this are pruned",
    )
    parser.add_argument(
        "--max-prune-fraction",
        type=float,
        default=0.5,
        help="never prune more than this fraction of the skeleton (safety budget)",
    )
    parser.add_argument(
        "--no-pruning",
        action="store_true",
        help="disable spur pruning entirely (raw skeleton is used)",
    )
    parser.add_argument(
        "--polarity",
        choices=("auto", "normal", "invert"),
        default="auto",
        help=(
            "how to read the input's free/obstacle polarity: 'auto' detects it "
            "(npy: 1=obstacle; image: the class framing the map is the "
            "obstacle), 'normal' and 'invert' force the decision"
        ),
    )
    parser.add_argument(
        "--thinning-backend",
        choices=VALID_THINNING_BACKENDS,
        default="auto",
        help="Guo-Hall implementation to use",
    )
    parser.add_argument(
        "--graph-container",
        choices=VALID_GRAPH_CONTAINERS,
        default="auto",
        help="networkx container for the extracted graph",
    )
    parser.add_argument(
        "--keep-corner-junctions",
        dest="keep_corner_junctions",
        action="store_true",
        default=False,
        help=(
            "keep degree-2 junction clusters as nodes instead of dissolving "
            "them back into a single edge"
        ),
    )
    parser.add_argument(
        "--pure-cycle-aux-nodes",
        type=int,
        choices=(1, 2),
        default=2,
        help="auxiliary nodes injected per pure-cycle component",
    )
    parser.add_argument("--dpi", type=int, default=150, help="figure DPI")
    parser.add_argument(
        "--no-node-id",
        dest="show_node_id",
        action="store_false",
        default=True,
        help="do not draw node ids on the graph overlay",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress progress logging"
    )
    return parser


def configs_from_args(args: argparse.Namespace) -> list[Config]:
    """Expand parsed CLI arguments into one :class:`Config` per input map."""
    maps = args.maps
    if not maps:
        maps = [
            os.path.join(DEFAULT_MAPS_DIR, f"{name}.npy") for name in MAZE_NAMES
        ]

    shared = dict(
        invert_input={"auto": None, "normal": False, "invert": True}[args.polarity],
        thinning_backend=args.thinning_backend,
        enable_pruning=not args.no_pruning,
        min_branch_length=args.min_branch_length,
        max_prune_fraction=args.max_prune_fraction,
        graph_container=args.graph_container,
        dissolve_degree2_junctions=not args.keep_corner_junctions,
        pure_cycle_aux_nodes=args.pure_cycle_aux_nodes,
        dpi=args.dpi,
        show_node_id=args.show_node_id,
        verbose=not args.quiet,
    )

    configs: list[Config] = []
    for path in maps:
        stem = os.path.splitext(os.path.basename(path))[0]
        configs.append(
            Config(map_path=path, output_dir=os.path.join(args.output_dir, stem), **shared)
        )
    return configs


def config_field_names() -> list[str]:
    """Names of all configuration fields (used by the report writer)."""
    return [f.name for f in fields(Config)]
