"""Occupancy Map -> Skeleton -> Node-Edge Graph -> Visualization.

Stage-1 only.  This package deliberately contains **no** path planning, no
diffusion and no start/goal handling.

Global conventions (see `config.py` docstring for the full statement):

    occupancy : uint8 (H, W)   0 = obstacle, 255 = free
    skeleton  : uint8 (H, W)   0 = background, 1 = skeleton pixel
    coordinate: (x, y) = (image column, image row)

Keep this module import-light: importing `skeleton_graph` must not pull in
numpy/networkx/matplotlib, so that `python skeleton_graph/main.py` and
`python -m skeleton_graph.main` both stay cheap and side-effect free.
"""

__all__ = [
    "config",
    "map_loader",
    "thinning",
    "pruning",
    "graph_extractor",
    "visualization",
]
