"""Visualisation of the stage-1 pipeline.

Produces, for one map:

======================================  ==========================================
``01_occupancy.png``                    the input occupancy map
``02_skeleton_raw.png``                 the Guo-Hall skeleton
``03_skeleton_pruned.png``              after pruning; removed hairs in red
``04_graph_overlay.png``                graph drawn **on top of the map**
``05_pipeline_overview.png``            the four panels as one 2x2 figure
======================================  ==========================================

The graph overlay shows

* every edge as its full skeleton polyline (not a straight line between
  nodes);
* endpoints, junctions and auxiliary nodes with three distinct markers;
* the node id next to every node, so the drawing can be checked against
  ``graph.json``.

Coordinates are ``(x, y) = (column, row)`` everywhere; ``imshow`` is used with
``origin="upper"`` so that plotting ``(x, y)`` lines up with the image.
"""

from __future__ import annotations

import os

import numpy as np

from skeleton_graph.graph_extractor import (
    NODE_AUXILIARY,
    NODE_ENDPOINT,
    NODE_JUNCTION,
    iter_edges,
)

#: Marker style per node type (marker, colour, size).
NODE_STYLE = {
    NODE_ENDPOINT: ("o", "#ff3b30", 46.0),
    NODE_JUNCTION: ("s", "#0a84ff", 52.0),
    NODE_AUXILIARY: ("^", "#30d158", 64.0),
}

_EDGE_CMAP = "hsv"


def visualize_pipeline(
    occupancy: np.ndarray,
    skeleton_raw: np.ndarray,
    skeleton_pruned: np.ndarray,
    graph,
    output_dir: str,
    dpi: int = 150,
    show_node_id: bool = True,
    node_id_fontsize: int = 7,
    edge_linewidth: float = 1.8,
    title_suffix: str = "",
    verbose: bool = True,
) -> dict:
    """Render all stage-1 figures; returns a ``{name: path}`` mapping."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    fig_occ = _single_panel(
        occupancy, "Occupancy Map (0=obstacle, 255=free)", dpi, cmap="gray"
    )
    paths["01_occupancy"] = os.path.join(output_dir, "01_occupancy.png")
    fig_occ.savefig(paths["01_occupancy"], dpi=dpi, bbox_inches="tight")
    plt.close(fig_occ)

    fig_raw = _single_panel(
        _as_display(skeleton_raw), "Guo-Hall Skeleton (raw)", dpi, cmap="gray"
    )
    paths["02_skeleton_raw"] = os.path.join(output_dir, "02_skeleton_raw.png")
    fig_raw.savefig(paths["02_skeleton_raw"], dpi=dpi, bbox_inches="tight")
    plt.close(fig_raw)

    fig_pruned = _pruned_panel(skeleton_raw, skeleton_pruned, dpi)
    paths["03_skeleton_pruned"] = os.path.join(
        output_dir, "03_skeleton_pruned.png"
    )
    fig_pruned.savefig(paths["03_skeleton_pruned"], dpi=dpi, bbox_inches="tight")
    plt.close(fig_pruned)

    fig_graph = _graph_panel(
        occupancy,
        graph,
        dpi,
        show_node_id=show_node_id,
        node_id_fontsize=node_id_fontsize,
        edge_linewidth=edge_linewidth,
    )
    paths["04_graph_overlay"] = os.path.join(output_dir, "04_graph_overlay.png")
    fig_graph.savefig(paths["04_graph_overlay"], dpi=dpi, bbox_inches="tight")
    plt.close(fig_graph)

    overview_path = os.path.join(output_dir, "05_pipeline_overview.png")
    _overview(
        occupancy,
        skeleton_raw,
        skeleton_pruned,
        graph,
        overview_path,
        dpi,
        show_node_id=show_node_id,
        node_id_fontsize=node_id_fontsize,
        edge_linewidth=edge_linewidth,
        title_suffix=title_suffix,
    )
    paths["05_pipeline_overview"] = overview_path

    if verbose:
        print(f"[visualisation] wrote {len(paths)} figures to {output_dir}")
    return paths


def save_binary_png(path: str, binary: np.ndarray, dpi: int = 150) -> str:
    """Save a 0/1 mask as a black-and-white PNG (``1`` -> white)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = _single_panel(_as_display(binary), os.path.basename(path), dpi, cmap="gray")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------


def _single_panel(image: np.ndarray, title: str, dpi: int, cmap: str = "gray"):
    import matplotlib.pyplot as plt

    height, width = image.shape[:2]
    scale = 4.6
    fig, ax = plt.subplots(figsize=(scale, scale * height / width), dpi=dpi)
    ax.imshow(image, cmap=cmap, vmin=0, vmax=255, origin="upper", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x  (column)", fontsize=8)
    ax.set_ylabel("y  (row)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def _pruned_panel(skeleton_raw: np.ndarray, skeleton_pruned: np.ndarray, dpi: int):
    """Pruned skeleton, with whatever pruning removed highlighted in red."""
    import matplotlib.pyplot as plt

    raw = np.asarray(skeleton_raw) > 0
    pruned = np.asarray(skeleton_pruned) > 0
    removed = raw & ~pruned

    height, width = pruned.shape
    scale = 4.6
    fig, ax = plt.subplots(figsize=(scale, scale * height / width), dpi=dpi)
    ax.imshow(
        _as_display(pruned), cmap="gray", vmin=0, vmax=255,
        origin="upper", interpolation="nearest",
    )
    if removed.any():
        overlay = np.zeros((*removed.shape, 4), dtype=np.float64)
        overlay[removed] = (1.0, 0.2, 0.2, 1.0)
        ax.imshow(overlay, origin="upper", interpolation="nearest")
    title = "Pruned Skeleton"
    if removed.any():
        title += f"  (red = {int(removed.sum())} removed px)"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x  (column)", fontsize=8)
    ax.set_ylabel("y  (row)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def _graph_panel(
    occupancy: np.ndarray,
    graph,
    dpi: int,
    show_node_id: bool = True,
    node_id_fontsize: int = 7,
    edge_linewidth: float = 1.8,
):
    import matplotlib.pyplot as plt

    height, width = occupancy.shape
    scale = 5.4
    fig, ax = plt.subplots(figsize=(scale, scale * height / width), dpi=dpi)
    ax.imshow(
        occupancy, cmap="gray", vmin=0, vmax=255, origin="upper",
        interpolation="nearest",
    )
    _draw_graph(
        ax, graph, node_id_fontsize=node_id_fontsize,
        edge_linewidth=edge_linewidth, show_node_id=show_node_id,
    )
    ax.set_title("Extracted Graph over Occupancy Map", fontsize=10)
    ax.set_xlabel("x  (column)", fontsize=8)
    ax.set_ylabel("y  (row)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(
        handles=_legend_handles(edge_linewidth),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=4,
        fontsize=8,
        framealpha=0.9,
    )
    fig.tight_layout()
    return fig


def _overview(
    occupancy: np.ndarray,
    skeleton_raw: np.ndarray,
    skeleton_pruned: np.ndarray,
    graph,
    out_path: str,
    dpi: int,
    show_node_id: bool,
    node_id_fontsize: int,
    edge_linewidth: float,
    title_suffix: str = "",
) -> None:
    import matplotlib.pyplot as plt

    height, width = occupancy.shape
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 11.0 * height / width), dpi=dpi)

    axes[0, 0].imshow(
        occupancy, cmap="gray", vmin=0, vmax=255, origin="upper",
        interpolation="nearest",
    )
    axes[0, 0].set_title("(a) Occupancy Map", fontsize=11)

    axes[0, 1].imshow(
        _as_display(skeleton_raw), cmap="gray", vmin=0, vmax=255,
        origin="upper", interpolation="nearest",
    )
    axes[0, 1].set_title("(b) Guo-Hall Skeleton (raw)", fontsize=11)

    raw = np.asarray(skeleton_raw) > 0
    pruned = np.asarray(skeleton_pruned) > 0
    removed = raw & ~pruned
    axes[1, 0].imshow(
        _as_display(pruned), cmap="gray", vmin=0, vmax=255,
        origin="upper", interpolation="nearest",
    )
    if removed.any():
        overlay = np.zeros((*removed.shape, 4), dtype=np.float64)
        overlay[removed] = (1.0, 0.2, 0.2, 1.0)
        axes[1, 0].imshow(overlay, origin="upper", interpolation="nearest")
    axes[1, 0].set_title(
        f"(c) Pruned Skeleton  (red = {int(removed.sum())} removed px)", fontsize=11
    )

    axes[1, 1].imshow(
        occupancy, cmap="gray", vmin=0, vmax=255, origin="upper",
        interpolation="nearest",
    )
    _draw_graph(
        axes[1, 1], graph, node_id_fontsize=node_id_fontsize,
        edge_linewidth=edge_linewidth, show_node_id=show_node_id,
    )
    axes[1, 1].set_title("(d) Extracted Graph over Map", fontsize=11)
    axes[1, 1].legend(
        handles=_legend_handles(edge_linewidth),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=4,
        fontsize=7,
        framealpha=0.9,
    )

    for ax in axes.ravel():
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.tick_params(labelsize=7)

    if title_suffix:
        fig.suptitle(title_suffix, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------


def _draw_graph(
    ax,
    graph,
    node_id_fontsize: int,
    edge_linewidth: float,
    show_node_id: bool,
) -> None:
    """Draw edge polylines plus typed, labelled nodes."""
    import matplotlib.patheffects as path_effects
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    edges = list(iter_edges(graph))
    cmap = colormaps[_EDGE_CMAP]

    for index, (u, v, data) in enumerate(edges):
        pixels = data["pixels"]
        if len(pixels) < 2:
            continue
        xs = [p[0] for p in pixels]
        ys = [p[1] for p in pixels]
        color = to_hex(cmap((index * 0.618033988749895) % 1.0))
        ax.plot(
            xs, ys, "-", color=color, linewidth=edge_linewidth,
            solid_capstyle="round", zorder=2,
        )

    for node_id, data in sorted(graph.nodes(data=True)):
        kind = data.get("type", NODE_AUXILIARY)
        marker, color, size = NODE_STYLE.get(kind, NODE_STYLE[NODE_AUXILIARY])
        x, y = float(data["x"]), float(data["y"])
        ax.scatter(
            [x], [y], marker=marker, s=size, c=color,
            edgecolors="black", linewidths=0.7, zorder=4,
        )
        if show_node_id:
            text = ax.annotate(
                str(node_id),
                (x, y),
                textcoords="offset points",
                xytext=(5, -5),
                fontsize=node_id_fontsize,
                color="#ffe066",
                zorder=5,
            )
            text.set_path_effects(
                [path_effects.withStroke(linewidth=1.6, foreground="black")]
            )


def _legend_handles(edge_linewidth: float):
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color="#888888", linewidth=edge_linewidth, label="Graph edge"),
    ]
    for kind, (marker, color, size) in NODE_STYLE.items():
        handles.append(
            Line2D(
                [0], [0], marker=marker, color="none", markerfacecolor=color,
                markeredgecolor="black", markersize=size ** 0.5,
                label=f"{kind} node",
            )
        )
    return handles


def _as_display(binary: np.ndarray) -> np.ndarray:
    """0/1 mask -> 0/255 image for display."""
    return (np.asarray(binary) > 0).astype(np.uint8) * 255
