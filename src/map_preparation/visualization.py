"""Simple occupancy overlays; visualization never changes graph geometry."""

import numpy as np

COLORS = {
    "endpoint": "#0072B2",
    "junction": "#D55E00",
    "isolated": "#CC79A7",
    "loop_anchor": "#009E73",
}


def draw_graph(ax, result, labels=True):
    """Plot upstream curve samples and one representative per node cluster."""
    ax.imshow(
        result.occupancy,
        cmap="gray_r",
        vmin=0,
        vmax=1,
        origin="lower",
        interpolation="nearest",
    )
    rgba = np.zeros((*result.occupancy.shape, 4))
    for node in result.graph["nodes"]:
        if node["node_type"] == "junction":
            p = np.asarray(node["pixel_cluster"])
            rgba[p[:, 0], p[:, 1]] = [0.84, 0.33, 0, 0.65]
    ax.imshow(rgba, origin="lower", interpolation="nearest")
    for edge in result.graph["edges"]:
        p = np.asarray(edge["curve_pixels_rc"])
        ax.plot(p[:, 1], p[:, 0], color="#009E73", lw=1)
        if labels:
            r, c = p[len(p) // 2]
            ax.annotate(
                f'e{edge["edge_id"]}',
                (c, r),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6,
                color="#663399",
                bbox=dict(fc="white", alpha=0.7, ec="none", pad=0.2),
            )
    for node in result.graph["nodes"]:
        r, c = node["pixel_rc"]
        ax.scatter(
            c,
            r,
            c=COLORS[node["node_type"]],
            s=25,
            edgecolors="white",
            linewidths=0.4,
            zorder=4,
        )
        if labels:
            ax.annotate(
                f'n{node["node_id"]}',
                (c, r),
                xytext=(4, -8),
                textcoords="offset points",
                fontsize=7,
                color="#003366",
            )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def save_map_preview(result, prefix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, mask, title in zip(
        axes[:2],
        [result.skeleton, result.thin_skeleton],
        ["Original DMA", "Thinned working copy"],
    ):
        ax.imshow(
            result.occupancy,
            cmap="gray_r",
            vmin=0,
            vmax=1,
            origin="lower",
            interpolation="nearest",
        )
        rgba = np.zeros((*mask.shape, 4))
        rgba[mask] = [0.88, 0.16, 0.04, 1]
        ax.imshow(rgba, origin="lower", interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
    draw_graph(axes[2], result)
    axes[2].set_title("sknw sparse multigraph")
    a = result.graph["audit"]
    delta = result.graph["parameters"]["delta_px"]
    fig.suptitle(
        f'DMA delta={delta:g}px | V={a["nodes"]} E={a["edges"]} C={a["components"]} cycles={a["cycle_rank"]}'
    )
    fig.savefig(str(prefix) + "_preview.png", dpi=180)
    plt.close(fig)


def save_summary(cases, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = min(4, len(cases))
    rows = int(np.ceil(len(cases) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, result) in zip(axes.flat, cases):
        draw_graph(ax, result, labels=False)
        a = result.graph["audit"]
        ax.set_title(f'{title} | V={a["nodes"]} E={a["edges"]}', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
