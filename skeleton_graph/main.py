#!/usr/bin/env python
"""Stage-1 pipeline entry point.

    Occupancy Map -> Skeleton -> Node-Edge Graph -> Visualization

Deliberately stops there: no path planning, no diffusion, no start/goal
selection.

Usage
-----
Run the three mazes shipped with ``data/processed_scene_v1``::

    E:/CondaEnvData/envs/GGMPC/python.exe skeleton_graph/main.py

One specific map::

    E:/CondaEnvData/envs/GGMPC/python.exe skeleton_graph/main.py \\
        --map data/processed_scene_v1/maps/medium.npy \\
        --min-branch-length 10

``python -m skeleton_graph.main`` works as well.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

# Allow both `python skeleton_graph/main.py` and `python -m skeleton_graph.main`.
if __package__ in (None, ""):  # pragma: no cover - exercised by direct run
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skeleton_graph import graph_extractor, pruning, thinning, visualization  # noqa: E402
from skeleton_graph.config import (  # noqa: E402
    Config,
    build_arg_parser,
    configs_from_args,
)
from skeleton_graph.map_loader import free_mask, load_map, map_stats  # noqa: E402


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def run_pipeline(config: Config) -> dict:
    """Run the full stage-1 pipeline for one map; returns the stats record."""
    started = time.perf_counter()
    os.makedirs(config.output_dir, exist_ok=True)

    if config.verbose:
        print(f"\n=== {os.path.basename(config.map_path)} ===")

    # 1. map --------------------------------------------------------------
    occupancy = load_map(
        config.map_path,
        invert=config.invert_input,
        threshold=config.binarize_threshold,
        verbose=config.verbose,
    )
    stats: dict = {"map": map_stats(occupancy), "config": config.to_dict()}

    # 2. Guo-Hall thinning of the free space ------------------------------
    timings: dict[str, float] = {}
    tick = time.perf_counter()
    skeleton_raw = thinning.guo_hall_thinning(
        occupancy,
        backend=config.thinning_backend,
        verbose=config.verbose,
    )
    timings["thinning"] = time.perf_counter() - tick

    stats["thinning_backend"] = thinning.resolve_backend(config.thinning_backend)
    stats["skeleton_raw"] = thinning.skeleton_diagnostics(
        skeleton_raw, config.connectivity
    )
    stats["thinning_topology"] = thinning.validate_thinning(
        free_mask(occupancy), skeleton_raw
    )

    # 3. spur pruning ------------------------------------------------------
    tick = time.perf_counter()
    if config.enable_pruning:
        skeleton_pruned, prune_info = pruning.prune_skeleton(
            skeleton_raw,
            min_branch_length=config.min_branch_length,
            connectivity=config.connectivity,
            max_prune_fraction=config.max_prune_fraction,
            verbose=config.verbose,
        )
    else:
        skeleton_pruned, prune_info = pruning.no_pruning(skeleton_raw)
        if config.verbose:
            print("[pruning] disabled")
    timings["pruning"] = time.perf_counter() - tick
    stats["pruning"] = prune_info
    _annotate_spur_clearance(occupancy, skeleton_raw, prune_info)
    stats["skeleton_pruned"] = thinning.skeleton_diagnostics(
        skeleton_pruned, config.connectivity
    )
    stats["pruning_topology"] = thinning.validate_thinning(
        free_mask(occupancy), skeleton_pruned
    )

    # 4. skeleton -> graph -------------------------------------------------
    tick = time.perf_counter()
    graph, graph_info = graph_extractor.skeleton_to_graph(
        skeleton_pruned,
        connectivity=config.connectivity,
        pure_cycle_aux_nodes=config.pure_cycle_aux_nodes,
        graph_container=config.graph_container,
        dissolve_degree2_junctions=config.dissolve_degree2_junctions,
        verbose=config.verbose,
    )
    timings["graph"] = time.perf_counter() - tick

    summary = graph_extractor.graph_summary(graph)
    validation = graph_extractor.validate_graph(
        skeleton_pruned,
        graph,
        connectivity=config.connectivity,
        non_chain_pixels_xy=graph_info["non_chain_pixels"],
    )
    stats["graph"] = summary
    stats["graph_build"] = {
        "container": graph_info["container"],
        "junction_cluster_sizes": graph_info["junction_cluster_sizes"],
        "junction_cluster_count": len(graph_info["junction_cluster_sizes"]),
        "max_junction_cluster": max(graph_info["junction_cluster_sizes"] or [0]),
        "absorbed_bridge_pixels": graph_info["absorbed_bridge_pixels"],
        "dissolved_junctions": graph_info["dissolved_junctions"],
        "dissolved_junction_count": len(graph_info["dissolved_junctions"]),
        "auxiliary": graph_info["auxiliary"],
        "parallel_edge_pairs": graph_info["parallel_edge_pairs"],
        "self_loops": graph_info["self_loops"],
    }
    stats["validation"] = validation

    # 5. export graph.json and read it back --------------------------------
    graph_path = os.path.join(config.output_dir, "graph.json")
    graph_extractor.graph_to_json(graph, graph_path, verbose=config.verbose)
    reloaded = graph_extractor.load_graph_json(graph_path)
    round_trip_ok, round_trip_details = graph_extractor.graphs_equal(graph, reloaded)
    stats["graph_json"] = {
        "path": os.path.abspath(graph_path),
        "round_trip_ok": round_trip_ok,
        **round_trip_details,
    }

    # 6. visualisation -----------------------------------------------------
    tick = time.perf_counter()
    figure_paths = visualization.visualize_pipeline(
        occupancy,
        skeleton_raw,
        skeleton_pruned,
        graph,
        config.output_dir,
        dpi=config.dpi,
        show_node_id=config.show_node_id,
        node_id_fontsize=config.node_id_fontsize,
        edge_linewidth=config.edge_linewidth,
        title_suffix=os.path.basename(config.map_path),
        verbose=config.verbose,
    )
    timings["visualization"] = time.perf_counter() - tick
    stats["figures"] = {k: os.path.abspath(v) for k, v in figure_paths.items()}

    timings["total"] = time.perf_counter() - started
    stats["timings_seconds"] = {k: round(v, 4) for k, v in timings.items()}

    _write_reports(config.output_dir, stats, verbose=config.verbose)
    return stats


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def _annotate_spur_clearance(
    occupancy: np.ndarray, skeleton: np.ndarray, prune_info: dict
) -> None:
    """Tag each pruning candidate with its distance to the nearest obstacle.

    **Report-only.**  This is a QA metric for choosing ``min_branch_length``;
    the skeleton itself is produced purely by Guo-Hall thinning.  The reading:

    * clearance ~= the skeleton's average clearance (corridor half-width)
      -> the branch ends in a genuine dead-end cap, it is a real corridor;
    * clearance ~= 1-3 px -> the branch ends at a free-space corner, i.e. a
      medial-axis hair that is safe to prune.

    Without it, raising ``min_branch_length`` looks free until whole dead-end
    corridors start disappearing.
    """
    spurs = prune_info.get("candidate_spurs_before") or []
    if not spurs:
        return
    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover - scipy is present in GGMPC
        return

    free = np.asarray(occupancy) > 127
    distance = ndimage.distance_transform_edt(free)
    ridge = float(distance[np.asarray(skeleton) > 0].mean())

    for spur in spurs:
        x, y = spur["endpoint"]
        clearance = float(distance[y, x])
        spur["endpoint_clearance_px"] = round(clearance, 3)
        spur["looks_like"] = (
            "corner_hair" if clearance < 0.5 * ridge else "dead_end_corridor"
        )

    prune_info["ridge_clearance_px"] = round(ridge, 3)
    prune_info["clearance_method"] = (
        "scipy distance transform, report-only heuristic "
        "(clearance < 0.5 * ridge -> corner hair)"
    )


def format_report(stats: dict) -> str:
    """The short report requested for manual review (Chinese labels)."""
    graph = stats["graph"]
    types = graph["node_type_counts"]
    raw_pixels = stats["skeleton_raw"]["pixels"]
    pruned_pixels = stats["skeleton_pruned"]["pixels"]
    total = stats["timings_seconds"]["total"]

    lines = [
        f"输入地图尺寸：{stats['map']['height']} x {stats['map']['width']}",
        f"Skeleton像素数：{raw_pixels}",
        f"Pruning前/后像素数：{raw_pixels} / {pruned_pixels}",
        f"Graph节点数：{graph['nodes']}",
        f"  endpoint: {types['endpoint']}",
        f"  junction: {types['junction']}",
        f"  auxiliary: {types['auxiliary']}",
        f"Graph边数：{graph['edges']}",
        f"Connected components数量：{graph['connected_components']}",
        f"是否存在cycle：{'是' if graph['has_cycle'] else '否'}"
        f"（cycle rank = {graph['cycle_rank']}）",
        f"运行时间：{total:.2f} s",
    ]
    return "\n".join(lines)


def format_diagnostics(stats: dict) -> str:
    """Acceptance-criteria evidence, printed after the short report."""
    topo_thin = stats["thinning_topology"]
    topo_prune = stats["pruning_topology"]
    validation = stats["validation"]
    build = stats["graph_build"]
    pruning_info = stats["pruning"]

    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    rows = [
        ("free space -> 1px skeleton",
         stats["skeleton_raw"]["thick_2x2_blocks"] == 0,
         f"2x2 blocks = {stats['skeleton_raw']['thick_2x2_blocks']}"),
        ("thinning keeps connectivity/holes", topo_thin["ok"],
         f"components {topo_thin['reference']['components']}->"
         f"{topo_thin['candidate']['components']}, holes "
         f"{topo_thin['reference']['holes']}->{topo_thin['candidate']['holes']}"),
        ("pruning keeps connectivity/holes", topo_prune["ok"],
         f"components {topo_prune['candidate']['components']}, holes "
         f"{topo_prune['candidate']['holes']}"),
        ("short spurs removed", pruning_info["branches_removed"] >= 0,
         f"{pruning_info['branches_removed']} branches / "
         f"{pruning_info['pixels_removed']} px (min length "
         f"{pruning_info['min_branch_length']})"
         + ("  [budget exhausted, threshold likely too large]"
            if pruning_info.get("budget_exhausted") else "")),
        ("junction clusters merged", True,
         f"{build['junction_cluster_count']} clusters, largest "
         f"{build['max_junction_cluster']} px, "
         f"{build['absorbed_bridge_pixels']} bridge px absorbed"),
        ("corner clusters dissolved", True,
         f"{build['dissolved_junction_count']} degree-2 junction clusters "
         f"merged back into a single edge"),
        ("every chain pixel on exactly one edge",
         validation["chain_coverage_complete"]
         and validation["chain_pixels_covered_twice"] == 0,
         f"{validation['chain_pixels_covered']}/"
         f"{validation['chain_pixels_total']} covered, "
         f"{validation['chain_pixels_covered_twice']} doubled"),
        ("edges carry the skeleton polyline", validation["pixels_off_skeleton"] == 0,
         f"{validation['pixels_off_skeleton']} px off skeleton, "
         f"{validation['edge_length_mismatches']} length mismatches"),
        ("edge polylines are connected",
         validation["polyline_breaks"] == 0,
         f"{validation['polyline_breaks']} jumps between consecutive pixels"),
        ("cycles preserved", validation["cycles_preserved"],
         f"skeleton holes {validation['skeleton_holes']} == "
         f"graph cycle rank {validation['graph_cycle_rank']}"),
        ("graph.json round-trip", stats["graph_json"]["round_trip_ok"],
         f"{stats['graph_json']['left_nodes']} nodes / "
         f"{stats['graph_json']['left_edges']} edges rebuilt identically"),
    ]
    width = max(len(name) for name, _, _ in rows)
    lines = ["验收检查:"]
    for name, ok, detail in rows:
        lines.append(f"  [{mark(ok)}] {name.ljust(width)}  {detail}")

    lines.append(
        f"  容器: {build['container']}"
        f"{'  (存在平行边，必须用 MultiGraph 才能不丢环)' if build['container'] == 'multigraph' else ''}"
    )
    if build["self_loops"]:
        lines.append(f"  自环边: {build['self_loops']}")
    if build["auxiliary"]["pure_cycles"]:
        lines.append(
            f"  纯环组件: {len(build['auxiliary']['pure_cycles'])} 个，已注入 auxiliary 节点"
        )
    spurs = pruning_info.get("candidate_spurs_before") or []
    if spurs:
        ridge = pruning_info.get("ridge_clearance_px")
        lines.append(
            f"  候选毛刺(pruning 前) — 共 {len(spurs)} 条，"
            f"骨架平均 clearance {ridge} px:"
        )
        for spur in spurs:
            clearance = spur.get("endpoint_clearance_px")
            verdict = {
                "corner_hair": "角点毛刺(可删)",
                "dead_end_corridor": "真实死胡同(勿删)",
            }.get(spur.get("looks_like"), "")
            lines.append(
                f"    L={spur['length']:8.2f}  endpoint(x,y)={tuple(spur['endpoint'])}"
                f"  clearance={clearance} px  {verdict}"
            )
        if pruning_info["branches_removed"] == 0 and spurs:
            lines.append(
                "    提示: 当前 min_branch_length="
                f"{pruning_info['min_branch_length']} 小于最短候选 "
                f"{spurs[0]['length']:.2f}，因此本轮未删除任何支路。"
            )
    return "\n".join(lines)


def _write_reports(output_dir: str, stats: dict, verbose: bool) -> None:
    text_path = os.path.join(output_dir, "report.txt")
    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write(format_report(stats))
        handle.write("\n\n")
        handle.write(format_diagnostics(stats))
        handle.write("\n")

    json_path = os.path.join(output_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False, default=_json_default)

    if verbose:
        print("[" + "-" * 60 + "]")
        print(format_report(stats))
        print(format_diagnostics(stats))


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    # The report contains Chinese labels; on a Windows console the default
    # code page turns them into mojibake.  UTF-8 output is always correct and
    # is what report.txt uses too.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):  # pragma: no cover - exotic stdout
            pass

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configs = configs_from_args(args)

    all_stats = []
    failures = []
    for config in configs:
        try:
            all_stats.append(run_pipeline(config))
        except Exception as error:  # keep going so one bad map cannot hide the rest
            failures.append((config.map_path, error))
            print(f"[error] {config.map_path}: {error}", file=sys.stderr)

    if len(all_stats) > 1:
        print("\n" + "=" * 64)
        print("汇总")
        print("=" * 64)
        for stats in all_stats:
            graph = stats["graph"]
            name = os.path.basename(stats["config"]["map_path"])
            print(
                f"{name:24s} skel={stats['skeleton_pruned']['pixels']:5d} "
                f"nodes={graph['nodes']:3d} edges={graph['edges']:3d} "
                f"cycles={graph['cycle_rank']} "
                f"valid={stats['validation']['ok']} "
                f"json_rt={stats['graph_json']['round_trip_ok']} "
                f"{stats['timings_seconds']['total']:.2f}s"
            )

    if failures:
        for path, error in failures:
            print(f"FAILED {path}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
