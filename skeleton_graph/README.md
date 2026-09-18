# skeleton_graph — Occupancy Map → Skeleton → Node-Edge Graph

Stage 1 of the planning stack, and **nothing beyond it**: no path planning, no
graph diffusion, no start/goal selection.

```
Occupancy Map  →  Guo-Hall Skeleton  →  Pruned Skeleton  →  Node-Edge Graph  →  Visualization
```

## Conventions (fixed for the whole package)

| thing | representation |
| --- | --- |
| occupancy | `uint8 (H, W)`, `0 = obstacle`, `255 = free` |
| skeleton | `uint8 (H, W)`, `0 = background`, `1 = skeleton pixel` |
| coordinates | `(x, y) = (image column, image row)` — never `(row, col)` |
| thinning input | **free space**, never the obstacle region |

`1 = obstacle` is the convention of `data/processed_scene_v1/maps/*.npy`; the
loader converts to the package convention and logs its polarity decision.

## Files

```
skeleton_graph/
├── main.py             # CLI entry point, report writer, graph.json round-trip check
├── config.py           # every tunable (no magic numbers in the algorithm modules)
├── map_loader.py       # .npy / PNG / JPG -> 0/255 occupancy, polarity handling
├── thinning.py         # Guo-Hall thinning + pixel degree + diagnostics
├── pruning.py          # short-hair removal (endpoint -> junction branches only)
├── graph_extractor.py  # junction clustering, edge tracing, cycle handling, graph.json
├── visualization.py    # 2x2 overview + individual panels
├── topology.py         # connected components / holes, used by the acceptance checks
├── data/test_map.png   # umaze exported as a plain black/white PNG
└── outputs/            # generated results (git-ignored)
```

## Run

```bash
# all three mazes in data/processed_scene_v1/maps
E:/CondaEnvData/envs/GGMPC/python.exe skeleton_graph/main.py

# one map, custom threshold
E:/CondaEnvData/envs/GGMPC/python.exe skeleton_graph/main.py \
    --map data/processed_scene_v1/maps/medium.npy --min-branch-length 25

# the PNG copy, to exercise the image + polarity path
E:/CondaEnvData/envs/GGMPC/python.exe skeleton_graph/main.py \
    --map skeleton_graph/data/test_map.png
```

`python -m skeleton_graph.main` works too. Tests:

```bash
E:/CondaEnvData/envs/GGMPC/python.exe -m pytest tests/test_skeleton_graph.py -q
```

## Guo-Hall backend

The task asks for `cv2.ximgproc.thinning(..., THINNING_GUOHALL)`. That lives in
`opencv-contrib-python`, which is **not installed** in the `GGMPC` environment
(and no package index was reachable to add it), so `thinning.py` ships a
faithful vectorised Guo-Hall implementation and uses it as the fallback:

* `thinning_backend = "auto"` (default) — `cv2.ximgproc` when importable,
  otherwise the bundled numpy version. Identical conditions, no dependency.
* `"cv2"` / `"numpy"` force one of them.

Both are the same algorithm, so results do not change when contrib is
available; `report.json` records which one actually ran.

## The three things that are easy to get wrong

**1. Junction cluster merging.** A crossing is 2–5 adjacent `d >= 3` pixels,
not one. Every 8-connected cluster of them becomes exactly one node
(`junction_cluster_sizes` in `report.json` shows 2–7 px clusters on the test
maps). Three refinements on top of plain clustering:

* a degree-2 pixel whose neighbours all belong to the *same* cluster is
  absorbed into it — otherwise it shows up as a bogus length-2 self-loop
  (`absorbed_bridge_pixels`);
* a cluster with exactly **two** incident edges is not a junction at all: the
  corridor merely turns there, and Guo-Hall leaves a short diagonal ladder of
  `d >= 3` pixels at every 90° bend. Those nodes are dissolved back into one
  edge, with the polylines concatenated through the cluster so the merged
  polyline stays 8-connected (`dissolved_junctions`; disable with
  `--keep-corner-junctions`);
* parallel edges are never silently merged: `graph_container="auto"` uses
  `networkx.Graph` unless parallel edges actually exist, in which case it
  switches to `MultiGraph`. `large.npy` needs this.

**2. Edge tracing.** From every node pixel the skeleton is walked through
`d == 2` chain pixels until another node is reached, then the pixel set of the
walk is used as the identity key so the same edge is not emitted twice (once
from each end). Each edge stores its **full polyline**, and the acceptance
check verifies that every chain pixel lands on exactly one edge and that no
polyline contains a jump.

**3. Cycle preservation.** A component in which every pixel has `d == 2` is a
pure loop with no endpoint and no junction — naively it yields zero nodes and
disappears. One or two auxiliary nodes are injected
(`pure_cycle_aux_nodes`, default 2 → the loop becomes a clean two-edge cycle;
`1` → a self-loop). The formal check is
`skeleton holes == graph cycle rank`, run on every map.

## Pruning rules

Only `endpoint → junction` branches shorter than `min_branch_length` are
removed. Deliberately never removed:

* branches between two junctions, however short — short corridors are real;
* branches that end at another endpoint (isolated path components) — removing
  them would delete real free-space structure.

`max_prune_fraction` (default `0.5`) is a safety budget: pruning never removes
more than that share of the skeleton, shortest branches first, so a mis-set
threshold degrades gracefully instead of erasing everything.

Because only endpoint-rooted trees are removed, pruning cannot disconnect the
skeleton and cannot destroy a cycle.

## Outputs (per map)

```
outputs/<map>/
├── 01_occupancy.png
├── 02_skeleton_raw.png
├── 03_skeleton_pruned.png      # red = pixels pruning removed
├── 04_graph_overlay.png        # graph drawn on top of the map
├── 05_pipeline_overview.png    # the four panels as one 2x2 figure
├── graph.json                  # networkx-independent {nodes, edges}
├── report.txt                  # short report + acceptance checks
└── report.json                 # full machine-readable record
```

`graph.json`:

```json
{
  "nodes": [{"id": 0, "x": 120, "y": 84, "type": "junction"}],
  "edges": [{"source": 0, "target": 1, "length": 35.4, "pixels": [[120, 84], "..."]}]
}
```

Node `type` is one of `endpoint`, `junction`, `auxiliary`. The pipeline reads
the file back and compares node set and edge multiset against the in-memory
graph, so every run proves the round trip.
