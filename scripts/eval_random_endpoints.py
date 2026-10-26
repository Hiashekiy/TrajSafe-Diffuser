#!/usr/bin/env python
"""复杂地图随机起终点补充实验。

从 160k4p_c48/test 的互异占用栅格中按确定性复杂度分数选 4 张地图，
在带安全间隙的最大自由连通域内随机取点，并按栅格最短路长度分成近/中/远三档。
生成调用正在运行的 Diffusion Lens API，因此与交互面板使用完全相同的模型和代码路径。
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import string
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "carla_processed_160k4p_c48" / "test"
DEFAULT_OUT = ROOT / "outputs" / "ch4" / "random_endpoints" / "results.json"
TIERS = ("near", "medium", "far")
TIER_ZH = {"near": "近距离", "medium": "中距离", "far": "远距离"}


def erode_square(free: np.ndarray, radius: int) -> np.ndarray:
    """仅保留以自身为中心的 (2r+1)^2 邻域全部自由的像素。"""
    h, w = free.shape
    padded = np.pad(free.astype(np.int32), 1)
    integral = padded.cumsum(0).cumsum(1)
    y0 = np.clip(np.arange(h) - radius, 0, h)
    y1 = np.clip(np.arange(h) + radius + 1, 0, h)
    x0 = np.clip(np.arange(w) - radius, 0, w)
    x1 = np.clip(np.arange(w) + radius + 1, 0, w)
    sums = (integral[y1[:, None], x1[None, :]]
            - integral[y0[:, None], x1[None, :]]
            - integral[y1[:, None], x0[None, :]]
            + integral[y0[:, None], x0[None, :]])
    expected = ((y1 - y0)[:, None] * (x1 - x0)[None, :])
    return sums == expected


def largest_component(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = []
    for y, x in np.argwhere(mask):
        if seen[y, x]:
            continue
        seen[y, x] = True
        q = deque([(int(y), int(x))])
        comp = []
        while q:
            cy, cx = q.popleft()
            comp.append((cy, cx))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        if len(comp) > len(best):
            best = comp
    out = np.zeros_like(mask, dtype=bool)
    if best:
        yy, xx = zip(*best)
        out[np.asarray(yy), np.asarray(xx)] = True
    return out


def bfs_dist(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    sy, sx = start
    dist[sy, sx] = 0
    q = deque([(sy, sx)])
    while q:
        y, x = q.popleft()
        nd = dist[y, x] + 1
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and dist[ny, nx] < 0:
                dist[ny, nx] = nd
                q.append((ny, nx))
    return dist


def pixel_to_scene(point: tuple[int, int], resolution: int = 256) -> list[float]:
    y, x = point
    return [(x + 0.5) * 2.0 / resolution - 1.0,
            (y + 0.5) * 2.0 / resolution - 1.0]


def unique_maps_and_scores(occupancy: np.ndarray, candidate_mask: np.ndarray,
                           candidate_lengths: np.ndarray) -> list[dict]:
    rows, hashes = [], set()
    for i in range(len(occupancy)):
        digest = hashlib.sha1(np.asarray(occupancy[i]).tobytes()).hexdigest()
        if digest in hashes:
            continue
        hashes.add(digest)
        free = occupancy[i] <= 0.5
        transitions = (
            np.abs(np.diff(free.astype(np.int8), axis=0)).sum()
            + np.abs(np.diff(free.astype(np.int8), axis=1)).sum()
        ) / free.size
        valid = candidate_mask[i]
        max_candidate = float(candidate_lengths[i][valid].max()) if valid.any() else 0.0
        count = int(valid.sum())
        # 道路边界越多、可行候选越长、分支候选越丰富，分数越高。
        score = float(transitions * (1.0 + max_candidate) * (1.0 + 0.1 * max(0, count - 1)))
        rows.append({
            "map_index": i,
            "occupancy_sha1": digest,
            "boundary_density": float(transitions),
            "max_candidate_length_scene": max_candidate,
            "candidate_count": count,
            "complexity_score": score,
        })
    return sorted(rows, key=lambda row: row["complexity_score"], reverse=True)


def sample_pairs(mask: np.ndarray, rng: np.random.Generator,
                 pairs_per_tier: int) -> dict[str, list[dict]]:
    points = np.argwhere(mask)
    if len(points) < 100:
        raise RuntimeError("安全收缩后的自由连通域过小")
    starts = points[rng.choice(len(points), size=min(32, len(points)), replace=False)]
    candidates = []
    cell_m = 160.0 / mask.shape[0]
    for start_arr in starts:
        start = (int(start_arr[0]), int(start_arr[1]))
        dist = bfs_dist(mask, start)
        goals = points[rng.choice(len(points), size=min(40, len(points)), replace=False)]
        for goal_arr in goals:
            goal = (int(goal_arr[0]), int(goal_arr[1]))
            cells = int(dist[goal])
            if cells < 32:  # 至少约 20 m，排除几乎重合的端点。
                continue
            euclid_cells = float(np.linalg.norm(np.asarray(goal) - np.asarray(start)))
            candidates.append({
                "start_px": list(start),
                "goal_px": list(goal),
                "geodesic_m": cells * cell_m,
                "euclidean_m": euclid_cells * cell_m,
                "detour_ratio": cells / max(euclid_cells, 1e-6),
            })
    if len(candidates) < 30:
        raise RuntimeError("可用随机点对不足")
    candidates.sort(key=lambda row: row["geodesic_m"])
    values = np.asarray([row["geodesic_m"] for row in candidates])
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    pools = {
        "near": [row for row in candidates if row["geodesic_m"] <= q1],
        "medium": [row for row in candidates if q1 < row["geodesic_m"] <= q2],
        "far": [row for row in candidates if row["geodesic_m"] > q2],
    }
    selected: dict[str, list[dict]] = {tier: [] for tier in TIERS}
    used: list[np.ndarray] = []
    for tier in TIERS:
        order = rng.permutation(len(pools[tier]))
        for idx in order:
            row = dict(pools[tier][int(idx)])
            endpoints = np.asarray(row["start_px"] + row["goal_px"], dtype=float)
            if any(np.linalg.norm(endpoints - old) < 18 for old in used):
                continue
            start = tuple(row["start_px"])
            goal = tuple(row["goal_px"])
            row["condition"] = [pixel_to_scene(start), pixel_to_scene(goal)]
            row["tier"] = tier
            selected[tier].append(row)
            used.append(endpoints)
            if len(selected[tier]) == pairs_per_tier:
                break
        if len(selected[tier]) < pairs_per_tier:
            raise RuntimeError(f"{tier} 档无法取得足够分散的点对")
    return selected


def post_generate(api: str, request_body: dict) -> tuple[int, dict]:
    body = json.dumps(request_body).encode("utf-8")
    request = Request(api.rstrip("/") + "/generate", data=body,
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=180) as response:
            return int(response.status), json.load(response)
    except HTTPError as error:
        try:
            payload = json.load(error)
        except Exception:
            payload = {"error": str(error)}
        return int(error.code), payload


def compact_result(payload: dict) -> dict:
    final = payload.get("final_validation") or {}
    topology = (payload.get("topology") or {}).get("metrics") or {}
    collision_free = bool(not final.get("final_collision", True))
    corridor_feasible = bool(final.get("final_constraint_feasible", False))
    return {
        "cache_hit": bool(payload.get("cache_hit")),
        "elapsed_ms": payload.get("elapsed_ms"),
        "steps": payload.get("steps"),
        "alm": {
            "enabled": (payload.get("alm") or {}).get("enabled"),
            "final_project": (payload.get("alm") or {}).get("final_project"),
            "status": (payload.get("alm") or {}).get("status"),
        },
        "final_projection": payload.get("final_projection"),
        "final_validation": final,
        "collision_free": collision_free,
        "strict_corridor_feasible": corridor_feasible,
        "topology_metrics": topology,
        "final_curve": payload.get("final_curve"),
        "raw_final_curve": ((payload.get("x0_raw_history") or [None])[-1]),
        "pre_projection": payload.get("pre_projection"),
        "corridor_cells": (payload.get("corridor") or {}).get("cells") or [],
    }


def summarize(records: list[dict]) -> dict:
    summary = {}
    for tier in TIERS:
        rows = [row for row in records if row["tier"] == tier]
        generated = [row for row in rows if row.get("http_status") == 200]
        collision_free = [row for row in generated
                          if row.get("result", {}).get("collision_free")]
        corridor_feasible = [row for row in generated
                             if row.get("result", {}).get("strict_corridor_feasible")]
        times = [float(row["result"]["elapsed_ms"]) for row in generated
                 if row["result"].get("elapsed_ms") is not None
                 and not row["result"].get("cache_hit")]
        summary[tier] = {
            "label": TIER_ZH[tier],
            "attempted": len(rows),
            "generated": len(generated),
            "collision_free": len(collision_free),
            "collision_free_rate": len(collision_free) / len(rows) if rows else None,
            "strict_corridor_feasible": len(corridor_feasible),
            "strict_corridor_feasible_rate": (
                len(corridor_feasible) / len(rows) if rows else None),
            "mean_geodesic_m": float(np.mean([row["geodesic_m"] for row in rows])),
            "min_geodesic_m": float(np.min([row["geodesic_m"] for row in rows])),
            "max_geodesic_m": float(np.max([row["geodesic_m"] for row in rows])),
            "mean_elapsed_ms_uncached": float(np.mean(times)) if times else None,
        }
    return summary


def render_figures(results_path: Path, pairs_per_tier: int,
                   figure_dir: Path) -> list[Path]:
    """每两个样本下标生成一张 4×6 组合图。"""
    if pairs_per_tier < 2:
        raise RuntimeError("自动绘图要求 --pairs-per-tier 至少为 2")
    figure_dir.mkdir(parents=True, exist_ok=True)
    tag = results_path.stem
    if tag.startswith("results_"):
        tag = tag[len("results_"):]
    elif tag == "results":
        tag = "default"
    outputs = []
    pair_groups = [(i, i + 1) for i in range(0, pairs_per_tier - 1, 2)]
    for figure_index, (first, second) in enumerate(pair_groups):
        suffix = (string.ascii_lowercase[figure_index]
                  if figure_index < len(string.ascii_lowercase)
                  else str(figure_index + 1))
        figure_path = figure_dir / f"random_endpoints_{tag}_{suffix}.png"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "plot_random_endpoints.py"),
            "--results", str(results_path),
            "--pair-indices", f"{first},{second}",
            "--out", str(figure_path),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        outputs.append(figure_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8765")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--maps", type=int, default=4)
    parser.add_argument("--pairs-per-tier", type=int, default=5)
    parser.add_argument("--sampling-seed", type=int, default=20260925)
    parser.add_argument("--generation-seed", type=int, default=43)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--plot", action="store_true",
                        help="评测结束后自动按每两个样本一组生成4×6组合图")
    parser.add_argument(
        "--figure-dir", default=str(ROOT / "outputs" / "ch4" / "figures"),
        help="--plot 生成图片的目录")
    args = parser.parse_args()

    occupancy = np.load(DATA / "occupancy.npy", mmap_mode="r")
    candidate_mask = np.load(DATA / "candidate_mask.npy", mmap_mode="r")
    candidate_lengths = np.load(DATA / "candidate_lengths.npy", mmap_mode="r")
    ranked = unique_maps_and_scores(occupancy, candidate_mask, candidate_lengths)
    selected_maps = ranked[:args.maps]
    rng = np.random.default_rng(args.sampling_seed)

    records = []
    total = args.maps * len(TIERS) * args.pairs_per_tier
    done = 0
    for map_rank, map_info in enumerate(selected_maps, 1):
        map_index = int(map_info["map_index"])
        free = occupancy[map_index] <= 0.5
        core = largest_component(erode_square(free, radius=4))
        pairs = sample_pairs(core, rng, args.pairs_per_tier)
        for tier in TIERS:
            for pair_no, pair in enumerate(pairs[tier], 1):
                done += 1
                body = {
                    "sample_key": f"test_{map_index:04d}",
                    "split": "test",
                    "dataset": "160k4p_c48",
                    "model_id": "K4P_c48_oneshot:best_task",
                    "seed": args.generation_seed,
                    "condition": pair["condition"],
                    "alm_enabled": True,
                    "final_project": True,
                    "steps": args.steps,
                }
                started = time.perf_counter()
                status, payload = post_generate(args.api, body)
                record = {
                    "map_rank": map_rank,
                    "map_index": map_index,
                    "map_complexity": map_info,
                    "tier": tier,
                    "pair_no": pair_no,
                    **pair,
                    "request": body,
                    "http_status": status,
                    "wall_elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                }
                if status == 200:
                    record["result"] = compact_result(payload)
                    if not record["result"]["collision_free"]:
                        result_text = "COLLISION"
                    elif not record["result"]["strict_corridor_feasible"]:
                        result_text = "SAFE_WITH_RESIDUAL"
                    else:
                        result_text = "SAFE"
                else:
                    record["error"] = payload.get("error", str(payload))
                    result_text = "ERROR"
                records.append(record)
                print(f"[{done:02d}/{total}] map={map_index} {tier}/{pair_no} "
                      f"d={pair['geodesic_m']:.1f}m -> {result_text}", flush=True)

    output = {
        "protocol": {
            "dataset": "160k4p_c48/test",
            "model": "K4P_c48_oneshot:best_task",
            "maps": args.maps,
            "pairs_per_tier_per_map": args.pairs_per_tier,
            "total": total,
            "sampling_seed": args.sampling_seed,
            "generation_seed": args.generation_seed,
            "steps": args.steps,
            "alm_enabled": True,
            "final_project": True,
            "clearance_radius_pixels": 4,
            "scene_width_m": 160.0,
            "distance_definition": "4-connected shortest path on eroded free grid",
            "map_selection": "top complexity score among unique occupancies",
            "complexity_formula": "boundary_density*(1+max_candidate_length)*(1+0.1*(candidate_count-1))",
        },
        "selected_maps": selected_maps,
        "records": records,
        "summary": summarize(records),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2), flush=True)
    print(out_path, flush=True)
    if args.plot:
        figures = render_figures(out_path.resolve(), args.pairs_per_tier,
                                 Path(args.figure_dir).resolve())
        print("generated figures:", flush=True)
        for figure in figures:
            print(figure, flush=True)


if __name__ == "__main__":
    main()
