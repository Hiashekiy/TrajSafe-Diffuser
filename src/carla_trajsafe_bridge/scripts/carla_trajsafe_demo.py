"""Command line entry point for the TrajSafe-Diffuser CARLA closed loop.

    # 1) plan ONCE with the guided ALM sampler and freeze the result
    python scripts/carla_trajsafe_demo.py --mode plan

    # 2) drive that frozen plan in CARLA (no re-planning)
    python scripts/carla_trajsafe_demo.py --mode drive

    # or both in one go
    python scripts/carla_trajsafe_demo.py --mode all --no-video

Sample test_0056 (episode 70, Town03, dataset index 56) is reproduced from the
recorded anchor pose; see src/carla_bridge/scenario.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.carla_bridge import demo as demo_mod  # noqa: E402
from src.carla_bridge.planner_adapter import check_acceptance  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "carla_demo.yaml"))
    parser.add_argument("--mode", choices=["plan", "drive", "all"], default="all")
    parser.add_argument("--out", default=None,
                        help="output directory (default from the config)")
    parser.add_argument("--plan", default=None,
                        help="existing plan .npz to drive (skips planning entirely)")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--occupancy-source", choices=["dataset", "live"], default=None)
    parser.add_argument("--ablation", choices=["none", "raw"], default="none",
                        help="raw = same checkpoint/seed/occupancy with ALM disabled")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = demo_mod.load_yaml(args.config)
    if args.occupancy_source:
        cfg["planner"]["occupancy_source"] = args.occupancy_source
    if args.seed is not None:
        cfg["planner"]["diffusion_seed"] = int(args.seed)
    out_dir = args.out or cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    if args.ablation == "raw":
        cfg["planner"]["alm_enabled"] = False
        cfg["planner"]["require_guided"] = False
    tag = "" if args.ablation == "none" else "_" + args.ablation
    npz_path = args.plan or os.path.join(
        out_dir, "plan_%s%s.npz" % (demo_mod.SAMPLE["key"], tag))
    json_path = os.path.splitext(npz_path)[0] + ".json"
    manifest: dict = {"mode": args.mode, "config": os.path.abspath(args.config),
                      "ablation": args.ablation, "out_dir": os.path.abspath(out_dir)}

    ctx = demo_mod.build_sample_context(cfg)
    latest = str(cfg["planner"].get("backend", "")).startswith("latest_160m")
    plan = None
    if args.mode in ("plan", "all"):
        print("=" * 78)
        print("[1/2] planning ONCE with %s (seed %s, alm=%s, ablation=%s)"
              % (cfg["planner"].get("model_id"), cfg["planner"].get("diffusion_seed"),
                 cfg["planner"].get("alm_enabled"), args.ablation))
        plan = (demo_mod.run_latest_plan(cfg, ctx) if latest
                else demo_mod.run_plan(cfg, ctx))
        gate = ctx.latest_acceptance or check_acceptance(
            plan,
            wheelbase_m=float(cfg["controller"]["wheelbase_m"]),
            max_steer_rad=float(cfg["controller"]["max_steer_rad"]),
            require_guided=bool(cfg["planner"].get("require_guided", True)),
            min_obstacle_body_gap_m=plan.quality.get("min_obstacle_body_gap_m"),
            body_free_rate=plan.quality.get("body_free_rate"))
        manifest["acceptance"] = gate
        print("[plan] guided=%s alm=%s selected=%d %.0f ms length=%.2f m"
              % (plan.guided, plan.alm_status, plan.selected_index, plan.planning_ms,
                 plan.length_m()))
        for check in gate["checks"]:
            print("       %-28s %-8s value=%s  limit %s"
                  % (check["name"], "OK" if check["ok"] else "FAIL",
                     check["value"], check["limit"]))
        payload = demo_mod.save_plan(plan, ctx, cfg, npz_path, json_path)
        manifest["plan"] = {"npz": npz_path, "json": json_path,
                            "planning_ms": plan.planning_ms,
                            "validation": plan.validation, "quality": plan.quality}
        print("[plan] saved %s" % npz_path)
        if latest:
            guided_path = os.path.join(out_dir, "plan_%s.npz" % demo_mod.SAMPLE["key"])
            raw_path = os.path.join(out_dir, "plan_%s_raw.npz" % demo_mod.SAMPLE["key"])
            if os.path.exists(guided_path) and os.path.exists(raw_path):
                from src.carla_bridge import occupancy as occ_mod
                from src.carla_bridge.visualize import plot_trajectories_together

                guided = demo_mod.load_plan(guided_path)
                raw = demo_mod.load_plan(raw_path)
                occupancy = np.asarray(np.load(guided_path)["occupancy"], dtype=np.uint8)
                comparison_path = os.path.join(
                    out_dir, "trajectory_guided_vs_raw_16step.png")
                plot_trajectories_together(
                    occupancy,
                    [{"name": "GUIDED (ALM on, 16 steps)", "colour": "magenta",
                      "curve_world": guided.curve_world,
                      "planned_free": occ_mod.free_mask(occupancy,
                                                         guided.curve_scene),
                      "start_scene": guided.condition_scene[0],
                      "goal_scene": guided.condition_scene[1]},
                     {"name": "RAW (ALM off, 16 steps)", "colour": "darkorange",
                      "curve_world": raw.curve_world,
                      "planned_free": occ_mod.free_mask(occupancy, raw.curve_scene),
                      "start_scene": raw.condition_scene[0],
                      "goal_scene": raw.condition_scene[1]}],
                    comparison_path, ctx.model_frame, obstacles=ctx.obstacles,
                    title="test_0056: latest model guided vs raw (full 16-step)")
                manifest["trajectory_comparison_png"] = comparison_path
                print("[plot] guided vs raw -> %s" % comparison_path)
        if not gate["ok"]:
            print("[plan] ACCEPTANCE FAILED (%s) -- refusing to drive"
                  % ", ".join(gate["failed"]))
            with open(os.path.join(
                    out_dir, "run_manifest%s.json"
                    % ("" if args.ablation == "none" else "_" + args.ablation)),
                    "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
            return 2
    else:
        if not os.path.exists(npz_path):
            raise SystemExit("no frozen plan at %s; run --mode plan first" % npz_path)
        print("[plan] reusing frozen plan %s (no re-planning)" % npz_path)

    if args.mode in ("drive", "all"):
        print("=" * 78)
        print("[2/2] driving the frozen plan in CARLA")
        if plan is None:
            plan = demo_mod.load_plan(npz_path)
            if latest:
                from src.carla_bridge.rolling_frame import WorldSceneFrame

                saved = np.load(npz_path)
                ctx.occupancy = np.asarray(saved["occupancy"], dtype=np.uint8)
                center = (np.asarray(plan.start_world, dtype=np.float64)
                          - np.asarray(plan.condition_scene[0], dtype=np.float64) * 80.0)
                ctx.model_frame = WorldSceneFrame(center)
        manifest["drive"] = demo_mod.run_drive(cfg, ctx, plan, out_dir,
                                               record_video=not args.no_video)
        drive = manifest["drive"]
        print("[drive] status=%s frames=%d sim=%.2fs final_dist=%.3f m "
              "cross_rms=%s collisions=%d"
              % (drive["status"], drive["frames"], drive["sim_seconds"],
                 drive["final_distance_to_goal_m"],
                 drive["cross_track"]["rms_m"], len(drive["collisions"])))

    manifest_path = os.path.join(
        out_dir, "run_manifest%s.json" % ("" if args.ablation == "none" else "_" + args.ablation))
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print("[done] manifest -> %s" % manifest_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
