"""Run long-distance CARLA planning with asynchronous four-step denoising."""

from __future__ import annotations

import argparse
import os
import sys
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.carla_bridge.continuous_demo import run_continuous, save_manifest
from src.carla_bridge.demo import load_yaml


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.path.join(
        ROOT, "configs", "carla_continuous.yaml"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--start-spawn", type=int, default=None)
    parser.add_argument("--goal-spawn", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = load_yaml(args.config)
    if args.start_spawn is not None:
        cfg["route"]["start_spawn_index"] = args.start_spawn
    if args.goal_spawn is not None:
        cfg["route"]["goal_spawn_index"] = args.goal_spawn
    out = os.path.abspath(args.out or cfg["output"]["dir"])
    manifest = run_continuous(cfg, out)
    path = os.path.join(out, "continuous_manifest.json")
    save_manifest(manifest, path)
    print("[done] status=%s route=%.1f/%.1f m replans=%d manifest=%s"
          % (manifest["status"], manifest.get("route_progress_m", 0.0),
             manifest.get("route", {}).get("length_m", 0.0),
             manifest.get("replans_activated", 0), path))
    return 0 if manifest["status"] == "arrived" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

