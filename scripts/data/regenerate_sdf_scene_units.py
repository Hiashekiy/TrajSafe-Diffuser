"""Regenerate scene SDF maps in *scene units* for all three mazes.

The stored data/processed_scene/maps/*_sdf.npy were previously produced by
distance_transform_edt, which returns per-pixel (grid-cell) distances, not scene
units.  This script rebuilds them with src.geometry.scene_frame.build_scene_sdf
(now scene-unit) so the SDF is consistent with the scene-coordinate occupancy
map.  It is idempotent: it recomputes from the binary occupancy each time.

Usage:
    python scripts/data/regenerate_sdf_scene_units.py
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.insert(0, ROOT)

from src.geometry.scene_frame import build_scene_sdf

MAPS_DIR = os.path.join(ROOT, "data", "processed_scene", "maps")
MAZES = ["umaze", "medium", "large"]

for m in MAZES:
    occ_path = os.path.join(MAPS_DIR, f"{m}.npy")
    out_path = os.path.join(MAPS_DIR, f"{m}_sdf.npy")
    occ = np.load(occ_path).astype(np.float32)
    sdf = build_scene_sdf(occ)          # scene units
    np.save(out_path, sdf)
    print(f"regenerated {m}_sdf.npy: range [{sdf.min():.3f}, {sdf.max():.3f}] (scene units)")
print("done")
