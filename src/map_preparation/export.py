"""Portable JSON metadata and pickle-free NumPy graph arrays."""

import json
from pathlib import Path
import numpy as np
from .skeleton_graph import save_arrays


def save_prepared_map(result, prefix):
    """Write PREFIX.json and PREFIX.npz, preserving both skeleton stages."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(result.graph, indent=2, ensure_ascii=False), encoding="utf8"
    )
    save_arrays(
        prefix.with_suffix(".npz"),
        result.graph,
        result.skeleton,
        result.thin_skeleton,
        result.occupancy,
        result.distance_px,
        result.nearest_background_rc,
    )
