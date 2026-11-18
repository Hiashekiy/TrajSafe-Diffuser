"""CLI: python -m src.map_preparation --help."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .pipeline import prepare_occupancy_map
from .export import save_prepared_map


def select_maps(input_path, count, seed, indices=None):
    """Memory-map occupancy batches; sample distinct maps without replacement."""
    paths = (
        sorted(input_path.rglob("occupancy.npy"))
        if input_path.is_dir()
        else [input_path]
    )
    if not paths:
        raise FileNotFoundError(f"No occupancy.npy under {input_path}")
    arrays = []
    for path in paths:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim == 2:
            array = array[None]
        if array.ndim == 4 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim != 3:
            raise ValueError(f"{path}: expected HxW, NxHxW or Nx1xHxW")
        arrays.append((path, array))
    if indices is not None and len(arrays) != 1:
        raise ValueError("--indices requires a single input .npy file")
    total = sum(len(a) for _, a in arrays)
    order = (
        indices
        if indices is not None
        else np.random.default_rng(seed).permutation(total)
    )
    selected = []
    seen = set()
    for flat in order:
        index = int(flat)
        if not 0 <= index < total:
            raise ValueError(f"Index {index} outside [0,{total})")
        for path, array in arrays:
            if index < len(array):
                occ = np.asarray(array[index])
                if not np.isin(occ, [0, 1]).all():
                    raise ValueError(f"{path}[{index}]: occupancy must be 0 or 1")
                digest = hashlib.sha256(occ.tobytes()).hexdigest()
                if indices is not None or digest not in seen:
                    selected.append((path, index, occ.copy(), digest))
                    seen.add(digest)
                break
            index -= len(array)
        if indices is None and len(selected) == count:
            break
    if indices is None and len(selected) < count:
        raise ValueError(f"Only {len(selected)} unique maps available; lower --count")
    return selected


def main():
    ap = argparse.ArgumentParser(
        description="Offline map preparation: binary occupancy -> DMA -> sknw curved multigraph."
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed"),
        help="occupancy .npy file or directory searched for occupancy.npy",
    )
    ap.add_argument("--output", type=Path, default=Path("outputs/map_preparation"))
    ap.add_argument(
        "--count", type=int, default=16, help="number of unique random maps"
    )
    ap.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help="explicit indices in a single input file (overrides count)",
    )
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--delta-px", type=float, default=3.0)
    ap.add_argument(
        "--resolution",
        type=float,
        default=0.625,
        help="meters per pixel; set for your dataset",
    )
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="allow writing into a nonempty output directory",
    )
    args = ap.parse_args()
    if args.count < 1:
        ap.error("--count must be positive")
    if not np.isfinite(args.delta_px) or args.delta_px <= 0:
        ap.error("--delta-px must be positive")
    if not np.isfinite(args.resolution) or args.resolution <= 0:
        ap.error("--resolution must be positive")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        ap.error("Output is nonempty; choose another --output or pass --overwrite")
    maps = select_maps(args.input, args.count, args.seed, args.indices)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    cases = []
    for serial, (path, index, occ, digest) in enumerate(maps):
        result = prepare_occupancy_map(occ, args.delta_px, args.resolution)
        source = dict(file=str(path.resolve()), index=index, sha256=digest)
        result.graph["source"] = source
        prefix = args.output / f"map_{serial:04d}"
        save_prepared_map(result, prefix)
        if not args.no_plots:
            from .visualization import save_map_preview

            save_map_preview(result, prefix)
            cases.append((f"{path.parent.name}[{index}]", result))
        a = result.graph["audit"]
        record = dict(
            output=prefix.name,
            **source,
            delta_px=args.delta_px,
            nodes=a["nodes"],
            edges=a["edges"],
            components=a["components"],
            digital_holes=a["thin_topology"]["holes"],
            cycle_rank=a["cycle_rank"],
            warnings="; ".join(a["warnings"]),
        )
        records.append(record)
        print(
            f'{serial+1}/{len(maps)} {path.parent.name}[{index}]: V={a["nodes"]} E={a["edges"]} C={a["components"]}',
            flush=True,
        )
        for warning in a["warnings"]:
            print("  WARNING:", warning, flush=True)
    if cases:
        from .visualization import save_summary

        save_summary(cases, args.output / "summary.png")
    with (args.output / "statistics.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(seed=args.seed, resolution_m_per_pixel=args.resolution, maps=records),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf8",
    )
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
