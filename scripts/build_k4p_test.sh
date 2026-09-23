#!/usr/bin/env bash
# Build a p4 (k=4) border-protected cache, TEST SPLIT ONLY, into data/carla_processed_160k4p.
set -euo pipefail
PY="E:/CondaEnvData/envs/GGMPC/python.exe"
OUT=data/carla_processed_160k4p
CFG=configs/config_160k8p.yaml
SPLITS=test

if [ ! -d "$OUT" ]; then
  echo "=== [1/5] erode k=4 protect (test) ==="
  "$PY" scripts/data/carla_full/05_erode_occupancy.py \
    --source data/carla_processed_160 --out "$OUT" --k 4 --border-mode protect --splits "$SPLITS"
else
  echo "=== [1/5] skip erode (already present) ==="
fi

echo "=== [2/5] candidates ==="
"$PY" scripts/data/carla/01_build_candidates.py --processed "$OUT" --config "$CFG" --splits "$SPLITS"

echo "=== [3/5] ellipse labels ==="
"$PY" scripts/data/carla/02_build_ellipse_labels.py --processed "$OUT" --config "$CFG" --splits "$SPLITS"

echo "=== [4/5] alm constraints ==="
"$PY" scripts/data/carla_full/04_build_alm_constraints.py --processed "$OUT" --config "$CFG" --splits "$SPLITS"

echo "=== [5/5] validate ==="
"$PY" scripts/data/carla/03_validate_processed.py --processed "$OUT" --config "$CFG" --splits "$SPLITS"

echo "=== DONE k4p test ==="
