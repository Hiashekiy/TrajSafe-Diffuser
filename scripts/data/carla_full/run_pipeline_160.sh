#!/usr/bin/env bash
# Full data-preparation pipeline for the carla_full_160_256 dataset.
#   00  raw scene/task  -> processed cache contract
#   01  skeleton candidates + topology_best
#   02  fixed-centre safe-ellipse labels
#   03  contract validation
# Each stage is idempotent / resumable; 01 and 02 cache per sample under
# <split>/_cache/ so a re-run skips finished work.
set -uo pipefail
cd /d/ProjectDirectory/Neural-IRISDiffuser || exit 1

PY="E:/CondaEnvData/envs/GGMPC/python.exe"
CFG="configs/config_160.yaml"
OUT="data/carla_processed_160"
WORKERS="${WORKERS:-12}"

stage () {
  echo ""
  echo "##################################################################"
  echo "### $1   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "##################################################################"
}

stage "00_build_processed"
"$PY" -u scripts/data/carla_full/00_build_processed.py --config "$CFG" --out "$OUT" || exit 10

stage "01_build_candidates"
"$PY" -u scripts/data/carla/01_build_candidates.py --processed "$OUT" --config "$CFG" --workers "$WORKERS" || exit 11

stage "02_build_ellipse_labels"
"$PY" -u scripts/data/carla/02_build_ellipse_labels.py --processed "$OUT" --config "$CFG" --workers "$WORKERS" || exit 12

stage "03_validate_processed"
"$PY" -u scripts/data/carla/03_validate_processed.py --processed "$OUT" --config "$CFG" || exit 13

stage "PIPELINE DONE"
echo "all stages finished successfully"
