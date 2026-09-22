#!/usr/bin/env bash
# Train TrajSafe-Diffuser on the carla_full_160_256 processed cache.
# Checkpoints go to outputs/bspline_carla_160/ckpt (NEVER the 80 m run dir).
set -uo pipefail
cd /d/ProjectDirectory/Neural-IRISDiffuser || exit 1

PY="E:/CondaEnvData/envs/GGMPC/python.exe"
CFG="${CFG:-configs/config_160.yaml}"
EXTRA="${EXTRA:-}"

"$PY" -u train.py --config "$CFG" $EXTRA
