#!/usr/bin/env bash
# k4p_c48_train.sh - 48-control B-spline on the k=4 border-protected cache.
#
# Same recipe as the C=32 run (scripts/k4p_finish_and_train.sh, outputs/oneshot_k4p):
#   data  : data/carla_processed_160k4p  (k=4, --border-mode protect)
#   recipe: 200 epoch, batch 8 x accum 2, lr 2e-4, feedback rollout on
#   only  : C = 32 -> 48  (model.num_controls + bspline.num_controls)
#
# Stage 0 re-fits the OFFLINE control labels for C=48 into a NEW root
# (everything else is copied verbatim, so the two runs differ in C only).
# Stages 1-2 are cheap / idempotent, so this script is safe to re-run; the
# supervisor restarts training from latest.pt after a crash.
set -uo pipefail
PY="E:/CondaEnvData/envs/GGMPC/python.exe"
SRC=data/carla_processed_160k4p
OUT=data/carla_processed_160k4p_c48
CFG=configs/config_160k4p_c48_oneshot.yaml
LOGD=outputs/logs
mkdir -p "$LOGD"

if [ ! -f "$OUT/train/control_gt.npy" ]; then
  echo "### refit control labels C=48  $(date +%H:%M)"
  "$PY" scripts/data/carla_full/06_refit_controls.py \
    --source "$SRC" --out "$OUT" --config "$CFG"
  rc=$?; echo "### refit rc=$rc  $(date +%H:%M)"; [ $rc -eq 0 ] || exit 1
fi

echo "### validate  $(date +%H:%M)"
"$PY" scripts/data/carla/03_validate_processed.py --processed "$OUT" --config "$CFG" \
  2>&1 | tee "$LOGD/k4p_c48_validate.log"
grep -q "VALID = True" "$LOGD/k4p_c48_validate.log" || {
  echo "### ABORT: not VALID=True"; exit 1; }

echo "### train C=48 on 160k4p, 200 epoch  $(date +%H:%M)"
"$PY" scripts/night_train.py --config "$CFG" \
  --ckpt-dir outputs/oneshot_k4p_c48/ckpt --out-dir outputs/oneshot_k4p_c48 \
  --epochs 200 --max-hours 11.0 --batch-size 8 --accum 2 --lr 2e-4
rc=$?; echo "### train rc=$rc  $(date +%H:%M)"
echo "### FINISHED $(date +%H:%M)"
