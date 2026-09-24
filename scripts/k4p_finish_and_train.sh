#!/usr/bin/env bash
# Finish the k4p build (labels resume -> ALM -> validate), then train 200 epoch.
# Resumable: the per-sample caches make stages 1-2 cheap on a re-run.
set -uo pipefail
PY="E:/CondaEnvData/envs/GGMPC/python.exe"
OUT=data/carla_processed_160k4p
CFG=configs/config_160k8p.yaml
VAL=outputs/logs/k4p_validate.log

echo "### labels (resume, workers=8)  $(date +%H:%M)"
"$PY" scripts/data/carla/02_build_ellipse_labels.py --processed "$OUT" --config "$CFG" --workers 8
rc=$?; echo "### labels rc=$rc  $(date +%H:%M)"; [ $rc -eq 0 ] || exit 1

echo "### ALM constraints (workers=8)  $(date +%H:%M)"
"$PY" scripts/data/carla_full/04_build_alm_constraints.py --processed "$OUT" --config "$CFG" --workers 8
rc=$?; echo "### ALM rc=$rc  $(date +%H:%M)"; [ $rc -eq 0 ] || exit 1

echo "### validate  $(date +%H:%M)"
"$PY" scripts/data/carla/03_validate_processed.py --processed "$OUT" --config "$CFG" 2>&1 | tee "$VAL"
grep -q "VALID = True" "$VAL" || { echo "### ABORT: not VALID=True"; exit 1; }

echo "### restart dashboard backend  $(date +%H:%M)"
( cd diffusion-dashboard && nohup "$PY" backend_carla.py > dashboard_backend.log 2>&1 & )
sleep 15

echo "### train OneShot k4p 200 epoch  $(date +%H:%M)"
"$PY" scripts/night_train.py --config configs/config_160k4p_oneshot.yaml \
  --ckpt-dir outputs/oneshot_k4p/ckpt --out-dir outputs/oneshot_k4p \
  --epochs 200 --max-hours 11.0 --batch-size 8 --accum 2 --lr 2e-4
rc=$?; echo "### train rc=$rc  $(date +%H:%M)"
echo "### FINISHED $(date +%H:%M)"
