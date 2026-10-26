#!/usr/bin/env bash
# v4_train_c48.sh —— v4（单地图特征 16x16 / 修正量反馈 / 统一连续安全损失）
# 在 data/processed/160m-k4p-c48 上从零训练的入口，可后台分离执行。
#
#   bash scripts/v4_train_c48.sh [epochs] [max-hours]
#
# ckpt 一律落在 <out-root>/ckpt，日志 <out-root>/train.log + outputs/logs/v4_*.log。
set -uo pipefail

PY="E:/CondaEnvData/envs/GGMPC/python.exe"
CFG="configs/config_160k4p_c48_v4_map16.yaml"
OUT="outputs/models/v4_map16_c48"
EPOCHS="${1:-200}"
MAXH="${2:-11.0}"

cd "$(dirname "$0")/.." || exit 1
LOGD=outputs/logs
mkdir -p "$LOGD" "$OUT"
TAG="$(basename "$OUT")"
LOG="$LOGD/v4_train_${TAG}.log"

{
  echo "=============================================================="
  echo "### v4 train START $(date '+%F %T')"
  echo "### config   $CFG"
  echo "### out      $OUT"
  echo "### epochs   $EPOCHS   max-hours $MAXH"
  echo "=============================================================="
} | tee -a "$LOG"

"$PY" scripts/night_train.py --config "$CFG" \
  --ckpt-dir "$OUT/ckpt" --out-dir "$OUT" \
  --epochs "$EPOCHS" --max-hours "$MAXH" \
  --batch-size 8 --accum 2 --lr 2e-4 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

echo "### v4 train rc=$rc  $(date '+%F %T')" | tee -a "$LOG"
exit "$rc"
