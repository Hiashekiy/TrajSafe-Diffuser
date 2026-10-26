#!/usr/bin/env bash
# ch4_train.sh —— 第4章实验的统一训练入口（可复用、可后台分离执行）
#
#   bash scripts/ch4_train.sh <config> <out-root> <epochs> <max-hours>
#
# 例：
#   bash scripts/ch4_train.sh configs/config_160k4p_c48_s_nofb.yaml outputs/models/oneshot_k4p_c48_s 100 6.0
#   bash scripts/ch4_train.sh configs/config_160k4p_c48_d_std.yaml  outputs/models/oneshot_k4p_c48_d 100 6.0
#
# 与 scripts/k4p_c48_train.sh 相同的 recipe（batch 8 x accum 2, lr 2e-4），
# 只把 epoch 预算与数据/配置参数化。ckpt 一律落在 <out-root>/ckpt，
# 日志同时写 <out-root>/train.log（night_train 负责）与 outputs/logs/ch4_*.log。
set -uo pipefail

PY="E:/CondaEnvData/envs/GGMPC/python.exe"
CFG="${1:?usage: ch4_train.sh <config> <out-root> <epochs> <max-hours>}"
OUT="${2:?}"
EPOCHS="${3:-100}"
MAXH="${4:-6.0}"

cd "$(dirname "$0")/.." || exit 1
LOGD=outputs/logs
mkdir -p "$LOGD"
TAG="$(basename "$OUT")"
LOG="$LOGD/ch4_train_${TAG}.log"

{
  echo "=============================================================="
  echo "### ch4 train START $(date '+%F %T')"
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

echo "### ch4 train rc=$rc  $(date '+%F %T')" | tee -a "$LOG"
exit "$rc"
