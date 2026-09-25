#!/usr/bin/env bash
# ch4_chain_d.sh —— 等 S-C48 训练结束，立刻接力训练 D-C48（100 epoch）。
# 目的是让 GPU 不空转：S-C48 与 D-C48 串行（12 GB 显存放不下两个 batch-8 训练）。
#
# 判定「S-C48 结束」用 night_train 的收尾标记 "[supervisor] done"，并额外要求
# best_task.pt 存在；两者都满足才接力。最多等 5 小时，超时就放弃并记录。
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
S_LOG=outputs/oneshot_k4p_c48_s/train.log
S_CKPT=outputs/oneshot_k4p_c48_s/ckpt/best_task.pt
LOGD=outputs/logs
mkdir -p "$LOGD"
LOG="$LOGD/ch4_chain_d.log"

echo "### chain started $(date '+%F %T'), waiting for S-C48 to finish" | tee -a "$LOG"
waited=0
while [ "$waited" -lt 18000 ]; do
  if grep -q "supervisor\] done" "$S_LOG" 2>/dev/null && [ -f "$S_CKPT" ]; then
    echo "### S-C48 finished after ${waited}s  $(date '+%F %T')" | tee -a "$LOG"
    break
  fi
  sleep 60
  waited=$((waited + 60))
done

if [ "$waited" -ge 18000 ]; then
  echo "### TIMEOUT waiting for S-C48, not starting D-C48" | tee -a "$LOG"
  exit 1
fi

sleep 30          # let the S-C48 process release the GPU
echo "### launching D-C48 training  $(date '+%F %T')" | tee -a "$LOG"
bash scripts/ch4_train.sh configs/config_160k4p_c48_d_std.yaml \
  outputs/oneshot_k4p_c48_d 100 6.0 >> "$LOG" 2>&1
echo "### D-C48 training finished rc=$?  $(date '+%F %T')" | tee -a "$LOG"
