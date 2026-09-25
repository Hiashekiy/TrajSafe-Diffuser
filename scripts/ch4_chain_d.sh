#!/usr/bin/env bash
# ch4_chain_d.sh —— 等 S-C48 训练结束，立刻接力训练 D-C48（100 epoch）。
# 目的是让 GPU 不空转：S-C48 与 D-C48 串行（12 GB 显存放不下两个 batch-8 训练）。
#
# 判定「S-C48 结束」必须用 night_train **实际打印**的收尾标记。注意它打印的是
#   [supervisor] exit code 0
# 而不是 "[supervisor] done"（train.py 自己打印 "[done] epochs=..."，supervisor
# 打印的是 exit code）。第一版哨兵写成了 "[supervisor] done"，结果永远不成立、
# 白白空等 —— 所以这里同时接受两种形式，并要求 best_task.pt 存在。
# 最多等 5 小时，超时就放弃并记录。
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
  if grep -qE "supervisor\] (done|exit code)|^\[done\] epochs=" "$S_LOG" 2>/dev/null      && [ -f "$S_CKPT" ]; then
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
