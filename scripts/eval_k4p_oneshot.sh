#!/usr/bin/env bash
# Two same-protocol evals for the k4p-eroded retrain (K4P_oneshot) on its OWN eroded test split.
# Protocol matches scripts/eval_raw160_oneshot.sh: test 420 / 16 steps / seed 0 / chunk 32.
# Full output goes to stdout -> outputs/logs/eval_k4p_own.log
PY="E:/CondaEnvData/envs/GGMPC/python.exe"
CFG=configs/config_160k4p_oneshot.yaml
CKPT=outputs/oneshot_k4p/ckpt/best_task.pt
M="K4P_oneshot=$CKPT::$CFG"

echo "=== 1/2 K4P test (own eroded map), ALM on ===  $(date +%H:%M)"
"$PY" scripts/eval_campaign_testset.py --config "$CFG" --split test --samples 0 \
  --chunk 32 --steps 16 --seed 0 --model "$M" \
  --out outputs/eval_k4p_own_alm.json --md outputs/eval_k4p_own_alm.md 2>&1

echo "=== 2/2 K4P test (own eroded map), ALM off (ablation A) ===  $(date +%H:%M)"
"$PY" scripts/eval_campaign_testset.py --config "$CFG" --split test --samples 0 \
  --chunk 32 --steps 16 --seed 0 --ablation A --model "$M" \
  --out outputs/eval_k4p_own_noalm.json --md outputs/eval_k4p_own_noalm.md 2>&1

echo "ALL DONE  $(date +%H:%M)"
