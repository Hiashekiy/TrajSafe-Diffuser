#!/usr/bin/env bash
# Two same-protocol evals for the C=48 retrain on its OWN eroded test split
# (data/carla_processed_160k4p_c48).  Protocol matches scripts/eval_k4p_oneshot.sh
# so the numbers are directly comparable with §11.3 (K4P_oneshot: 5/420 with ALM,
# 50/420 without) -- same test 420 / 16 steps / seed 0 / chunk 32.
# Full output goes to stdout -> outputs/logs/eval_k4p_c48_own.log
PY="E:/CondaEnvData/envs/GGMPC/python.exe"
CFG=configs/config_160k4p_c48_oneshot.yaml
CKPT=outputs/oneshot_k4p_c48/ckpt/best_task.pt
M="K4P_c48_oneshot=$CKPT::$CFG"

echo "=== 1/2 k4p_c48 test (own eroded map), ALM on ===  $(date +%H:%M)"
"$PY" scripts/eval_campaign_testset.py --config "$CFG" --split test --samples 0 \
  --chunk 32 --steps 16 --seed 0 --model "$M" \
  --out outputs/eval_k4p_c48_own_alm.json --md outputs/eval_k4p_c48_own_alm.md 2>&1

echo "=== 2/2 k4p_c48 test (own eroded map), ALM off (ablation A) ===  $(date +%H:%M)"
"$PY" scripts/eval_campaign_testset.py --config "$CFG" --split test --samples 0 \
  --chunk 32 --steps 16 --seed 0 --ablation A --model "$M" \
  --out outputs/eval_k4p_c48_own_noalm.json --md outputs/eval_k4p_c48_own_noalm.md 2>&1

echo "ALL DONE  $(date +%H:%M)"
