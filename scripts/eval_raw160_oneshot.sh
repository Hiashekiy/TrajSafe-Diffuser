#!/usr/bin/env bash
# Four same-protocol evals for the pre-erosion retrain (RAW160_oneshot).
# Full output goes to stdout -> outputs/logs/eval_raw160_oneshot.log
PY="E:/CondaEnvData/envs/GGMPC/python.exe"
CFG=configs/config_160raw_oneshot.yaml
CKPT=outputs/oneshot_raw160/ckpt/best_task.pt
M="RAW160_oneshot=$CKPT::$CFG"

echo "=== 1/4 raw160 test, ALM on ==="
"$PY" scripts/eval_campaign_testset.py --config "$CFG" --split test --samples 0 \
  --chunk 32 --steps 16 --seed 0 --model "$M" \
  --out outputs/eval_raw160_oneshot_alm.json --md outputs/eval_raw160_oneshot_alm.md 2>&1

echo "=== 2/4 raw160 test, ALM off (ablation A) ==="
"$PY" scripts/eval_campaign_testset.py --config "$CFG" --split test --samples 0 \
  --chunk 32 --steps 16 --seed 0 --ablation A --model "$M" \
  --out outputs/eval_raw160_oneshot_noalm.json --md outputs/eval_raw160_oneshot_noalm.md 2>&1

echo "=== 3/4 k8p test, ALM on (cross-cache) ==="
"$PY" scripts/eval_campaign_testset.py --config "$CFG" --split test --samples 0 \
  --chunk 32 --steps 16 --seed 0 --processed data/carla_processed_160k8p --model "$M" \
  --out outputs/eval_raw160_on_k8p_alm.json --md outputs/eval_raw160_on_k8p_alm.md 2>&1

echo "=== 4/4 k8p test, ALM off (cross-cache) ==="
"$PY" scripts/eval_campaign_testset.py --config "$CFG" --split test --samples 0 \
  --chunk 32 --steps 16 --seed 0 --ablation A --processed data/carla_processed_160k8p --model "$M" \
  --out outputs/eval_raw160_on_k8p_noalm.json --md outputs/eval_raw160_on_k8p_noalm.md 2>&1

echo "ALL DONE"
