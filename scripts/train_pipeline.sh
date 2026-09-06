#!/usr/bin/env bash
# ============================================================================
# V1 joint trajectory-ellipse diffusion training pipeline
#   Windows (Git Bash): bash scripts/train_pipeline.sh
#   Ubuntu:             bash scripts/train_pipeline.sh
#
#   Configurable env vars (defaults):
#     PY        python executable (auto-detected)
#     CONFIG    config file (configs/config_v1.yaml)
#     EPOCHS    training epochs (default from config = 100)
#
#   The whole V1 training is ONE stage (L = ||x0_P - x0_P_hat||^2 +
#   lambda_e * ||x0_E - x0_E_hat||^2), no phase curriculum.
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

if [[ -z "${PY:-}" ]]; then
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)  PY="E:/CondaEnvData/envs/GGMPC/python.exe" ;;
    *)                      PY="python" ;;
  esac
fi

CONFIG="${CONFIG:-configs/config_v1.yaml}"
EPOCHS="${EPOCHS:-}"

ARGS=(--config "$CONFIG")
if [[ -n "$EPOCHS" ]]; then
  ARGS+=(--epochs "$EPOCHS")
fi

echo "==> train.py ${ARGS[*]}"
"$PY" train.py "${ARGS[@]}"
