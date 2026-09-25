#!/usr/bin/env bash
# ch4_run_all.sh —— 第4章正式实验的无人值守编排（选参 -> 主表 -> 消融 -> 表 -> 图）
#
#   bash scripts/ch4_run_all.sh [D-C48 的 config] [D-C48 的 ckpt]
#
# 每一步都可断点续跑：eval_chapter4.py 的产物存在就跳过（ch4_campaign.py 负责），
# 选参结果存在也会重跑（选参是幂等的，但耗时，所以默认只跑一次，见下）。
set -uo pipefail

PY="E:/CondaEnvData/envs/GGMPC/python.exe"
D_CFG="${1:-configs/config_160k4p_c48_d_std.yaml}"
D_CKPT="${2:-outputs/oneshot_k4p_c48_d/ckpt/best_task.pt}"
A_CFG="configs/config_160k4p_c48_oneshot.yaml"
A_CKPT="outputs/oneshot_k4p_c48/ckpt/best_task.pt"
S_CFG="configs/config_160k4p_c48_s_nofb.yaml"
S_CKPT="outputs/oneshot_k4p_c48_s/ckpt/best_task.pt"
VAL_SAMPLES="${VAL_SAMPLES:-256}"
LOGD=outputs/logs
mkdir -p "$LOGD" outputs/ch4/runs

step () { echo; echo "############ $*  $(date '+%F %T')"; }

step "0/6 前置检查"
for f in "$D_CFG" "$D_CKPT" "$A_CFG" "$A_CKPT"; do
  if [ ! -f "$f" ]; then echo "### MISSING $f -- abort"; exit 1; fi
done
[ -f "$S_CKPT" ] && echo "### S-C48 present: $S_CKPT" || echo "### S-C48 MISSING（T 表会跳过）"

step "1/6 验证集选参（四个外部处理器，网格来自指南 4.1.2）"
for mode in penalty projection primal_dual alm_sdf; do
  if [ -f "outputs/ch4/sweep_${mode}.json" ]; then
    echo "### sweep $mode already done, skip"
  else
    "$PY" scripts/ch4_sweep_val.py --config "$D_CFG" --ckpt "$D_CKPT" \
      --guidance "$mode" --val-samples "$VAL_SAMPLES"
  fi
done

step "2/6 冻结参数"
"$PY" - <<'EOF'
import json, glob
for path in sorted(glob.glob('outputs/ch4/sweep_*.json')):
    d = json.load(open(path, encoding='utf-8'))
    print('  %-14s -> %s' % (d['guidance'], json.dumps(d['winner'])))
EOF

step "3/6 4.2 主表：6 方法 × seed 0/1/2 × test 420"
"$PY" scripts/ch4_campaign.py --stage main --seeds 0 1 2 \
  --d-ckpt "$D_CKPT" --d-config "$D_CFG" \
  --a-ckpt "$A_CKPT" --a-config "$A_CFG"

step "4/6 4.3.1 推理期消融 A0..A3"
"$PY" scripts/ch4_campaign.py --stage ablation_a --seeds 0 \
  --a-ckpt "$A_CKPT" --a-config "$A_CFG"

step "5/6 4.3.2 训练期消融 T0..T3"
if [ -f "$S_CKPT" ]; then
  "$PY" scripts/ch4_campaign.py --stage ablation_t --seeds 0 \
    --d-ckpt "$D_CKPT" --d-config "$D_CFG" \
    --a-ckpt "$A_CKPT" --a-config "$A_CFG" \
    --s-ckpt "$S_CKPT" --s-config "$S_CFG"
else
  echo "### skipped (no S-C48 checkpoint)"
fi

step "6/6 汇总表 + 图"
"$PY" scripts/ch4_report.py --runs-dir outputs/ch4/runs \
  --out outputs/ch4/REPORT.md
mkdir -p outputs/ch4/figures
# NOTE: method names contain spaces ("Full (A-C48+ALM+FB+Proj)__s0.json"), so the
# run list MUST stay an array: an unquoted $(ls ...) word-splits it and the figure
# is silently never written (and a trailing `|| true` would hide that).
mapfile -t RUNS < <(ls outputs/ch4/runs/*__s0.json 2>/dev/null | sort)
echo "### plotting ${#RUNS[@]} run(s)"
"$PY" scripts/ch4_plot.py curves \
  --runs "${RUNS[@]}" --out outputs/ch4/figures/process_curves.png
"$PY" scripts/ch4_plot.py smoothness \
  --runs "${RUNS[@]}" --out outputs/ch4/figures/smoothness.png
"$PY" scripts/ch4_plot.py table \
  --runs "${RUNS[@]}" --out outputs/ch4/figures/main_table.png

echo
echo "############ ALL DONE  $(date '+%F %T')"
