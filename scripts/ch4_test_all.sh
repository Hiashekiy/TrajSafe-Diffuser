#!/usr/bin/env bash
# ch4_test_all.sh —— 第4章报告的【纯测试】一键全流程（不含任何训练）
#
#   bash scripts/ch4_test_all.sh          # 前台
#   scripts\ch4_test_all.cmd             # 分离式后台（关窗口不死）
#
# 只用现成 ckpt，每一步都幂等（产物存在即跳过），可反复重跑 / 断点续跑：
#
#   1 单元测试
#   2 验证集选参（4 个外部处理器）          -> outputs/chapter4/sweep_*.json
#   3 主表 6 方法 x seed 0/1/2 x test 420  -> outputs/chapter4/runs/
#   4 消融 A0-A3（推理期）                 -> outputs/chapter4/runs_abl/
#   5 消融 T0-T3（训练期）                 -> outputs/chapter4/runs_abl/
#   6 汇总表 + 图 + 可选案例图             -> outputs/chapter4/REPORT.md, figures/
#   7 效率套件（计时 / 走廊加速 A-B / 区域共享扫描）
#
# 环境变量：
#   FORCE=1       第 2-5 步忽略已有产物全部重跑
#   SKIP_SWEEP=1  跳过第 2 步（复用已有 sweep_*.json）
#   SKIP_EFF=1    跳过第 7 步（效率套件必须在空闲机器上跑）
#   PLOT_CASES=1  第 6 步额外出 5 张中间轨迹案例图
#   VAL_SAMPLES=N 第 2 步验证集样本数（默认 256）
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

# --- 解释器定位 -----------------------------------------------------------
# 不能只判 `-x`:在 cmd 里敲 `bash xxx.sh` 时,若 PATH 上的 bash 不是 Git Bash
# (常见的是 C:\Windows\System32\bash.exe = WSL),`E:/...` 这种 MSYS 风格路径
# 根本不存在,`-x` 会假失败。这里改成"逐个候选真正跑一次 import"。
PY_DEFAULT="E:/CondaEnvData/envs/GGMPC/python.exe"
pick_python () {
  local cand
  for cand in "${PY_OVERRIDE:-}" "$PY_DEFAULT" \
              "/e/CondaEnvData/envs/GGMPC/python.exe" \
              "/mnt/e/CondaEnvData/envs/GGMPC/python.exe" \
              "$(command -v python 2>/dev/null)" \
              "$(command -v python3 2>/dev/null)"; do
    [ -n "$cand" ] || continue
    if "$cand" -c "import sys" >/dev/null 2>&1; then printf '%s' "$cand"; return 0; fi
  done
  return 1
}
if ! PY="$(pick_python)"; then
  echo "### 找不到可用的 python。试过的路径:$PY_DEFAULT 及其 /e/ 与 /mnt/e/ 形式、PATH 上的 python。"
  echo "### 请显式指定,例如:PY_OVERRIDE=/e/CondaEnvData/envs/GGMPC/python.exe bash scripts/ch4_test_all.sh"
  exit 2
fi
echo "### bash      $(command -v bash) ($BASH_VERSION)"
if ! "$PY" -c "import torch, sys; print('### python   ', sys.executable); print('### torch    ', torch.__version__, '| cuda', torch.cuda.is_available())"; then
  echo "### $PY 不能 import torch —— 解释器找错了,请用 E:/CondaEnvData/envs/GGMPC/python.exe -- abort"
  exit 2
fi

LOGD=outputs/logs
mkdir -p "$LOGD" outputs/chapter4/runs outputs/chapter4/runs_abl outputs/chapter4/figures
LOG="$LOGD/ch4_test_all.log"

D_CFG=configs/config_160k4p_c48_d_std.yaml
D_CKPT=outputs/models/oneshot_k4p_c48_d/ckpt/best_task.pt
A_CFG=configs/config_160k4p_c48_oneshot.yaml
A_CKPT=outputs/models/oneshot_k4p_c48/ckpt/best_task.pt
S_CFG=configs/config_160k4p_c48_s_nofb.yaml
S_CKPT=outputs/models/oneshot_k4p_c48_s/ckpt/best_task.pt
VAL_SAMPLES="${VAL_SAMPLES:-256}"

FORCE_ARG=()
if [ "${FORCE:-0}" = 1 ]; then FORCE_ARG=(--force); fi

FAILED=()
STARTED=$(date +%s)

# 同时进终端与总日志
exec > >(tee -a "$LOG") 2>&1

step () { echo; echo "############ $*  $(date '+%F %T')"; }

run () {                       # run <tag> <cmd...>
  local tag="$1"; shift
  echo "---- $tag"
  local t0=$(date +%s)
  if "$@"; then
    echo "---- $tag OK ($(( $(date +%s) - t0 ))s)"
  else
    echo "---- $tag FAILED (rc=$?) ($(( $(date +%s) - t0 ))s)"
    FAILED+=("$tag")
  fi
}

echo "=============================================================="
echo "### ch4 test-only pipeline START $(date '+%F %T')"
echo "### root      $ROOT"
echo "### force=${FORCE:-0}  skip_sweep=${SKIP_SWEEP:-0}  skip_eff=${SKIP_EFF:-0}  val_samples=$VAL_SAMPLES"
echo "### log       $LOG"
echo "=============================================================="

# ---------------------------------------------------------------- 0 前置检查
step "0/7 前置检查（本脚本不含训练，训练产物必须已存在）"
MISS=0
for f in "$D_CFG" "$D_CKPT" "$A_CFG" "$A_CKPT" "$S_CFG" "$S_CKPT" \
         data/processed/160m-k4p-c48/test/occupancy.npy \
         data/processed/160m-k4p-c48/val/occupancy.npy; do
  if [ -f "$f" ]; then echo "  OK   $f"; else echo "  MISS $f"; MISS=1; fi
done
if [ "$MISS" != 0 ]; then
  echo "### 缺少前置产物；本脚本刻意不含训练步骤，请先补训练 -- abort"
  exit 2
fi

if [ "${DRYRUN:-0}" = 1 ]; then
  echo
  echo "### DRYRUN=1:解释器与前置产物检查通过,未运行任何评测 -- exit 0"
  exit 0
fi

# ---------------------------------------------------------------- 1 单元测试
step "1/7 单元测试（全套；-p no:cacheprovider 避免 .pytest_cache 权限告警）"
run pytest "$PY" -m pytest -q -p no:cacheprovider

# ---------------------------------------------------------------- 2 选参
if [ "${SKIP_SWEEP:-0}" = 1 ]; then
  step "2/7 验证集选参（SKIP_SWEEP=1，跳过）"
else
  step "2/7 验证集选参（网格来自指南 4.1.2，内迭代预算固定为 10）"
  for m in penalty projection primal_dual alm_sdf; do
    if [ -f "outputs/chapter4/sweep_$m.json" ] && [ "${FORCE:-0}" != 1 ]; then
      echo "  sweep $m 已存在，跳过"
    else
      run "sweep_$m" "$PY" scripts/ch4_sweep_val.py \
        --config "$D_CFG" --ckpt "$D_CKPT" \
        --guidance "$m" --val-samples "$VAL_SAMPLES"
    fi
  done
fi
echo "---- 冻结参数"
"$PY" -c "
import glob, json
for p in sorted(glob.glob('outputs/chapter4/sweep_*.json')):
    d = json.load(open(p, encoding='utf-8'))
    print('  %-12s -> %s' % (d['guidance'], json.dumps(d['winner'])))
" || FAILED+=("print_sweep_winners")

# ---------------------------------------------------------------- 3 主表
step "3/7 主表：6 方法 x seed 0/1/2 x test 420"
run main "$PY" scripts/ch4_campaign.py --stage main --seeds 0 1 2 \
  ${FORCE_ARG[@]+"${FORCE_ARG[@]}"} \
  --d-ckpt "$D_CKPT" --d-config "$D_CFG" \
  --a-ckpt "$A_CKPT" --a-config "$A_CFG"

# ---------------------------------------------------------------- 4 消融 A
step "4/7 推理期消融 A0-A3（同一 A-C48，只切推理开关）"
run ablation_a "$PY" scripts/ch4_campaign.py --stage ablation_a --seeds 0 \
  --out-dir outputs/chapter4/runs_abl ${FORCE_ARG[@]+"${FORCE_ARG[@]}"} \
  --a-ckpt "$A_CKPT" --a-config "$A_CFG"

# ---------------------------------------------------------------- 5 消融 T
step "5/7 训练期消融 T0-T3（S-C48 与 A-C48，全部关闭末端投影）"
run ablation_t "$PY" scripts/ch4_campaign.py --stage ablation_t --seeds 0 \
  --out-dir outputs/chapter4/runs_abl ${FORCE_ARG[@]+"${FORCE_ARG[@]}"} \
  --d-ckpt "$D_CKPT" --d-config "$D_CFG" \
  --a-ckpt "$A_CKPT" --a-config "$A_CFG" \
  --s-ckpt "$S_CKPT" --s-config "$S_CFG"

# ---------------------------------------------------------------- 6 表 + 图
step "6/7 汇总表 + 图"
run report "$PY" scripts/ch4_report.py \
  --runs-dir outputs/chapter4/runs outputs/chapter4/runs_abl --out outputs/chapter4/REPORT.md

# 方法名里有空格（"Full (A-C48+ALM+FB+Proj)__s0.json"），运行列表必须保持为数组：
# 未加引号的 $(ls ...) 会被词分割，图会静默不生成。
mapfile -t RUNS < <(ls outputs/chapter4/runs/*__s0.json 2>/dev/null | sort)
if [ "${#RUNS[@]}" = 0 ]; then
  echo "  ### 没有找到主表运行，跳过绘图"; FAILED+=("figures(no runs)")
else
  echo "  ### plotting ${#RUNS[@]} run(s)"
  run fig_curves     "$PY" scripts/ch4_plot.py curves     --runs "${RUNS[@]}" --out outputs/chapter4/figures/process_curves.png
  run fig_smoothness "$PY" scripts/ch4_plot.py smoothness --runs "${RUNS[@]}" --out outputs/chapter4/figures/smoothness.png
  run fig_table      "$PY" scripts/ch4_plot.py table      --runs "${RUNS[@]}" --out outputs/chapter4/figures/main_table.png
fi

if [ "${PLOT_CASES:-0}" = 1 ]; then
  echo "---- 可选：§5.2 中间轨迹案例图（失败不影响结论）"
  "$PY" scripts/ch4_plot.py cases --ckpt "$A_CKPT" --guidance alm \
    --samples 0,1,2,63,293 --exact --campaign-samples 0 \
    --feedback on --final-project on \
    --out-dir outputs/chapter4/figures/cases || echo "  ### cases(alm) 失败，忽略"
  "$PY" scripts/ch4_plot.py cases --ckpt "$D_CKPT" --guidance none \
    --samples 0,1,2 --out-dir outputs/chapter4/figures/cases || echo "  ### cases(none) 失败，忽略"
fi

# ---------------------------------------------------------------- 7 效率套件
if [ "${SKIP_EFF:-0}" = 1 ]; then
  step "7/7 效率套件（SKIP_EFF=1，跳过）"
else
  step "7/7 效率套件（必须在空闲机器上跑）"
  LOAD=$("$PY" -c "
try:
    import psutil; print('%.0f' % psutil.cpu_percent(2))
except Exception:
    print('?')
")
  echo "  ### 当前 CPU 负载 ${LOAD}%（>25% 时计时列不可引用，只当相对比较）"

  # 7A+7B: 分方法计时(128 条,16 步) + 步数x投影矩阵(420 条)
  run eff_timing bash -c "'$PY' -u scripts/ch4_perf_timing.py 2>&1 | tee outputs/logs/ch4_perf_report.log"
  # 7C: 走廊向量化 A/B（同一进程背靠背，比值不受负载影响）
  run eff_corridor bash -c "'$PY' -u scripts/ch4_perf_corridor_ab.py 32 2>&1 | tee outputs/logs/ch4_ab_batch.log"
  # 7D: 区域共享扫描 stride 1/2/4/8（420 条 x 4 配置，4 步去噪，逐 chunk 交替）
  run eff_stride bash -c "'$PY' -u scripts/ch4_perf_stride.py 0 2>&1 | tee outputs/logs/ch4_stride.log"
fi

# ---------------------------------------------------------------- 汇总
step "汇总"
"$PY" - <<'EOF'
import glob
import json
import os

out = []
runs = sorted(glob.glob('outputs/chapter4/runs/*.json'))
abl = sorted(glob.glob('outputs/chapter4/runs_abl/*.json'))
out.append('main runs %d (expect 18)   abl runs %d (expect 8)'
           % (len(runs), len(abl)))
missing_commit = [p for p in runs + abl
                  if 'git_commit' not in json.load(open(p, encoding='utf-8'))]
out.append('runs without git_commit: %s' % (missing_commit or 'none'))
for path in ('outputs/chapter4/REPORT.md',
             'outputs/chapter4/figures/process_curves.png',
             'outputs/chapter4/figures/smoothness.png',
             'outputs/chapter4/figures/main_table.png'):
    out.append('%-52s %s' % (path, 'OK' if os.path.exists(path) else 'MISS'))
for path in ('outputs/logs/ch4_perf_report.log',
             'outputs/logs/ch4_ab_batch.log',
             'outputs/logs/ch4_stride.log'):
    out.append('##### %s' % path)
    if not os.path.exists(path):
        out.append('  (missing)')
        continue
    text = open(path, 'rb').read().decode('utf-8', 'replace')
    hits = [l for l in text.splitlines()
            if l.startswith(('RESULT', 'ALL DONE'))]
    if hits:
        out.extend('  ' + h for h in hits)
    else:
        out.append('  (no RESULT yet)')

print('\n'.join(out))
with open('outputs/logs/ch4_test_all_summary.txt', 'w',
          encoding='utf-8') as handle:
    handle.write('\n'.join(out) + '\n')
EOF

ELAPSED=$(( $(date +%s) - STARTED ))
echo
if [ "${#FAILED[@]}" = 0 ]; then
  echo "############ ALL DONE  $(date '+%F %T')  用时 $((ELAPSED / 60)) min"
else
  echo "############ DONE WITH FAILURES  $(date '+%F %T')  用时 $((ELAPSED / 60)) min"
  for item in "${FAILED[@]}"; do echo "   FAILED $item"; done
fi
echo "### 简报  outputs/logs/ch4_test_all_summary.txt"
echo "### 总日志 outputs/logs/ch4_test_all.log"
[ "${#FAILED[@]}" = 0 ] || exit 1
