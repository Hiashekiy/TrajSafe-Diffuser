#!/usr/bin/env bash
# ============================================================================
# 三阶段训练流水线： 轨迹 -> 椭圆 -> 联合
#   Windows: 在 Git Bash 里运行  bash scripts/train_pipeline.sh
#   Ubuntu : 在 bash 里运行        bash scripts/train_pipeline.sh
#
#  可配置（环境变量，默认值如下）：
#    PY        Python 可执行文件（默认按系统自动选）
#    CONFIG    配置文件（默认 configs/config.yaml）
#    EPOCHS1   阶段1(轨迹) 轮数（默认 30）
#    EPOCHS2   阶段2(椭圆) 额外轮数（默认 30）
#    EPOCHS3   阶段3(联合) 额外轮数（默认 100）
#
#  说明：train.py 里 start_epoch = resume_epoch + 1，循环为 range(start, epochs)。
#  要让每个阶段真正训练 EPOCHSx 个 epoch，步骤：
#      累计目标 = 当前 best.pt 的 epoch + EPOCHSx + 1
#  这样 range(start, T) 恰好跑 EPOCHSx 个 epoch（含阶段 3）。
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."   # 切到仓库根目录

if [[ -z "${PY:-}" ]]; then
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)  PY="E:/CondaEnvData/envs/GGMPC/python.exe" ;;
    *)                      PY="python" ;;
  esac
fi

CONFIG="${CONFIG:-configs/config.yaml}"
EPOCHS1="${EPOCHS1:-30}"
EPOCHS2="${EPOCHS2:-30}"
EPOCHS3="${EPOCHS3:-100}"

# 读取 best.pt 记录的学习轮数（没有则为 0）
current_epoch() {
  "$PY" -c "import torch,os; p='outputs/ckpt/best.pt'; d=torch.load(p,map_location='cpu',weights_only=False) if os.path.exists(p) else {}; print(int(d.get('epoch',0)))" 2>/dev/null || echo 0
}

# ---- 阶段 1：先训轨迹（从 0 开始） ----
echo "[pipeline] ===== 阶段 1/3：轨迹 (--phase traj, epochs=$EPOCHS1) ====="
"$PY" train.py --config "$CONFIG" --phase traj --epochs "$EPOCHS1"
cp outputs/ckpt/best.pt outputs/ckpt/best_phase1.pt 2>/dev/null || echo "[pipeline] 阶段1 best.pt 不存在，跳过备份"

# ---- 阶段 2：再训椭圆 ----
C2=$(current_epoch); T2=$((C2 + EPOCHS2 + 1))
echo "[pipeline] ===== 阶段 2/3：椭圆 (累计 $C2 -> $T2, 额外 ${EPOCHS2}) ====="
"$PY" train.py --config "$CONFIG" --phase ellipse --resume outputs/ckpt/best.pt --epochs "$T2"
cp outputs/ckpt/best.pt outputs/ckpt/best_phase2.pt 2>/dev/null || echo "[pipeline] 阶段2 best.pt 不存在，跳过备份"

# ---- 阶段 3：联合微调 ----
C3=$(current_epoch); T3=$((C3 + EPOCHS3 + 1))
echo "[pipeline] ===== 阶段 3/3：联合 (累计 $C3 -> $T3, 额外 ${EPOCHS3}) ====="
"$PY" train.py --config "$CONFIG" --phase joint --resume outputs/ckpt/best.pt --epochs "$T3"
cp outputs/ckpt/best.pt outputs/ckpt/best_phase3.pt 2>/dev/null || echo "[pipeline] 阶段3 best.pt 不存在，跳过备份"

echo "[pipeline] 三阶段完成。最终模型：outputs/ckpt/best.pt"
