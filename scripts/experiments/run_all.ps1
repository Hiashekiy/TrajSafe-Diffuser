# 一键复现 实验 1/2/3（见 实验/docs/*.md）
#
#   powershell -ExecutionPolicy Bypass -File scripts\experiments\run_all.ps1
#
# 每一步都可单独运行；日志写到 实验\logs\，结果写到 实验\results\。
# 需要：E:\CondaEnvData\envs\GGMPC\python.exe（torch 2.9.1+cu126, CUDA 可用）。

$ErrorActionPreference = "Stop"
$py = "E:\CondaEnvData\envs\GGMPC\python.exe"
$root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location $root
$logs = Join-Path $root "实验\logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

function Step($name, $args) {
    Write-Host "==> $name" -ForegroundColor Cyan
    & $py -u @args *> (Join-Path $logs "$name.log")
    if ($LASTEXITCODE -ne 0) { throw "$name failed (exit $LASTEXITCODE), see 实验\logs\$name.log" }
}

# --- 实验 1 ---------------------------------------------------------------
Step "exp1_run"     @("scripts\experiments\exp1_methods.py", "--stage", "run", "--seeds", "0", "1", "2")
Step "exp1_metrics" @("scripts\experiments\exp1_methods.py", "--stage", "metrics")

# --- 实验 2 ---------------------------------------------------------------
Step "exp2_ablation_run" @("scripts\experiments\exp2_ablation.py", "--stage", "run", "--seeds", "0", "1", "2")
Step "exp2_ablation_metrics" @("scripts\experiments\exp2_ablation.py", "--stage", "metrics")
Step "exp2_feedback_run" @("scripts\experiments\exp2_feedback_process.py", "--stage", "run", "--seeds", "0", "1", "2")
Step "exp2_feedback_summary" @("scripts\experiments\exp2_feedback_process.py", "--stage", "summarise")

# --- 实验 3（ALM 迭代预算与安全-效率权衡）-------------------------------
Step "exp3_p0"      @("scripts\experiments\exp3_alm_budget.py", "--stage", "p0")
Step "exp3_p1"      @("scripts\experiments\exp3_alm_budget.py", "--stage", "p1", "--p1-limit", "60")
Step "exp3_run"     @("scripts\experiments\exp3_alm_budget.py", "--stage", "run", "--seeds", "0", "1", "2")
Step "exp3_timing"  @("scripts\experiments\exp3_alm_budget.py", "--stage", "timing")
Step "exp3_latency" @("scripts\experiments\exp3_alm_budget.py", "--stage", "latency")
Step "exp3_table"   @("scripts\experiments\exp3_alm_budget.py", "--stage", "table")
Step "exp3_p1table" @("scripts\experiments\exp3_alm_budget.py", "--stage", "p1table")
Step "exp3_figure"  @("scripts\experiments\exp3_alm_budget.py", "--stage", "figure")

# --- 汇总 ---------------------------------------------------------------
Step "summarize"    @("scripts\experiments\summarize.py")

Write-Host "所有实验完成。结果：实验\results\{exp1,exp2,exp3}" -ForegroundColor Green
