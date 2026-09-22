<#
.SYNOPSIS
    Finish the carla_full_160_256 data preparation (processed cache + labels).

.DESCRIPTION
    00  raw scene/task  -> processed cache contract   (fast, idempotent)
    01  skeleton candidates + topology_best           (idempotent, per-sample cache)
    02  fixed-centre safe-ellipse labels              (SLOW: ~20 min, resumable)
    03  contract validation                           (must print VALID = True)

    Stages 01 and 02 cache per sample under <split>\_cache\, so re-running the
    script after an interruption resumes instead of restarting.

.EXAMPLE
    .\scripts\data\carla_full\run_pipeline_160.ps1
    .\scripts\data\carla_full\run_pipeline_160.ps1 -SkipBuild -Workers 14
#>
[CmdletBinding()]
param(
    [int]    $Workers   = 12,
    [switch] $SkipBuild
)

$ErrorActionPreference = 'Stop'

$Repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $Repo

$Py  = 'E:\CondaEnvData\envs\GGMPC\python.exe'
$Cfg = 'configs/config_160.yaml'
$Out = 'data/carla_processed_160'

if (-not (Test-Path $Py))  { throw "python not found: $Py" }
if (-not (Test-Path $Cfg)) { throw "config not found: $Cfg" }

function Invoke-Stage {
    param([string] $Name, [string[]] $CmdArgs)
    Write-Host ''
    Write-Host ('=' * 72) -ForegroundColor Cyan
    Write-Host ("  {0}   {1}" -f $Name, (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -ForegroundColor Cyan
    Write-Host ('=' * 72) -ForegroundColor Cyan
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $Py -u @CmdArgs
    $code = $LASTEXITCODE
    $sw.Stop()
    if ($code -ne 0) {
        Write-Host ("[FAIL] {0} exited with {1} after {2:n0}s" -f $Name, $code, $sw.Elapsed.TotalSeconds) -ForegroundColor Red
        exit $code
    }
    Write-Host ("[ok] {0} in {1:n0}s" -f $Name, $sw.Elapsed.TotalSeconds) -ForegroundColor Green
}

Write-Host "repo    : $Repo"
Write-Host "python  : $Py"
Write-Host "config  : $Cfg"
Write-Host "out     : $Out"
Write-Host "workers : $Workers"

if (-not $SkipBuild) {
    Invoke-Stage '00_build_processed' @(
        'scripts/data/carla_full/00_build_processed.py', '--config', $Cfg, '--out', $Out)
    Invoke-Stage '01_build_candidates' @(
        'scripts/data/carla/01_build_candidates.py', '--processed', $Out, '--config', $Cfg, '--workers', "$Workers")
}

Invoke-Stage '02_build_ellipse_labels' @(
    'scripts/data/carla/02_build_ellipse_labels.py', '--processed', $Out, '--config', $Cfg, '--workers', "$Workers")

Invoke-Stage '03_validate_processed' @(
    'scripts/data/carla/03_validate_processed.py', '--processed', $Out, '--config', $Cfg)

Write-Host ''
Write-Host 'ALL STAGES DONE - check that 03 printed VALID = True' -ForegroundColor Green
Write-Host 'next:  .\scripts\train_160.ps1' -ForegroundColor Green
