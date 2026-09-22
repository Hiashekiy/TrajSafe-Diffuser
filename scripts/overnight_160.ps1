<#
.SYNOPSIS
    Sleep-safe one-shot for the carla_full_160_256 run.

.DESCRIPTION
    Waits for any in-flight 02_build_ellipse_labels run to finish, then:
        02  finish the fixed-centre safe-ellipse labels   (resumable, no-op when cached)
        03  validate the processed cache                  (GATE: stops if errors)
        --  train TrajSafe-Diffuser                       (self-stops after train.max_hours)

    Launch it detached so it survives closing the window:
        Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
            '-File','D:\ProjectDirectory\Neural-IRISDiffuser\scripts\overnight_160.ps1' -WindowStyle Hidden

    Progress log: logs\overnight_160.log

.EXAMPLE
    .\scripts\overnight_160.ps1
    .\scripts\overnight_160.ps1 -Workers 14
    .\scripts\overnight_160.ps1 -TrainAnyway     # ignore a failed validation
#>
[CmdletBinding()]
param(
    [string] $Config = 'configs/config_160.yaml',
    [int]    $Workers = 12,
    [string] $Out = 'data/carla_processed_160',
    [switch] $TrainAnyway
)

$ErrorActionPreference = 'Continue'

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Py  = 'E:\CondaEnvData\envs\GGMPC\python.exe'
$Log = Join-Path $Repo 'logs\overnight_160.log'
New-Item -ItemType Directory -Force -Path (Join-Path $Repo 'logs') | Out-Null

$script:LastCode = 0

function Say([string] $m) {
    $line = '[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m
    Write-Host $line
    Add-Content -Path $Log -Value $line
}

function Run([string] $name, [string[]] $cmdArgs) {
    Say ('START ' + $name)
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $Py -u @cmdArgs 2>&1 | Tee-Object -FilePath $Log -Append
    $script:LastCode = $LASTEXITCODE
    $sw.Stop()
    Say ('END   {0}  exit={1}  {2:n0}s' -f $name, $script:LastCode, $sw.Elapsed.TotalSeconds)
}

if (-not (Test-Path $Py))  { Say ('FATAL python not found: ' + $Py); exit 1 }
if (-not (Test-Path $Config)) { Say ('FATAL config not found: ' + $Config); exit 1 }

Say '=============================================================='
Say 'overnight_160 start'
Say ('repo    : ' + $Repo)
Say ('config  : ' + $Config)
Say ('out     : ' + $Out)
Say ('workers : ' + $Workers)
Say '=============================================================='

# --- 1. never run two ellipse pools at once -------------------------------
Say 'waiting for any in-flight 02_build_ellipse_labels to finish ...'
$waited = 0
while ($true) {
    $other = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
               Where-Object { $_.CommandLine -and $_.CommandLine -like '*02_build_ellipse_labels*' })
    if ($other.Count -eq 0) { break }
    Start-Sleep -Seconds 20
    $waited += 20
    if (($waited % 300) -eq 0) { Say ('  still waiting ({0:n0} min)' -f ($waited / 60.0)) }
}
Say 'clear - no other ellipse run active'

# --- 2. finish the ellipse labels (resume; no-op when fully cached) -------
Run '02_build_ellipse_labels' @(
    'scripts/data/carla/02_build_ellipse_labels.py',
    '--processed', $Out, '--config', $Config, '--workers', "$Workers")
if ($script:LastCode -ne 0) {
    Say 'FATAL 02_build_ellipse_labels failed - NOT starting training'
    exit 2
}

# --- 3. validate the processed cache -------------------------------------
Run '03_validate_processed' @(
    'scripts/data/carla/03_validate_processed.py',
    '--processed', $Out, '--config', $Config)
if ($script:LastCode -ne 0 -and -not $TrainAnyway) {
    Say 'FATAL validation reported errors - NOT starting training'
    Say '      inspect logs\overnight_160.log, fix, or re-run with -TrainAnyway'
    exit 3
}

# --- 4. train -------------------------------------------------------------
Say 'data is ready - starting training'
Run 'train' @('train.py', '--config', $Config)

Say ('training exited with ' + $script:LastCode)
Say 'checkpoints : outputs/bspline_carla_160/ckpt'
Say 'train log   : logs/overnight_160.log'
Say 'overnight_160 done'
exit $script:LastCode
