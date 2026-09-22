<#
.SYNOPSIS
    Train TrajSafe-Diffuser on the carla_full_160_256 processed cache.

.DESCRIPTION
    Writes stdout+stderr to logs\train_160.out.log so the run can be launched
    detached and still be inspected later.  Checkpoints go to
    outputs/bspline_carla_160/ckpt (NEVER the 80 m run dir).

.EXAMPLE
    # foreground - keep this window open
    .\scripts\train_160.ps1

    # detached - survives closing the window (run from a throwaway window)
    Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
        '-File','D:\ProjectDirectory\Neural-IRISDiffuser\scripts\train_160.ps1' -WindowStyle Hidden

    # override anything
    .\scripts\train_160.ps1 -Extra '--epochs','50'

    # watch it
    Get-Content logs\train_160.out.log -Wait -Tail 20
#>
[CmdletBinding()]
param(
    [string]   $Config = 'configs/config_160.yaml',
    [string[]] $Extra  = @(),
    [string]   $LogName = 'train_160.out.log'
)

$ErrorActionPreference = 'Continue'

# this script lives in <repo>\scripts\, so the repo root is ONE level up
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

# --- kill switch -----------------------------------------------------------
# Create logs\NO_TRAIN to turn this script into a no-op.  Batch drivers that
# chain "prepare data -> train" keep running their data stages but will not
# start training while the sentinel exists.  Delete the file to arm training.
$Log = Join-Path $Repo ('logs\' + $LogName)
if (Test-Path (Join-Path $Repo 'logs\NO_TRAIN')) {
    Write-Host '[train_160] logs\NO_TRAIN exists -> training SKIPPED'
    'training SKIPPED because logs\NO_TRAIN exists' | Add-Content -Path $Log
    exit 0
}

$Py = 'E:\CondaEnvData\envs\GGMPC\python.exe'
if (-not (Test-Path $Py))  { throw "python not found: $Py" }
if (-not (Test-Path $Config)) { throw "config not found: $Config" }

New-Item -ItemType Directory -Force -Path (Join-Path $Repo 'logs') | Out-Null
$Log = Join-Path $Repo ('logs\' + $LogName)

$cmdArgs = @('-u', 'train.py', '--config', $Config) + $Extra

"=== train_160 START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  pid=$PID ===" |
    Tee-Object -FilePath $Log
"    config : $Config" | Tee-Object -FilePath $Log -Append
"    extra  : $($Extra -join ' ')" | Tee-Object -FilePath $Log -Append

& $Py @cmdArgs 2>&1 | Tee-Object -FilePath $Log -Append

"=== train_160 END   $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  exit=$LASTEXITCODE ===" |
    Tee-Object -FilePath $Log -Append
