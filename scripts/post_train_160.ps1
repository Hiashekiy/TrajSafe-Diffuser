<#
.SYNOPSIS
    Wait for the 160 m training run to finish, then evaluate it automatically.

.DESCRIPTION
    Polls for the python process running train.py, then runs, in order:

        evaluate.py --compare-raw   on val   (ALM on vs off, collision delta)
        evaluate.py --compare-raw   on test
        preview_zoom.py             on val   (zoomed, collision-annotated grid)
        preview_samples.py          on val   (the standard grid)

    Everything lands in outputs/bspline_carla_160/post/ and is appended to
    logs/post_train_160.log.

    Launch it detached through scripts\launch_detached.ps1 so it survives the
    calling shell:
        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\launch_detached.ps1 `
            -CommandLine 'powershell -NoProfile -ExecutionPolicy Bypass -File D:\...\scripts\post_train_160.ps1'
#>
[CmdletBinding()]
param(
    [string] $Config = 'configs/config_160.yaml',
    [string] $TrainLog = 'logs\train_160.out.log'
)
$ErrorActionPreference = 'Continue'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = 'E:\CondaEnvData\envs\GGMPC\python.exe'
$Log = Join-Path $Repo 'logs\post_train_160.log'
New-Item -ItemType Directory -Force -Path (Join-Path $Repo 'logs') | Out-Null

function Say([string] $m) {
    $line = '[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m
    Write-Host $line
    Add-Content -Path $Log -Value $line
}

Say 'post_train_160: waiting for train.py to exit ...'
$waited = 0
while ($true) {
    $p = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
           Where-Object { $_.CommandLine -and $_.CommandLine -like '*train.py*' })
    if ($p.Count -eq 0) { break }
    Start-Sleep -Seconds 60
    $waited += 60
    if (($waited % 1800) -eq 0) { Say ('  still waiting ({0:n1} h)' -f ($waited / 3600.0)) }
}
Say ('train.py gone after {0:n1} h; starting evaluation' -f ($waited / 3600.0))

# newest checkpoint of this run: prefer best_task, fall back to latest
$ckpt = Join-Path $Repo 'outputs\bspline_carla_160\ckpt\best_task.pt'
if (-not (Test-Path $ckpt)) { $ckpt = Join-Path $Repo 'outputs\bspline_carla_160\ckpt\latest.pt' }
Say ('checkpoint: ' + $ckpt)

function Run([string] $name, [string[]] $cmdArgs) {
    Say ('START ' + $name)
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $Py -u @cmdArgs 2>&1 | Tee-Object -FilePath $Log -Append
    $code = $LASTEXITCODE
    $sw.Stop()
    Say ('END   {0} exit={1} {2:n0}s' -f $name, $code, $sw.Elapsed.TotalSeconds)
}

$out = Join-Path $Repo 'outputs\bspline_carla_160\post'
New-Item -ItemType Directory -Force -Path $out | Out-Null

Run 'eval_val' @('evaluate.py', '--config', $Config, '--ckpt', $ckpt,
                 '--split', 'val', '--num-batches', '6', '--runs', '4',
                 '--compare-raw', '--out', (Join-Path $out 'eval_val'))
Run 'eval_test' @('evaluate.py', '--config', $Config, '--ckpt', $ckpt,
                  '--split', 'test', '--num-batches', '6', '--runs', '4',
                  '--compare-raw', '--out', (Join-Path $out 'eval_test'))
Run 'zoom_val' @('scripts/preview_zoom.py', '--config', $Config, '--ckpt', $ckpt,
                 '--split', 'val', '--pool', '256', '--num', '4',
                 '--out', (Join-Path $out 'zoom_val'))
Run 'zoom_train' @('scripts/preview_zoom.py', '--config', $Config, '--ckpt', $ckpt,
                   '--split', 'train', '--pool', '256', '--num', '3',
                   '--out', (Join-Path $out 'zoom_train'))
Run 'preview_val' @('scripts/preview_samples.py', '--config', $Config, '--ckpt', $ckpt,
                    '--split', 'val', '--pool', '256', '--num', '8',
                    '--out', (Join-Path $out 'preview_val'))

Say 'post_train_160 DONE - results in outputs/bspline_carla_160/post'
