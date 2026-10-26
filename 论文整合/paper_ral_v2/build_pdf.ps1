$ErrorActionPreference = 'Stop'
$compiler = Join-Path $env:APPDATA 'TinyTeX\bin\windows\pdflatex.exe'
if (-not (Test-Path -LiteralPath $compiler)) {
    $compiler = (Get-Command pdflatex -ErrorAction Stop).Source
}
Push-Location -LiteralPath $PSScriptRoot
try {
    foreach ($pass in 1..2) {
        & $compiler -interaction=nonstopmode -halt-on-error -file-line-error main.tex
        if ($LASTEXITCODE -ne 0) { throw "LaTeX compilation failed on pass $pass" }
    }
    $destination = Join-Path $PSScriptRoot 'output\pdf'
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'main.pdf') -Destination (Join-Path $destination 'TrajSafe_Diffuser_Compact_Illustrated.pdf') -Force
} finally {
    Pop-Location
}
