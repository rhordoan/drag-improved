$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
Set-Location $rootDir

if (-not (Test-Path "data")) { New-Item -ItemType Directory -Path "data" }

Write-Host "Downloading datasets via src/data/download_hf.py..."
python -m src.data.download_hf --output_dir data

Write-Host "Datasets downloaded to data/webqsp and data/cwq"
