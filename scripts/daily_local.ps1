# Local fallback for the GitHub Actions daily run.
#
# Pulls first, so when the scheduled cloud run already succeeded the forecast
# file exists and nothing is regenerated; only if the cloud run failed (for
# example because Yahoo blocked the runner) is the forecast made here. Running
# the next morning, before the US open, the forecast still precedes its outcome.
#
# Register once, from an elevated PowerShell (adjust the path if needed):
#   schtasks /Create /TN "StockForecastDaily" /SC WEEKLY /D TUE,WED,THU,FRI,SAT /ST 08:00 `
#     /TR "powershell -NoProfile -ExecutionPolicy Bypass -File D:\BME_2025-26_1\Onlab\Anomaly_detector\scripts\daily_local.ps1"

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw "git pull failed; resolve the local branch before the next run." }

.\.venv\Scripts\python.exe -m src.cli daily
if ($LASTEXITCODE -ne 0) { throw "The daily run failed." }

git add live/
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -m "Live predictions $(Get-Date -Format yyyy-MM-dd) (local run)"
    git push
}
