@echo off
setlocal

set BASE=%~dp0

if exist "%BASE%.venv\Scripts\python.exe" (
    set PYTHON=%BASE%.venv\Scripts\python.exe
) else if exist "%BASE%python_env\Scripts\python.exe" (
    set PYTHON=%BASE%python_env\Scripts\python.exe
) else (
    set PYTHON=python
)

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set RUNDATE=%%i
set OUTDIR=%BASE%results\thesis_ml_runs\%RUNDATE%

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo Running ML-track weekly thesis benchmark: %RUNDATE%
"%PYTHON%" "%BASE%thesis_runner.py" --profile "%BASE%thesis_profile_ml.json" --run-tag ml-%RUNDATE% --out-dir "%OUTDIR%" %*

echo.
echo ML weekly run finished. Output folder:
echo %OUTDIR%
pause
