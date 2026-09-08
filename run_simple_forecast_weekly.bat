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
set OUTDIR=%BASE%results\simple_forecast_runs\%RUNDATE%

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo Running weekly simple forecast evaluation: %RUNDATE%
"%PYTHON%" "%BASE%simple_forecast_runner.py" --profile "%BASE%simple_forecast_profile.json" --run-tag %RUNDATE% --out-dir "%OUTDIR%" %*

echo.
echo Weekly simple run finished. Output folder:
echo %OUTDIR%
pause
