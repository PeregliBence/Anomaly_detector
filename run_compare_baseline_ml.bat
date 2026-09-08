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

set BASELINE_DIR=%BASE%results\thesis
set CANDIDATE_DIR=%BASE%results\thesis_ml
set OUTDIR=%BASE%results\comparisons\baseline_vs_ml

echo Comparing baseline vs ML runs...
"%PYTHON%" "%BASE%compare_forecast_runs.py" --baseline-dir "%BASELINE_DIR%" --candidate-dir "%CANDIDATE_DIR%" --label-baseline baseline --label-candidate ml --out-dir "%OUTDIR%"

echo.
echo Comparison finished. Output folder:
echo %OUTDIR%
pause
