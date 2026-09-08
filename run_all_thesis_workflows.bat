@echo off
setlocal ENABLEDELAYEDEXPANSION

set BASE=%~dp0

if exist "%BASE%.venv\Scripts\python.exe" (
    set PYTHON=%BASE%.venv\Scripts\python.exe
) else if exist "%BASE%python_env\Scripts\python.exe" (
    set PYTHON=%BASE%python_env\Scripts\python.exe
) else (
    set PYTHON=python
)

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set RUNDATE=%%i

echo ================================================
echo Thesis master workflow started: %RUNDATE%
echo Python: %PYTHON%
echo ================================================

set BASELINE_OUT=%BASE%results\thesis_runs\%RUNDATE%
set ML_OUT=%BASE%results\thesis_ml_runs\%RUNDATE%
set LSTM_OUT=%BASE%results\thesis_lstm_runs\%RUNDATE%
set COMPARE_OUT=%BASE%results\comparisons\%RUNDATE%

if not exist "%BASELINE_OUT%" mkdir "%BASELINE_OUT%"
if not exist "%ML_OUT%" mkdir "%ML_OUT%"
if not exist "%LSTM_OUT%" mkdir "%LSTM_OUT%"
if not exist "%COMPARE_OUT%" mkdir "%COMPARE_OUT%"

set FAIL=0

echo.
echo [1/4] Running baseline profile...
"%PYTHON%" "%BASE%thesis_runner.py" --profile "%BASE%thesis_profile.json" --run-tag baseline-%RUNDATE% --out-dir "%BASELINE_OUT%"
if errorlevel 1 (
    echo Baseline run FAILED
    set FAIL=1
) else (
    echo Baseline run OK
)

echo.
echo [2/4] Running ML profile...
"%PYTHON%" "%BASE%thesis_runner.py" --profile "%BASE%thesis_profile_ml.json" --run-tag ml-%RUNDATE% --out-dir "%ML_OUT%"
if errorlevel 1 (
    echo ML run FAILED
    set FAIL=1
) else (
    echo ML run OK
)

echo.
echo [3/4] Running LSTM profile...
"%PYTHON%" "%BASE%thesis_runner.py" --profile "%BASE%thesis_profile_lstm.json" --run-tag lstm-%RUNDATE% --out-dir "%LSTM_OUT%"
if errorlevel 1 (
    echo LSTM run FAILED
    set FAIL=1
) else (
    echo LSTM run OK
)

echo.
echo [4/4] Comparing baseline vs ML...
"%PYTHON%" "%BASE%compare_forecast_runs.py" --baseline-dir "%BASELINE_OUT%" --candidate-dir "%ML_OUT%" --label-baseline baseline --label-candidate ml --out-dir "%COMPARE_OUT%\baseline_vs_ml"
if errorlevel 1 (
    echo Baseline vs ML comparison FAILED
    set FAIL=1
) else (
    echo Baseline vs ML comparison OK
)

echo.
echo Comparing baseline vs LSTM...
"%PYTHON%" "%BASE%compare_forecast_runs.py" --baseline-dir "%BASELINE_OUT%" --candidate-dir "%LSTM_OUT%" --label-baseline baseline --label-candidate lstm --out-dir "%COMPARE_OUT%\baseline_vs_lstm"
if errorlevel 1 (
    echo Baseline vs LSTM comparison FAILED
    set FAIL=1
) else (
    echo Baseline vs LSTM comparison OK
)

echo.
echo ================================================
if "%FAIL%"=="0" (
    echo Thesis master workflow COMPLETED SUCCESSFULLY.
) else (
    echo Thesis master workflow finished WITH ERRORS.
)
echo Output root: %COMPARE_OUT%
echo Weekly checklist template: %BASE%WEEKLY_CHECKLIST.md
echo ================================================

pause
exit /b %FAIL%
