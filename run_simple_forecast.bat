@echo off
setlocal
cd /d "%~dp0"

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else if exist python_env\Scripts\activate.bat (
    call python_env\Scripts\activate.bat
)

python simple_forecast_runner.py --profile simple_forecast_profile.json

echo.
echo Simple forecast run finished.
echo Output: results\simple_forecast\
pause
