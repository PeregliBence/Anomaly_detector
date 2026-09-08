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

echo Running thesis research pipeline...
if exist "%BASE%thesis_profile.json" (
    "%PYTHON%" "%BASE%thesis_runner.py" --profile "%BASE%thesis_profile.json" %*
) else (
    "%PYTHON%" "%BASE%thesis_runner.py" %*
)

echo.
echo Finished. Check results\thesis\
pause
