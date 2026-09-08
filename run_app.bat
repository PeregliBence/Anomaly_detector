@echo off
setlocal

REM A BASE változóba a .bat fájl mappája kerül
set BASE=%~dp0

if exist "%BASE%.venv\Scripts\python.exe" (
	set PYTHON=%BASE%.venv\Scripts\python.exe
) else if exist "%BASE%python_env\Scripts\python.exe" (
	set PYTHON=%BASE%python_env\Scripts\python.exe
) else (
	set PYTHON=python
)

"%PYTHON%" -m streamlit run "%BASE%streamlit_app.py"

pause