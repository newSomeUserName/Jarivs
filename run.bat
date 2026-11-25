@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto run
echo Installing dependencies, please wait a few minutes...
py -m venv .venv
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
:run
".venv\Scripts\python.exe" "%~dp0jarvis.py" %*
if errorlevel 1 goto fail
exit /b 0
:fail
echo.
echo Something went wrong. See the error message above.
pause
