@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo [ERROR] Run setup_windows.bat first.& pause & exit /b 1)
.venv\Scripts\python.exe -m scout_genesis smoke --backend cpu
if errorlevel 1 (echo [ERROR] Physics check failed. Do not train yet.& pause & exit /b 1)
echo [OK] Physical forward movement verified.
pause
