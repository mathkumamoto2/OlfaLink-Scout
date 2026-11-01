@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo [ERROR] Run setup_windows.bat first.& pause & exit /b 1)
.venv\Scripts\python.exe -m scout_genesis train --backend cpu --steps 20000
if errorlevel 1 (echo [ERROR] Training failed.& pause & exit /b 1)
echo [OK] Checkpoint stored in outputs\ppo_genesis.zip.
pause
