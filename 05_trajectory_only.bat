@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo [ERROR] Run setup_windows.bat first.& pause & exit /b 1)
.venv\Scripts\python.exe -m scout_genesis demo --backend cpu --policy scripted
if errorlevel 1 (echo [ERROR] Demo failed.& pause & exit /b 1)
echo [OK] Trajectory: outputs\demo_trajectory.png. (Scripted, NOT an RL policy.)
pause
