@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo [ERROR] Run setup_windows.bat first.& pause & exit /b 1)
.venv\Scripts\python.exe -m scout_genesis demo --backend cpu --policy scripted --record
if errorlevel 1 (echo [ERROR] 3D camera recording failed.& pause & exit /b 1)
echo [OK] Scripted demonstration, not a trained AI: outputs\genesis_3d_demo.mp4.
pause
