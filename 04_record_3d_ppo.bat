@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo [ERROR] Run setup_windows.bat first.& pause & exit /b 1)
if not exist outputs\ppo_genesis.zip (echo [ERROR] Model missing. Run 02_train_ppo.bat first.& pause & exit /b 1)
.venv\Scripts\python.exe -m scout_genesis demo --backend cpu --policy ppo --record
if errorlevel 1 (echo [ERROR] Video generation failed. Try 05_trajectory_only.bat.& pause & exit /b 1)
echo [OK] Video saved to outputs\genesis_3d_demo.mp4.
pause
