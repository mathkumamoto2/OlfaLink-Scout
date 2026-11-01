@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo [ERROR] Run setup_windows.bat first.& pause & exit /b 1)
if not exist outputs\ppo_genesis.zip (echo [ERROR] Model missing. Run 02_train_ppo.bat first.& pause & exit /b 1)
.venv\Scripts\python.exe -m scout_genesis evaluate --backend cpu --episodes 20
if errorlevel 1 (echo [ERROR] Evaluation failed.& pause & exit /b 1)
echo [OK] Open outputs\eval_summary.json and outputs\eval.csv.
pause
