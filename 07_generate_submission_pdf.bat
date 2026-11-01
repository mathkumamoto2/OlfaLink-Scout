@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo [ERROR] Run setup_windows.bat first.& pause & exit /b 1)
if not exist outputs\smoke_result.json (echo [ERROR] Run 01_physics_test.bat first.& pause & exit /b 1)
if not exist outputs\eval_summary.json (echo [ERROR] Run 03_evaluate.bat first.& pause & exit /b 1)
echo Enter the OmniCampus ACCOUNT ID, not your real name.
set /p ACCOUNT_ID=Account ID: 
if "%ACCOUNT_ID%"=="" (echo [ERROR] No account ID.& pause & exit /b 1)
.venv\Scripts\python.exe -m scout_genesis.report --account "%ACCOUNT_ID%"
if errorlevel 1 (echo [ERROR] PDF generation failed.& pause & exit /b 1)
echo [OK] Report created in outputs\. Review the PDF before submission.
pause
