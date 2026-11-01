@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
where py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python launcher "py" was not found. Install Python 3.11 from python.org.
  pause
  exit /b 1
)
py -3.11 -c "import sys; assert sys.maxsize>2**32; print(sys.version)" || (
  echo [ERROR] 64-bit Python 3.11 was not found. Install CPython 3.11 x64.
  pause
  exit /b 1
)
if not exist .venv\Scripts\python.exe (
  py -3.11 -m venv .venv || goto :fail
)
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel || goto :fail
rem Official PyTorch Windows CPU wheel. CUDA can be configured later.
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu || goto :fail
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :fail
.venv\Scripts\python.exe -m scout_genesis doctor || goto :fail
.venv\Scripts\python.exe -m pytest -q tests || goto :fail
echo.
echo [OK] Environment is installed. Run 01_physics_test.bat next.
pause
exit /b 0
:fail
echo.
echo [ERROR] Setup failed. See the message above. No learning or demo was run.
pause
exit /b 1
