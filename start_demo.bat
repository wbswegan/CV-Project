@echo off
setlocal

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found in PATH.
    echo Install Python first, then run: python -m pip install -r requirements.txt
    exit /b 1
)

echo Opening local demo page in your browser...
start "" cmd /c "timeout /t 3 >nul && start http://127.0.0.1:8000"

echo Starting FastAPI demo server...
python app.py
