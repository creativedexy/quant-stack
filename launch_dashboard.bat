@echo off
title Quant Stack Dashboard
cd /d "%~dp0"
echo Starting Quant Stack Dashboard...
echo.
echo Dashboard will open at: http://localhost:8000/ui/overview
echo Login: admin / changeme
echo.
echo Press Ctrl+C to stop the server.
echo.

REM Start the browser after a short delay
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8000/ui/overview"

REM Launch the FastAPI server
.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
