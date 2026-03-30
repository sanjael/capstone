@echo off
cd /d "%~dp0"
echo Starting backend with venv Python...
"%~dp0venv\Scripts\python.exe" -m uvicorn main:app --reload --port 8000
pause
