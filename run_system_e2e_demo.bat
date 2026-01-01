@echo off
echo ===================================================
echo   Candidate Matching System - Automated Demo Run
echo   Maintained by Misem (Project Lead)
echo ===================================================

echo [1/4] Checking python environment...
if not exist ".venv" (
    echo    Creating virtual environment...
    python -m venv .venv
)

echo [2/4] Installing dependencies...
.venv\Scripts\pip install -q -r requirements.txt
if %errorlevel% neq 0 (
    echo    Error installing dependencies!
    pause
    exit /b %errorlevel%
)

echo [3/4] Running Training Pipeline (Prefect)...
echo    This might take a minute as we train the Ensemble...
set PYTHONPATH=.
.venv\Scripts\python src/workflow.py

if %errorlevel% neq 0 (
    echo    Pipeline failed! Check the logs above.
    pause
    exit /b %errorlevel%
)

echo [4/4] Starting Inference Server...
echo    The API will be available at http://127.0.0.1:8000
echo    Press Ctrl+C to stop the server.
set PYTHONPATH=.
.venv\Scripts\uvicorn src.inference:app --reload

pause
