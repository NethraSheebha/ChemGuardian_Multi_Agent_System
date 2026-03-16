@echo off
echo ========================================
echo Chemical Leak Monitoring System
echo ========================================
echo.
echo Starting Backend API...
start "Backend API" cmd /k "python backend_api.py"
timeout /t 5 /nobreak >nul
echo.
echo Starting Frontend Dashboard...
start "Frontend Dashboard" cmd /k "cd frontend && streamlit run app.py"
echo.
echo ========================================
echo System Started!
echo ========================================
echo Backend: http://localhost:8000
echo Frontend: http://localhost:8501
echo.
echo Press any key to exit...
pause >nul
