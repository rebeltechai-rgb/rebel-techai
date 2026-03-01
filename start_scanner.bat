@echo off
chcp 65001 >nul
title REBEL Intelligent Scanner - Account 10018026 
color 0D
echo ============================================
echo   REBEL INTELLIGENT SCANNER
echo   Account: 10018026 (AXI Select)
echo   MT5: C:\Program Files\Axi MetaTrader 5 Terminal
echo ============================================
echo.

REM Load API key from scanner env file if it exists
if exist "C:\Rebel Technologies\Rebel intelligent Scanner\.env" (
    for /f "tokens=1,2 delims==" %%a in ("C:\Rebel Technologies\Rebel intelligent Scanner\.env") do (
        if "%%a"=="OPENAI_API_KEY" set OPENAI_API_KEY=%%b
    )
)

REM Check if API key is set
if not defined OPENAI_API_KEY (
    echo [WARNING] OPENAI_API_KEY not set - AI features disabled
    echo Create .env file with: OPENAI_API_KEY=sk-your-key
    echo.
)

REM Change to Scanner directory
cd /d "C:\Rebel Technologies\Rebel intelligent Scanner"

REM Start MT5 terminal first (minimized)
echo Starting MT5 Terminal (SELECT)...
start "" /min "C:\Program Files\Axi MetaTrader 5 Terminal"
timeout /t 5 /nobreak > nul

REM Run Scanner
echo Starting Rebel Intelligent Scanner...
py -3.12 rebel_intelligent_scanner.py

pause
