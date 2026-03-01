@echo off
chcp 65001 >nul
title REBEL Trading Systems - Launcher
color 0E
echo ============================================
echo   REBEL TRADING SYSTEMS - FULL STARTUP
echo ============================================
echo.
echo This will start all 3 systems:
echo   1. REBEL Trader  (10018026)
echo   2. REBEL Master  (10020166)
echo   3. Scanner       (60078307)
echo.
echo ============================================
pause

REM Start all MT5 terminals first
echo.
echo [1/3] Starting MT5 Terminals...
start "" /min "C:\mt5\terminal64.exe"
start "" /min "C:\Master Pro\terminal64.exe"
start "" /min "C:\MT5-SELECT\terminal64.exe"
echo     Waiting for terminals to initialize...
timeout /t 8 /nobreak > nul

REM Start REBEL Trader in new window
echo.
echo [2/3] Starting REBEL Trader...
start "REBEL Trader" cmd /k "cd /d "C:\Rebel Technologies\Rebel Trader" && color 0A && py -3.12 run.py"
timeout /t 3 /nobreak > nul

REM Start REBEL Master in new window
echo.
echo [3/3] Starting REBEL Master...
start "REBEL Master" cmd /k "cd /d "C:\Rebel Technologies\Rebel Master\Python" && color 0B && py -3.12 main.py"
timeout /t 3 /nobreak > nul

REM Start Scanner in new window
echo.
echo [4/4] Starting Scanner...
start "REBEL Scanner" cmd /k "cd /d "C:\Rebel Technologies\Rebel intelligent Scanner" && color 0D && py -3.12 rebel_intelligent_scanner.py"

echo.
echo ============================================
echo   ALL SYSTEMS LAUNCHED!
echo ============================================
echo.
echo Each system is running in its own window.
echo Close this window when ready.
echo.
pause
