@echo off
chcp 65001 >nul
title REBEL Trader - Account 10018026
color 0A
echo ============================================
echo   REBEL TRADER - Rules-Based AI System
echo   Account: 10018026
echo   MT5: C:\MT5__TRADER\terminal64.exe
echo ============================================
echo.

REM Change to REBEL Trader directory
cd /d "C:\Rebel Technologies\Rebel Trader"

REM Start MT5 terminal first (minimized)
echo Starting MT5 Terminal...
start "" /min "C:\MT5__TRADER\terminal64.exe"
timeout /t 5 /nobreak > nul

REM Run REBEL Trader
echo Starting REBEL Trader...
py -3.12 run.py

pause
