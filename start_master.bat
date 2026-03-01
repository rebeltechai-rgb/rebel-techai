@echo off
chcp 65001 >nul
title REBEL Master - Account 10020166
color 0B
echo ============================================
echo   REBEL MASTER - ML/RF Trading System
echo   Account: 10020166
echo   MT5: C:\Master Pro\terminal64.exe
echo ============================================
echo.

REM Change to REBEL Master directory
cd /d "C:\Rebel Technologies\Rebel Master\Python"

REM Start MT5 terminal first (minimized)
echo Starting MT5 Terminal...
start "" /min "C:\Master Pro\terminal64.exe"
timeout /t 5 /nobreak > nul

REM Run REBEL Master
echo Starting REBEL Master...
py -3.12 main.py

pause
