@echo off
chcp 65001 >nul
title REBEL Trader Rules — Scanner-Driven (No GPT)
color 0B
echo ============================================
echo   REBEL TRADER RULES — Scanner-Driven
echo   No GPT / Pure TA Multi-Timeframe
echo   MT5: C:\MT5__TRADER\terminal64.exe
echo ============================================
echo.

REM Change to REBEL Trader Rules directory
cd /d "C:\Rebel Technologies\Rebel Trader Rules"

REM MT5 should already be running — don't restart it
REM If MT5 is not running, start it manually first.

REM Run REBEL Trader Rules
echo Starting REBEL Trader Rules...
py -3.12 run.py

pause
