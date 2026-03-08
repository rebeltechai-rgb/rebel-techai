@echo off
title REBEL Intelligent Scanner
cd /d "%~dp0"
echo ============================================
echo  REBEL INTELLIGENT SCANNER
echo  TA-Driven Market Analysis Engine
echo ============================================
echo.
echo Make sure MetaTrader 5 is open and logged in.
echo.
python rebel_scanner.py
echo.
pause
