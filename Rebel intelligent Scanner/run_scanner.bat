@echo off
title REBEL Intelligent Scanner - Run
cd /d "C:\Rebel Technologies\Rebel intelligent Scanner"
echo ============================================
echo  REBEL INTELLIGENT SCANNER - RUN
echo ============================================
echo.
echo Ensure MT5 is open at: C:\mt5_scanner_live\terminal64.exe
echo.
py -3.12 rebel_intelligent_scanner.py
echo.
pause
