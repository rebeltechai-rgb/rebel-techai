@echo off
title REBEL Intelligent Scanner Lite - Run
cd /d "C:\Rebel Technologies\Rebel Intelligent scanner Lite"
echo ============================================
echo  REBEL INTELLIGENT SCANNER LITE - RUN
echo ============================================
echo.
echo Ensure MT5 is open at: C:\mt5_scanner_live\terminal64.exe
echo.
py -3.12 rebel_intelligent_scanner_lite.py
echo.
pause
