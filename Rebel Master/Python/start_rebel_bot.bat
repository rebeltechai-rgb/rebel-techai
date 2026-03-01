@echo off
REM REBEL Trading Bot - Auto Start Script
REM Starts the bot with proper Python environment

cd /d "C:\Rebel Technologies\Rebel Master\Python"

REM Log startup
echo [%date% %time%] REBEL Bot starting... >> "C:\Rebel Technologies\Rebel Master\logs\autostart.log"

REM Start the bot
python main.py

REM Log if it exits
echo [%date% %time%] REBEL Bot stopped >> "C:\Rebel Technologies\Rebel Master\logs\autostart.log"

