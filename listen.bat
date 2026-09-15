@echo off
title Hey Joy wake-word demo
cd /d "%~dp0"
echo Listening for Hey Joy on the selected microphone.
echo Say "Hey Joy"  -  Ctrl+C to quit.
echo.
python -u demo.py --skip-enroll --device 3 --channel 0 %*
if errorlevel 1 pause
