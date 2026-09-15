@echo off
title Hey Joy wake-word demo
cd /d "%~dp0"
python demo.py %*
if errorlevel 1 pause
