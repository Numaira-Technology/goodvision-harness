@echo off
setlocal
python "%~dp0neural_container.py" status %*
if errorlevel 1 exit /b 1
