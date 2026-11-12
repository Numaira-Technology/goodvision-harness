@echo off
setlocal
python "%~dp0neural_container.py" start %*
if errorlevel 1 exit /b 1
