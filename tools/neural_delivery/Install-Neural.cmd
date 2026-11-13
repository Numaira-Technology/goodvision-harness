@echo off
setlocal
python "%~dp0neural_container.py" install %*
if errorlevel 1 exit /b 1
