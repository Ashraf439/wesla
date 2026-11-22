@echo off
cd /d "%~dp0"
set "HF_HOME=%CD%\.venv\huggingface"
set "HF_HUB_DISABLE_SYMLINKS=1"
set "GRADIO_ANALYTICS_ENABLED=False"
set "GRADIO_SERVER_NAME=127.0.0.1"
set "PYTHONUTF8=1"
".venv\Scripts\python.exe" -m wesla ui
pause
