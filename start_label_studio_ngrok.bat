@echo off
REM Launcher double-click untuk start_label_studio_ngrok.ps1 (Windows).
REM Menjalankan PowerShell dengan bypass execution policy agar skrip bisa langsung jalan.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_label_studio_ngrok.ps1" %*
pause
