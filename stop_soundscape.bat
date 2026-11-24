@echo off
rem Stops the Earth Soundscape kiosk and any leftover FluidSynth. Double-click to run.
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*kiosk.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
taskkill /F /IM fluidsynth.exe >nul 2>&1
echo Stopped.
timeout /t 2 >nul
