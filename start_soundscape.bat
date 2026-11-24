@echo off
rem Earth Soundscape launcher: double-click to start. Edit the two lines below if needed.
cd /d "%~dp0"
set WORLD=worlddata\world.u8
set SITE=
rem  ^ after the site bake, set SITE=sitedata\site.u8

if not exist ".venv\Scripts\python.exe" ( echo .venv not found in %CD% & pause & exit /b 1 )
set ARGS=--world %WORLD% --open
if not "%SITE%"=="" set ARGS=%ARGS% --site %SITE%

echo Starting Earth Soundscape... (close this window to stop it)
.venv\Scripts\python.exe kiosk.py %ARGS%
echo.
echo The kiosk stopped. Scroll up for the reason.
pause
