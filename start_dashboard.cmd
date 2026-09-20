@echo off
setlocal EnableExtensions
title Diffusion Lens launcher
set "ROOT=D:\ProjectDirectory\Neural-IRISDiffuser"
set "DASH=%ROOT%\diffusion-dashboard"
set "PY=E:\CondaEnvData\envs\GGMPC\python.exe"

echo ============================================================
echo  Diffusion Lens  -  CARLA + 32-control B-spline dashboard
echo ============================================================
echo.

if not exist "%DASH%\backend.py" goto :badroot
if not exist "%PY%" goto :badpy

echo [1/3] stopping any OLD inference service on 8765 ...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /C:"127.0.0.1:8765" ^| findstr /C:"LISTENING"') do (
  echo     killing PID %%p
  taskkill /F /PID %%p >nul 2>&1
)
ping -n 3 127.0.0.1 >nul

echo [2/3] starting inference service on http://localhost:8765 ...
start "Diffusion Lens API - 8765" /D "%DASH%" "%PY%" backend.py

netstat -ano | findstr /C:":3000" | findstr /C:"LISTENING" >nul 2>&1
if errorlevel 1 goto :startui
echo [3/3] web UI already running on 3000 - skipping start
goto :afterui
:startui
echo [3/3] starting web UI on http://localhost:3000 ...
start "Diffusion Lens UI - 3000" /D "%DASH%" cmd /k npm run dev
:afterui

echo waiting 22 seconds for the services ...
ping -n 23 127.0.0.1 >nul
echo backend health:
curl -s http://127.0.0.1:8765/health
echo.
start "" "http://localhost:3000/"

echo.
echo  If "format" above is not 3, the backend did not restart - close its
echo  window manually and run this file again.
goto :end

:badroot
echo [ERROR] cannot find "%DASH%\backend.py" - edit the ROOT line.
goto :end
:badpy
echo [ERROR] python not found at "%PY%" - edit the PY line.
goto :end

:end
echo.
pause
