@echo off
rem Detached launcher for scripts/ch4_perf_stride.py (region sharing scan).
rem argv %1 = sample limit (0/absent = all 420), %2 = log name
set LIMIT=%1
if "%LIMIT%"=="" set LIMIT=0
set LOG=%2
if "%LOG%"=="" set LOG=ch4_stride
cd /d D:\ProjectDirectory\Neural-IRISDiffuser
"D:\Program Files\Git\usr\bin\bash.exe" -lc "cd /d/ProjectDirectory/Neural-IRISDiffuser && E:/CondaEnvData/envs/GGMPC/python.exe -u scripts/ch4_perf_stride.py %LIMIT% >> outputs/logs/%LOG%.log 2>&1"
