@echo off
cd /d D:\ProjectDirectory\Neural-IRISDiffuser\diffusion-dashboard
echo ==== backend restart %DATE% %TIME% ==== > backend.log
"E:\CondaEnvData\envs\GGMPC\python.exe" -u backend.py >> backend.log 2>&1
