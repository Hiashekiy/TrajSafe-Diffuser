@echo off
cd /d D:\ProjectDirectory\Neural-IRISDiffuser
"E:\CondaEnvData\envs\GGMPC\python.exe" -u scripts\_diag35.py > logs\diag35.log 2>&1
echo DONE_EXIT_%ERRORLEVEL% >> logs\diag35.log
