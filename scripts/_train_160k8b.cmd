@echo off
cd /d D:\ProjectDirectory\Neural-IRISDiffuser
echo ==== train_160k8b START %DATE% %TIME% ==== > logs\train_160k8b.out.log
"E:\CondaEnvData\envs\GGMPC\python.exe" -u train.py --config configs/config_160k8.yaml --init-best-task 28.0124 >> logs\train_160k8b.out.log 2>&1
echo ==== train_160k8b END %DATE% %TIME% exit=%ERRORLEVEL% ==== >> logs\train_160k8b.out.log
