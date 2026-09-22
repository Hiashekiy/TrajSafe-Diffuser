@echo off
cd /d D:\ProjectDirectory\Neural-IRISDiffuser
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train_160.ps1 -Config configs/config_160k8.yaml -LogName train_160k8.out.log
