@echo off
rem 第4章正式实验全流程（选参 -> 主表 -> 消融 -> 表 -> 图）—— 分离式后台执行
cd /d D:\ProjectDirectory\Neural-IRISDiffuser
"D:\Program Files\Git\usr\bin\bash.exe" -lc "cd /d/ProjectDirectory/Neural-IRISDiffuser && bash scripts/ch4_run_all.sh >> outputs/logs/ch4_run_all.log 2>&1"
