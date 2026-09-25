@echo off
rem 等 S-C48 训练结束 -> 自动接力训练 D-C48（分离式后台执行）
cd /d D:\ProjectDirectory\Neural-IRISDiffuser
"D:\Program Files\Git\usr\bin\bash.exe" -lc "cd /d/ProjectDirectory/Neural-IRISDiffuser && bash scripts/ch4_chain_d.sh"
