@echo off
rem v4_map16_c48: 单 16x16 地图特征 + 修正量(Delta)反馈 + 统一连续安全损失 L_safe
rem 数据 data/processed/160m-k4p-c48，从零训练 200 epoch / 11 h 预算。
cd /d D:\ProjectDirectory\Neural-IRISDiffuser
"D:\Program Files\Git\usr\bin\bash.exe" -lc "cd /d/ProjectDirectory/Neural-IRISDiffuser && bash scripts/v4_train_c48.sh 200 11.0" >> logs\v4_map16_c48.out.log 2>&1
