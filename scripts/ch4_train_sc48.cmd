@echo off
rem S-C48（100 epoch，无历史安全反馈训练）—— 分离式后台执行
cd /d D:\ProjectDirectory\Neural-IRISDiffuser
"D:\Program Files\Git\usr\bin\bash.exe" -lc "cd /d/ProjectDirectory/Neural-IRISDiffuser && bash scripts/ch4_train.sh configs/config_160k4p_c48_s_nofb.yaml outputs/models/oneshot_k4p_c48_s 100 6.0"
