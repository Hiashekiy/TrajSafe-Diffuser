@echo off
rem D-C48（100 epoch，标准条件扩散：无椭圆安全特征/无历史反馈）—— 分离式后台执行
cd /d D:\ProjectDirectory\Neural-IRISDiffuser
"D:\Program Files\Git\usr\bin\bash.exe" -lc "cd /d/ProjectDirectory/Neural-IRISDiffuser && bash scripts/ch4_train.sh configs/config_160k4p_c48_d_std.yaml outputs/oneshot_k4p_c48_d 100 6.0"
