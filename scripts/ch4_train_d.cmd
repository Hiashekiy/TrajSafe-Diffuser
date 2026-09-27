@echo off
rem D-C48 training: 100 epochs, plain conditional diffusion (no ellipse safety
rem features, no history feedback). Keep ASCII-only (cmd.exe reads .cmd as GBK).
cd /d D:\ProjectDirectory\Neural-IRISDiffuser
"D:\Program Files\Git\usr\bin\bash.exe" -lc "cd /d/ProjectDirectory/Neural-IRISDiffuser && bash scripts/ch4_train.sh configs/config_160k4p_c48_d_std.yaml outputs/oneshot_k4p_c48_d 100 6.0"
