@echo off
cd /d D:\ProjectDirectory\Neural-IRISDiffuser
set PY=E:\CondaEnvData\envs\GGMPC\python.exe
echo ===== ZOOM ===== > logs\viz_alm.log
"%PY%" -u scripts\preview_zoom.py --config configs/config_160.yaml --ckpt outputs\bspline_carla_160\ckpt\best_task.pt --split val --pool 128 --num 4 --out outputs\bspline_carla_160\live\zoom_alm2 >> logs\viz_alm.log 2>&1
echo ===== GRID ===== >> logs\viz_alm.log
"%PY%" -u scripts\preview_samples.py --config configs/config_160.yaml --ckpt outputs\bspline_carla_160\ckpt\best_task.pt --split val --pool 128 --num 8 --out outputs\bspline_carla_160\live\grid_alm >> logs\viz_alm.log 2>&1
echo DONE_EXIT_%ERRORLEVEL% >> logs\viz_alm.log
