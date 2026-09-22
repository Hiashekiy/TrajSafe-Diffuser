@echo off
cd /d D:\ProjectDirectory\Neural-IRISDiffuser
set PY=E:\CondaEnvData\envs\GGMPC\python.exe
set CK=outputs\bspline_carla_160k8\ckpt\latest.pt
set CFG=configs/config_160k8.yaml
set L=logs\random_latest.log
echo ===== BATCH START %DATE% %TIME% ===== > %L%
"%PY%" -u scripts\random_preview.py --config %CFG% --ckpt %CK% --split val --num 8 --seed 1001 --out outputs\bspline_carla_160k8\randombatch_latest\val >> %L% 2>&1
"%PY%" -u scripts\random_preview.py --config %CFG% --ckpt %CK% --split val --num 8 --seed 1002 --out outputs\bspline_carla_160k8\randombatch_latest\val >> %L% 2>&1
"%PY%" -u scripts\random_preview.py --config %CFG% --ckpt %CK% --split val --num 8 --seed 1003 --out outputs\bspline_carla_160k8\randombatch_latest\val >> %L% 2>&1
"%PY%" -u scripts\random_preview.py --config %CFG% --ckpt %CK% --split val --num 8 --seed 1004 --out outputs\bspline_carla_160k8\randombatch_latest\val >> %L% 2>&1
"%PY%" -u scripts\random_preview.py --config %CFG% --ckpt %CK% --split val --num 8 --seed 1005 --out outputs\bspline_carla_160k8\randombatch_latest\val >> %L% 2>&1
"%PY%" -u scripts\random_preview.py --config %CFG% --ckpt %CK% --split val --num 8 --seed 1006 --out outputs\bspline_carla_160k8\randombatch_latest\val >> %L% 2>&1
"%PY%" -u scripts\random_preview.py --config %CFG% --ckpt %CK% --split val --num 8 --seed 1007 --out outputs\bspline_carla_160k8\randombatch_latest\val >> %L% 2>&1
"%PY%" -u scripts\random_preview.py --config %CFG% --ckpt %CK% --split val --num 8 --seed 1008 --out outputs\bspline_carla_160k8\randombatch_latest\val >> %L% 2>&1
"%PY%" -u scripts\random_preview.py --config %CFG% --ckpt %CK% --split test --num 8 --seed 2001 --out outputs\bspline_carla_160k8\randombatch_latest\test >> %L% 2>&1
"%PY%" -u scripts\random_preview.py --config %CFG% --ckpt %CK% --split test --num 8 --seed 2002 --out outputs\bspline_carla_160k8\randombatch_latest\test >> %L% 2>&1
"%PY%" -u scripts\random_preview.py --config %CFG% --ckpt %CK% --split test --num 8 --seed 2003 --out outputs\bspline_carla_160k8\randombatch_latest\test >> %L% 2>&1
"%PY%" -u scripts\random_preview.py --config %CFG% --ckpt %CK% --split test --num 8 --seed 2004 --out outputs\bspline_carla_160k8\randombatch_latest\test >> %L% 2>&1
"%PY%" -u scripts\random_preview.py --config %CFG% --ckpt %CK% --split train --num 8 --seed 3001 --out outputs\bspline_carla_160k8\randombatch_latest\train >> %L% 2>&1
"%PY%" -u scripts\random_preview.py --config %CFG% --ckpt %CK% --split train --num 8 --seed 3002 --out outputs\bspline_carla_160k8\randombatch_latest\train >> %L% 2>&1
"%PY%" -u scripts\random_preview.py --config %CFG% --ckpt %CK% --split train --num 8 --seed 3003 --out outputs\bspline_carla_160k8\randombatch_latest\train >> %L% 2>&1
"%PY%" -u scripts\random_preview.py --config %CFG% --ckpt %CK% --split train --num 8 --seed 3004 --out outputs\bspline_carla_160k8\randombatch_latest\train >> %L% 2>&1
echo ===== BATCH DONE %DATE% %TIME% ===== >> %L%
