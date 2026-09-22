@echo off
cd /d D:\ProjectDirectory\Neural-IRISDiffuser
set PY=E:\CondaEnvData\envs\GGMPC\python.exe
set CFG=configs/config_160k8.yaml
set OUT=data/carla_processed_160k8
set LOG=logs\pipeline_160k8.log

echo ===== START %DATE% %TIME% ===== > %LOG%

echo ===== 02_build_ellipse_labels ===== >> %LOG%
"%PY%" -u scripts\data\carla\02_build_ellipse_labels.py --processed %OUT% --config %CFG% --workers 12 >> %LOG% 2>&1
if errorlevel 1 ( echo 02_FAILED >> %LOG% & goto :fail )

echo ===== 04_build_alm_constraints ===== >> %LOG%
"%PY%" -u scripts\data\carla_full\04_build_alm_constraints.py --processed %OUT% --config %CFG% --workers 12 >> %LOG% 2>&1
if errorlevel 1 ( echo 04_FAILED >> %LOG% & goto :fail )

echo ===== 03_validate_processed ===== >> %LOG%
"%PY%" -u scripts\data\carla\03_validate_processed.py --processed %OUT% --config %CFG% >> %LOG% 2>&1
if errorlevel 1 ( echo VALIDATION_FAILED_not_training >> %LOG% & goto :fail )

echo ===== train_160k8 ===== >> %LOG%
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train_160.ps1 -Config %CFG% -LogName train_160k8.out.log
echo ===== ALL_DONE %DATE% %TIME% ===== >> %LOG%
goto :eof
:fail
echo ===== ABORTED %DATE% %TIME% ===== >> %LOG%
