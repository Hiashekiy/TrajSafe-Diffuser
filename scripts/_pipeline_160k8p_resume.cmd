@echo off
REM ===========================================================================
REM  Resume the border-protected 160 m k=8 cache AFTER 05 (erosion) and
REM  01 (candidates) already completed.
REM
REM    data/carla_processed_160k8p : occupancy + candidates done
REM    next                        : 02 ellipse labels -> 04 ALM -> 03 validate
REM
REM  02 and 04 cache per sample under <split>\_cache\, so re-running after an
REM  interruption resumes instead of restarting.
REM ===========================================================================
cd /d D:\ProjectDirectory\Neural-IRISDiffuser
set PY=E:\CondaEnvData\envs\GGMPC\python.exe
set CFG=configs/config_160k8p.yaml
set OUT=data/carla_processed_160k8p
set LOG=logs\pipeline_160k8p_resume.log

echo ===== RESUME START %DATE% %TIME% ===== > %LOG%

echo ===== 02_build_ellipse_labels ===== >> %LOG%
%PY% -u scripts/data/carla/02_build_ellipse_labels.py --processed %OUT% --config %CFG% --workers 12 >> %LOG% 2>&1
if errorlevel 1 goto fail

echo ===== 04_build_alm_constraints ===== >> %LOG%
%PY% -u scripts/data/carla_full/04_build_alm_constraints.py --processed %OUT% --config %CFG% --workers 12 >> %LOG% 2>&1
if errorlevel 1 goto fail

echo ===== 03_validate_processed ===== >> %LOG%
%PY% -u scripts/data/carla/03_validate_processed.py --processed %OUT% --config %CFG% >> %LOG% 2>&1
if errorlevel 1 goto fail

echo ===== ALL STAGES DONE %DATE% %TIME% ===== >> %LOG%
exit /b 0

:fail
echo ===== FAILED exit=%ERRORLEVEL% %DATE% %TIME% ===== >> %LOG%
exit /b 1
