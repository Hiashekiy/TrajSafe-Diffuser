@echo off
REM ===========================================================================
REM  Rebuild the 160 m / k=8 cache with the crop border PROTECTED.
REM
REM  The first k=8 cache (data/carla_processed_160k8) was eroded with scipy's
REM  default border_value=0, which counts everything outside the 256^2 crop as
REM  FREE: the crop's outer wall got eaten from the outside as well and a 5 m
REM  free "moat" opened around every scene (outermost 8 cells went from
REM  0.80-0.90 obstacle to 0.000 in ALL samples).  --border-mode protect keeps
REM  the outer k cells at the source occupancy and treats the outside as
REM  obstacle, so the moat is impossible by construction.
REM
REM  Occupancy changed -> every occupancy-dependent label must be rebuilt:
REM    05 occupancy  ->  01 candidates  ->  02 ellipse labels  ->  04 ALM  ->  03 validate
REM ===========================================================================
cd /d D:\ProjectDirectory\Neural-IRISDiffuser
set PY=E:\CondaEnvData\envs\GGMPC\python.exe
set CFG=configs/config_160k8p.yaml
set OUT=data/carla_processed_160k8p
set LOG=logs\pipeline_160k8p.log

echo ===== START %DATE% %TIME% ===== > %LOG%

echo ===== 05_erode_occupancy (border-mode protect) ===== >> %LOG%
%PY% -u scripts/data/carla_full/05_erode_occupancy.py --source data/carla_processed_160 --out %OUT% --k 8 --border-mode protect >> %LOG% 2>&1
if errorlevel 1 goto fail

echo ===== 01_build_candidates ===== >> %LOG%
%PY% -u scripts/data/carla/01_build_candidates.py --processed %OUT% --config %CFG% --workers 12 >> %LOG% 2>&1
if errorlevel 1 goto fail

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
