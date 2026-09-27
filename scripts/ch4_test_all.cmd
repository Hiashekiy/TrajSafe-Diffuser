@echo off
rem Chapter-4 test-only pipeline (no training).
rem Keep this file ASCII-only and SHORT: cmd.exe reads .cmd as GBK, so UTF-8
rem Chinese comments become garbage and get executed as a command.
rem Env knobs need no passthrough -- just "set FORCE=1" (or SKIP_EFF / SKIP_SWEEP
rem / PLOT_CASES / VAL_SAMPLES) before running this file; the child inherits them.
cd /d D:\ProjectDirectory\Neural-IRISDiffuser
"D:\Program Files\Git\usr\bin\bash.exe" -lc "cd /d/ProjectDirectory/Neural-IRISDiffuser && bash scripts/ch4_test_all.sh"
