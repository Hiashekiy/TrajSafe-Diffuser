@echo off
rem Chapter-4 full pipeline (sweep -> main table -> ablations -> report -> figures).
rem Keep this file ASCII-only: cmd.exe reads .cmd as GBK, so UTF-8 Chinese
rem comments turn into garbage and get executed as a command.
cd /d D:\ProjectDirectory\Neural-IRISDiffuser
"D:\Program Files\Git\usr\bin\bash.exe" -lc "cd /d/ProjectDirectory/Neural-IRISDiffuser && bash scripts/ch4_run_all.sh >> outputs/logs/ch4_run_all.log 2>&1"
