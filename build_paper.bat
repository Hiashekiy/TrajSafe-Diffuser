@echo off
setlocal
title TrajSafe-Diffuser / RA-L - build
cd /d "%~dp0"

echo ============================================================
echo   TrajSafe-Diffuser  -  IEEE RA-L paper build
echo   (pdflatex -> bibtex -> pdflatex -> pdflatex)
echo ============================================================
echo.

where pdflatex >nul 2>&1
if errorlevel 1 (
  echo  [FAIL] pdflatex not found on PATH.
  echo         expected: C:\Users\zz\AppData\Roaming\TinyTeX\bin\windows
  goto :end
)

REM ---------------------------------------------------------------
REM  0. sync source figures into the LaTeX project
REM ---------------------------------------------------------------
echo [0/4] syncing figures ...
python scripts\paper\sync_figures.py
if errorlevel 1 echo   [WARN] figure sync reported a problem
echo.

cd paper_ral_english

REM ---------------------------------------------------------------
REM  pick a job name: normally "main"; if main.pdf is held open by a
REM  PDF viewer we build "main_build" instead of failing.
REM ---------------------------------------------------------------
set "JOB=main"
if exist main.pdf (
  del /q main.pdf >nul 2>&1
  if exist main.pdf (
    set "JOB=main_build"
    echo  [NOTE] main.pdf is open in a viewer - building as main_build.pdf
    echo.
  )
)

for %%e in (aux bbl blg log out) do (
  if exist main.%%e        del /q main.%%e        >nul 2>&1
  if exist main_build.%%e  del /q main_build.%%e  >nul 2>&1
)
if "%JOB%"=="main_build" if exist main_build.pdf del /q main_build.pdf >nul 2>&1

echo [1/4] pdflatex  pass 1 ...
pdflatex -interaction=nonstopmode -jobname=%JOB% main.tex >nul
if not exist %JOB%.pdf goto :fail

echo [2/4] bibtex ...
bibtex %JOB% >nul

echo [3/4] pdflatex  pass 2 ...
pdflatex -interaction=nonstopmode -jobname=%JOB% main.tex >nul

echo [4/4] pdflatex  pass 3 ...
pdflatex -interaction=nonstopmode -jobname=%JOB% main.tex >nul
if not exist %JOB%.pdf goto :fail

REM ---------------------------------------------------------------
REM  report
REM ---------------------------------------------------------------
echo.
echo ------------------------- result -------------------------
findstr /c:"Output written" %JOB%.log
echo.
set ERR=0
set OVF=0
findstr /c:"! LaTeX Error" %JOB%.log >nul && set ERR=1
findstr /c:"! Package"     %JOB%.log >nul && set ERR=1
findstr /c:"Citation"      %JOB%.log >nul && set ERR=1
findstr /c:"undefined"     %JOB%.log >nul && set ERR=1
findstr /c:"Overfull"      %JOB%.log >nul && set OVF=1
if "%ERR%"=="1" echo  [WARN] log has errors or undefined citations/refs
if "%OVF%"=="1" echo  [WARN] log has overfull boxes
if "%ERR%%OVF%"=="00" echo  [OK]   log is clean (no errors, no overfull)
echo.

if not exist output mkdir output
copy /y %JOB%.pdf "output\TrajSafe_Diffuser_RAL.pdf" >nul 2>&1
if errorlevel 1 (
  copy /y %JOB%.pdf "output\TrajSafe_Diffuser_RAL_latest.pdf" >nul
  echo  [WARN] output PDF is locked - wrote output\TrajSafe_Diffuser_RAL_latest.pdf
) else (
  echo  [OK]   output\TrajSafe_Diffuser_RAL.pdf updated
)

cd ..
echo.
echo Done.  PDF: paper_ral_english\output\TrajSafe_Diffuser_RAL.pdf
echo        log: paper_ral_english\%JOB%.log
goto :end

:fail
cd ..
echo.
echo  [FAIL] compile failed - open paper_ral_english\%JOB%.log and search for "! "
goto :end

:end
echo.
echo Press any key to close this window.
pause >nul
