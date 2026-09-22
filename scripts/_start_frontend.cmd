@echo off
cd /d D:\ProjectDirectory\Neural-IRISDiffuser\diffusion-dashboard
echo ==== frontend restart %DATE% %TIME% ==== > frontend.log
npm run dev >> frontend.log 2>&1
