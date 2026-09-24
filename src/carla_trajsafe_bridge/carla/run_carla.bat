@echo off
cd /d E:\CARLA_0.9.16
"E:\CARLA_0.9.16\CarlaUE4.exe" -quality-level=Low -windowed -ResX=800 -ResY=450 -nosound -carla-rpc-port=2000 -stdout > "E:\CarDataSample\outputs\carla_t0056\server.log" 2>&1
echo EXITCODE=%ERRORLEVEL% >> "E:\CarDataSample\outputs\carla_t0056\server.log"