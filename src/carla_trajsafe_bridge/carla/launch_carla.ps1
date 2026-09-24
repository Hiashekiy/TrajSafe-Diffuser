$r = ([wmiclass]"Win32_Process").Create("cmd.exe /c E:\CarDataSample\outputs\carla_t0056\run_carla.bat", "E:\CARLA_0.9.16")
Write-Output ("WMI ReturnValue=" + $r.ReturnValue + " PID=" + $r.ProcessId)