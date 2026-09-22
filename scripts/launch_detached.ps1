# Launch a long job so that it is NOT a child of the calling shell.
#
# The DSH bash tool tears down the process tree it started when the call ends,
# which kills "nohup ... &" and even "Start-Process" children.  A process
# created through the WMI service (Win32_Process.Create) is parented to
# WmiPrvSE.exe instead, so it survives.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\launch_detached.ps1
#   powershell ... -File scripts\launch_detached.ps1 -CommandLine 'python -c "print(1)"'
[CmdletBinding()]
param(
    [string] $Repo = 'D:\ProjectDirectory\Neural-IRISDiffuser',
    [string] $CommandLine = ''
)
if ([string]::IsNullOrWhiteSpace($CommandLine)) {
    $CommandLine = 'powershell -NoProfile -ExecutionPolicy Bypass -File ' +
                   $Repo + '\scripts\train_160.ps1'
}
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine      = $CommandLine
    CurrentDirectory = $Repo
}
if ($r.ReturnValue -eq 0) {
    Write-Host ("launched  pid=" + $r.ProcessId)
    Write-Host ("cmd       " + $CommandLine)
    Write-Host ("log       " + $Repo + "\logs\train_160.out.log")
} else {
    Write-Host ("FAILED ReturnValue=" + $r.ReturnValue) -ForegroundColor Red
    exit 1
}
