# Keep the PC awake while pipeline jobs run (a sleeping PC froze E016 scoring for 2 h 40 min on 26 Sep).
#   powershell -File scripts/keep_awake.ps1
# Uses SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED): a request held only by this process, no power-plan
# change. The display may still turn off. Exits (and the request lapses) once no pipeline job has run for $IdleMin minutes.
param([int]$IdleMin = 10, [int]$EverySec = 60)

Add-Type -TypeDefinition @'
using System; using System.Runtime.InteropServices;
public static class Awake { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f); }
'@ -ErrorAction SilentlyContinue
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM_REQUIRED = [uint32]"0x00000001"

$pattern = 'src\.er_[a-z0-9_]+'
$lastSeen = Get-Date
[void][Awake]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
"$(Get-Date -Format HH:mm:ss) keep-awake on"
while (((Get-Date) - $lastSeen).TotalMinutes -lt $IdleMin) {
  $jobs = Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -match $pattern }
  if ($jobs) { $lastSeen = Get-Date }
  [void][Awake]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
  Start-Sleep -Seconds $EverySec
}
[void][Awake]::SetThreadExecutionState($ES_CONTINUOUS)
"$(Get-Date -Format HH:mm:ss) keep-awake off (no pipeline job for $IdleMin min)"
