# RAM guard: while a protected job runs, suspend a lower-priority job (and page its memory out) when free RAM gets low,
# and resume it once RAM recovers. Nothing is killed, so no progress is lost.
#   powershell -File scripts/ram_guard.ps1 -Victim "er_crossenc train" -Until runs/E015-stage2/exit.json [-AssumeSuspended]
param([string]$Victim = "er_crossenc train", [string]$Until = "runs/E015-stage2/exit.json",
      [int]$LowGB = 4, [int]$HighGB = 14, [int]$EverySec = 10, [switch]$AssumeSuspended)

Add-Type -TypeDefinition @'
using System; using System.Runtime.InteropServices;
public static class Guard {
  [DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h);
  [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);
  [DllImport("psapi.dll")] public static extern bool EmptyWorkingSet(IntPtr h);
}
'@ -ErrorAction SilentlyContinue

$start = Get-Date
$suspended = @()
function FreeGB { [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1) }
function VictimProcs { Get-CimInstance Win32_Process -Filter "name='python.exe'" |
    Where-Object { $_.CommandLine -match [regex]::Escape($Victim) -and $_.CommandLine -notmatch 'monitor\.launch' } |
    ForEach-Object { Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue } }
if ($AssumeSuspended) { $suspended = @(VictimProcs) }   # take over from a previous guard that left the victim suspended

while (-not ((Test-Path $Until) -and ((Get-Item $Until).LastWriteTime -gt $start))) {
  $free = FreeGB
  if ($free -lt $LowGB -and $suspended.Count -eq 0) {
    foreach ($p in VictimProcs) { [void][Guard]::NtSuspendProcess($p.Handle); [void][Guard]::EmptyWorkingSet($p.Handle); $suspended += $p }
    if ($suspended.Count) { "$(Get-Date -Format HH:mm:ss) SUSPENDED $Victim ($($suspended.Count) procs) at $free GB free; now $(FreeGB) GB" }
  } elseif ($free -gt $HighGB -and $suspended.Count -gt 0) {
    foreach ($p in $suspended) { [void][Guard]::NtResumeProcess($p.Handle) }
    "$(Get-Date -Format HH:mm:ss) RESUMED $Victim at $free GB free"
    $suspended = @()
  }
  Start-Sleep -Seconds $EverySec
}
foreach ($p in $suspended) { [void][Guard]::NtResumeProcess($p.Handle) }
if ($suspended.Count) { "$(Get-Date -Format HH:mm:ss) RESUMED $Victim (protected job finished)" }
"$(Get-Date -Format HH:mm:ss) guard exit"
