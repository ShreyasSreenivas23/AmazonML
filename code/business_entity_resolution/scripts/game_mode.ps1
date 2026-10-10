# Game mode: keep a game smooth while pipeline jobs run in the background.
#   powershell -File scripts/game_mode.ps1 [-Game Overwatch] [-LowGB 5] [-HighGB 12]
# Every few seconds: pipeline python processes -> BelowNormal CPU priority (the game always wins the CPU).
# If the game is running and free RAM < LowGB: suspend the pipeline jobs and page their memory out; resume them when
# free RAM > HighGB or the game has closed. Nothing is killed, so no progress is lost. Stop with Ctrl+C / TaskStop;
# on exit it resumes anything it suspended.
param([string]$Game = "Overwatch", [int]$LowGB = 5, [int]$HighGB = 12, [int]$EverySec = 5)

Add-Type -TypeDefinition @'
using System; using System.Runtime.InteropServices;
public static class GameMode {
  [DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h);
  [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);
  [DllImport("psapi.dll")] public static extern bool EmptyWorkingSet(IntPtr h);
}
'@ -ErrorAction SilentlyContinue

$pattern = 'src\.er_[a-z0-9_]+'
$suspended = @{}
function FreeGB { [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1) }
function Jobs { Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -match $pattern } |
    ForEach-Object { Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue } }
function ResumeAll($why) {
  foreach ($p in $suspended.Values) { [void][GameMode]::NtResumeProcess($p.Handle) }
  if ($suspended.Count) { "$(Get-Date -Format HH:mm:ss) RESUMED $($suspended.Count) job procs ($why)" }
  $script:suspended = @{}
}

"$(Get-Date -Format HH:mm:ss) game mode on (game '$Game', suspend below $LowGB GB free)"
try {
  while ($true) {
    $jobs = @(Jobs)
    foreach ($p in $jobs) {
      if ($p.PriorityClass -ne 'BelowNormal' -and $p.PriorityClass -ne 'Idle') {
        try { $p.PriorityClass = 'BelowNormal'; "$(Get-Date -Format HH:mm:ss) priority BelowNormal: pid $($p.Id)" } catch {}
      }
    }
    $gameOn = [bool](Get-Process -Name $Game -ErrorAction SilentlyContinue)
    $free = FreeGB
    if ($gameOn -and $free -lt $LowGB -and $suspended.Count -eq 0 -and $jobs.Count) {
      foreach ($p in $jobs) { [void][GameMode]::NtSuspendProcess($p.Handle); [void][GameMode]::EmptyWorkingSet($p.Handle); $suspended[$p.Id] = $p }
      "$(Get-Date -Format HH:mm:ss) SUSPENDED $($jobs.Count) job procs for the game at $free GB free; now $(FreeGB) GB"
    } elseif ($suspended.Count -and (-not $gameOn -or $free -gt $HighGB)) {
      ResumeAll $(if (-not $gameOn) { "game closed" } else { "$free GB free" })
    }
    Start-Sleep -Seconds $EverySec
  }
} finally { ResumeAll "game mode exit" }
