# Pause or resume every running pipeline job (training, embedding, scoring) without losing progress.
#   powershell -File scripts/pause_jobs.ps1 pause
#   powershell -File scripts/pause_jobs.ps1 resume
#   powershell -File scripts/pause_jobs.ps1 status
# Suspended processes keep their RAM and GPU memory but use no CPU/GPU time. Chain scripts just keep waiting.
param([ValidateSet("pause", "resume", "status")][string]$Action = "status")

Add-Type -TypeDefinition @'
using System; using System.Runtime.InteropServices;
public static class JobCtl {
  [DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h);
  [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);
}
'@ -ErrorAction SilentlyContinue

$pattern = 'src\.er_[a-z0-9_]+|scripts/recall_gain'
$jobs = Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -match $pattern -and $_.CommandLine -notmatch 'monitor\.launch' }
if (-not $jobs) { "no pipeline jobs running"; exit 0 }
foreach ($j in $jobs) {
  $p = Get-Process -Id $j.ProcessId -ErrorAction SilentlyContinue
  if (-not $p) { continue }
  $what = ($j.CommandLine -replace '^.*?-m ', '').Substring(0, [Math]::Min(70, ($j.CommandLine -replace '^.*?-m ', '').Length))
  switch ($Action) {
    "pause"  { $r = [JobCtl]::NtSuspendProcess($p.Handle); "paused  $($j.ProcessId) (status $r)  $what" }
    "resume" { $r = [JobCtl]::NtResumeProcess($p.Handle);  "resumed $($j.ProcessId) (status $r)  $what" }
    "status" { $t = ($p.Threads | Where-Object { $_.WaitReason -eq 'Suspended' }).Count
               "{0} {1,-9} {2:N1} GB  {3}" -f $j.ProcessId, ($(if ($t -eq $p.Threads.Count) { 'SUSPENDED' } else { 'running' })), ($p.WorkingSet64 / 1GB), $what }
  }
}
