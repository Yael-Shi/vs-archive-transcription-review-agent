# Identity-based helpers for the isolated Grok pipe experiment.
# This file does not launch Grok Bot.

function Write-ExperimentLog {
  param($State, [string]$Message)
  Add-Content -LiteralPath $State.LogPath -Value ((Get-Date).ToString('o') + ' ' + $Message) -Encoding utf8
}

function Write-Stop {
  param($State, [string]$Reason)
  if (-not $State.StopReason) { $State.StopReason = $Reason }
  Write-ExperimentLog $State ('STOP reason=' + $State.StopReason)
}

function New-ExperimentState {
  param(
    [Parameter(Mandatory=$true)][string]$ProfileDir,
    [Parameter(Mandatory=$true)][string]$LogPath,
    [Parameter(Mandatory=$true)][string]$ExpectedExe,
    [int[]]$BaselinePids = @(),
    [int]$ControllerPid = $PID
  )
  $dir = New-Object System.Collections.ArrayList
  return @{
    ProfileDir = $ProfileDir
    LogPath = $LogPath
    ExpectedExe = $ExpectedExe
    BaselinePids = @($BaselinePids)
    ControllerPid = $ControllerPid
    Owned = $dir
    StopReason = $null
    StopWaitMs = 2000
  }
}

function Test-LaunchAllowed {
  param([bool]$ConfirmLaunch, [string]$ConfirmEnv)
  return ($ConfirmLaunch -and $ConfirmEnv -eq '1')
}

function Get-ProfileSuffix {
  param([string]$ProfileDir)
  $leaf = Split-Path -Leaf $ProfileDir
  if ($leaf -match '^grok-iso-exp-([0-9a-f]{12})$') { return $Matches[1] }
  return $null
}

function Test-SameExe {
  param([string]$Left, [string]$Right)
  if (-not $Left -or -not $Right) { return $false }
  return [string]::Equals($Left, $Right, [StringComparison]::OrdinalIgnoreCase)
}

function Convert-ToIdentity {
  param($Process)
  if (-not $Process) { return $null }
  if (-not $Process.CreationDate) { return $null }
  $exe = [string]$Process.ExecutablePath
  if (-not $exe) { return $null }
  return [pscustomobject]@{
    Pid = [int]$Process.ProcessId
    ParentPid = [int]$Process.ParentProcessId
    CreationTicks = $Process.CreationDate.ToUniversalTime().Ticks
    Executable = $exe
  }
}

function Get-ProcessIdentity {
  param([int]$ProcessId)
  if ($ProcessId -le 0) { return $null }
  $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction SilentlyContinue
  return Convert-ToIdentity $proc
}

function Find-Owned {
  param($State, [int]$ProcessId)
  foreach ($rec in @($State.Owned)) {
    if ([int]$rec.Pid -eq $ProcessId) { return $rec }
  }
  return $null
}

function Test-SameIdentity {
  param($Record, $Process)
  $now = Convert-ToIdentity $Process
  if (-not $Record -or -not $now) { return $false }
  return ($Record.Pid -eq $now.Pid -and $Record.CreationTicks -eq $now.CreationTicks -and (Test-SameExe $Record.Executable $now.Executable))
}

function Test-OwnedDirectory {
  param($State)
  $suffix = Get-ProfileSuffix $State.ProfileDir
  if (-not $suffix) {
    return @{ Abort = $true; Ready = $false; Reason = 'directory not owned' }
  }
  $expected = [IO.Path]::GetFullPath((Join-Path $env:TEMP ('grok-iso-exp-' + $suffix)))
  $given = [IO.Path]::GetFullPath($State.ProfileDir)
  if (-not [string]::Equals($given, $expected, [StringComparison]::OrdinalIgnoreCase)) {
    return @{ Abort = $true; Ready = $false; Reason = 'directory not owned' }
  }
  if (-not (Test-Path -LiteralPath $expected)) {
    return @{ Abort = $true; Ready = $false; Reason = 'directory missing' }
  }
  $item = Get-Item -LiteralPath $expected -Force
  if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
    return @{ Abort = $true; Ready = $false; Reason = 'directory reparse' }
  }
  $markerPath = Join-Path $expected 'EXPERIMENT_MARKER.txt'
  if (-not (Test-Path -LiteralPath $markerPath)) {
    return @{ Abort = $true; Ready = $false; Reason = 'marker missing' }
  }
  $info = Get-Item -LiteralPath $markerPath -Force
  if ($info.Length -gt 64) {
    return @{ Abort = $true; Ready = $false; Reason = 'marker mismatch' }
  }
  $marker = ([IO.File]::ReadAllText($markerPath)).Trim()
  if ($marker -cne $suffix) {
    return @{ Abort = $true; Ready = $false; Reason = 'marker mismatch' }
  }
  return @{ Abort = $false; Ready = $false; Reason = 'directory owned'; Suffix = $suffix }
}

function Get-ProfileStatus {
  param($State)
  $owned = Test-OwnedDirectory $State
  if ($owned.Abort) { return $owned }
  $hasSand = $false
  $sandPath = Join-Path $State.ProfileDir 'sand-data'
  if (Test-Path -LiteralPath $sandPath) {
    $sandItem = Get-Item -LiteralPath $sandPath -Force
    $hasSand = $sandItem.PSIsContainer -and -not ($sandItem.Attributes -band [IO.FileAttributes]::ReparsePoint)
  }
  $rootRec = $null
  if ($State.ContainsKey('RootPid') -and $State.RootPid) {
    $rootRec = Find-Owned $State ([int]$State.RootPid)
  }
  if ($rootRec) {
    $now = Get-ProcessIdentity ([int]$rootRec.Pid)
    if (-not $now) {
      return @{ Abort = $true; Ready = $false; Reason = 'child exited' }
    }
    if ($now.CreationTicks -ne $rootRec.CreationTicks -or -not (Test-SameExe $now.Executable $rootRec.Executable)) {
      return @{ Abort = $true; Ready = $false; Reason = 'pid reuse' }
    }
    if (-not (Test-SameExe $now.Executable $State.ExpectedExe)) {
      return @{ Abort = $true; Ready = $false; Reason = 'executable mismatch' }
    }
  }
  if (-not $hasSand) {
    return @{ Abort = $false; Ready = $false; Reason = 'waiting for isolated profile' }
  }
  if (-not $rootRec) {
    return @{ Abort = $true; Ready = $false; Reason = 'root identity missing' }
  }
  return @{ Abort = $false; Ready = $true; Reason = 'isolated profile confirmed' }
}

function Wait-ProfileReady {
  param($State, [datetime]$Until, [scriptblock]$Sleep = { Start-Sleep -Seconds 2 })
  while ((Get-Date) -lt $Until) {
    $status = Get-ProfileStatus $State
    if ($status.Abort) {
      Write-Stop $State $status.Reason
      return $status
    }
    if ($status.Ready) {
      Write-ExperimentLog $State 'isolated profile confirmed'
      return $status
    }
    Write-ExperimentLog $State 'waiting for isolated profile'
    & $Sleep
  }
  Write-Stop $State 'isolated profile not confirmed'
  return @{ Abort = $true; Ready = $false; Reason = 'isolated profile not confirmed' }
}

function Add-OwnedProcess {
  param($State, [int]$ProcessId)
  if ($ProcessId -eq $State.ControllerPid) {
    return @{ Ok = $false; Reason = 'controller' }
  }
  if ($State.BaselinePids -contains $ProcessId) {
    return @{ Ok = $false; Reason = 'baseline' }
  }
  $identity = Get-ProcessIdentity $ProcessId
  if (-not $identity) {
    return @{ Ok = $false; Reason = 'identity unknown' }
  }
  if (-not (Test-SameExe $identity.Executable $State.ExpectedExe)) {
    return @{ Ok = $false; Reason = 'executable mismatch' }
  }
  $existing = Find-Owned $State $ProcessId
  if ($existing) {
    if ($existing.CreationTicks -ne $identity.CreationTicks -or -not (Test-SameExe $existing.Executable $identity.Executable)) {
      return @{ Ok = $false; Reason = 'pid reuse' }
    }
    return @{ Ok = $true; Reason = 'already tracked' }
  }
  [void]$State.Owned.Add($identity)
  if (-not (Save-OwnedLedger $State)) {
    return @{ Ok = $false; Reason = 'ledger unpublished' }
  }
  return @{ Ok = $true; Reason = 'tracked' }
}

function Save-OwnedLedger {
  param($State)
  if (-not $State.LogPath -or $State.EvidenceId -notmatch '^[0-9a-f]{16}$') { return $false }
  if (@($State.Owned).Count -lt 1) { return $false }
  try {
    $records = @()
    foreach ($rec in @($State.Owned)) {
      $records += [ordered]@{
        pid = [int]$rec.Pid
        creationTicks = [string]$rec.CreationTicks
        executable = [string]$rec.Executable
      }
    }
    $integrity = 'unchecked'
    if ($State.ContainsKey('BaselineIntegrity') -and $State.BaselineIntegrity) {
      $integrity = [string]$State.BaselineIntegrity
    }
    $payload = [ordered]@{
      evidenceId = [string]$State.EvidenceId
      phase = 'owned'
      expectedExe = [string]$State.ExpectedExe
      controllerPid = [int]$State.ControllerPid
      baseline = @($State.BaselinePids | ForEach-Object { [int]$_ })
      baselineIntegrity = $integrity
      owned = @($records)
    }
    $json = ConvertTo-Json -InputObject $payload -Compress -Depth 6
    $path = [string]$State.LogPath + '.run.' + $State.EvidenceId + '.json'
    $tmp = $path + '.tmp'
    [IO.File]::WriteAllText($tmp, $json)
    [IO.File]::Copy($tmp, $path, $true)
    Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
    return $true
  } catch {
    Write-ExperimentLog $State ('ledger save failed ' + $_.Exception.GetType().Name)
    return $false
  }
}

function Update-OwnedChildren {
  param($State)
  $procs = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
  $byPid = @{}
  foreach ($proc in $procs) { $byPid[[int]$proc.ProcessId] = $proc }
  $guard = 0
  while ($guard -lt 20) {
    $guard += 1
    $added = $false
    foreach ($proc in $procs) {
      $childPid = [int]$proc.ProcessId
      if ($childPid -eq $State.ControllerPid) { continue }
      if ($State.BaselinePids -contains $childPid) { continue }
      $parentId = [int]$proc.ParentProcessId
      $parentRec = Find-Owned $State $parentId
      if (-not $parentRec) { continue }
      $parentNow = $byPid[$parentId]
      if (-not (Test-SameIdentity $parentRec $parentNow)) {
        Write-Stop $State 'parent identity changed'
        $null = Save-OwnedLedger $State
        return $false
      }
      $existing = Find-Owned $State $childPid
      $childId = Convert-ToIdentity $proc
      if (-not $childId) {
        Write-ExperimentLog $State ('child identity unknown pid=' + $childPid)
        Write-Stop $State 'child identity unknown'
        $null = Save-OwnedLedger $State
        return $false
      }
      if ($existing) {
        if ($existing.CreationTicks -ne $childId.CreationTicks -or -not (Test-SameExe $existing.Executable $childId.Executable)) {
          Write-Stop $State 'pid reuse'
          $null = Save-OwnedLedger $State
          return $false
        }
        continue
      }
      if (-not (Test-SameExe $childId.Executable $State.ExpectedExe)) {
        Write-ExperimentLog $State ('excluded unexpected executable pid=' + $childPid)
        continue
      }
      [void]$State.Owned.Add($childId)
      $added = $true
    }
    if (-not $added) { break }
  }
  if (-not (Save-OwnedLedger $State)) { return $false }
  return $true
}

function Test-MayStop {
  param($State, $Record, $Current)
  if (-not $Record) { return 'identity unknown' }
  if ([int]$Record.Pid -eq $State.ControllerPid) { return 'controller' }
  if ($State.BaselinePids -contains [int]$Record.Pid) { return 'baseline' }
  if (-not $Current) { return 'already gone' }
  if ($Current.CreationTicks -ne $Record.CreationTicks) { return 'pid reuse' }
  if (-not (Test-SameExe $Current.Executable $Record.Executable)) { return 'executable mismatch' }
  if (-not (Test-SameExe $Current.Executable $State.ExpectedExe)) { return 'executable mismatch' }
  return 'allow'
}

function Test-BaselineIntegrity {
  param($State)
  $failures = @()
  foreach ($id in @($State.BaselinePids)) {
    $baselinePid = [int]$id
    $expected = ''
    if ($State.ContainsKey('BaselineTicks') -and $State.BaselineTicks.ContainsKey($baselinePid)) {
      $expected = [string]$State.BaselineTicks[$baselinePid]
    }
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$baselinePid" -ErrorAction SilentlyContinue
    if (-not $proc) {
      $failures += ($baselinePid.ToString() + ':missing')
      continue
    }
    $ticks = ''
    if ($proc.CreationDate) { $ticks = [string]$proc.CreationDate.ToUniversalTime().Ticks }
    $exe = [string]$proc.ExecutablePath
    if (-not $expected -or $ticks -ne $expected -or -not (Test-SameExe $exe $State.ExpectedExe)) {
      $failures += ($baselinePid.ToString() + ':identity')
    }
  }
  if ($failures.Count -gt 0) {
    Write-ExperimentLog $State ('baseline integrity failed ' + ($failures -join ','))
    $State.BaselineIntegrity = 'failed'
    return $false
  }
  Write-ExperimentLog $State 'baseline integrity intact'
  $State.BaselineIntegrity = 'intact'
  return $true
}

function Publish-BaselineIntegrity {
  param($State)
  if ($State.BaselineIntegrity -ne 'intact' -and $State.BaselineIntegrity -ne 'failed') { return }
  if (-not $State.LogPath -or $State.EvidenceId -notmatch '^[0-9a-f]{16}$') { return }
  $path = [string]$State.LogPath + '.run.' + $State.EvidenceId + '.json'
  if (-not (Test-Path -LiteralPath $path)) { return }
  try {
    $payload = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    $payload | Add-Member -NotePropertyName baselineIntegrity -NotePropertyValue ([string]$State.BaselineIntegrity) -Force
    $json = ConvertTo-Json -InputObject $payload -Compress -Depth 6
    $tmp = $path + '.tmp'
    [IO.File]::WriteAllText($tmp, $json)
    [IO.File]::Copy($tmp, $path, $true)
    Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
  } catch {
    Write-ExperimentLog $State ('baseline integrity unpublished ' + $_.Exception.GetType().Name)
  }
}

function Stop-OwnedProcesses {
  param($State)
  $ids = @($State.Owned | ForEach-Object { [int]$_.Pid })
  Write-ExperimentLog $State ('cleanup candidates: ' + ($(if ($ids.Count) { $ids -join ',' } else { 'none' })))
  $completed = $true
  $action = $State.StopAction
  foreach ($rec in @($State.Owned)) {
    $current = Get-ProcessIdentity ([int]$rec.Pid)
    $decision = Test-MayStop $State $rec $current
    if ($decision -eq 'already gone') {
      Write-ExperimentLog $State ('already gone ' + $rec.Pid)
      continue
    }
    if ($decision -ne 'allow') {
      Write-ExperimentLog $State ('REFUSE ' + $decision + ' ' + $rec.Pid)
      if ($decision -eq 'controller' -or $decision -eq 'baseline') { continue }
      $completed = $false
      continue
    }
    try {
      if ($action) { & $action ([int]$rec.Pid) }
      else { Stop-Process -Id ([int]$rec.Pid) -ErrorAction Stop }
      Write-ExperimentLog $State ('stopped ' + $rec.Pid)
    } catch {
      Write-ExperimentLog $State ('cleanup failed ' + $rec.Pid + ' ' + $_.Exception.GetType().Name)
      $completed = $false
    }
  }
  if (-not $State.StopAction) {
    $pending = @()
    foreach ($rec in @($State.Owned)) {
      $current = Get-ProcessIdentity ([int]$rec.Pid)
      if ((Test-MayStop $State $rec $current) -eq 'allow') { $pending += $rec }
    }
    if ($pending.Count -gt 0) {
      Start-Sleep -Milliseconds $State.StopWaitMs
      foreach ($rec in $pending) {
        $current = Get-ProcessIdentity ([int]$rec.Pid)
        $decision = Test-MayStop $State $rec $current
        if ($decision -eq 'already gone') {
          Write-ExperimentLog $State ('already gone ' + $rec.Pid)
          continue
        }
        if ($decision -ne 'allow') {
          Write-ExperimentLog $State ('REFUSE ' + $decision + ' ' + $rec.Pid)
          $completed = $false
          continue
        }
        try {
          Stop-Process -Id ([int]$rec.Pid) -Force -ErrorAction Stop
          Write-ExperimentLog $State ('force-stopped ' + $rec.Pid)
        } catch {
          Write-ExperimentLog $State ('cleanup failed ' + $rec.Pid + ' ' + $_.Exception.GetType().Name)
          $completed = $false
        }
      }
    }
  }
  Write-ExperimentLog $State ('cleanup completed=' + $completed)
  return @{ Completed = $completed }
}

function Test-LiveDataMention {
  param([string]$CommandLine)
  if (-not $CommandLine) { return $false }
  return ($CommandLine.Contains('\AppData\Roaming\Grok Bot') -or $CommandLine.Contains('\.grokbot'))
}

function Test-OwnedLiveData {
  param($State)
  foreach ($rec in @($State.Owned)) {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$rec.Pid)" -ErrorAction SilentlyContinue
    if (-not $proc) { continue }
    if (-not (Test-SameIdentity $rec $proc)) { return 'pid reuse' }
    if (Test-LiveDataMention ([string]$proc.CommandLine)) { return 'live data path' }
  }
  return $null
}

function Test-OwnedListener {
  param($State)
  foreach ($rec in @($State.Owned)) {
    $current = Get-ProcessIdentity ([int]$rec.Pid)
    if ((Test-MayStop $State $rec $current) -ne 'allow') { continue }
    $hits = @(Get-NetTCPConnection -OwningProcess ([int]$rec.Pid) -State Listen -ErrorAction SilentlyContinue)
    if ($hits.Count -gt 0) { return $true }
  }
  return $false
}

function Resolve-ExperimentStop {
  param(
    [string]$ProfileReason,
    [bool]$ProfileAbort,
    [bool]$BaselineMissing,
    [bool]$IdentityUncertain,
    [bool]$Listener,
    [bool]$ChildExited,
    [bool]$TimedOut,
    [bool]$GotDiscovery
  )
  if ($BaselineMissing) { return 'baseline pid missing' }
  if ($IdentityUncertain) { return 'identity uncertain' }
  if ($ProfileAbort -and $ProfileReason) { return $ProfileReason }
  if ($Listener) { return 'tcp listener' }
  if ($ChildExited) { return 'child exited' }
  if ($TimedOut -and -not $GotDiscovery) { return 'timeout without cdp response' }
  return $null
}

function Send-Cdp {
  param([System.IO.StreamWriter]$Writer, [string]$Json)
  $Writer.Write($Json)
  $Writer.Write([char]0)
  $Writer.Flush()
}

function New-DebugPipes {
  $toBrowser = New-Object System.IO.Pipes.AnonymousPipeServerStream ([System.IO.Pipes.PipeDirection]::Out, [System.IO.HandleInheritability]::Inheritable)
  $fromBrowser = New-Object System.IO.Pipes.AnonymousPipeServerStream ([System.IO.Pipes.PipeDirection]::In, [System.IO.HandleInheritability]::Inheritable)
  return [pscustomobject]@{
    ToBrowser = $toBrowser
    FromBrowser = $fromBrowser
    ReadHandle = $toBrowser.GetClientHandleAsString()
    WriteHandle = $fromBrowser.GetClientHandleAsString()
  }
}

function Complete-Experiment {
  param($State, [string]$Reason)
  if ($Reason) { Write-Stop $State $Reason }
  $result = Stop-OwnedProcesses $State
  Write-ExperimentLog $State 'parent finished'
  return $result
}
