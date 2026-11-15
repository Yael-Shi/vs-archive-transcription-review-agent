# One isolated Grok Bot process and a line protocol in front of harness.ps1.
# CDP writes happen only after the browser pipe has been idle, matching the
# pilot. This host does not delete a bot or open a TCP debug port.
param(
  [switch]$ConfirmLaunch,
  [string]$ProfileDir = 'C:\Users\User\AppData\Local\Grok Bot Isolated Pilot',
  [string]$ExePath = 'C:\Users\User\AppData\Local\Programs\Grok Bot\Grok Bot.exe',
  [string]$LiveSettings = 'C:\Users\User\.grokbot\settings.json',
  [string]$LogPath = '',
  [string]$EvidenceId = '',
  [switch]$ProtocolSelfTest
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# The Windows PowerShell console writer uses a legacy code page and replaces
# characters it cannot encode with "?". The CDP pipe is already UTF-8. This
# protocol stream is the other boundary, in both directions.
$script:Utf8Protocol = New-Object System.Text.UTF8Encoding $false
$script:ProtocolIn = New-Object System.IO.StreamReader([Console]::OpenStandardInput(), $script:Utf8Protocol)
$script:ProtocolOut = New-Object System.IO.StreamWriter([Console]::OpenStandardOutput(), $script:Utf8Protocol)
$script:ProtocolOut.AutoFlush = $true

function Write-Protocol([string]$Json) { $script:ProtocolOut.WriteLine($Json) }
function Read-ProtocolLine { return $script:ProtocolIn.ReadLine() }

if ($ProtocolSelfTest) {
  $line = Read-ProtocolLine
  if ($null -eq $line) { exit 2 }
  $cmd = $line | ConvertFrom-Json
  $value = $cmd.value
  $out = @{ id = 1; ok = $true; value = $value } | ConvertTo-Json -Compress -Depth 12
  Write-Protocol $out
  exit 0
}

. (Join-Path $PSScriptRoot 'harness.ps1')

if (-not $LogPath) { $LogPath = Join-Path $env:TEMP 'grok-desktop-host.log' }
if (Test-Path -LiteralPath $LogPath) { Remove-Item -LiteralPath $LogPath -Force }
New-Item -ItemType File -Path $LogPath | Out-Null

function Test-StopRequested {
  $path = $LogPath + '.stop'
  if (-not (Test-Path -LiteralPath $path)) { return $false }
  try { $text = [IO.File]::ReadAllText($path).Trim() } catch { return $false }
  return ($text -eq $EvidenceId)
}

function Publish-RunPhase([string]$Phase) {
  $payload = [ordered]@{ evidenceId = $EvidenceId; phase = $Phase }
  $json = ConvertTo-Json -InputObject $payload -Compress -Depth 4
  $path = $LogPath + '.run.' + $EvidenceId + '.json'
  $tmp = $path + '.tmp'
  [IO.File]::WriteAllText($tmp, $json)
  [IO.File]::Copy($tmp, $path, $true)
  Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
}

function Write-Refusal([string]$ErrorText) {
  try {
    Add-Content -LiteralPath $LogPath -Value ((Get-Date).ToString('o') + ' refusal=' + $ErrorText) -Encoding utf8
  } catch { }
  Write-Protocol ('{"ready":false,"error":"' + $ErrorText + '"}')
}

function Read-Perm([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) { return 'absent' }
  $text = [IO.File]::ReadAllText($Path)
  $found = [regex]::Matches($text, '"localToolPermission"\s*:\s*"(always|ask|never)"')
  if ($found.Count -ne 1) { return 'unreadable' }
  return $found[0].Groups[1].Value
}

function Get-NormalSnapshot {
  $map = @{}
  $all = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'Grok Bot.exe' -and $_.CommandLine -like '*\AppData\Roaming\Grok Bot*'
  })
  foreach ($proc in $all) {
    $map[[int]$proc.ProcessId] = $proc.CreationDate.ToUniversalTime().Ticks
  }
  return $map
}

if ($EvidenceId -notmatch '^[0-9a-f]{16}$') {
  Write-Refusal 'evidence id rejected'
  exit 2
}
Publish-RunPhase 'prelaunch'

$profileFull = [IO.Path]::GetFullPath($ProfileDir)
if ($profileFull -match '\\AppData\\Roaming\\Grok Bot' -or $profileFull -match '\\\.grokbot') {
  Write-Refusal 'normal profile'
  exit 2
}
$state = New-ExperimentState -ProfileDir $profileFull -LogPath $LogPath -ExpectedExe $ExePath -BaselinePids @() -ControllerPid $PID
$state.EvidenceId = $EvidenceId
$normalBefore = Get-NormalSnapshot
$state.BaselinePids = @($normalBefore.Keys)
$state.BaselineTicks = $normalBefore
$state.BaselineIntegrity = 'unchecked'

if (-not (Test-LaunchAllowed -ConfirmLaunch ([bool]$ConfirmLaunch) -ConfirmEnv ([string]$env:GROK_ISO_CONFIRM))) {
  Write-Refusal 'launch not confirmed'
  exit 2
}
$markerPath = Join-Path $profileFull 'pilot-marker.txt'
if (-not (Test-Path -LiteralPath $markerPath)) { Write-Refusal 'marker missing'; exit 2 }
$marker = ([IO.File]::ReadAllText($markerPath)).Trim()
if ($marker -cne 'grok-bot-isolated-pilot') { Write-Refusal 'marker mismatch'; exit 2 }
if ((Get-Item -LiteralPath $profileFull).Attributes -band [IO.FileAttributes]::ReparsePoint) {
  Write-Refusal 'profile reparse'; exit 2
}
$pilotPerm = Read-Perm (Join-Path $profileFull 'sand-data\settings.json')
$livePerm = Read-Perm $LiveSettings
Write-ExperimentLog $state "preflight pilotPerm=$pilotPerm livePerm=$livePerm"
if ($pilotPerm -ne 'never' -or $livePerm -ne 'ask') {
  Write-Protocol '{"ready":false,"error":"permission preflight mismatch"}'; exit 3
}
$busy = @(Get-CimInstance Win32_Process | Where-Object {
  $_.Name -eq 'Grok Bot.exe' -and $_.CommandLine -like '*Grok Bot Isolated Pilot*'
})
if ($busy.Count -gt 0) { Write-Protocol '{"ready":false,"error":"isolated instance already open"}'; exit 3 }

$pipes = $null
$writer = $null
$proc = $null
$script:carry = New-Object System.Collections.Generic.List[byte]
$script:readTask = $null
$script:readBuf = $null
$script:sid = ''
$script:nextId = 10

$script:texts = New-Object System.Collections.Generic.List[string]

function Read-CdpTexts([int]$WaitMs) {
  $script:texts.Clear()
  if ($null -eq $script:readTask) {
    $script:readBuf = New-Object byte[] 16384
    $script:readTask = $pipes.FromBrowser.ReadAsync($script:readBuf, 0, $script:readBuf.Length)
  }
  if (-not $script:readTask.Wait($WaitMs)) { return }
  if ($script:readTask.IsFaulted) { throw 'pipe read fault' }
  $count = [int]$script:readTask.Result
  $script:readTask = $null
  if ($count -le 0) { throw 'pipe closed' }
  for ($i = 0; $i -lt $count; $i++) { [void]$script:carry.Add($script:readBuf[$i]) }
  while ($true) {
    $zero = $script:carry.IndexOf([byte]0)
    if ($zero -lt 0) { break }
    $bytes = New-Object byte[] $zero
    for ($i = 0; $i -lt $zero; $i++) { $bytes[$i] = $script:carry[$i] }
    $script:carry.RemoveRange(0, $zero + 1)
    if ($bytes.Length -gt 0) { [void]$script:texts.Add([Text.Encoding]::UTF8.GetString($bytes)) }
  }
}

function Wait-CdpId([int]$Id, [int]$Seconds) {
  $deadline = (Get-Date).AddSeconds($Seconds)
  while ((Get-Date) -lt $deadline) {
    if (Test-StopRequested) { throw 'stop requested' }
    Read-CdpTexts 500
    foreach ($text in @($script:texts.ToArray())) {
      if (-not $text.StartsWith('{')) { continue }
      try { $obj = $text | ConvertFrom-Json } catch { continue }
      $idText = ''
      if ($obj.PSObject.Properties.Name -contains 'id') { $idText = [string]$obj.id }
      if ($idText -eq [string]$Id) { return $obj }
    }
  }
  return $null
}

function Drain-Idle {
  $quiet = 0
  while ($quiet -lt 2) {
    if (Test-StopRequested) { throw 'stop requested' }
    Read-CdpTexts 200
    if ($script:texts.Count -eq 0) { $quiet += 1 } else { $quiet = 0 }
  }
}

try {
  $pipes = New-DebugPipes
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = $ExePath
  $psi.Arguments = "--remote-debugging-pipe --remote-debugging-io-pipes=$($pipes.ReadHandle),$($pipes.WriteHandle)"
  $psi.UseShellExecute = $false
  $psi.WorkingDirectory = $env:TEMP
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.EnvironmentVariables['SAND_USER_DATA_DIR'] = $profileFull
  if ($psi.EnvironmentVariables.ContainsKey('SAND_DATA_ROOT')) { $psi.EnvironmentVariables.Remove('SAND_DATA_ROOT') }
  # Child process only. Grok Bot 1.0.1 copies SAND_DISABLE_UPDATES into
  # environment.updatesDisabled before the updater is constructed. This does
  # not change the parent environment or the normal profile.
  $psi.EnvironmentVariables['SAND_DISABLE_UPDATES'] = '1'
  $proc = New-Object System.Diagnostics.Process
  $proc.StartInfo = $psi
  Publish-RunPhase 'launching'
  [void]$proc.Start()
  $pipes.ToBrowser.DisposeLocalCopyOfClientHandle()
  $pipes.FromBrowser.DisposeLocalCopyOfClientHandle()
  $script:childOut = [IO.File]::Create($LogPath + '.child-out')
  $script:childErr = [IO.File]::Create($LogPath + '.child-err')
  [void]$proc.StandardOutput.BaseStream.CopyToAsync($script:childOut)
  [void]$proc.StandardError.BaseStream.CopyToAsync($script:childErr)
  $added = Add-OwnedProcess $state ([int]$proc.Id)
  if (-not $added.Ok) { throw ('identity uncertain ' + $added.Reason) }
  $state.RootPid = [int]$proc.Id
  $readyUntil = (Get-Date).AddSeconds(25)
  $ready = $false
  do {
    if (Test-StopRequested) { throw 'stop requested' }
    Start-Sleep -Milliseconds 500
    if ($proc.HasExited) { throw 'child exited' }
    if (-not (Update-OwnedChildren $state)) { throw 'owned children' }
    $kids = @(Get-CimInstance Win32_Process | Where-Object {
      $_.ParentProcessId -eq $proc.Id -and $_.CommandLine -like '*Grok Bot Isolated Pilot*'
    })
    if ($kids.Count -ge 1 -and (Test-Path -LiteralPath (Join-Path $profileFull 'sand-data'))) { $ready = $true; break }
  } while ((Get-Date) -lt $readyUntil)
  if (-not $ready) { throw 'isolated profile not confirmed' }
  $liveData = Test-OwnedLiveData $state
  if ($liveData) { throw $liveData }
  $badNormal = $null
  $normalNow = Get-NormalSnapshot
  foreach ($id in $normalBefore.Keys) {
    if (-not $normalNow.ContainsKey($id) -or $normalNow[$id] -ne $normalBefore[$id]) { $badNormal = "normal changed $id"; break }
  }
  if ($badNormal) { throw $badNormal }
  if (Test-OwnedListener $state) { throw 'tcp listener' }

  $utf8 = New-Object System.Text.UTF8Encoding $false
  $writer = New-Object System.IO.StreamWriter($pipes.ToBrowser, $utf8)
  $writer.AutoFlush = $true
  $match = @()
  foreach ($attempt in 1..3) {
    Drain-Idle
    Send-Cdp $writer '{"id":1,"method":"Target.getTargets"}'
    $targets = Wait-CdpId 1 30
    if (-not $targets -or -not $targets.result.targetInfos) { throw 'no targets' }
    $match = @($targets.result.targetInfos | Where-Object { $_.type -eq 'page' -and [string]$_.url -like '*renderer/index.html*' })
    if ($match.Count -ge 1) { break }
    Write-ExperimentLog $state ("renderer missing attempt=$attempt")
  }
  if ($match.Count -lt 1) { throw 'renderer missing' }
  $targetId = [string]$match[0].targetId
  if ($targetId -notmatch '^[A-Za-z0-9._-]{1,80}$') { throw 'renderer target rejected' }
  Drain-Idle
  $attach = @{ id = 2; method = 'Target.attachToTarget'; params = @{ targetId = $targetId; flatten = $true } }
  Send-Cdp $writer ($attach | ConvertTo-Json -Compress -Depth 6)
  $attached = Wait-CdpId 2 30
  if (-not $attached -or -not $attached.result.sessionId) { throw 'attach failed' }
  $script:sid = [string]$attached.result.sessionId
  if ($script:sid -notmatch '^[A-Za-z0-9._-]{1,80}$') { throw 'session rejected' }
  Write-Protocol '{"ready":true}'

  while ($true) {
    $line = Read-ProtocolLine
    if ($null -eq $line) { break }
    if (-not $line.Trim()) { continue }
    $cmd = $line | ConvertFrom-Json
    $op = [string]$cmd.op
    if ($op -eq 'close') { break }
    $commandId = $cmd.id
    if ($op -eq 'eval') {
      $script:nextId += 1
      $cdpId = $script:nextId
      $payload = @{ id = $cdpId; sessionId = $script:sid; method = 'Runtime.evaluate'; params = @{ expression = [string]$cmd.expression; returnByValue = $true } }
      Drain-Idle
      Send-Cdp $writer ($payload | ConvertTo-Json -Compress -Depth 8)
      $response = Wait-CdpId $cdpId 30
      if (-not $response) { Write-Protocol ((@{ id = $commandId; ok = $false; error = 'timeout' } | ConvertTo-Json -Compress)); continue }
      $value = $null
      if ($response.result.result) { $value = $response.result.result.value }
      $out = @{ id = $commandId; ok = $true; value = $value } | ConvertTo-Json -Compress -Depth 12
      Write-Protocol $out
    } elseif ($op -eq 'insert') {
      $script:nextId += 1
      $cdpId = $script:nextId
      $payload = @{ id = $cdpId; sessionId = $script:sid; method = 'Input.insertText'; params = @{ text = [string]$cmd.text } }
      Drain-Idle
      Send-Cdp $writer ($payload | ConvertTo-Json -Compress -Depth 6)
      $response = Wait-CdpId $cdpId 30
      if (-not $response) { Write-Protocol ((@{ id = $commandId; ok = $false; error = 'timeout' } | ConvertTo-Json -Compress)); continue }
      Write-Protocol ((@{ id = $commandId; ok = $true } | ConvertTo-Json -Compress))
    } else {
      Write-Protocol ((@{ id = $commandId; ok = $false; error = 'unknown op' } | ConvertTo-Json -Compress))
    }
  }
} catch {
  Write-ExperimentLog $state ('host exception ' + $_.Exception.GetType().Name)
  if (-not $script:sid) { Write-Protocol ('{"ready":false,"error":"' + $_.Exception.GetType().Name + '"}') }
} finally {
  if ($writer -and $proc -and -not $proc.HasExited) {
    try { Send-Cdp $writer '{"id":90,"method":"Browser.close"}' } catch { Write-ExperimentLog $state 'close send failed' }
    $closeUntil = (Get-Date).AddSeconds(15)
    while ((Get-Date) -lt $closeUntil -and -not $proc.HasExited) { Start-Sleep -Milliseconds 300 }
  }
  if ($writer) { try { $writer.Dispose() } catch { } }
  if ($pipes) {
    try { $pipes.ToBrowser.Dispose() } catch { }
    try { $pipes.FromBrowser.Dispose() } catch { }
  }
  try { [void](Test-BaselineIntegrity $state) } catch {
    $state.BaselineIntegrity = 'failed'
    Write-ExperimentLog $state ('baseline integrity failed check ' + $_.Exception.GetType().Name)
  }
  [void](Update-OwnedChildren $state)
  [void](Stop-OwnedProcesses $state)
  Publish-BaselineIntegrity $state
  $pilotAfter = Read-Perm (Join-Path $profileFull 'sand-data\settings.json')
  $liveAfter = Read-Perm $LiveSettings
  Write-ExperimentLog $state "after pilotPerm=$pilotAfter livePerm=$liveAfter profile=$((Test-Path -LiteralPath $profileFull))"
  Write-ExperimentLog $state 'parent finished'
}
exit 0
