# One JSON object on stdin, one JSON object on stdout.
# This process does not activate Codex Desktop and does not send input.
param()
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if (-not ('CodexDesktop.Native' -as [type])) {
  Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;
namespace CodexDesktop {
  public static class Native {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern SafeFileHandle CreateFile(string fileName, uint access, uint share, IntPtr security, uint disposition, uint flags, IntPtr template);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetNamedPipeServerProcessId(SafeFileHandle pipe, out uint pid);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint pid);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr OpenProcess(uint access, bool inherit, uint pid);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool QueryFullProcessImageName(IntPtr process, int flags, StringBuilder buffer, ref int size);
    [DllImport("kernel32.dll")] private static extern bool CloseHandle(IntPtr handle);
    public static uint ServerPid(string pipePath) {
      using (SafeFileHandle handle = CreateFile(pipePath, 0x80000000 | 0x40000000, 0, IntPtr.Zero, 3, 0, IntPtr.Zero)) {
        if (handle.IsInvalid) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
        uint pid;
        if (!GetNamedPipeServerProcessId(handle, out pid)) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
        return pid;
      }
    }
    public static string Exe(uint pid) {
      IntPtr handle = OpenProcess(0x1000, false, pid);
      if (handle == IntPtr.Zero) return "";
      try {
        StringBuilder buffer = new StringBuilder(1024);
        int size = buffer.Capacity;
        if (!QueryFullProcessImageName(handle, 0, buffer, ref size)) return "";
        return buffer.ToString();
      } finally { CloseHandle(handle); }
    }
  }
}
'@
}
function ConvertTo-Focus {
  $hwnd = [CodexDesktop.Native]::GetForegroundWindow()
  $processId = [uint32]0
  [void][CodexDesktop.Native]::GetWindowThreadProcessId($hwnd, [ref]$processId)
  [pscustomobject]@{
    hwnd = [int64]$hwnd
    pid = [int]$processId
    executable = [CodexDesktop.Native]::Exe($processId)
  }
}
function Write-Result([hashtable]$Payload) {
  $Payload | ConvertTo-Json -Compress -Depth 20
}
$raw = [Console]::In.ReadToEnd()
if (-not $raw) { Write-Result @{ ok = $false; error = 'missing request' }; exit 1 }
$request = $raw | ConvertFrom-Json
if ($request.action -eq 'discover') {
  $pipes = @()
  foreach ($path in [System.IO.Directory]::GetFiles('\\.\pipe\')) {
    if ($path -notmatch '\\codex-app-tools-[A-Za-z0-9-]+$') { continue }
    try {
      $serverPid = [CodexDesktop.Native]::ServerPid($path)
      $pipes += [pscustomobject]@{
        name = $path
        pid = [int]$serverPid
        executable = [CodexDesktop.Native]::Exe($serverPid)
      }
    } catch { }
  }
  Write-Result @{ ok = $true; pipes = $pipes }
  exit 0
}
if ($request.action -ne 'call') { Write-Result @{ ok = $false; error = 'unknown action' }; exit 1 }
if (-not $request.frameBase64) { Write-Result @{ ok = $false; error = 'missing frame' }; exit 1 }
$before = ConvertTo-Focus
$paths = @([System.IO.Directory]::GetFiles('\\.\pipe\') | Where-Object { $_ -match '\\codex-app-tools-[A-Za-z0-9-]+$' })
$chosen = $null
foreach ($path in $paths) {
  try {
    $serverPid = [CodexDesktop.Native]::ServerPid($path)
    $exe = [CodexDesktop.Native]::Exe($serverPid)
    if ($exe -match '\\ChatGPT\.exe$' -and $exe -match '\\WindowsApps\\OpenAI\.Codex_') {
      if ($null -ne $chosen) { Write-Result @{ ok = $false; error = 'more than one Codex Desktop pipe'; focus = @{ before = $before; after = (ConvertTo-Focus) } }; exit 1 }
      $chosen = $path
    }
  } catch { }
}
if ($null -eq $chosen) { Write-Result @{ ok = $false; error = 'Codex Desktop pipe is not running'; focus = @{ before = $before; after = (ConvertTo-Focus) } }; exit 1 }
$pipeName = $chosen.Substring($chosen.LastIndexOf('\') + 1)
if (-not $pipeName) { Write-Result @{ ok = $false; error = 'pipe name was empty'; focus = @{ before = $before; after = (ConvertTo-Focus) } }; exit 1 }
$client = New-Object System.IO.Pipes.NamedPipeClientStream('.', $pipeName, [System.IO.Pipes.PipeDirection]::InOut)
try {
  $client.Connect([int]$request.timeoutMs)
  $bytes = [Convert]::FromBase64String([string]$request.frameBase64)
  if ($bytes.Length -gt 8388608) { throw 'frame is too large' }
  $header = [BitConverter]::GetBytes([uint32]$bytes.Length)
  $client.Write($header, 0, 4)
  $client.Write($bytes, 0, $bytes.Length)
  $prefix = New-Object byte[] 4
  $got = 0
  while ($got -lt 4) { $n = $client.Read($prefix, $got, 4 - $got); if ($n -le 0) { throw 'pipe closed' }; $got += $n }
  $length = [BitConverter]::ToUInt32($prefix, 0)
  if ($length -gt 8388608) { throw 'frame is too large' }
  $body = New-Object byte[] $length
  $got = 0
  while ($got -lt $length) { $n = $client.Read($body, $got, [Math]::Min(65536, $length - $got)); if ($n -le 0) { throw 'pipe closed' }; $got += $n }
} catch {
  Write-Result @{ ok = $false; error = $_.Exception.Message; focus = @{ before = $before; after = (ConvertTo-Focus) } }
  exit 1
} finally {
  if ($client) { $client.Dispose() }
}
Write-Result @{ ok = $true; responseBase64 = [Convert]::ToBase64String($body); focus = @{ before = $before; after = (ConvertTo-Focus) } }
exit 0
