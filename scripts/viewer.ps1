param(
    [string]$Config = "configs\stage7\p1.yaml",
    [string]$Checkpoint = "",
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8765,
    [switch]$Open
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$ArgsList = @("-m", "adaptive_ct.viewer", "--config", $Config, "--host", $HostName, "--port", "$Port")
if ($Checkpoint -ne "") {
    $ArgsList += @("--checkpoint", $Checkpoint)
}
if ($Open) {
    $ArgsList += "--open"
}

& $Python @ArgsList

