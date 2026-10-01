param(
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $connections) {
    Write-Host "No adaptive CT viewer listener found on port $Port."
    exit 0
}

foreach ($connection in $connections) {
    $pidValue = [int]$connection.OwningProcess
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue"
    if ($process.CommandLine -and $process.CommandLine.Contains("adaptive_ct.viewer")) {
        Stop-Process -Id $pidValue -Force
        Write-Host "Stopped adaptive CT viewer process $pidValue on port $Port."
    } else {
        Write-Host "Skipping process $pidValue on port $Port because it is not adaptive_ct.viewer."
    }
}

