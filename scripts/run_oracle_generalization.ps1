param(
    [ValidateSet("all", "fandisk", "rocker_arm", "bunny")]
    [string]$Case = "all"
)

$ErrorActionPreference = "Stop"
$Python = "D:\Projects\3.3DGS\SAD-GS\.venv\Scripts\python.exe"
$ConfigMap = [ordered]@{
    fandisk = @{ Config = "configs\fandisk_48v.yaml"; Out = "output\fandisk_48v\oracle.json" }
    rocker_arm = @{ Config = "configs\rocker_arm_48v.yaml"; Out = "output\rocker_arm_48v\oracle.json" }
    bunny = @{ Config = "configs\bunny.yaml"; Out = "output\bunny\oracle.json" }
}

foreach ($name in $ConfigMap.Keys) {
    if ($Case -ne "all" -and $Case -ne $name) {
        continue
    }
    $entry = $ConfigMap[$name]
    Write-Host "Running oracle for $name..."
    & $Python -m adaptive_ct.oracle --config $entry.Config --out $entry.Out
}
