param(
    [ValidateSet("fandisk", "rocker_arm", "bunny")]
    [string]$Case = "fandisk"
)

$ErrorActionPreference = "Stop"
$Python = "D:\Projects\3.3DGS\SAD-GS\.venv\Scripts\python.exe"
$ConfigMap = @{
    fandisk = "configs\fandisk_48v.yaml"
    rocker_arm = "configs\rocker_arm_48v.yaml"
    bunny = "configs\bunny.yaml"
}

& $Python -m adaptive_ct.train --config $ConfigMap[$Case]

