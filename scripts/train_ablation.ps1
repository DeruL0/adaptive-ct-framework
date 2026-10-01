param(
    [string]$Config = "configs\ablations\figure473_48v_hybrid.yaml"
)

$ErrorActionPreference = "Stop"
$Python = "D:\Projects\3.3DGS\SAD-GS\.venv\Scripts\python.exe"
& $Python -m adaptive_ct.train --config $Config

