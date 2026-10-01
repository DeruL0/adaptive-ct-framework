param(
    [string]$Out = "output\generalization_report.json"
)

$ErrorActionPreference = "Stop"
$Python = "D:\Projects\3.3DGS\SAD-GS\.venv\Scripts\python.exe"
& $Python -m adaptive_ct.generalization --out $Out
