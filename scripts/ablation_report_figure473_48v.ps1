param(
    [string]$Out = "output\figure473_48v_ablation_report.json"
)

$ErrorActionPreference = "Stop"
$Python = "D:\Projects\3.3DGS\SAD-GS\.venv\Scripts\python.exe"

& $Python -m adaptive_ct.ablation `
    --report output\figure473_48v_gradient\training_report.json `
    --report output\figure473_48v_hybrid\training_report.json `
    --report output\figure473_48v_hybrid_volume\training_report.json `
    --out $Out
