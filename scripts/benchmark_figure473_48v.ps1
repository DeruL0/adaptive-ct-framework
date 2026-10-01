param(
    [string]$Config = "configs\figure473_48v.yaml",
    [string]$Checkpoint = "output\figure473_48v\checkpoint.pt",
    [string]$Out = "output\figure473_48v\benchmark_report.json"
)

$ErrorActionPreference = "Stop"
$Python = "D:\Projects\3.3DGS\SAD-GS\.venv\Scripts\python.exe"

& $Python -m adaptive_ct.benchmark `
    --config $Config `
    --checkpoint $Checkpoint `
    --out $Out `
    --baseline "D:\Projects\3.3DGS\SAD-GS\paper\results\fair_comparison_figure473_common96_i10000.json" `
    --baseline "D:\Projects\3.3DGS\SAD-GS\paper\results\material_projection_figure473_48v_i15000_thr010\material_projection_comparison_metrics.json" `
    --baseline "D:\Projects\3.3DGS\SAD-GS\paper\results\mixed_protocol_projection_fandisk_figure_arm_48v_i15000\*.json"

