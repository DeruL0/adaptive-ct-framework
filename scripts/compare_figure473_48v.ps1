param(
    [string]$Benchmark = "output\figure473_48v\benchmark_report.json",
    [string]$Baseline = "D:\Projects\3.3DGS\SAD-GS\paper\results\fair_comparison_figure473_common96_i10000.json",
    [string]$Out = "output\figure473_48v\comparison_tables.json"
)

$ErrorActionPreference = "Stop"
$Python = "D:\Projects\3.3DGS\SAD-GS\.venv\Scripts\python.exe"

& $Python -m adaptive_ct.compare --benchmark $Benchmark --baseline $Baseline --out $Out
