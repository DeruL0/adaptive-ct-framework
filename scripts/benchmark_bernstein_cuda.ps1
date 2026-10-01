param(
    [string]$Config = "configs\figure473_48v.yaml",
    [int]$BatchRays = 2048,
    [int]$Warmup = 5,
    [int]$Iterations = 20,
    [string]$Output = "output\benchmarks\bernstein_cuda.json"
)

$ErrorActionPreference = "Stop"
$Python = "D:\Projects\3.3DGS\SAD-GS\.venv\Scripts\python.exe"

& $Python -m adaptive_ct.benchmark_bernstein_cuda `
    --config $Config `
    --batch-rays $BatchRays `
    --warmup $Warmup `
    --iterations $Iterations `
    --out $Output
