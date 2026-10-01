param(
    [string]$Python = "C:\Users\Derul\AppData\Local\Programs\Python\Python310\python.exe"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path
$Config = "configs/research/industrial_detail_v5_256v_equal_ratio_anisotropic_fixedray.yaml"
$Reference = "output/industrial_detail_v5_256v_equal_ratio_anisotropic_fixedray/checkpoint.pt"
$P5 = "output/industrial_detail_v5_256v_equal_ratio_anisotropic_fixedray_p5"

& $Python -m adaptive_ct.train --config $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -m adaptive_ct.functional_compression `
    --config $Config `
    --checkpoint $Reference `
    --r-max-bytes 81750000 `
    --samples-per-axis 4 `
    --max-workspace-mb 2048 `
    --finetune-iterations 15000 `
    --finetune-batch-rays 65536 `
    --finetune-lr 0.015 `
    --eval-ray-chunk 65536 `
    --score-ray-chunk 8192 `
    --score-views 256 `
    --score-rays-per-view 32768 `
    --device cuda `
    --out-checkpoint "$P5/checkpoint.pt" `
    --out-compact "$P5/compact_octree.npz" `
    --compact-quantization float16 `
    --out-report "$P5/compression_report.json"
exit $LASTEXITCODE
