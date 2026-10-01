param(
    [string]$Python = "C:\Users\Derul\AppData\Local\Programs\Python\Python310\python.exe"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path
$Config = "configs/research/industrial_detail_v5_256v_equal_ratio_anisotropic_nodal.yaml"
$Reference = "output/industrial_detail_v5_256v_equal_ratio_anisotropic/checkpoint.pt"
$P5 = "output/industrial_detail_v5_256v_equal_ratio_anisotropic_p5"
$Tied = "output/nodal_p1_256v_equal_ratio_anisotropic_tied"
$Score = "output/nodal_p1_256v_equal_ratio_anisotropic_score30m"
$Stage20 = "output/nodal_p1_256v_equal_ratio_anisotropic_private30m_ft20"
$Stage220 = "output/nodal_p1_256v_equal_ratio_anisotropic_private30m_ft220"
$Final = "output/nodal_p1_256v_equal_ratio_anisotropic_private30m_ft720"

if (-not (Test-Path "$P5/checkpoint.pt")) {
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
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $Python scripts/run_nodal_p1.py `
    --config $Config `
    --checkpoint "$P5/checkpoint.pt" `
    --output $Tied `
    --device cuda `
    --iterations 0 `
    --eval-ray-chunk 65536
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$TiedReport = Get-Content -Raw "$Tied/report.json" | ConvertFrom-Json
$BaseRawBytes = [int64]$TiedReport.compact.raw_payload_bytes

& $Python scripts/score_nodal_private.py `
    --config $Config `
    --reference-checkpoint "$P5/checkpoint.pt" `
    --tied-checkpoint "$Tied/checkpoint.pt" `
    --output $Score `
    --device cuda `
    --batches 2048 `
    --batch-rays 4096 `
    --raw-budget-bytes 30000000 `
    --base-raw-bytes $BaseRawBytes `
    --bytes-per-private-leaf 16 `
    --rank-mode projection_jump
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python scripts/run_nodal_p1.py `
    --config $Config `
    --checkpoint "$P5/checkpoint.pt" `
    --private-actions "$Score/private_actions.pt" `
    --output $Stage20 `
    --device cuda `
    --iterations 20 `
    --batch-rays 4096 `
    --lr 0.001 `
    --eval-ray-chunk 65536
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python scripts/run_nodal_p1.py `
    --config $Config `
    --checkpoint "$Stage20/checkpoint.pt" `
    --already-nodal `
    --output $Stage220 `
    --device cuda `
    --iterations 200 `
    --batch-rays 8192 `
    --lr 0.0005 `
    --eval-ray-chunk 65536
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python scripts/run_nodal_p1.py `
    --config $Config `
    --checkpoint "$Stage220/checkpoint.pt" `
    --already-nodal `
    --output $Final `
    --device cuda `
    --iterations 500 `
    --batch-rays 16384 `
    --lr 0.0002 `
    --eval-ray-chunk 65536
exit $LASTEXITCODE
