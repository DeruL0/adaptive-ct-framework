# 96-view equal-ratio anisotropic nodal test

Date: 2026-07-05

## Spatial levels

The anisotropic grid follows the source-volume aspect ratio:

```text
[22, 48, 48]
[44, 96, 96]
[88, 192, 192]
[176, 384, 384]
```

Finest-axis compression:

```text
x: 473 / 176 = 2.688
y: 1024 / 384 = 2.667
z: 1024 / 384 = 2.667
```

Configuration:
`configs/research/industrial_detail_v5_96v_equal_ratio_anisotropic.yaml`

## Pipeline

1. 25k anisotropic h-only reconstruction.
2. p5 projection-domain keep/p0/p1 compression.
3. 15k fixed-topology p5 coefficient finetuning.
4. 2:1 balance.
5. Shared-vertex nodal p1 conversion.
6. 3,145,728 training rays for projection-domain private-leaf scoring.
7. Select 176,563 private leaves under the 30 MB raw target.
8. Fixed nodal topology finetuning:
   - 20 steps, 4096 rays, `lr=1e-3`;
   - 200 steps, 8192 rays, `lr=5e-4`;
   - 500 steps, 16384 rays, `lr=2e-4`.
9. Float16 compact nodal v4 export.

## Results

| Metric | Old anisotropic h-only | Anisotropic p5 | Anisotropic nodal |
|---|---:|---:|---:|
| Test projection PSNR | 42.6074 dB | 43.8810 dB | 43.9668 dB |
| Test projection MAE | 0.0020350 | 0.0017879 | 0.0018533 |
| Leaves | 5,708,529 | 1,850,724 | 2,025,290 |
| Coefficients/masters | 5,708,529 | 3,765,812 | 2,926,140 |
| Compact file | 19,315,504 B | 9,816,350 B | 8,141,784 B |
| 15-slice PSNR | 30.2396 dB | 30.5816 dB | 30.8889 dB |
| 15-slice MAE | 0.0130779 | 0.0128023 | 0.0123164 |
| Top-1% gradient PSNR | 18.3604 dB | 19.1903 dB | 19.4503 dB |
| Top-1% gradient MAE | 0.0605713 | 0.0554322 | 0.0542965 |
| Gradient ratio | 0.4120 | 0.3976 | 0.3411 |

## Axis-wise comparison against cubic nodal

| Axis | Cubic nodal PSNR | Anisotropic nodal PSNR | Difference |
|---|---:|---:|---:|
| x | 32.5512 dB | 32.6594 dB | +0.1081 dB |
| y | 31.8960 dB | 32.1302 dB | +0.2343 dB |
| z | 27.7658 dB | 27.8770 dB | +0.1113 dB |

Overall comparison:

| Metric | Cubic nodal | Anisotropic nodal |
|---|---:|---:|
| Test projection PSNR | 43.2965 dB | 43.9668 dB |
| Test projection MAE | 0.0019026 | 0.0018533 |
| Compact file | 8,419,206 B | 8,141,784 B |
| 15-slice PSNR | 30.7377 dB | 30.8889 dB |
| Top-1% gradient PSNR | 19.1808 dB | 19.4503 dB |
| Gradient ratio | 0.3259 | 0.3411 |

## Conclusion

The previous negative anisotropic conclusion applied only to the h-only p0
checkpoint. Under the complete p5 plus nodal pipeline, equal-ratio anisotropic
levels improve projection accuracy, all three slice axes, high-gradient value
accuracy, parameter count, and compact file size relative to the cubic nodal
model.

The remaining limitation is edge amplitude: the nodal result is more
value-accurate but its mean gradient ratio remains below both h-only and p5.

Selected output:

```text
output/nodal_p1_96v_equal_ratio_anisotropic_private30m_ft720/
```
