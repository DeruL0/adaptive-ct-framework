# Reversible nodal coarse-to-fine trial

Date: 2026-07-05

## Scope

This trial changes only the structural optimisation schedule. It preserves the
96-view equal-ratio anisotropic octree, constrained nodal p1 representation,
projection loss, 30 MB raw budget, and compact nodal v4 schema.

Each scale performs:

1. coefficient optimisation at the current detector scale;
2. projection-domain rescoring from a fully tied reference;
3. up to 8,192 paired `retie + untie` swaps under the fixed private-leaf budget;
4. reconstruction of the nodal constraint graph;
5. acceptance on a fixed full-resolution training-ray batch.

Rejected rounds do not overwrite the last accepted checkpoint.

## Run

```powershell
python -m scripts.run_nodal_coarse_to_fine `
  --config configs/research/industrial_detail_v5_96v_equal_ratio_anisotropic.yaml `
  --checkpoint output/nodal_p1_96v_equal_ratio_anisotropic_private30m_ft720/checkpoint.pt `
  --output output/nodal_p1_96v_equal_ratio_anisotropic_c2f_v1 `
  --scales 4,2,1 `
  --score-rays-per-view 32768 `
  --max-swaps-per-round 8192
```

## Result

| Scale | Added | Retied | Full-resolution loss before | Candidate loss | Accepted |
|---:|---:|---:|---:|---:|:---:|
| 4 | 8,192 | 8,192 | 2.511895e-6 | 2.746713e-6 | no |
| 2 | 8,192 | 8,192 | 2.511895e-6 | 2.710738e-6 | no |
| 1 | 8,192 | 8,192 | 2.511895e-6 | 2.715001e-6 | no |

All three structural proposals were rolled back. The final output is therefore
the previous accepted anisotropic nodal model:

- test projection PSNR: 43.966829 dB;
- test projection MAE: 0.00185331;
- compact file: 8,141,784 bytes;
- private leaves: 176,563;
- total runtime: 51.7 seconds.

## Interpretation

The reversible loop works as intended and prevents a coarse-scale proposal
from permanently damaging the model. The negative result also shows that
coarse-to-fine scheduling alone does not fix the visible boundary blur: the
current additive leaf score did not discover a better private mask, and the
leaf-local Q1 space still cannot place an oblique discontinuity inside a leaf.

