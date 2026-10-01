# Ambrosio--Tortorelli nodal CT Stage A result

Date: 2026-07-05

## Implemented scope

- Fixed 96-view equal-ratio anisotropic octree topology.
- Fully tied constrained nodal p1 attenuation field `mu`.
- Independent constrained nodal phase field `v` on the same master/hanging graph.
- Standard convention: `v=1` smooth region, `v=0` diffuse interface.
- Two-point tensor Gauss integration of the 3D AT bulk and phase energies.
- Volume-weighted adaptive-leaf sampling.
- Alternating projected optimisation of `mu` and `v`.
- Epsilon continuation at `epsilon/h_finest = 8 -> 4`.
- Existing CUDA ray projector unchanged.

No h-refinement, private thresholding, cut-cell conversion, support mask, TV,
or compact schema change was added.

## Energy

```text
0.5 ||A mu - y||^2
+ alpha integral (v^2 + eta) |grad mu|^2
+ beta  integral [epsilon |grad v|^2 + (1-v)^2 / (4 epsilon)]
```

## Calibration

The initial `alpha=1e-5` smoke test made the AT term about 100 times larger
than the projection loss. The calibrated run used `alpha=1e-8`, for an initial
AT/data ratio of about 0.12. Two surface weights were tested:

- `beta=1e-7`;
- `beta=1e-8`, with twice the leaf batch and more v steps.

## Result

| Metric | Source private nodal | Fully tied initial | AT beta=1e-7 | AT beta=1e-8 |
|---|---:|---:|---:|---:|
| Test projection PSNR | 43.9668 | 42.2730 | 42.3244 | 42.3245 |
| y=512 PSNR | 29.8951 | 29.9812 | 29.9817 | 29.9816 |
| y=512 gradient ratio | 0.1970 | 0.1146 | 0.1145 | 0.1145 |
| y=512 near-air halo MAE | 0.01823 | 0.02461 | 0.02460 | 0.02460 |
| phase minimum | n/a | 1.0 | 0.8785 | 0.7680 |
| phase masters below 0.5 | n/a | 0 | 0 | 0 |

The lower-beta run covered 131,072 volume-weighted leaves per v step and used
500 v steps. It still produced no `v < 0.5` interface. Continuing to sweep beta
was stopped by design.

At `epsilon/h_finest=4`, about 96% of volume-weighted sampled leaves violate
`h_K <= epsilon/4`. The fixed adaptive topology therefore does not resolve a
globally valid AT transition width, even though fine interface leaves may do so
locally.

## Conclusion

The implementation is numerically valid and fully compatible with the current
projector, but the proposed fixed-topology AT model is not an accuracy
improvement. Converting the current private field to global C0 removes edge
amplitude before AT optimisation; the diffuse phase field did not recreate a
free-discontinuity set and did not improve halo error.

AT should not replace the selected model. A future AT path would require an
adaptive phase-resolving topology plus an explicit phase-to-face/cut-cell
transition, or a DG formulation that charges existing jumps instead of first
projecting them into a fully tied field.

Artifacts:

- `output/at_nodal_96v_stage_a_v1/report.json`
- `output/at_nodal_96v_stage_a_beta1e8/report.json`
- `adaptive_ct/ambrosio_tortorelli.py`
- `scripts/run_at_nodal.py`

