# Interface-aware nodal p1 execution plan

## Objective

Replace independent per-leaf p1 coefficients with a shared-vertex trilinear
attenuation field in smooth regions, while retaining explicit discontinuities
at material interfaces, cracks, pores, and thin walls. The CT projection data
term remains the reconstruction objective.

## Representation

- Packed octree levels `[48, 96, 192, 384]`.
- Every leaf evaluates a trilinear Bernstein (`Q1`) function.
- Smooth neighboring leaves share master corner vertices.
- Projection-selected discontinuous leaves keep private corner vertices.
- 2:1 coarse/fine interfaces eliminate hanging vertices through exact edge or
  face interpolation from coarse masters.
- Rays use the existing exact Bernstein CUDA projector.

## Implemented pipeline

1. Load the fixed p5 reconstruction and enforce 2:1 balance.
2. Convert all p0/p1 leaves to one-sided p1 corner samples.
3. Build the fully tied nodal field and exact hanging-node constraints.
4. For every leaf, evaluate the projection-loss change obtained by restoring
   its independent one-sided p1 function.
5. Select only positive predicted-gain untie actions under one global raw-byte
   budget. No material label, spatial quota, minimum level, or fixed boundary
   fraction is used.
6. Rebuild the nodal field with selected private leaves, freeze topology and
   constraints, and jointly optimize master/private vertices using 96 training
   views.
7. Evaluate on the four held-out test views and 15 fixed representative volume
   slices.
8. Export compact nodal v4: packed topology, float16 master values, and a
   private-leaf bitmask. Hanging constraints are reconstructed deterministically
   instead of being stored in the file.

The implemented action is conservative `untie(all faces of leaf)`. A true
face-granular connected-component action is a possible future storage
optimization, not required by the current runtime.

## Runtime implementation

- The exact CPU and native CUDA projectors consume reconstructed leaf-local p1
  coefficients, so ray traversal did not need to be replaced.
- Constraint evaluation caches only real nonzero entries and uses flattened
  gather/index-add. A 4096-ray training step dropped from about 7.9 seconds to
  0.86 seconds.
- Compact loading currently rebuilds the full constraint graph from topology.
  The 2.02M-leaf model takes roughly four minutes to load; caching a compact CSR
  graph is the main remaining deployment-performance issue.

## Verified 96-view result

| Metric | p5 | Selected nodal p1 |
|---|---:|---:|
| Held-out projection PSNR | 43.3745 dB | 43.2965 dB |
| Held-out projection MAE | 0.0018015 | 0.0019026 |
| Independent coefficients/masters | 3,844,560 | 2,980,422 |
| Compact file | 10,041,162 B | 8,419,206 B |
| Mean PSNR over 15 slices | 30.3990 dB | 30.7377 dB |
| Mean MAE over 15 slices | 0.0129016 | 0.0124216 |
| Top-1% gradient PSNR | 18.8976 dB | 19.1808 dB |
| Top-1% gradient MAE | 0.0566187 | 0.0549285 |
| Mean gradient ratio | 0.3860 | 0.3259 |

The selected model is 0.0781 dB below p5 in projection PSNR, 16.2% smaller on
disk, and better on representative slice/value and high-gradient value errors.
Its gradient magnitude ratio is lower, so the sharpest interfaces remain more
smoothed despite improved high-gradient PSNR/MAE.

The projection-times-interface-jump ablation assigned more private leaves to
L3 and partly recovered gradient magnitude, but reduced aggregate projection
and slice accuracy. It is retained as an ablation and is not the selected
model.

## Verification

- Same-level and hanging-face continuity tests pass.
- Explicit private-leaf discontinuity test passes.
- CPU/native CUDA projection gradient propagation passes.
- Legacy compact v1-v3 and nodal compact v4 round trips pass.
- Real compact v4 independently loads and reproduces the checkpoint slice to
  approximately `6e-5 dB`.
- Full suite: 100 tests passed.

