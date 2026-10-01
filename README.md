# Adaptive Projection-Domain CT Reconstruction

CUDA-first research code for sparse adaptive parallel-beam CT reconstruction.
The first-paper target is a compact neural voxel field: a packed adaptive
octree, one low-dimensional latent per leaf, and one shared linear decoder that
emits the local P1 attenuation field used by exact CT projection.

This repository is a research prototype. It is not medical software and must
not be used for diagnosis or treatment.

## Method

The first-paper reconstruction path is built in stages.

Stage 0 freezes the non-neural reference:

1. Load sparse-view projections and known scanner geometry.
2. Initialize a packed octree with fixed topology.
3. Store one independent P1 Bernstein block per leaf, `c_K in R^8`.
4. Project with the exact segment integrator.
5. Optimize only projection-domain discrepancy, `D(p, A_T[c])`.

The exact projector keeps the leaf-intersection path: exact segment lengths,
local P1 evaluation, two-point Gauss-Legendre quadrature, and ray accumulation.
A trilinear P1 field restricted to a ray is at most cubic, so this quadrature
integrates each segment exactly.

This direct-P1 baseline answers whether the later latent decoder improves
anything beyond a more neural name.

The later compact neural path is:

1. Load sparse-view projections and known scanner geometry.
2. Initialize a packed adaptive octree.
3. Store one latent vector per active leaf.
4. Decode each latent with `c_K = W z_K + b`, where `W in R^(8x4)`.
5. Optimize only against projection-domain data.
6. Use held-out projection evidence for split and merge decisions.
7. Export the compact representation `(T, {z_K}, D_phi)`.

Stage 2 treats reconstruction and compression as one rate-distortion objective:

```text
min_{T,Z,phi} D(y, A_T[D_phi(Z)]) + lambda * B(T,Z,phi)
```

`D` is weighted projection MSE. `B` is the compact representation byte count,
using the same packed topology and latent payload accounting as export. If
enabled, projection noise variance is estimated once from low-attenuation
training detector pixels and then frozen, so split scores can be reported as
noise-normalized improvement per byte.

Stage 3 uses projection-driven h-splitting:

1. Score all eligible leaves with cheap residual exposure.
2. Keep only the top candidates.
3. For each candidate, split training rays into fit rays and decision rays.
4. Fit only the candidate child latents on fit rays; freeze the shared decoder
   and the rest of the volume.
5. Accept only if held-out decision-ray improvement per compact byte exceeds
   `lambda`.

This avoids fixed attenuation thresholds: low attenuation may be weak, but it
must be supported by projection evidence that survives the fit/decision split.

Stage 4 runs structural split rounds:

1. Globally optimize the current fixed topology.
2. Score cheap residual exposure and shortlist candidate leaves.
3. Evaluate candidate splits with fit/decision rays from training projections.
4. Rank candidates by held-out decision gain per compact byte.
5. Apply only the highest-value splits up to the per-round and byte caps.
6. Continue global finetuning on the updated octree.

Children inherit the parent latent at split time, so the topology change starts
near the previous attenuation field before local micro-fitting separates the
child latents.

Stage 5 adds projection-gated merge after split is stable:

1. Enumerate complete 8-sibling leaf groups.
2. Temporarily replace each group with its parent latent.
3. Fit only that parent latent on fit rays.
4. Measure decision-ray penalty and compact bytes saved.
5. Merge only when penalty per saved byte is below `merge_lambda`.

Stage 6 exports the deployment artifact directly. For the latent P1 path this
is an `ANVO.npz` file containing packed topology, FP16 leaf latents, and the
shared linear decoder weights. It does not store optimizer state, gradients,
training rays, decoded coefficients, dense volumes, or ground truth.

Ground-truth volume is not used for reconstruction, split/merge, neural
representation training, topology selection, or hyperparameter selection.

## Included Components

- Packed adaptive octree with arbitrary dyadic depth.
- Per-leaf latent P1 representation and shared linear decoder.
- Exact function-preserving hierarchy operations.
- CUDA traversal, line integration, coefficient gradients, and residual
  attribution.
- Parallel-beam projector and R2 geometry loader.
- Projection-domain training and global adaptive refinement.
- Compact packed-hierarchy export and loading.
- Legacy dynamic leaf-voxel representation used by compatible checkpoints.
- Python HTTP backend and Vite + TypeScript + Three.js viewer.

Generated datasets, checkpoints, experiment reports, audit documents, and test
code are not included in this public source release.

## Layout

```text
adaptive_ct/
  bernstein.py          Packed Bernstein hierarchy
  latent_p1.py          Per-leaf latent + shared linear local-P1 decoder
  backend.py            Native CUDA bindings
  geometry.py           Parallel-beam ray construction
  projection_domain.py  Residual attribution and adaptive decisions
  functional_compression.py  Legacy keep/p0/p1 subtree compression
  industrial_detail_pipeline.py  Legacy reference-to-compact entry point
  compression.py        Compact model export and loading
  train.py              Training, refinement, evaluation, and export
  viewer/               Python server and Three.js frontend
native/
  projector.cu          Dense parallel-beam projector
  dynamic_voxel.cu      Dynamic leaf-voxel integration
  bernstein_octree.cu   Packed Bernstein integration
configs/
  figure473_48v.yaml
  research/figure473_128v_pure_adaptive.yaml
  research/first_paper_latent_p1.yaml
  research/stage3_latent_p1_crossfit_split.yaml
  research/stage4_latent_p1_structural_rounds.yaml
  research/stage5_6_latent_p1_merge_anvo.yaml
  research/industrial_detail_v5_h_reference.yaml
  research/industrial_detail_v5_20mb.yaml
  research/industrial_detail_v5_20mb_final.yaml
setup.py
```

## Requirements

The currently verified Windows environment is:

- Python 3.10.11
- PyTorch 2.11.0+cu128
- CUDA Toolkit/runtime 12.8
- NVIDIA GPU with CUDA support
- Visual Studio C++ build tools
- Node.js 24.13 and npm 11.6 for the viewer

Core Python dependencies are PyTorch, NumPy, PyYAML, and Pillow. Selected
evaluation utilities additionally use SciPy and Matplotlib.

## Installation

Create an environment and install a CUDA-enabled PyTorch build matching the
local CUDA toolkit:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install numpy pyyaml pillow scipy matplotlib
```

Build the native extension:

```powershell
python setup.py build_ext --inplace
```

Check that the CUDA backend loads:

```powershell
python -c "from adaptive_ct.backend import require_native; require_native(); print('native backend: ok')"
```

## Dataset Format

The loader expects an R2-style parallel-beam directory:

```text
dataset_root/
  meta_data.json
  vol_gt.npy
  proj_train/
    proj_train_0000.npy
    ...
  proj_test/
    proj_test_0000.npy
    ...
```

`meta_data.json` contains scanner geometry and projection frame lists:

```json
{
  "scanner": {
    "mode": "parallel",
    "nVoxel": [96, 96, 96],
    "nDetector": [96, 96]
  },
  "vol": "vol_gt.npy",
  "proj_train": [
    {"file_path": "proj_train/proj_train_0000.npy", "angle": 0.0}
  ],
  "proj_test": [
    {"file_path": "proj_test/proj_test_0000.npy", "angle": 0.0654498}
  ]
}
```

An optional `proj_val` list may define a validation split. Update
`dataset.root` in the selected YAML before running.

## Training

Run the compact 48-view configuration:

```powershell
python -m adaptive_ct.train --config configs\figure473_48v.yaml
```

Run the Stage 0 direct-P1 baseline:

```powershell
python -m adaptive_ct.train `
  --config configs\research\figure473_128v_pure_adaptive.yaml
```

The output directory is selected by the YAML configuration. A completed run may
produce:

```text
checkpoint.pt
training_report.json
compact_octree.npz
```

The checkpoint contains training state and framework metadata.
`compact_octree.npz` is the deployment-oriented representation.

## Industrial Detail Pipeline

The industrial-detail entry point reconstructs its own h-only reference from
the configured sparse projections, then performs projection-aware bottom-up
subtree compression. Each internal subtree is either kept, replaced by a p0
constant leaf, or replaced by a p1 trilinear leaf under one packed-byte budget.
It does not accept an externally supplied reference checkpoint.

Validate the complete pipeline contract without training:

```powershell
python -m adaptive_ct.industrial_detail_pipeline `
  --config configs\research\industrial_detail_v5_20mb.yaml `
  --validate-only
```

Run reconstruction, functional compression, fixed-topology refinement, and
compact export:

```powershell
python -m adaptive_ct.industrial_detail_pipeline `
  --config configs\research\industrial_detail_v5_20mb.yaml
```

The pipeline writes its h-only reference to the reference config's output
directory and the final checkpoint, compact hierarchy, compression report, and
pipeline report to the pipeline config's output directory. Use
`industrial_detail_v5_20mb_final.yaml` to load the final checkpoint in the
viewer.

## Viewer

Build the frontend:

```powershell
cd adaptive_ct\viewer\web
npm.cmd install
npm.cmd run build
cd ..\..\..
```

Start the viewer:

```powershell
python -m adaptive_ct.viewer `
  --config configs\figure473_48v.yaml `
  --checkpoint output\figure473_48v\checkpoint.pt `
  --host 127.0.0.1 `
  --port 8765 `
  --open
```

The application opens at `http://127.0.0.1:8765/`. It provides adaptive leaf
inspection, orbit/zoom/pan controls, model slices, native projection
comparisons, artifact summaries, and runtime model loading.

For frontend development:

```powershell
# Terminal 1
python -m adaptive_ct.viewer --config configs\figure473_48v.yaml

# Terminal 2
cd adaptive_ct\viewer\web
npm.cmd run dev
```

Viewer-specific API documentation is in
[`adaptive_ct/viewer/README.md`](adaptive_ct/viewer/README.md).

## Evaluation Scope

Projection fidelity, volume fidelity, material-region error, boundary quality,
storage, and runtime should be reported together under one consistent dataset
and view protocol. Projection PSNR alone is not evidence of accurate volume
reconstruction.
