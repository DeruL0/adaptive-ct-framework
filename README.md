# Adaptive Projection-Domain CT Reconstruction

CUDA-first research code for sparse adaptive parallel-beam CT reconstruction.
The main representation is a packed Bernstein octree optimized directly from
measured projections and refined under a global storage budget.

This repository is a research prototype. It is not medical software and must
not be used for diagnosis or treatment.

## Method

The current reconstruction path is:

1. Initialize a coarse packed octree with nonnegative degree-0 Bernstein leaves.
2. Optimize all active coefficients with a projection-domain WLS objective.
3. Detect convergence plateaus before changing topology.
4. Put every eligible leaf into one global cross-level refinement queue.
5. Rank h-splits by a scale-aware face-jump indicator per added packed byte.
6. Apply function-preserving splits within a global byte budget.
7. Continue joint optimization after each refinement round.
8. Export the packed hierarchy and quantized coefficients.

The core selector does not require a support mask, attenuation threshold,
minimum active resolution, per-level quota, forced material coverage, or test
views.

## Included Components

- Packed Bernstein octree with arbitrary dyadic depth.
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
  backend.py            Native CUDA bindings
  geometry.py           Parallel-beam ray construction
  projection_domain.py  Residual attribution and adaptive decisions
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
setup.py
```

## Requirements

The currently verified Windows environment is:

- Python 3.10.11
- PyTorch 2.10.0+cu130
- CUDA Toolkit 13.0
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

Run the packed global adaptive configuration:

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
