# Adaptive Projection-Domain CT Reconstruction

CUDA-first research code for sparse-view parallel-beam CT. A packed adaptive
octree of trilinear (P1) or constant (P0) Bernstein leaves is fitted directly to
the projections with an exact CUDA ray integrator; the tree is refined by
projection-driven Dörfler marking under a deployment byte budget, then exported
as a compact latent-P1 codec (per-leaf latent + one shared linear decoder).

No FDK, reference volume, or ground-truth volume is used for initialization,
topology, or scoring. The ground-truth volume is loaded only after training for
evaluation.

This repository is a research prototype. It is not medical software and must
not be used for diagnosis or treatment.

See [`docs/architecture.md`](docs/architecture.md) for the code framework
diagram, module responsibilities, and the training protocol.

## Pipeline

1. Start from the L0 root grid with coefficients set to the projection mean.
2. Train with random exact-projected rays (weighted MSE, Adam).
3. When the fixed-ray loss plateaus, the Dörfler controller estimates per-leaf
   residual energy, marks the smallest set carrying a fraction `theta` of it
   (rate-ranked, byte-budget truncated), and splits those leaves with exact
   de Casteljau subdivision. A held-out dev check can roll a harmful batch back.
4. Stop at the byte budget, when nothing is splittable, or when rollback is
   exhausted; fine-tune, convert to the codec, round to fp16, and evaluate.

## Layout

```text
adaptive_ct/
  core/            config, ray geometry, loss, native CUDA backend (projector)
  data/            dataset, train/dev view split, fixed ray sets
  representation/  Bernstein octree, latent-P1 octree, model factory, state snapshot
  codec/           byte model, fp16 quantization, fine-tuning, compact artifact
  refinement/      residual estimator, Dörfler marking, refinement controller
  evaluation/      metrics, rendering, ground-truth volume evaluation
  training/        training loop, plateau detection, export, config checks
  viewer/          Python server and Three.js frontend
native/            CUDA extension sources
configs/stage7/    p1, p0, p0_longft, p1_nogate configurations
docs/              architecture.md
tests/             mirrors adaptive_ct/ (core, data, representation, codec, refinement, training, viewer)
```

## Requirements

The currently verified Windows environment is:

- Python 3.10.11
- PyTorch 2.11.0+cu130
- CUDA Toolkit 13.0 (must match the PyTorch CUDA build to compile the extension)
- NVIDIA GPU with CUDA support
- Visual Studio C++ build tools
- Node.js 24.13 and npm 11.6 for the viewer

Core Python dependencies are PyTorch, NumPy, PyYAML, and Pillow.

## Installation

Create an environment and install a CUDA-enabled PyTorch build matching the
local CUDA toolkit:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install numpy pyyaml pillow
```

Build the native extension:

```powershell
python setup.py build_ext --inplace   # or scriptsuild_native.ps1
```

Check that the CUDA backend loads:

```powershell
python -c "from adaptive_ct.core.projector import require_native; require_native(); print('native backend: ok')"
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

```powershell
.venv\Scripts\python.exe -m adaptive_ct.training --config configs\stage7\p1.yaml
```

Controls: `p0.yaml` (P0 leaves), `p0_longft.yaml` (compute-matched P0),
`p1_nogate.yaml` (no rollback). Edit `dataset.root` and `output.dir` first. A run
writes `checkpoint.pt` (trained octree), `checkpoint_codec.pt` (deployed codec
model), `ANVO.npz` (compact export), `training_report.json`, and slice previews.

Run the tests with `scripts	est.ps1` or
`.venv\Scripts\python.exe -m unittest discover -s tests -t .`.

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
  --config configs\stage7\p1.yaml `
  --checkpoint output\stage7_dorfler_p1_256v\checkpoint.pt `
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
python -m adaptive_ct.viewer --config configs\stage7\p1.yaml

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
