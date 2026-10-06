# Adaptive Projection-Domain CT Reconstruction

CUDA-first research code for parallel-beam CT. A packed adaptive octree whose
leaves are constant, trilinear or adjustable-width step functions is fitted
directly to the projections with an exact CUDA ray integrator; tree and leaf
types are chosen under a byte budget, and everything is stored in fp16 (what is
stored is what is counted).

No FDK, reference volume, or ground-truth volume is used for initialization,
topology, or scoring. The ground-truth volume is loaded only after training for
evaluation.

This repository is a research prototype. It is not medical software and must
not be used for diagnosis or treatment.

See [`docs/architecture.md`](docs/architecture.md) for the code framework
diagram, module responsibilities, and the training protocol.

## Pipeline

1. Start from the 48³ root grid with every coefficient set to half the mean projection value.
2. Each round, with the tree fixed, solve `min_{c>=0} 1/2 ||A c - y||^2` over the physical leaf
   coefficients: residual `r = A c - y`, gradient `g = A^T r`, per-coefficient step `1/d` with
   `d = A^T A 1`, update `c <- max(0, c - g/d)`; angle-interleaved grouped updates first, then
   full-data updates with extrapolation. The optimality residual is reported per round.
3. On the solved model, refit every leaf's options (keep / constant / trilinear / step / split)
   with all other leaves fixed, on half of 16 evenly spaced training views (alternate views),
   and score them on the other half (statistics accumulated in the CUDA traversal); each leaf takes the option with the largest held-out gain minus a byte price that keeps
   the model within budget; per round, actions are executed in order of net benefit until they
   spend 70% of the remaining bytes (byte-saving actions always); a leaf whose type just changed
   may only be kept or split in the next round; refinement stops below 1% remaining budget.
   Each solve stops at an optimality-residual tolerance (1e-3 per round); while leaves are still
   coarse it uses 2x2 / 4x4 averaged detector pixels.
4. When no action has a positive net benefit, solve strictly (tolerance 1e-4, full resolution), round the
   values and step geometry to fp16, evaluate, and write the compact file.

## Layout

```text
adaptive_ct/
  core/            config, ray geometry, native CUDA backend (projector)
  data/            dataset, train/dev view split, fixed ray sets
  representation/  octree with constant / trilinear / step leaves, model factory
  codec/           byte model, fp16 rounding, compact file
  refinement/      held-out refit decision (keep / change type / split) and its controller
  evaluation/      metrics, rendering, ground-truth volume evaluation
  training/        fixed-tree solver, rounds of solve + structure update, export, checks
  viewer/          Python server and Three.js frontend
native/            CUDA extension sources
configs/stage7/    p1 (main) and p0 (constant-leaf control) configurations
scripts/audit/     gate_chain.py: offline diagnostics with the GT volume (not part of training)
docs/              architecture.md
tests/             mirrors adaptive_ct/
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
python setup.py build_ext --inplace   # or scripts\build_native.ps1
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

Control: `p0.yaml` (constant leaves). Edit `dataset.root` and `output.dir` first. A run
writes `checkpoint.pt` (deployed octree, coefficients rounded to fp16), `ANVO.npz`
(compact file), `training_report.json`, and slice previews.

Run the tests with `scripts\test.ps1` or
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
  --checkpoint output\main_p1_256v\checkpoint.pt `
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
