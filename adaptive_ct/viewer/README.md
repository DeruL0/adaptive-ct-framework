# Adaptive CT Viewer

The viewer inspects projection-domain Bernstein-octree and legacy dynamic
leaf-voxel checkpoints. The main 3D view renders active leaves at their actual
cell sizes, exposing adaptive topology without first converting it to a dense
volume.

The backend is a small Python HTTP server in `server.py`. The frontend is a
Vite + TypeScript + Three.js app under `web/`.

## Modes

- **Voxels**: render active leaves and color by level, mean attenuation, or cell
  size. Filters control minimum attenuation, visible levels, opacity, gaps, and
  wireframe. Mouse controls provide orbit, zoom, and pan.
- **Slice**: compare axial, coronal, or sagittal model planes with `vol_gt`.
- **Projection**: compare native model projections with stored projections by
  split and view.
- **Inspector**: show model/config paths, projection-domain metrics, adaptive
  events, compact-export summaries, and optional uncertainty artifacts.

## Production Build

```powershell
cd adaptive_ct\viewer\web
npm.cmd install
npm.cmd run build
cd ..\..\..

python -m adaptive_ct.viewer `
  --config configs\figure473_48v.yaml `
  --checkpoint output\figure473_48v\checkpoint.pt `
  --open
```

The Python server hosts `web/dist/`. If the frontend has not been built, it
returns a build hint instead of the application.

The server uses model configuration embedded in a checkpoint when available, so
a Bernstein checkpoint is not interpreted through an incompatible legacy
schema. CUDA is used for native projection rendering when available. State,
leaf, artifact, and selected plane-sampling paths can run without full-volume
CUDA decoding, but high-resolution projections are substantially slower on CPU.

## Frontend Development

```powershell
# Terminal 1: backend and /api on port 8765
python -m adaptive_ct.viewer `
  --config configs\figure473_48v.yaml `
  --host 127.0.0.1 `
  --port 8765

# Terminal 2: Vite with /api proxying
cd adaptive_ct\viewer\web
npm.cmd run dev
```

Open `http://localhost:5173/` for the Vite development server.

## HTTP API

- `GET /api/state`: dataset, model, checkpoint, leaf, and training-report
  metadata.
- `GET /api/artifacts`: summaries for checkpoint, training report, compact
  export, MACT, and uncertainty artifacts when present.
- `GET /api/leaves?max_leaves=&min_mu=`: binary base64 leaf positions, sizes,
  attenuation values, and levels.
- `GET /api/slice?axis=&index=`: model/reference slice comparison and metrics.
- `GET /api/projection?split=&view=`: rendered/stored projection comparison and
  metrics.
- `GET /api/volume_texture`: quantized dense texture helper.
- `GET /api/volume_projection`: interactive dense-volume projection helper.
- `GET /api/sources`: compatible configs and `.pt`/`.npz` model sources.
- `POST /api/load`: hot-load a selected config and optional model.
- `GET /healthz`: service and checkpoint status.

Original-resolution slice requests sample only the requested 2D model plane and
read the matching reference plane from a memory-mapped array. They do not decode
or cache the complete original-resolution volume.
