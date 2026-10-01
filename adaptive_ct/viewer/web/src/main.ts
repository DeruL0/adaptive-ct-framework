import "./styles.css";
import {
  getArtifacts,
  getLeaves,
  getProjection,
  getSlice,
  getState,
  getVolumeTexture,
  loadModel,
  selectModel,
  type ArtifactsPayload,
  type LeavesPayload,
  type StatePayload,
} from "./api";
import { VoxelScene, type ColorMode, type VoxelOptions } from "./voxelScene";
import { LEVEL_COLORS, RAMP_CSS } from "./colormaps";

type Mode = "voxel" | "slice" | "projection";

const APP = document.getElementById("app") as HTMLElement;

APP.innerHTML = `
  <div class="app-shell">
    <header class="toolbar">
      <div class="brand">
        <span class="eyebrow">Adaptive CT Viewer</span>
        <h1 id="title">RD-CVF</h1>
      </div>
      <div class="statline" id="statline"></div>
      <span class="model-feedback" id="modelFeedback" hidden></span>
      <button class="model-button" id="openSource" type="button">Open model...</button>
      <div class="segmented" id="modes">
        <button data-mode="voxel" class="is-active" type="button">Voxels</button>
        <button data-mode="slice" type="button">Slice</button>
        <button data-mode="projection" type="button">Projection</button>
      </div>
    </header>
    <main class="workspace">
      <section class="viewport">
        <div class="control-strip" id="controlStrip">
          <div class="ctl-group" data-controls="voxel" style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;">
            <label class="field"><span>Color</span>
              <select id="colorMode">
                <option value="level">by level</option>
                <option value="mu">by &mu; (attenuation)</option>
                <option value="size">by voxel size</option>
                <option value="xray">X-ray projection</option>
              </select>
            </label>
            <label class="field"><span>Min &mu;</span>
              <input id="minMu" type="range" min="0" max="1" step="0.001" value="0">
              <span id="minMuVal" style="min-width:42px">0.000</span>
            </label>
            <span class="field" id="levelToggles" style="gap:10px"></span>
            <label class="field"><span>Opacity</span>
              <input id="opacity" type="range" min="0.15" max="1" step="0.05" value="1">
            </label>
            <label class="field"><span>Gap</span>
              <input id="shrink" type="range" min="0" max="0.4" step="0.02" value="0.04">
            </label>
            <label class="field"><span>Leaves</span>
              <select id="leafBudget">
                <option value="100000">100k</option>
                <option value="250000" selected>250k</option>
                <option value="500000">500k</option>
                <option value="1000000">1M</option>
                <option value="0">all</option>
              </select>
            </label>
            <label class="field check"><input id="wireframe" type="checkbox"><span>Wireframe</span></label>
            <label class="field"><span>Density</span>
              <input id="xrayDensity" type="range" min="0.2" max="8" step="0.1" value="2.2">
              <span id="xrayDensityVal" style="min-width:32px">2.2</span>
            </label>
            <button class="field" id="resetView" type="button" style="cursor:pointer;border:1px solid var(--border);border-radius:6px;padding:6px 10px;background:rgba(8,12,17,0.74)">Reset view</button>
          </div>
          <div class="ctl-group" data-controls="slice" hidden style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;">
            <label class="field"><span>Axis</span>
              <select id="axis"><option value="z">z</option><option value="y">y</option><option value="x">x</option></select>
            </label>
            <label class="field"><span>Index</span>
              <input id="sliceSlider" type="range" min="0" max="95" value="48">
              <span id="sliceVal" style="min-width:32px">48</span>
            </label>
          </div>
          <div class="ctl-group" data-controls="projection" hidden style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;">
            <label class="field"><span>Split</span>
              <select id="split"><option value="test">test</option><option value="train">train</option></select>
            </label>
            <label class="field"><span>View</span>
              <input id="viewSlider" type="range" min="0" max="11" value="0">
              <span id="viewVal" style="min-width:32px">0</span>
            </label>
          </div>
        </div>

        <div class="stage">
          <div id="voxelStage"></div>
          <div class="hud" id="voxelHud"></div>
          <div class="legend" id="voxelLegend"></div>
          <div class="image-grid" id="imageGrid" hidden>
            <article class="image-panel"><div class="head"><strong>Prediction</strong><span id="imgPredTag"></span></div><div class="body"><img id="imgPred" alt="Prediction"></div></article>
            <article class="image-panel"><div class="head"><strong>Ground Truth</strong><span id="imgGtTag"></span></div><div class="body"><img id="imgGt" alt="Ground truth"></div></article>
            <article class="image-panel"><div class="head"><strong>Error</strong><span>pred &minus; gt</span></div><div class="body"><img id="imgErr" alt="Error"></div></article>
          </div>
          <div class="overlay" id="overlay" hidden>Loading</div>
        </div>

        <div class="metricbar" id="metricbar"></div>
      </section>

      <aside class="inspector" id="inspector"></aside>
    </main>
  </div>
`;

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const state = {
  mode: "voxel" as Mode,
  meta: null as StatePayload | null,
  artifacts: null as ArtifactsPayload | null,
  leaves: null as LeavesPayload | null,
  busy: false,
  leafBudget: 250000,
  render3dMode: "voxels" as "voxels" | "xray",
  volumeTextureLoaded: false,
};

const voxelOptions: VoxelOptions = {
  colorMode: "level",
  minMu: 0,
  levelVisible: [true, true, true, true, true],
  opacity: 1,
  wireframe: false,
  shrink: 0.04,
};

let scene: VoxelScene | null = null;

function fmt(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function fmtBytes(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let scaled = Number(value);
  let unit = 0;
  while (scaled >= 1024 && unit < units.length - 1) {
    scaled /= 1024;
    unit += 1;
  }
  return `${fmt(scaled, unit === 0 ? 0 : 2)} ${units[unit]}`;
}

function bindModelPicker(): void {
  const button = $("openSource") as HTMLButtonElement;
  const feedback = $("modelFeedback");
  button.addEventListener("click", async () => {
    let reloading = false;
    feedback.hidden = true;
    feedback.textContent = "";
    button.disabled = true;
    button.textContent = "Selecting...";
    try {
      const result = await selectModel();
      if (result.status === "cancelled" || !result.checkpoint) return;
      // The picker returned fast; loading the dataset + decoding the model can
      // take up to a minute (longer on CPU), so surface a visible progress note
      // instead of leaving the button on a misleading "Selecting..." label.
      const name = result.checkpoint.split(/[\\/]/).pop() || result.checkpoint;
      button.textContent = "Loading...";
      feedback.textContent = `Loading ${name}… this can take up to a minute.`;
      feedback.hidden = false;
      await loadModel(result.checkpoint);
      reloading = true;
      window.location.reload();
    } catch (error) {
      feedback.textContent = error instanceof Error ? error.message : "Could not load the selected model.";
      feedback.hidden = false;
    } finally {
      if (!reloading) {
        button.disabled = false;
        button.textContent = "Open model...";
      }
    }
  });
}

function bestRdScore(meta: StatePayload): number | null {
  const curve = meta.report.rate_distortion_curve || [];
  let best: number | null = null;
  for (const event of curve) {
    for (const evaluation of event.evaluations || []) {
      const score = Number(evaluation.score_per_byte);
      if (Number.isFinite(score)) best = best === null ? score : Math.max(best, score);
    }
  }
  return best;
}

function sci(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "-";
  return Number(value).toExponential(digits);
}

function setOverlay(text: string | null): void {
  const overlay = $("overlay");
  overlay.textContent = text ?? "";
  overlay.hidden = !text;
}

function setMetrics(cells: Array<[string, string]>): void {
  $("metricbar").innerHTML = cells
    .map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

// ---------------------------------------------------------------- inspector
function renderInspector(meta: StatePayload, artifacts: ArtifactsPayload | null = state.artifacts): void {
  $("title").textContent = meta.name || "RD-CVF";
  $("statline").innerHTML = [
    meta.dataset_available !== false && meta.volume_shape ? `${meta.volume_shape.join("×")} volume` : "model only (no dataset)",
    meta.dataset_available !== false && meta.detector_shape ? `${meta.detector_shape.join("×")} detector` : null,
    meta.dataset_available !== false ? `${meta.train_views}/${meta.test_views} views` : null,
    meta.device,
    meta.model.representation,
    meta.checkpoint_loaded ? "checkpoint loaded" : "no checkpoint",
  ].filter(Boolean).map((item) => `<span>${item}</span>`).join("");

  const m = meta.model;
  const ls = meta.leaf_structure;
  const totalLeaves = ls.leaf_cells || m.leaf_cells || 1;
  const levelBars = ls.levels
    .map((lvl) => {
      const pct = Math.round((lvl.count / Math.max(totalLeaves, 1)) * 100);
      const color = LEVEL_COLORS[lvl.level % LEVEL_COLORS.length];
      return `<div class="level-bar">
        <span class="tag" style="color:${color}">L${lvl.level}</span>
        <span class="track"><span class="fill" style="width:${pct}%;background:${color}"></span></span>
        <span class="count">${lvl.count.toLocaleString()} @ ${fmt(lvl.voxel_size, 4)}</span>
      </div>`;
    })
    .join("");

  const adaptiveEvents = meta.report.adaptive_events || [];
  const growthEvents = meta.report.growth_events || [];
  const rdBest = bestRdScore(meta);
  const totalCandidates = adaptiveEvents.reduce((acc, ev) => acc + (Number(ev.candidate_count) || 0), 0);
  const totalAccepted = adaptiveEvents.reduce((acc, ev) => acc + (Number(ev.accepted_count) || 0), 0);
  const hGrowthLeaves = growthEvents
    .filter((ev) => (ev.operation || "h_split") !== "p_elevate")
    .reduce((acc, ev) => acc + (Number(ev.active) || 0), 0);
  const pElevatedLeaves = growthEvents
    .filter((ev) => ev.operation === "p_elevate")
    .reduce((acc, ev) => acc + (Number(ev.active) || 0), 0);
  const pCoefficientDelta = growthEvents
    .filter((ev) => ev.operation === "p_elevate")
    .reduce((acc, ev) => {
      const reportedDelta = Number(ev.added_coefficients);
      const derivedDelta = Math.max(
        0,
        Number(ev.new_coefficient_count || 0) - Number(ev.old_coefficient_count || 0),
      );
      return acc + (Number.isFinite(reportedDelta) ? reportedDelta : derivedDelta);
    }, 0);
  const adaptiveTimeline = adaptiveEvents.length
    ? adaptiveEvents
        .map((ev) => `<div class="event" style="border-left-color:${Number(ev.accepted_count) > 0 ? "var(--accent)" : "var(--danger)"}">
          <div class="row"><strong>iter ${ev.iteration}</strong><span class="muted">${ev.accepted_count}/${ev.candidate_count} accepted</span></div>
          <div class="row"><span class="muted">${fmtBytes(ev.model_bytes)} · ${Number(ev.coefficient_count || 0).toLocaleString()} coeffs</span></div>
        </div>`)
        .join("")
    : "";
  const growthTimeline = growthEvents.length
    ? growthEvents
        .map((ev) => {
          const extra = [
            ev.gradient_mean !== undefined ? `grad ${fmt(ev.gradient_mean, 3)}` : "",
            ev.projected_residual_mean !== undefined ? `resid ${fmt(ev.projected_residual_mean, 3)}` : "",
            ev.sigma_mean !== undefined ? `&sigma; ${fmt(ev.sigma_mean, 3)}` : "",
          ].filter(Boolean).join(" · ");
          const isP = ev.operation === "p_elevate";
          const coeffDelta = ev.added_coefficients !== undefined
            ? Math.max(0, Number(ev.added_coefficients))
            : Math.max(0, Number(ev.new_coefficient_count || 0) - Number(ev.old_coefficient_count || 0));
          const detail = isP
            ? `${ev.active.toLocaleString()} leaves elevated · +${coeffDelta.toLocaleString()} coeffs · degree ${Array.isArray(ev.target_degree) ? ev.target_degree.join(",") : ev.target_degree ?? "-"}`
            : `+${ev.active.toLocaleString()} child leaves · top ${Math.round(ev.active_fraction * 100)}%`;
          return `<div class="event" style="border-left-color:${LEVEL_COLORS[ev.level % LEVEL_COLORS.length]}">
            <div class="row"><strong>iter ${ev.iteration} → L${ev.level}</strong><span class="muted">${ev.operation || "h_split"} · ${ev.strategy}</span></div>
            <div class="row"><span class="muted">${detail}</span></div>
            ${extra ? `<div class="row"><span class="muted">${extra}</span></div>` : ""}
          </div>`;
        })
        .join("")
    : "";
  const timeline = adaptiveTimeline || growthTimeline || `<span class="muted" style="font-size:11px">No topology events recorded.</span>`;
  const topologyTitle = adaptiveEvents.length ? "RD-CVF Held-out Gate" : "Coefficient-score H-Refinement";
  const topologyRounds = adaptiveEvents.length + growthEvents.length;
  const topologyChips = adaptiveEvents.length
    ? `
        <span>${totalCandidates.toLocaleString()} candidates</span>
        <span>${totalAccepted.toLocaleString()} accepted</span>
        <span>best S ${sci(rdBest)}</span>
      `
    : `
        <span>${growthEvents.length.toLocaleString()} growth events</span>
        <span>${hGrowthLeaves.toLocaleString()} activated child leaves</span>
        <span>${pElevatedLeaves.toLocaleString()} p-elevated leaves</span>
        <span>${pCoefficientDelta.toLocaleString()} p coeffs</span>
        <span>projection-domain scores</span>
      `;

  const report = meta.report;
  const proj = report.projection_test;
  const mat = report.material_volume;
  const boundary = report.boundary_sharpness;
  const compact = artifacts?.compact_octree;
  const rayIntegration = report.ray_integration || {};
  const volumeLoss = report.training_objective?.volume_loss as { enabled?: boolean } | undefined;
  const denseLoaded = report.training_objective?.dense_volume_loaded_during_training;

  $("inspector").innerHTML = `
    <section class="section">
      <h2>Model <span class="meta">${m.representation}</span></h2>
      <dl class="kv">
        <div><dt>Parameters</dt><dd>${m.parameter_count.toLocaleString()} (${(m.model_bytes / 1048576).toFixed(2)} MB)</dd></div>
        <div><dt>Bernstein degree</dt><dd>${m.max_degree ? `max (${m.max_degree.join(", ")}) · ${(m.coefficient_count || m.parameter_count).toLocaleString()} coefficients` : "-"}</dd></div>
        <div><dt>Leaf voxels</dt><dd>${(ls.leaf_cells || m.leaf_cells).toLocaleString()} active leaves · max depth L${m.max_depth}</dd></div>
        <div><dt>Voxel size range</dt><dd>${ls.size_range ? `${fmt(ls.size_range.min, 4)} – ${fmt(ls.size_range.max, 4)}` : "-"}</dd></div>
      </dl>
    </section>
    <section class="section">
      <h2>Leaf Structure <span class="meta">per level</span></h2>
      <div class="level-bars">${levelBars}</div>
    </section>
    <section class="section">
      <h2>${topologyTitle} <span class="meta">${topologyRounds} rounds</span></h2>
      <div class="chips">
        ${topologyChips}
        <span>${rayIntegration.compact_cuda_traversal ? "CUDA compact rays" : "torch rays"}</span>
      </div>
      <div class="timeline">${timeline}</div>
    </section>
    <section class="section">
      <h2>Training Report <span class="meta">${report.elapsed_sec ? `${fmt(report.elapsed_sec, 1)} s` : "-"}</span></h2>
      <div class="chips">
        <span>proj ${proj ? `${fmt(proj.psnr, 2)} dB` : "-"}</span>
        <span>proj MAE ${proj ? fmt(proj.mae, 5) : "-"}</span>
        <span>vol ${mat ? `${fmt(mat.psnr, 2)} dB` : "-"}</span>
        <span>vol MAE ${mat ? fmt(mat.mae, 5) : "-"}</span>
        <span>boundary ${boundary ? fmt(boundary.gradient_ratio, 3) : "-"}</span>
        <span>dense train ${denseLoaded === false ? "off" : denseLoaded === true ? "on" : "-"}</span>
        <span>volume loss ${volumeLoss?.enabled === false ? "off" : volumeLoss?.enabled === true ? "on" : "-"}</span>
      </div>
    </section>
    <section class="section">
      <h2>Compact Octree <span class="meta">${compact?.exists ? "loaded" : "missing"}</span></h2>
      <dl class="kv">
        <div><dt>File</dt><dd>${compact?.path || "-"}</dd></div>
        <div><dt>Size</dt><dd>${compact?.exists ? `${fmtBytes(compact.size_bytes)} vs checkpoint ${fmtBytes(artifacts?.checkpoint?.size_bytes || 0)}` : "-"}</dd></div>
        <div><dt>Encoding</dt><dd>${compact?.metadata?.quantization ? String(compact.metadata.quantization) : "-"}</dd></div>
        <div><dt>Payload</dt><dd>${compact ? `${(compact.leaf_count || 0).toLocaleString()} leaves · ${(compact.coefficient_count || 0).toLocaleString()} coeffs` : "-"}</dd></div>
      </dl>
      ${compact?.error ? `<p class="error-text">${compact.error}</p>` : ""}
    </section>
    <section class="section">
      <h2>Checkpoint <span class="meta">${meta.checkpoint_loaded ? "loaded" : "not loaded"}</span></h2>
      <dl class="kv">
        <div><dt>Config</dt><dd>${meta.paths.config || "— (model only, no YAML)"}</dd></div>
        <div><dt>Checkpoint</dt><dd>${meta.paths.checkpoint || meta.checkpoint_error || "-"}</dd></div>
        <div><dt>Dataset</dt><dd>${meta.paths.dataset || "— (no dataset loaded)"}</dd></div>
        <div><dt>Output</dt><dd>${meta.paths.output}</dd></div>
      </dl>
      ${meta.checkpoint_error ? `<p class="error-text">${meta.checkpoint_error}</p>` : ""}
      ${meta.dataset_error ? `<p class="error-text">${meta.dataset_error}</p>` : ""}
    </section>
  `;
}

// ---------------------------------------------------------------- voxel mode
function buildLevelToggles(meta: StatePayload): void {
  const host = $("levelToggles");
  host.innerHTML = meta.leaf_structure.levels
    .map((lvl) => {
      const color = LEVEL_COLORS[lvl.level % LEVEL_COLORS.length];
      return `<label class="field check" title="${lvl.count.toLocaleString()} leaves">
        <input type="checkbox" data-level="${lvl.level}" checked>
        <span style="color:${color}">L${lvl.level}</span>
      </label>`;
    })
    .join("");
  host.querySelectorAll<HTMLInputElement>("input[data-level]").forEach((input) => {
    input.addEventListener("change", () => {
      const level = Number(input.dataset.level);
      voxelOptions.levelVisible[level] = input.checked;
      scene?.setOptions({ levelVisible: [...voxelOptions.levelVisible] });
      updateVoxelHud();
    });
  });
}

function updateVoxelLegend(): void {
  const legend = $("voxelLegend");
  if (voxelOptions.colorMode === "level") {
    const rows = (state.meta?.leaf_structure.levels || [])
      .map((lvl) => `<div class="legend-row"><span class="legend-swatch" style="background:${LEVEL_COLORS[lvl.level % LEVEL_COLORS.length]}"></span>L${lvl.level} · size ${fmt(lvl.voxel_size, 3)}</div>`)
      .join("");
    legend.innerHTML = `<div class="legend-title">Refinement level</div>${rows}`;
  } else if (voxelOptions.colorMode === "mu") {
    const r = state.leaves?.mu_range;
    legend.innerHTML = `<div class="legend-title">Attenuation &mu;</div>
      <div class="legend-bar" style="background:${RAMP_CSS}"></div>
      <div class="legend-ends"><span>${fmt(r?.min ?? 0, 3)}</span><span>${fmt(r?.max ?? 1, 3)}</span></div>`;
  } else {
    const r = state.leaves?.size_range;
    legend.innerHTML = `<div class="legend-title">Voxel size (fine → coarse)</div>
      <div class="legend-bar" style="background:${RAMP_CSS}"></div>
      <div class="legend-ends"><span>${fmt(r?.min ?? 0, 3)}</span><span>${fmt(r?.max ?? 1, 3)}</span></div>`;
  }
}

function updateVoxelHud(): void {
  const leaves = state.leaves;
  $("voxelHud").innerHTML = [
    `${(scene?.visible ?? 0).toLocaleString()} / ${(leaves?.total ?? 0).toLocaleString()} leaves`,
    `render: ${state.render3dMode}`,
    `min μ ${fmt(voxelOptions.minMu, 3)}`,
  ].map((t) => `<span>${t}</span>`).join("");
  setMetrics([
    ["Visible leaves", (scene?.visible ?? 0).toLocaleString()],
    ["Total leaves", (leaves?.total ?? 0).toLocaleString()],
    ["Returned", (leaves?.count ?? 0).toLocaleString()],
    ["Render", state.render3dMode],
  ]);
}

async function loadLeaves(): Promise<void> {
  if (!state.meta) return;
  setOverlay(state.leafBudget > 0 ? `Loading ${Math.round(state.leafBudget / 1000)}k display leaves` : "Loading all voxels");
  try {
    const leaves = await getLeaves(state.leafBudget > 0 ? state.leafBudget : null, 0);
    state.leaves = leaves;
    // Configure the min-mu slider to the actual attenuation range.
    const slider = $("minMu") as HTMLInputElement;
    slider.max = String(leaves.mu_range.max);
    slider.step = String(Math.max(leaves.mu_range.max / 200, 1e-4));
    if (!scene) {
      scene = new VoxelScene($("voxelStage"), { ...voxelOptions });
    }
    scene.setData(leaves);
    scene.setOptions({ ...voxelOptions });
    scene.setRenderMode(state.render3dMode);
    if (state.render3dMode === "xray" && !state.volumeTextureLoaded) {
      await setRender3dMode("xray");
    }
    // Ensure the render loop runs even if the voxel tab was already active on
    // load (setMode's scene?.start() is a no-op while the scene is still null).
    if (state.mode === "voxel") scene.start();
    updateVoxelLegend();
    updateVoxelHud();
  } finally {
    setOverlay(null);
  }
}

function decodeBase64Bytes(encoded: string): Uint8Array<ArrayBuffer> {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

async function setRender3dMode(mode: "voxels" | "xray"): Promise<void> {
  state.render3dMode = mode;
  if (!scene) return;
  $("voxelLegend").hidden = mode === "xray";
  if (mode === "voxels") {
    scene.setRenderMode("voxels");
    updateVoxelHud();
    return;
  }
  scene.setRenderMode("xray");
  if (state.volumeTextureLoaded) {
    updateVoxelHud();
    return;
  }
  setOverlay("Building 3D X-ray volume");
  try {
    const payload = await getVolumeTexture(128);
    scene.setVolumeData(
      decodeBase64Bytes(payload.data),
      payload.width,
      payload.height,
      payload.depth,
      payload.value_range.min,
      payload.value_range.max,
    );
    state.volumeTextureLoaded = true;
    scene.setRenderMode("xray");
    updateVoxelHud();
  } finally {
    setOverlay(null);
  }
}

// ----------------------------------------------------------- 2D image modes
function showImages(
  payload: {
    images: { prediction: string; target: string; error: string };
    metrics: { psnr: number; ssim: number; mae: number };
    elapsed_ms?: number | null;
    device?: string;
    native_cuda_integrator?: boolean;
  },
  predTag: string,
  gtTag: string,
): void {
  ($("imgPred") as HTMLImageElement).src = payload.images.prediction;
  ($("imgGt") as HTMLImageElement).src = payload.images.target;
  ($("imgErr") as HTMLImageElement).src = payload.images.error;
  $("imgPredTag").textContent = predTag;
  $("imgGtTag").textContent = gtTag;
  const deviceLabel = payload.device
    ? payload.native_cuda_integrator
      ? `${payload.device} (native)`
      : payload.device === "cuda"
        ? `${payload.device} (torch fallback, slow)`
        : payload.device
    : null;
  setMetrics([
    ["PSNR", `${fmt(payload.metrics.psnr, 2)} dB`],
    ["SSIM", fmt(payload.metrics.ssim, 4)],
    ["MAE", fmt(payload.metrics.mae, 5)],
    ["Render", payload.elapsed_ms ? `${fmt(payload.elapsed_ms, 1)} ms` : "cached"],
    ...(deviceLabel ? ([["Device", deviceLabel]] as Array<[string, string]>) : []),
  ]);
}

async function renderSlice(): Promise<void> {
  const axis = ($("axis") as HTMLSelectElement).value;
  const index = Number(($("sliceSlider") as HTMLInputElement).value);
  setOverlay("Rendering slice");
  try {
    const payload = await getSlice(axis, index);
    showImages(payload, `${axis} = ${payload.index}`, "vol_gt");
  } finally {
    setOverlay(null);
  }
}

async function renderProjection(): Promise<void> {
  const split = ($("split") as HTMLSelectElement).value;
  const view = Number(($("viewSlider") as HTMLInputElement).value);
  setOverlay("Rendering projection");
  try {
    const payload = await getProjection(split, view);
    showImages(payload, `${split} view ${payload.view}`, "stored");
  } finally {
    setOverlay(null);
  }
}

// ----------------------------------------------------------------- mode mgmt
function setMode(mode: Mode): void {
  state.mode = mode;
  document.querySelectorAll<HTMLButtonElement>("#modes button").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.mode === mode);
  });
  document.querySelectorAll<HTMLElement>("[data-controls]").forEach((el) => {
    el.hidden = el.dataset.controls !== mode;
  });
  const isVoxel = mode === "voxel";
  $("voxelStage").hidden = !isVoxel;
  $("voxelHud").hidden = !isVoxel;
  $("voxelLegend").hidden = !isVoxel;
  $("imageGrid").hidden = isVoxel;

  if (isVoxel) {
    scene?.start();
    if (!state.leaves) void loadLeaves();
    else { updateVoxelHud(); }
  } else {
    scene?.stop();
    if (mode === "slice") void renderSlice();
    else void renderProjection();
  }
}

// Hide slice/projection mode buttons when there is no dataset to compare
// against, leaving only the model-intrinsic voxel view.
function setModeAvailability(meta: StatePayload): void {
  // Only hide the comparison modes when the server *explicitly* reports no
  // dataset. An older server that predates this field returns undefined, in
  // which case it always has a dataset and the modes must stay visible.
  const comparisonModes = new Set(["slice", "projection"]);
  const hideComparisons = meta.dataset_available === false;
  document.querySelectorAll<HTMLButtonElement>("#modes button").forEach((btn) => {
    if (comparisonModes.has(btn.dataset.mode || "")) {
      btn.hidden = hideComparisons;
    }
  });
}

function bindControls(meta: StatePayload): void {
  document.querySelectorAll<HTMLButtonElement>("#modes button").forEach((btn) => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode as Mode));
  });

  $("colorMode").addEventListener("change", (e) => {
    const value = (e.target as HTMLSelectElement).value;
    if (value === "xray") {
      void setRender3dMode("xray");
      return;
    }
    voxelOptions.colorMode = value as ColorMode;
    scene?.setOptions({ colorMode: voxelOptions.colorMode });
    void setRender3dMode("voxels");
    updateVoxelLegend();
  });
  $("minMu").addEventListener("input", (e) => {
    voxelOptions.minMu = Number((e.target as HTMLInputElement).value);
    $("minMuVal").textContent = fmt(voxelOptions.minMu, 3);
    scene?.setOptions({ minMu: voxelOptions.minMu });
    updateVoxelHud();
  });
  $("opacity").addEventListener("input", (e) => {
    voxelOptions.opacity = Number((e.target as HTMLInputElement).value);
    scene?.setOptions({ opacity: voxelOptions.opacity });
  });
  $("shrink").addEventListener("input", (e) => {
    voxelOptions.shrink = Number((e.target as HTMLInputElement).value);
    scene?.setOptions({ shrink: voxelOptions.shrink });
  });
  $("wireframe").addEventListener("change", (e) => {
    voxelOptions.wireframe = (e.target as HTMLInputElement).checked;
    scene?.setOptions({ wireframe: voxelOptions.wireframe });
  });
  $("leafBudget").addEventListener("change", (e) => {
    state.leafBudget = Number((e.target as HTMLSelectElement).value);
    void loadLeaves();
  });
  $("xrayDensity").addEventListener("input", (event) => {
    const density = Number((event.target as HTMLInputElement).value);
    $("xrayDensityVal").textContent = density.toFixed(1);
    scene?.setXrayDensity(density);
  });
  $("resetView").addEventListener("click", () => scene?.resetView());

  // Slice and projection compare the model against ground truth, so they only
  // work when a dataset was loaded. In model-only mode the mode buttons are
  // hidden (see setModeAvailability) and these controls stay inert.
  const volumeShape = meta.volume_shape;
  if (meta.dataset_available === false || !volumeShape) return;

  // slice
  const sliceSlider = $("sliceSlider") as HTMLInputElement;
  sliceSlider.max = String(volumeShape[2] - 1);
  sliceSlider.value = String(Math.floor(volumeShape[2] / 2));
  $("sliceVal").textContent = sliceSlider.value;
  $("axis").addEventListener("change", () => {
    const axis = ($("axis") as HTMLSelectElement).value as "x" | "y" | "z";
    const dim = { x: 0, y: 1, z: 2 }[axis];
    sliceSlider.max = String(volumeShape[dim] - 1);
    sliceSlider.value = String(Math.floor(volumeShape[dim] / 2));
    $("sliceVal").textContent = sliceSlider.value;
    void renderSlice();
  });
  sliceSlider.addEventListener("input", () => { $("sliceVal").textContent = sliceSlider.value; });
  sliceSlider.addEventListener("change", () => { void renderSlice(); });

  // projection
  const viewSlider = $("viewSlider") as HTMLInputElement;
  viewSlider.max = String(Math.max(0, meta.test_views - 1));
  $("split").addEventListener("change", () => {
    const split = ($("split") as HTMLSelectElement).value;
    viewSlider.max = String(Math.max(0, (split === "train" ? meta.train_views : meta.test_views) - 1));
    if (Number(viewSlider.value) > Number(viewSlider.max)) viewSlider.value = "0";
    $("viewVal").textContent = viewSlider.value;
    void renderProjection();
  });
  viewSlider.addEventListener("input", () => { $("viewVal").textContent = viewSlider.value; });
  viewSlider.addEventListener("change", () => { void renderProjection(); });
}

async function boot(): Promise<void> {
  setOverlay("Connecting");
  try {
    const [meta, artifacts] = await Promise.all([getState(), getArtifacts()]);
    state.meta = meta;
    state.artifacts = artifacts;
    renderInspector(meta, artifacts);
    buildLevelToggles(meta);
    setModeAvailability(meta);
    bindControls(meta);
    bindModelPicker();
    setOverlay(null);
    setMode("voxel");
  } catch (error) {
    setOverlay(error instanceof Error ? error.message : "Failed to initialize viewer.");
  }
}

void boot();
