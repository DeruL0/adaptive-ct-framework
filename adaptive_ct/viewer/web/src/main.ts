import "./styles.css";
import {
  getArtifacts,
  getLeaves,
  getProjection,
  getSlice,
  getSources,
  getState,
  loadSource,
  type ArtifactsPayload,
  type LeavesPayload,
  type SourcesPayload,
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
      <button class="model-button" id="openSource" type="button">Choose model</button>
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
            <label class="field check"><input id="wireframe" type="checkbox"><span>Wireframe</span></label>
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
    <div class="source-backdrop" id="sourceBackdrop" hidden>
      <section class="source-dialog" role="dialog" aria-modal="true" aria-labelledby="sourceTitle">
        <div class="source-head">
          <div>
            <span class="eyebrow">Local workspace files</span>
            <h2 id="sourceTitle">Choose reconstruction</h2>
          </div>
          <button id="closeSource" type="button" aria-label="Close model chooser">&times;</button>
        </div>
        <label class="source-field">
          <span>Config (.yaml)</span>
          <input id="configPath" list="configFiles" autocomplete="off">
          <datalist id="configFiles"></datalist>
        </label>
        <label class="source-field">
          <span>Model (.pt / .npz)</span>
          <input id="checkpointPath" list="checkpointFiles" autocomplete="off" placeholder="Blank = checkpoint from config output">
          <datalist id="checkpointFiles"></datalist>
        </label>
        <p class="source-note" id="sourceWorkspace"></p>
        <p class="error-text" id="sourceError" hidden></p>
        <div class="source-actions">
          <button id="refreshSources" type="button">Refresh list</button>
          <button class="primary" id="loadSource" type="button">Load</button>
        </div>
      </section>
    </div>
  </div>
`;

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const state = {
  mode: "voxel" as Mode,
  meta: null as StatePayload | null,
  artifacts: null as ArtifactsPayload | null,
  sources: null as SourcesPayload | null,
  leaves: null as LeavesPayload | null,
  busy: false,
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

function populateSourceLists(sources: SourcesPayload, meta: StatePayload, resetInputs = false): void {
  state.sources = sources;
  const fill = (id: string, files: SourcesPayload["configs"]): void => {
    const list = $(id) as HTMLDataListElement;
    list.replaceChildren();
    files.forEach((file) => {
      const option = document.createElement("option");
      option.value = file.path;
      option.label = `${file.relative_path} · ${fmtBytes(file.size_bytes)}`;
      list.appendChild(option);
    });
  };
  fill("configFiles", sources.configs);
  fill("checkpointFiles", sources.checkpoints);
  $("sourceWorkspace").textContent =
    `${sources.workspace} · ${sources.configs.length} configs · ${sources.checkpoints.length} models`;
  if (resetInputs || !(($("configPath") as HTMLInputElement).value)) {
    ($("configPath") as HTMLInputElement).value = meta.paths.config;
  }
  if (resetInputs || !(($("checkpointPath") as HTMLInputElement).value)) {
    ($("checkpointPath") as HTMLInputElement).value = meta.paths.checkpoint || "";
  }
}

function setSourceDialog(open: boolean): void {
  $("sourceBackdrop").hidden = !open;
  if (open) {
    $("sourceError").hidden = true;
    ($("configPath") as HTMLInputElement).focus();
  }
}

function bindSourcePicker(meta: StatePayload, sources: SourcesPayload): void {
  populateSourceLists(sources, meta, true);
  $("openSource").addEventListener("click", () => setSourceDialog(true));
  $("closeSource").addEventListener("click", () => setSourceDialog(false));
  $("sourceBackdrop").addEventListener("click", (event) => {
    if (event.target === $("sourceBackdrop")) setSourceDialog(false);
  });
  $("refreshSources").addEventListener("click", async () => {
    const button = $("refreshSources") as HTMLButtonElement;
    button.disabled = true;
    try {
      populateSourceLists(await getSources(), meta);
    } catch (error) {
      $("sourceError").textContent = error instanceof Error ? error.message : "Could not refresh files.";
      $("sourceError").hidden = false;
    } finally {
      button.disabled = false;
    }
  });
  $("loadSource").addEventListener("click", async () => {
    const button = $("loadSource") as HTMLButtonElement;
    const config = ($("configPath") as HTMLInputElement).value.trim();
    const checkpoint = ($("checkpointPath") as HTMLInputElement).value.trim();
    $("sourceError").hidden = true;
    button.disabled = true;
    button.textContent = "Loading…";
    try {
      await loadSource(config, checkpoint);
      window.location.reload();
    } catch (error) {
      $("sourceError").textContent = error instanceof Error ? error.message : "Could not load reconstruction.";
      $("sourceError").hidden = false;
      button.disabled = false;
      button.textContent = "Load";
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
    `${meta.volume_shape.join("×")} volume`,
    `${meta.detector_shape.join("×")} detector`,
    `${meta.train_views}/${meta.test_views} views`,
    meta.device,
    meta.model.representation,
    meta.checkpoint_loaded ? "checkpoint loaded" : "no checkpoint",
  ].map((item) => `<span>${item}</span>`).join("");

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
  const mact = artifacts?.mact;
  const compact = artifacts?.compact_octree;
  const surface = artifacts?.surface;
  const mactSummary = mact?.summary;
  const surfaceStd = surface?.surface_std;
  const coeffStd = surface?.coefficient_std;
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
      <h2>MACT Export <span class="meta">${mact?.exists ? "loaded" : "missing"}</span></h2>
      <dl class="kv">
        <div><dt>File</dt><dd>${mact?.path || "-"}</dd></div>
        <div><dt>Compression</dt><dd>${mactSummary ? `${fmt(mactSummary.compression_ratio, 3)}× · ${fmtBytes(mactSummary.original_coefficient_bytes)} → ${fmtBytes(mactSummary.compressed_payload_bytes)}` : "-"}</dd></div>
        <div><dt>Groups</dt><dd>${mactSummary ? `${mactSummary.template_groups} template groups · ${mactSummary.material_clusters} material clusters` : "-"}</dd></div>
        <div><dt>Material histogram</dt><dd>${mact?.material_histogram?.join(" / ") || "-"}</dd></div>
        <div><dt>Orientation histogram</dt><dd>${mact?.orientation_histogram?.join(" / ") || "-"}</dd></div>
      </dl>
      ${mact?.error ? `<p class="error-text">${mact.error}</p>` : ""}
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
      <h2>Surface Uncertainty <span class="meta">${surface?.exists ? "loaded" : "missing"}</span></h2>
      <dl class="kv">
        <div><dt>File</dt><dd>${surface?.path || "-"}</dd></div>
        <div><dt>Surface points</dt><dd>${surface?.surface_point_count?.toLocaleString() || "-"} @ threshold ${fmt(surface?.threshold, 3)}</dd></div>
        <div><dt>Surface std</dt><dd>${surfaceStd ? `mean ${fmt(surfaceStd.mean, 4)} · p95 ${fmt(surfaceStd.p95, 4)} · n=${surfaceStd.count.toLocaleString()}` : "-"}</dd></div>
        <div><dt>Coefficient std</dt><dd>${coeffStd ? `mean ${fmt(coeffStd.mean, 4)} · p95 ${fmt(coeffStd.p95, 4)} · n=${coeffStd.count.toLocaleString()}` : "-"}</dd></div>
      </dl>
      ${surface?.error ? `<p class="error-text">${surface.error}</p>` : ""}
    </section>
    <section class="section">
      <h2>Checkpoint <span class="meta">${meta.checkpoint_loaded ? "loaded" : "not loaded"}</span></h2>
      <dl class="kv">
        <div><dt>Config</dt><dd>${meta.paths.config}</dd></div>
        <div><dt>Checkpoint</dt><dd>${meta.paths.checkpoint || meta.checkpoint_error || "-"}</dd></div>
        <div><dt>Dataset</dt><dd>${meta.paths.dataset}</dd></div>
        <div><dt>Output</dt><dd>${meta.paths.output}</dd></div>
      </dl>
      ${meta.checkpoint_error ? `<p class="error-text">${meta.checkpoint_error}</p>` : ""}
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
    `color: ${voxelOptions.colorMode}`,
    `min μ ${fmt(voxelOptions.minMu, 3)}`,
  ].map((t) => `<span>${t}</span>`).join("");
  setMetrics([
    ["Visible leaves", (scene?.visible ?? 0).toLocaleString()],
    ["Total leaves", (leaves?.total ?? 0).toLocaleString()],
    ["Returned", (leaves?.count ?? 0).toLocaleString()],
    ["Color mode", voxelOptions.colorMode],
  ]);
}

async function loadLeaves(): Promise<void> {
  if (!state.meta) return;
  setOverlay("Loading all voxels");
  try {
    const leaves = await getLeaves(null, 0);
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
    // Ensure the render loop runs even if the voxel tab was already active on
    // load (setMode's scene?.start() is a no-op while the scene is still null).
    if (state.mode === "voxel") scene.start();
    updateVoxelLegend();
    updateVoxelHud();
  } finally {
    setOverlay(null);
  }
}

// ----------------------------------------------------------- 2D image modes
function showImages(payload: { images: { prediction: string; target: string; error: string }; metrics: { psnr: number; ssim: number; mae: number }; elapsed_ms?: number | null }, predTag: string, gtTag: string): void {
  ($("imgPred") as HTMLImageElement).src = payload.images.prediction;
  ($("imgGt") as HTMLImageElement).src = payload.images.target;
  ($("imgErr") as HTMLImageElement).src = payload.images.error;
  $("imgPredTag").textContent = predTag;
  $("imgGtTag").textContent = gtTag;
  setMetrics([
    ["PSNR", `${fmt(payload.metrics.psnr, 2)} dB`],
    ["SSIM", fmt(payload.metrics.ssim, 4)],
    ["MAE", fmt(payload.metrics.mae, 5)],
    ["Render", payload.elapsed_ms ? `${fmt(payload.elapsed_ms, 1)} ms` : "cached"],
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

function bindControls(meta: StatePayload): void {
  document.querySelectorAll<HTMLButtonElement>("#modes button").forEach((btn) => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode as Mode));
  });

  $("colorMode").addEventListener("change", (e) => {
    voxelOptions.colorMode = (e.target as HTMLSelectElement).value as ColorMode;
    scene?.setOptions({ colorMode: voxelOptions.colorMode });
    updateVoxelLegend();
    updateVoxelHud();
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
  $("resetView").addEventListener("click", () => scene?.resetView());

  // slice
  const sliceSlider = $("sliceSlider") as HTMLInputElement;
  sliceSlider.max = String(meta.volume_shape[2] - 1);
  sliceSlider.value = String(Math.floor(meta.volume_shape[2] / 2));
  $("sliceVal").textContent = sliceSlider.value;
  $("axis").addEventListener("change", () => {
    const axis = ($("axis") as HTMLSelectElement).value as "x" | "y" | "z";
    const dim = { x: 0, y: 1, z: 2 }[axis];
    sliceSlider.max = String(meta.volume_shape[dim] - 1);
    sliceSlider.value = String(Math.floor(meta.volume_shape[dim] / 2));
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
    const [meta, artifacts, sources] = await Promise.all([getState(), getArtifacts(), getSources()]);
    state.meta = meta;
    state.artifacts = artifacts;
    state.sources = sources;
    renderInspector(meta, artifacts);
    buildLevelToggles(meta);
    bindControls(meta);
    bindSourcePicker(meta, sources);
    setOverlay(null);
    setMode("voxel");
  } catch (error) {
    setOverlay(error instanceof Error ? error.message : "Failed to initialize viewer.");
  }
}

void boot();
