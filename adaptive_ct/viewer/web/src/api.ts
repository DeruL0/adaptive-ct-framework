// Typed wrappers around the Python viewer HTTP API plus binary decoders.

export interface LevelStructure {
  level: number;
  resolution: number;
  count: number;
  voxel_size: number;
}

export interface GrowthEvent {
  iteration: number;
  level: number;
  operation?: string;
  strategy: string;
  active_fraction: number;
  halo: number;
  active: number;
  target_degree?: number | number[];
  old_coefficient_count?: number;
  new_coefficient_count?: number;
  added_coefficients?: number;
  resolution?: number;
  score_mean?: number;
  gradient_mean?: number;
  sigma_mean?: number;
  projected_residual_mean?: number;
}

export interface AdaptiveEvent {
  iteration: number;
  accepted_count: number;
  candidate_count: number;
  model_bytes?: number;
  coefficient_count?: number;
}

export interface RateDistortionEvaluation {
  validation_gain: number;
  rate_delta_bytes: number;
  score_per_byte: number;
  accepted: boolean;
}

export interface RateDistortionEvent {
  iteration: number;
  model_bytes: number;
  accepted_count: number;
  evaluations: RateDistortionEvaluation[];
}

export interface ModelStats {
  representation: string;
  parameter_count: number;
  model_bytes: number;
  leaf_cells: number;
  max_depth: number;
  l0_active: number;
  l1_active: number;
  l2_active: number;
  l3_active?: number;
  active_by_level?: number[];
  coefficient_count?: number;
  max_degree?: [number, number, number];
}

export interface TensorStats {
  count: number;
  min: number | null;
  max: number | null;
  mean: number | null;
  p95: number | null;
}

export interface ArtifactsPayload {
  kind: "rd_cvf_artifacts";
  output: string;
  checkpoint: { exists: boolean; path: string | null; size_bytes: number };
  report: { exists: boolean; path: string | null; size_bytes: number };
  mact: {
    exists: boolean;
    path: string;
    size_bytes: number;
    error: string | null;
    representation?: string;
    summary?: {
      leaf_count: number;
      material_clusters: number;
      template_groups: number;
      original_coefficient_bytes: number;
      compressed_payload_bytes: number;
      compression_ratio: number;
    };
    material_centres?: number[];
    material_histogram?: number[];
    orientation_histogram?: number[];
    top_groups?: Array<{ degree: number[]; material_id: number; orientation: number; leaf_count: number; rank: number }>;
  };
  compact_octree: {
    exists: boolean;
    path: string;
    size_bytes: number;
    error: string | null;
    metadata?: Record<string, unknown>;
    leaf_count?: number;
    coefficient_count?: number;
    arrays?: Record<string, { shape: number[]; dtype: string; bytes: number }>;
  };
  surface: {
    exists: boolean;
    path: string;
    size_bytes: number;
    error: string | null;
    threshold?: number;
    surface_point_count?: number;
    coefficient_std?: TensorStats;
    surface_std?: TensorStats;
  };
}

export interface StatePayload {
  name: string;
  device: string;
  checkpoint_loaded: boolean;
  checkpoint_error: string | null;
  paths: { config: string; checkpoint: string | null; dataset: string; output: string; report: string | null };
  volume_shape: [number, number, number];
  detector_shape: [number, number];
  train_views: number;
  test_views: number;
  train_angles: number[];
  test_angles: number[];
  samples_per_ray: number;
  material_threshold: number;
  model: ModelStats;
  leaf_structure: { levels: LevelStructure[]; size_range: { min: number; max: number } | null; leaf_cells: number };
  report: {
    elapsed_sec: number | null;
    projection_test: { psnr: number; mae: number } | null;
    material_volume: { psnr: number; mae: number; count: number } | null;
    boundary_sharpness: {
      boundary_voxels: number;
      gt_gradient_mean: number;
      pred_gradient_mean: number;
      gradient_ratio: number;
      high_gradient_mae: number;
      boundary_quantile: number;
    } | null;
    ray_integration: Record<string, unknown> | null;
    training_objective: Record<string, unknown> | null;
    exports: Record<string, unknown> | null;
    growth_events: GrowthEvent[] | null;
    adaptive_events: AdaptiveEvent[] | null;
    rate_distortion_curve: RateDistortionEvent[] | null;
    history: Array<Record<string, number>> | null;
  };
}

export interface LeavesPayload {
  kind: "leaves";
  count: number;
  total: number;
  level_counts: number[];
  level_sizes: number[];
  level_sizes_xyz: number[][];
  resolutions: number[][];
  mu_range: { min: number; max: number };
  size_range: { min: number; max: number } | null;
  min_mu: number;
  coords: Uint16Array;
  mu: Float32Array;
  levels: Uint8Array;
  /** True when any leaf is a non-constant (p1) Bernstein block; only then
   * are `cornersLow`/`cornersHigh` present, so p1 leaves can be shaded by
   * their real corner values instead of a single flat block color. */
  hasCorners: boolean;
  cornersLow?: Float32Array;
  cornersHigh?: Float32Array;
}

export interface ImageTriple {
  prediction: string;
  target: string;
  error: string;
}

export interface Metrics {
  psnr: number;
  ssim: number;
  mae: number;
  max_abs_error: number;
}

export interface ProjectionPayload {
  kind: "projection";
  split: string;
  view: number;
  angle_rad: number;
  elapsed_ms: number | null;
  device: string;
  native_cuda_integrator: boolean;
  metrics: Metrics;
  images: ImageTriple;
}

export interface SlicePayload {
  kind: "slice";
  axis: string;
  index: number;
  elapsed_ms: number | null;
  device: string;
  metrics: Metrics;
  images: ImageTriple;
}

export interface SourceFile {
  path: string;
  relative_path: string;
  size_bytes: number;
  modified_time: number;
}

export interface SourcesPayload {
  workspace: string;
  configs: SourceFile[];
  checkpoints: SourceFile[];
}

export interface LoadSourcePayload {
  status: "ok";
  state: StatePayload;
  artifacts: ArtifactsPayload;
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, { cache: "no-store", ...init });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload?.error || `Request failed: ${response.status}`);
  return payload as T;
}

export async function getState(): Promise<StatePayload> {
  return fetchJson<StatePayload>("/api/state");
}

export async function getArtifacts(): Promise<ArtifactsPayload> {
  return fetchJson<ArtifactsPayload>("/api/artifacts");
}

export async function getSources(): Promise<SourcesPayload> {
  return fetchJson<SourcesPayload>("/api/sources");
}

export async function loadSource(config: string, checkpoint: string): Promise<LoadSourcePayload> {
  return fetchJson<LoadSourcePayload>("/api/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config, checkpoint }),
  });
}

export async function getLeaves(maxLeaves: number | null, minMu: number): Promise<LeavesPayload> {
  const budget = maxLeaves === null ? "" : `&max_leaves=${Math.round(maxLeaves)}`;
  const response = await fetch(
    `/api/leaves.bin?min_mu=${encodeURIComponent(String(minMu))}${budget}`,
    { cache: "no-store" },
  );
  if (!response.ok) throw new Error(`Leaf stream failed: ${response.status}`);
  const buffer = await response.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const magic = new TextDecoder().decode(bytes.subarray(0, 8));
  if (magic !== "ACTLEAF1") throw new Error(`Unsupported leaf stream: ${magic}`);
  const headerBytes = new DataView(buffer).getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + headerBytes))) as Record<string, unknown>;
  const payloadOffset = 12 + headerBytes;
  const count = Number(header.count) || 0;
  const coordsOffset = payloadOffset + Number(header.coords_offset);
  const muOffset = payloadOffset + Number(header.mu_offset);
  const levelsOffset = payloadOffset + Number(header.levels_offset);
  const hasCorners = Boolean(header.has_corners);
  let cornersLow: Float32Array | undefined;
  let cornersHigh: Float32Array | undefined;
  if (hasCorners) {
    const cornersOffset = payloadOffset + Number(header.corners_offset);
    const interleaved = new Float32Array(buffer, cornersOffset, count * 8);
    cornersLow = new Float32Array(count * 4);
    cornersHigh = new Float32Array(count * 4);
    for (let leaf = 0; leaf < count; leaf += 1) {
      const base = leaf * 8;
      cornersLow.set(interleaved.subarray(base, base + 4), leaf * 4);
      cornersHigh.set(interleaved.subarray(base + 4, base + 8), leaf * 4);
    }
  }
  return {
    kind: "leaves",
    count,
    total: Number(header.total) || 0,
    level_counts: (header.level_counts as number[]) || [],
    level_sizes: (header.level_sizes as number[]) || [],
    level_sizes_xyz: (header.level_sizes_xyz as number[][]) || [],
    resolutions: (header.resolutions as number[][]) || [],
    mu_range: header.mu_range as { min: number; max: number },
    size_range: (header.size_range as { min: number; max: number } | null) ?? null,
    min_mu: Number(header.min_mu) || 0,
    coords: new Uint16Array(buffer, coordsOffset, count * 3),
    mu: new Float32Array(buffer, muOffset, count),
    levels: new Uint8Array(buffer, levelsOffset, count),
    hasCorners,
    cornersLow,
    cornersHigh,
  };
}

export async function getProjection(split: string, view: number): Promise<ProjectionPayload> {
  return fetchJson<ProjectionPayload>(`/api/projection?split=${encodeURIComponent(split)}&view=${view}`);
}

export async function getSlice(axis: string, index: number): Promise<SlicePayload> {
  return fetchJson<SlicePayload>(`/api/slice?axis=${encodeURIComponent(axis)}&index=${index}`);
}
