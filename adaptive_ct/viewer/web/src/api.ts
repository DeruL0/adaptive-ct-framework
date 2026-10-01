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

export interface ArtifactsPayload {
  kind: "rd_cvf_artifacts";
  output: string;
  checkpoint: { exists: boolean; path: string | null; size_bytes: number };
  report: { exists: boolean; path: string | null; size_bytes: number };
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
}

export interface StatePayload {
  name: string;
  device: string;
  checkpoint_loaded: boolean;
  checkpoint_error: string | null;
  /** False when the model was opened without a dataset/parameter YAML: only the
   * voxel view and model stats are available, slice/projection are hidden. */
  dataset_available: boolean;
  dataset_error: string | null;
  paths: { config: string | null; checkpoint: string | null; dataset: string | null; output: string; report: string | null };
  volume_shape: [number, number, number] | null;
  detector_shape: [number, number] | null;
  train_views: number;
  test_views: number;
  train_angles: number[];
  test_angles: number[];
  samples_per_ray: number | null;
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
   * is `corners` present, so p1 leaves can be shaded by
   * their real corner values instead of a single flat block color. */
  hasCorners: boolean;
  /** Interleaved low/high corner groups, eight floats per leaf. */
  corners?: Float32Array;
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

export interface VolumeTexturePayload {
  kind: "volume_texture";
  source: "prediction";
  shape: [number, number, number];
  width: number;
  height: number;
  depth: number;
  value_range: { min: number; max: number };
  encoding: "uint8_base64_x_fastest";
  data: string;
}

export interface SelectModelPayload {
  status: "ok" | "cancelled";
  checkpoint?: string;
}

export interface LoadModelPayload {
  status: "ok";
  config: string | null;
  checkpoint: string;
  dataset_available: boolean;
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

export async function selectModel(): Promise<SelectModelPayload> {
  return fetchJson<SelectModelPayload>("/api/select-model", {
    method: "POST",
  });
}

export async function loadModel(checkpoint: string): Promise<LoadModelPayload> {
  return fetchJson<LoadModelPayload>("/api/load-model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ checkpoint }),
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
  let corners: Float32Array | undefined;
  if (hasCorners) {
    const cornersOffset = payloadOffset + Number(header.corners_offset);
    corners = new Float32Array(buffer, cornersOffset, count * 8);
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
    corners,
  };
}

export async function getProjection(split: string, view: number): Promise<ProjectionPayload> {
  return fetchJson<ProjectionPayload>(`/api/projection?split=${encodeURIComponent(split)}&view=${view}`);
}

export async function getSlice(axis: string, index: number): Promise<SlicePayload> {
  return fetchJson<SlicePayload>(`/api/slice?axis=${encodeURIComponent(axis)}&index=${index}`);
}

export async function getVolumeTexture(resolution = 192): Promise<VolumeTexturePayload> {
  return fetchJson<VolumeTexturePayload>(`/api/volume_texture?resolution=${Math.round(resolution)}`);
}
