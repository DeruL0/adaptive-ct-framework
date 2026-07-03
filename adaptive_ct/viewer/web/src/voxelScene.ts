import {
  AmbientLight,
  BoxGeometry,
  Color,
  DirectionalLight,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  Mesh,
  PerspectiveCamera,
  Scene,
  ShaderMaterial,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { levelColor } from "./colormaps";
import type { LeavesPayload } from "./api";

export type ColorMode = "level" | "mu" | "size";

export interface VoxelOptions {
  colorMode: ColorMode;
  minMu: number;
  levelVisible: boolean[];
  opacity: number;
  wireframe: boolean;
  shrink: number;
}

const MAX_SHADER_LEVELS = 8;

const VERTEX_SHADER = `
  attribute vec3 instanceCoord;
  attribute float instanceMu;
  attribute float instanceLevel;
  #ifdef HAS_CORNERS
  // Corner order matches bernstein._all_coords(2): (i,j,k) with k fastest.
  // low = i=0 side (000,001,010,011), high = i=1 side (100,101,110,111).
  attribute vec4 instanceCornersLow;
  attribute vec4 instanceCornersHigh;
  #endif

  uniform int uColorMode;
  uniform float uMinMu;
  uniform float uMuMin;
  uniform float uMuSpan;
  uniform float uSizeMin;
  uniform float uSizeSpan;
  uniform float uKeep;
  uniform vec3 uResolutions[${MAX_SHADER_LEVELS}];
  uniform float uLevelVisible[${MAX_SHADER_LEVELS}];
  uniform vec3 uLevelColors[${MAX_SHADER_LEVELS}];

  varying vec3 vColor;
  varying vec3 vNormal;
  varying float vVisible;

  vec3 ramp(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 cold = vec3(0.08, 0.35, 0.92);
    vec3 middle = vec3(0.10, 0.88, 0.74);
    vec3 warm = vec3(1.0, 0.72, 0.10);
    vec3 hot = vec3(0.95, 0.12, 0.08);
    if (t < 0.4) return mix(cold, middle, t / 0.4);
    if (t < 0.75) return mix(middle, warm, (t - 0.4) / 0.35);
    return mix(warm, hot, (t - 0.75) / 0.25);
  }

  void main() {
    int levelIndex = int(clamp(floor(instanceLevel + 0.5), 0.0, ${MAX_SHADER_LEVELS - 1}.0));
    float shown = step(uMinMu, instanceMu) * uLevelVisible[levelIndex];
    vec3 ctSize = 2.0 / uResolutions[levelIndex];
    vec3 ctCenter = -1.0 + (instanceCoord + 0.5) * ctSize;
    vec3 instanceOffset = vec3(ctCenter.x, ctCenter.z, ctCenter.y);
    vec3 instanceSize = vec3(ctSize.x, ctSize.z, ctSize.y);
    vec3 scaledSize = instanceSize * uKeep * shown;
    vec3 localPosition = position * scaledSize + instanceOffset;

    // A p1 (trilinear) leaf is not a flat block: evaluate its own corner
    // values at this vertex instead of reusing the leaf-wide mean, so the
    // box's shading actually shows the stored gradient (pipeline v5 step 9).
    float effectiveMu = instanceMu;
    #ifdef HAS_CORNERS
    {
      // position is the box's own local axis; the renderer swaps CT y/z
      // for a Y-up scene (see instanceOffset/instanceSize above), so local
      // y carries CT z and local z carries CT y.
      float u = position.x + 0.5;
      float v = position.z + 0.5;
      float w = position.y + 0.5;
      float wu0 = 1.0 - u;
      float wv0 = 1.0 - v;
      float ww0 = 1.0 - w;
      effectiveMu =
        wu0 * wv0 * ww0 * instanceCornersLow.x +
        wu0 * wv0 * w   * instanceCornersLow.y +
        wu0 * v   * ww0 * instanceCornersLow.z +
        wu0 * v   * w   * instanceCornersLow.w +
        u   * wv0 * ww0 * instanceCornersHigh.x +
        u   * wv0 * w   * instanceCornersHigh.y +
        u   * v   * ww0 * instanceCornersHigh.z +
        u   * v   * w   * instanceCornersHigh.w;
    }
    #endif

    float geometricSize = pow(max(instanceSize.x * instanceSize.y * instanceSize.z, 1e-20), 1.0 / 3.0);
    if (uColorMode == 0) {
      vColor = uLevelColors[levelIndex];
    } else if (uColorMode == 1) {
      vColor = ramp((effectiveMu - uMuMin) / uMuSpan);
    } else {
      vColor = ramp(1.0 - (geometricSize - uSizeMin) / uSizeSpan);
    }

    vVisible = shown;
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(localPosition, 1.0);
  }
`;

const FRAGMENT_SHADER = `
  uniform float uOpacity;
  varying vec3 vColor;
  varying vec3 vNormal;
  varying float vVisible;

  void main() {
    if (vVisible < 0.5) discard;
    vec3 lightDirection = normalize(vec3(0.45, 0.75, 0.55));
    float diffuse = 0.38 + 0.62 * abs(dot(normalize(vNormal), lightDirection));
    gl_FragColor = vec4(vColor * diffuse, uOpacity);
  }
`;

/**
 * GPU-driven adaptive voxel renderer.
 *
 * One compact vec3 offset, vec3 size, scalar attenuation, and byte level are
 * uploaded per leaf. Scaling, filtering, level visibility, and coloring happen
 * in the vertex shader. In particular, this avoids allocating one 4x4 matrix
 * and running setMatrixAt/setColorAt in JavaScript for every leaf.
 */
export class VoxelScene {
  readonly host: HTMLElement;
  private renderer: WebGLRenderer;
  private scene: Scene;
  private camera: PerspectiveCamera;
  private controls: OrbitControls;
  private baseGeometry: BoxGeometry;
  private instanceGeometry: InstancedBufferGeometry | null = null;
  private material: ShaderMaterial;
  private mesh: Mesh<InstancedBufferGeometry, ShaderMaterial> | null = null;
  private data: LeavesPayload | null = null;
  private options: VoxelOptions;
  private running = false;
  private dirty = true;
  private resizeObserver: ResizeObserver;
  private visibleCount = 0;

  constructor(host: HTMLElement, options: VoxelOptions) {
    this.host = host;
    this.options = options;

    this.renderer = new WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    host.replaceChildren(this.renderer.domElement);

    this.scene = new Scene();
    this.scene.background = new Color("#080d12");

    this.camera = new PerspectiveCamera(38, 1, 0.01, 100);
    this.camera.position.set(2.4, 1.8, 2.6);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0, 0);
    this.controls.addEventListener("change", () => { this.dirty = true; });

    // Kept in the scene for compatibility with future materials. The current
    // shader uses a fixed view-space light to minimize per-frame state.
    this.scene.add(new AmbientLight(0xffffff, 0.62));
    const key = new DirectionalLight(0xffffff, 0.85);
    key.position.set(2, 3, 2);
    this.scene.add(key);

    this.baseGeometry = new BoxGeometry(1, 1, 1);
    const levelColors = Array.from({ length: MAX_SHADER_LEVELS }, (_, level) => levelColor(level).clone());
    this.material = new ShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      uniforms: {
        uColorMode: { value: 0 },
        uMinMu: { value: 0 },
        uMuMin: { value: 0 },
        uMuSpan: { value: 1 },
        uSizeMin: { value: 0 },
        uSizeSpan: { value: 1 },
        uKeep: { value: 0.96 },
        uResolutions: {
          value: Array.from({ length: MAX_SHADER_LEVELS }, () => new Vector3(1, 1, 1)),
        },
        uLevelVisible: { value: new Float32Array(MAX_SHADER_LEVELS).fill(1) },
        uLevelColors: { value: levelColors },
        uOpacity: { value: 1 },
      },
    });
    this.applyMaterialOptions();

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(host);
    this.resize();
  }

  setData(data: LeavesPayload): void {
    this.data = data;
    if (this.mesh) this.scene.remove(this.mesh);
    this.instanceGeometry?.dispose();
    this.mesh = null;
    this.instanceGeometry = null;

    const hasCorners = data.hasCorners && !!data.cornersLow && !!data.cornersHigh;
    if (this.material.defines?.HAS_CORNERS !== (hasCorners ? 1 : undefined)) {
      this.material.defines = hasCorners ? { HAS_CORNERS: 1 } : {};
      this.material.needsUpdate = true;
    }

    if (data.count > 0) {
      const geometry = new InstancedBufferGeometry();
      geometry.index = this.baseGeometry.index;
      geometry.setAttribute("position", this.baseGeometry.getAttribute("position"));
      geometry.setAttribute("normal", this.baseGeometry.getAttribute("normal"));
      geometry.setAttribute("instanceCoord", new InstancedBufferAttribute(data.coords, 3, false));
      geometry.setAttribute("instanceMu", new InstancedBufferAttribute(data.mu, 1));
      geometry.setAttribute("instanceLevel", new InstancedBufferAttribute(data.levels, 1, false));
      if (hasCorners) {
        geometry.setAttribute("instanceCornersLow", new InstancedBufferAttribute(data.cornersLow!, 4));
        geometry.setAttribute("instanceCornersHigh", new InstancedBufferAttribute(data.cornersHigh!, 4));
      }
      geometry.instanceCount = data.count;
      this.instanceGeometry = geometry;
      this.mesh = new Mesh(geometry, this.material);
      this.mesh.frustumCulled = false;
      this.scene.add(this.mesh);
    }

    this.updateUniforms();
    this.updateVisibleCount();
    this.dirty = true;
  }

  setOptions(partial: Partial<VoxelOptions>): void {
    const filterChanged =
      (partial.minMu !== undefined && partial.minMu !== this.options.minMu)
      || (partial.levelVisible !== undefined
        && partial.levelVisible.some((value, index) => value !== this.options.levelVisible[index]));
    this.options = { ...this.options, ...partial };
    this.applyMaterialOptions();
    this.updateUniforms();
    if (filterChanged) this.updateVisibleCount();
    this.dirty = true;
  }

  get visible(): number {
    return this.visibleCount;
  }

  private applyMaterialOptions(): void {
    this.material.wireframe = this.options.wireframe;
    this.material.transparent = this.options.opacity < 0.999;
    this.material.depthWrite = !this.material.transparent;
    this.material.uniforms.uOpacity.value = this.options.opacity;
    this.material.needsUpdate = true;
  }

  private updateUniforms(): void {
    const data = this.data;
    const uniforms = this.material.uniforms;
    uniforms.uColorMode.value = this.options.colorMode === "level" ? 0 : this.options.colorMode === "mu" ? 1 : 2;
    uniforms.uMinMu.value = this.options.minMu;
    uniforms.uKeep.value = 1 - Math.max(0, Math.min(0.5, this.options.shrink));
    uniforms.uOpacity.value = this.options.opacity;
    const visibility = uniforms.uLevelVisible.value as Float32Array;
    for (let level = 0; level < MAX_SHADER_LEVELS; level += 1) {
      visibility[level] = (this.options.levelVisible[level] ?? true) ? 1 : 0;
    }
    if (data) {
      const resolutions = uniforms.uResolutions.value as Vector3[];
      for (let level = 0; level < MAX_SHADER_LEVELS; level += 1) {
        const shape = data.resolutions[level] ?? [1, 1, 1];
        resolutions[level].set(shape[0], shape[1], shape[2]);
      }
      uniforms.uMuMin.value = data.mu_range.min;
      uniforms.uMuSpan.value = Math.max(data.mu_range.max - data.mu_range.min, 1e-6);
      uniforms.uSizeMin.value = data.size_range?.min ?? Math.min(...data.level_sizes);
      uniforms.uSizeSpan.value = Math.max(
        (data.size_range?.max ?? Math.max(...data.level_sizes)) - uniforms.uSizeMin.value,
        1e-6,
      );
    }
  }

  private updateVisibleCount(): void {
    const data = this.data;
    if (!data) {
      this.visibleCount = 0;
      return;
    }
    const allLevelsVisible = data.level_counts.every((_, level) => this.options.levelVisible[level] ?? true);
    if (allLevelsVisible && this.options.minMu <= data.mu_range.min) {
      this.visibleCount = data.count;
      return;
    }
    let visible = 0;
    for (let index = 0; index < data.count; index += 1) {
      if (data.mu[index] >= this.options.minMu && (this.options.levelVisible[data.levels[index]] ?? true)) {
        visible += 1;
      }
    }
    this.visibleCount = visible;
  }

  resetView(): void {
    this.camera.position.set(2.4, 1.8, 2.6);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
    this.dirty = true;
  }

  private resize(): void {
    const rect = this.host.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.resize();
    const loop = () => {
      if (!this.running) return;
      const damping = this.controls.update();
      if (this.dirty || damping) {
        this.dirty = false;
        this.renderer.render(this.scene, this.camera);
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  stop(): void {
    this.running = false;
  }

  dispose(): void {
    this.stop();
    this.resizeObserver.disconnect();
    this.controls.dispose();
    if (this.mesh) this.scene.remove(this.mesh);
    this.instanceGeometry?.dispose();
    this.baseGeometry.dispose();
    this.material.dispose();
    this.renderer.dispose();
  }
}
