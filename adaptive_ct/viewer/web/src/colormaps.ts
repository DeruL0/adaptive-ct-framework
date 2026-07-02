import { Color } from "three";

// Distinct hues per refinement level (L0 coarse -> L2 fine).
export const LEVEL_COLORS = ["#5b8cff", "#f0b450", "#ff6b8a", "#8b5bff", "#5bffce"];

export function levelColor(level: number): Color {
  return new Color(LEVEL_COLORS[level % LEVEL_COLORS.length]);
}

// Teal viridis-ish ramp for attenuation / scalar fields.
const RAMP: Array<[number, string]> = [
  [0.0, "#0b1118"],
  [0.35, "#1f6f63"],
  [0.7, "#6ed6c8"],
  [1.0, "#f6fffb"],
];

const _rampColors = RAMP.map(([, hex]) => new Color(hex));
const _rampStops = RAMP.map(([stop]) => stop);

export function rampColor(t: number, out: Color = new Color()): Color {
  const x = Math.max(0, Math.min(1, t));
  for (let i = 1; i < _rampStops.length; i += 1) {
    if (x <= _rampStops[i]) {
      const a = _rampStops[i - 1];
      const b = _rampStops[i];
      const f = (x - a) / Math.max(b - a, 1e-6);
      return out.copy(_rampColors[i - 1]).lerp(_rampColors[i], f);
    }
  }
  return out.copy(_rampColors[_rampColors.length - 1]);
}

export const RAMP_CSS = `linear-gradient(90deg, ${RAMP.map(([, hex]) => hex).join(", ")})`;
