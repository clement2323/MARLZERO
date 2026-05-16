import { measureNaturalWidth, prepareWithSegments } from "@chenglou/pretext";

// Pretext measures via the browser font engine (canvas). SVG renders via the
// same engine, so a natural width measured here applies to <text> rendered
// with the same `font-family` / `font-size`. We use that to pick a
// letter-spacing that justifies a string to an exact target advance.

export interface JustifiedText {
  text: string;
  letterSpacing: number;
}

const widthCache = new Map<string, number>();

function naturalWidth(text: string, font: string): number {
  if (text.length === 0) return 0;
  const key = `${font} ${text}`;
  const hit = widthCache.get(key);
  if (hit !== undefined) return hit;
  const w = measureNaturalWidth(prepareWithSegments(text, font));
  widthCache.set(key, w);
  return w;
}

function evenLetterSpacing(text: string, font: string, target: number): number {
  const w = naturalWidth(text, font);
  const gaps = Math.max(1, [...text].length - 1);
  return (target - w) / gaps;
}

// Repeat `unit` enough times to cover roughly `target`, then space it to land exactly.
export function justifyRepeatToWidth(
  unit: string,
  font: string,
  target: number,
): JustifiedText {
  const unitW = naturalWidth(unit, font);
  if (unitW <= 0 || target <= 0) return { text: "", letterSpacing: 0 };
  const repeats = Math.max(1, Math.floor(target / unitW));
  const text = unit.repeat(repeats);
  return { text, letterSpacing: evenLetterSpacing(text, font, target) };
}

// Repeat `unit` around a closed circular path of given radius.
export function justifyRepeatToCircumference(
  unit: string,
  font: string,
  radius: number,
): JustifiedText {
  return justifyRepeatToWidth(unit, font, 2 * Math.PI * radius);
}
