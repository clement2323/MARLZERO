import { useEffect, useRef, useState } from "react";
import {
  layoutNextLineRange,
  materializeLineRange,
  prepareWithSegments,
  type LayoutCursor,
} from "@chenglou/pretext";

// Two pretext layers in one SVG: a static-ish rules paragraph, and a snake of
// "LOSER LOSER…" that crosses the area on click. Each animation frame, the
// snake's body samples become circular obstacles, the per-row available width
// is recomputed, and pretext re-flows the paragraph. The line breaks shift
// continuously as LOSER passes through — that's the demo.

const RULES_TEXT =
  "Each player starts with nine pieces in hand. In the placement phase they alternate dropping a piece on any empty intersection until the eighteen are down. In the movement phase a piece slides along a board line to an adjacent empty intersection — there is no flying. Aligning three of your pieces along a line forms a mill, which lets you remove one of your opponent's pieces; pieces sitting inside an opponent's mill are protected unless every opposing piece is itself in a mill. A player loses when reduced to two pieces or when no legal move remains; draws come from threefold repetition or 300 halfmoves without progress.";

const PARAGRAPH_FONT_FAMILY =
  '"Inter", "SF Pro Text", system-ui, -apple-system, sans-serif';
const PARAGRAPH_FONT_SIZE = 14;
const PARAGRAPH_FONT = `${PARAGRAPH_FONT_SIZE}px ${PARAGRAPH_FONT_FAMILY}`;
const LINE_HEIGHT = 22;

const SNAKE_FONT_FAMILY =
  'ui-monospace, "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace';
const SNAKE_FONT_SIZE = 18;
const SNAKE_TEXT = "LOSER ".repeat(80);

const AREA_WIDTH = 560;
const AREA_HEIGHT = 168;
const PARAGRAPH_PAD_X = 4;
const PARAGRAPH_TOP = 18;

const ANIM_DURATION_MS = 3200;
const SAMPLE_COUNT = 18;
const BLOB_RADIUS = 13;
const BODY_FRACTION = 0.55;
const MIN_GAP_WIDTH = 60;

interface Obstacle {
  x: number;
  y: number;
  r: number;
}

interface SnakeParams {
  baseY: number;
  amplitude: number;
  period: number;
  phase: number;
  direction: 1 | -1; // +1 = left to right, -1 = right to left
  yTilt: number; // additional drift along y across the run
}

interface SnakeFrame {
  obstacles: Obstacle[];
  pathD: string;
}

function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : 1 - 2 * (1 - t) * (1 - t);
}

function pseudoRandom(seed: number, salt: number): number {
  // 0..1 deterministic pick that varies clearly across triggers
  const v = Math.sin(seed * 9301.7 + salt * 49297.3) * 43758.5453;
  return v - Math.floor(v);
}

function pickSnakeParams(seed: number): SnakeParams {
  return {
    baseY: AREA_HEIGHT * 0.35 + pseudoRandom(seed, 1) * AREA_HEIGHT * 0.3,
    amplitude: 14 + pseudoRandom(seed, 2) * 22,
    period: 110 + pseudoRandom(seed, 3) * 90,
    phase: pseudoRandom(seed, 4) * Math.PI * 2,
    direction: pseudoRandom(seed, 5) > 0.5 ? 1 : -1,
    yTilt: (pseudoRandom(seed, 6) - 0.5) * AREA_HEIGHT * 0.25,
  };
}

function computeSnake(t: number, p: SnakeParams): SnakeFrame {
  const bodyLen = AREA_WIDTH * BODY_FRACTION;
  const total = AREA_WIDTH + bodyLen * 2;
  const eased = easeInOut(t);
  const advance = eased * total;
  // Head sweeps in the chosen direction; tail trails behind by bodyLen.
  const xHead = p.direction > 0 ? -bodyLen + advance : AREA_WIDTH + bodyLen - advance;
  const yDriftHead = (eased - 0.5) * 2 * p.yTilt;

  const points: [number, number][] = [];
  const obstacles: Obstacle[] = [];
  for (let i = 0; i < SAMPLE_COUNT; i++) {
    const frac = i / (SAMPLE_COUNT - 1);
    const xSample = xHead - p.direction * frac * bodyLen;
    const yBase = p.baseY + (frac - 0.5) * (p.yTilt * 0.6) + yDriftHead * (1 - frac);
    const ySample =
      yBase + Math.sin((xSample / p.period) * Math.PI * 2 + p.phase) * p.amplitude;
    points.push([xSample, ySample]);
    if (xSample >= -BLOB_RADIUS && xSample <= AREA_WIDTH + BLOB_RADIUS) {
      obstacles.push({ x: xSample, y: ySample, r: BLOB_RADIUS });
    }
  }

  // SVG textPath needs a left-to-right path so glyphs aren't reversed.
  const ordered = p.direction > 0 ? points : points.slice().reverse();
  let pathD = "";
  if (ordered.length > 0) {
    const [x0, y0] = ordered[0];
    pathD = `M ${x0.toFixed(1)} ${y0.toFixed(1)}`;
    for (let i = 1; i < ordered.length; i++) {
      const [x, y] = ordered[i];
      pathD += ` L ${x.toFixed(1)} ${y.toFixed(1)}`;
    }
  }
  return { obstacles, pathD };
}

function widestGapAt(
  y: number,
  obstacles: Obstacle[],
  minX: number,
  maxX: number,
): { x: number; width: number } {
  const blockers: [number, number][] = [];
  for (const o of obstacles) {
    const dy = o.y - y;
    if (Math.abs(dy) <= o.r) {
      const half = Math.sqrt(o.r * o.r - dy * dy);
      blockers.push([o.x - half, o.x + half]);
    }
  }
  blockers.sort((a, b) => a[0] - b[0]);
  const merged: [number, number][] = [];
  for (const b of blockers) {
    const last = merged[merged.length - 1];
    if (last && b[0] <= last[1]) {
      last[1] = Math.max(last[1], b[1]);
    } else {
      merged.push([b[0], b[1]]);
    }
  }
  let best = { x: minX, width: maxX - minX };
  if (merged.length === 0) return best;
  best = { x: minX, width: 0 };
  let cursor = minX;
  for (const [a, b] of merged) {
    if (a > cursor) {
      const w = Math.min(a, maxX) - cursor;
      if (w > best.width) best = { x: cursor, width: w };
    }
    cursor = Math.max(cursor, b);
    if (cursor >= maxX) break;
  }
  if (cursor < maxX) {
    const w = maxX - cursor;
    if (w > best.width) best = { x: cursor, width: w };
  }
  return best;
}

interface PlacedLine {
  x: number;
  y: number;
  text: string;
}

function layoutParagraph(obstacles: Obstacle[]): PlacedLine[] {
  const prepared = prepareWithSegments(RULES_TEXT, PARAGRAPH_FONT);
  let cursor: LayoutCursor = { segmentIndex: 0, graphemeIndex: 0 };
  const lines: PlacedLine[] = [];
  const minX = PARAGRAPH_PAD_X;
  const maxX = AREA_WIDTH - PARAGRAPH_PAD_X;
  for (let y = PARAGRAPH_TOP; y < AREA_HEIGHT - 6; y += LINE_HEIGHT) {
    const gap = widestGapAt(y, obstacles, minX, maxX);
    if (gap.width < MIN_GAP_WIDTH) continue;
    const range = layoutNextLineRange(prepared, cursor, gap.width);
    if (range === null) break;
    const line = materializeLineRange(prepared, range);
    lines.push({ x: gap.x, y, text: line.text });
    cursor = range.end;
  }
  return lines;
}

interface RulesTheaterProps {
  triggerKey: number;
}

export default function RulesTheater({ triggerKey }: RulesTheaterProps) {
  const [animT, setAnimT] = useState<number | null>(null);
  const paramsRef = useRef<SnakeParams | null>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (triggerKey === 0) return;
    paramsRef.current = pickSnakeParams(triggerKey);
    const start = performance.now();
    const tick = (now: number) => {
      const elapsed = now - start;
      const t = elapsed / ANIM_DURATION_MS;
      if (t >= 1) {
        rafRef.current = null;
        paramsRef.current = null;
        setAnimT(null);
        return;
      }
      setAnimT(t);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [triggerKey]);

  let snake: SnakeFrame | null = null;
  let obstacles: Obstacle[] = [];
  if (animT !== null && paramsRef.current) {
    snake = computeSnake(animT, paramsRef.current);
    obstacles = snake.obstacles;
  }
  const lines = layoutParagraph(obstacles);

  return (
    <svg
      className="rules-theater"
      width={AREA_WIDTH}
      height={AREA_HEIGHT}
      viewBox={`0 0 ${AREA_WIDTH} ${AREA_HEIGHT}`}
    >
      {lines.map((line, i) => (
        <text
          key={i}
          x={line.x}
          y={line.y}
          fontFamily={PARAGRAPH_FONT_FAMILY}
          fontSize={PARAGRAPH_FONT_SIZE}
          dominantBaseline="middle"
          style={{ fill: "var(--ink-dim)" }}
        >
          {line.text}
        </text>
      ))}

      {snake && snake.pathD.length > 0 && (
        <>
          <defs>
            <path id={`snake-path-${triggerKey}`} d={snake.pathD} />
          </defs>
          <text
            fontFamily={SNAKE_FONT_FAMILY}
            fontSize={SNAKE_FONT_SIZE}
            fontWeight={700}
            fill="#ff4d6d"
            style={{
              filter: "drop-shadow(0 0 9px rgba(255,77,109,0.6))",
              letterSpacing: "0.04em",
            }}
          >
            <textPath href={`#snake-path-${triggerKey}`}>{SNAKE_TEXT}</textPath>
          </text>
        </>
      )}
    </svg>
  );
}
