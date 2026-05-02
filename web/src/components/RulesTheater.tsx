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

const ANIM_DURATION_MS = 31000;
const SAMPLE_COUNT = 22;
const BLOB_RADIUS = 12;
const BODY_PIXELS = 340;
const OFFSCREEN_PAD = 90;
const MIN_GAP_WIDTH = 60;

interface Obstacle {
  x: number;
  y: number;
  r: number;
}

interface SnakeParams {
  // Path goes from `entry` (offscreen at one corner) to `exit` (offscreen at
  // the opposite corner). The diagonal guarantees the body crosses the full
  // text region, so reflow is visible the whole way.
  entry: [number, number];
  exit: [number, number];
  amplitude: number;
  period: number;
  phase: number;
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

const CORNER_PAIRS: Array<[[number, number], [number, number]]> = [
  // top-left → bottom-right
  [
    [-OFFSCREEN_PAD, -OFFSCREEN_PAD],
    [AREA_WIDTH + OFFSCREEN_PAD, AREA_HEIGHT + OFFSCREEN_PAD],
  ],
  // top-right → bottom-left
  [
    [AREA_WIDTH + OFFSCREEN_PAD, -OFFSCREEN_PAD],
    [-OFFSCREEN_PAD, AREA_HEIGHT + OFFSCREEN_PAD],
  ],
  // bottom-right → top-left
  [
    [AREA_WIDTH + OFFSCREEN_PAD, AREA_HEIGHT + OFFSCREEN_PAD],
    [-OFFSCREEN_PAD, -OFFSCREEN_PAD],
  ],
  // bottom-left → top-right
  [
    [-OFFSCREEN_PAD, AREA_HEIGHT + OFFSCREEN_PAD],
    [AREA_WIDTH + OFFSCREEN_PAD, -OFFSCREEN_PAD],
  ],
];

function pickSnakeParams(seed: number): SnakeParams {
  const cornerIdx = Math.floor(pseudoRandom(seed, 0) * CORNER_PAIRS.length) % CORNER_PAIRS.length;
  const [entry, exit] = CORNER_PAIRS[cornerIdx];
  return {
    entry,
    exit,
    amplitude: 14 + pseudoRandom(seed, 2) * 24,
    period: 120 + pseudoRandom(seed, 3) * 90,
    phase: pseudoRandom(seed, 4) * Math.PI * 2,
  };
}

function computeSnake(t: number, p: SnakeParams): SnakeFrame {
  const eased = easeInOut(t);
  const dx = p.exit[0] - p.entry[0];
  const dy = p.exit[1] - p.entry[1];
  const L = Math.hypot(dx, dy);
  const Dx = dx / L; // unit vector along travel
  const Dy = dy / L;
  const Px = -Dy; // perpendicular for sin modulation
  const Py = Dx;
  const totalTravel = L + BODY_PIXELS;
  const advance = eased * totalTravel;

  const points: [number, number][] = [];
  const obstacles: Obstacle[] = [];
  for (let i = 0; i < SAMPLE_COUNT; i++) {
    const along = advance - (i / (SAMPLE_COUNT - 1)) * BODY_PIXELS;
    const xBase = p.entry[0] + along * Dx;
    const yBase = p.entry[1] + along * Dy;
    const sinPhase = (along / p.period) * Math.PI * 2 + p.phase;
    const offset = Math.sin(sinPhase) * p.amplitude;
    const x = xBase + Px * offset;
    const y = yBase + Py * offset;
    points.push([x, y]);
    if (
      x >= -BLOB_RADIUS &&
      x <= AREA_WIDTH + BLOB_RADIUS &&
      y >= -BLOB_RADIUS &&
      y <= AREA_HEIGHT + BLOB_RADIUS
    ) {
      obstacles.push({ x, y, r: BLOB_RADIUS });
    }
  }

  // Emit path from tail (last sample) to head (first sample) so the glyphs
  // along <textPath> read in the same direction as the snake's motion.
  const ordered = points.slice().reverse();
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
