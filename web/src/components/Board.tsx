import {
  justifyRepeatToCircumference,
  justifyRepeatToWidth,
  justifyToWidth,
} from "../utils/pretextLayout";
import type { Jitter } from "../hooks/useShake";
import { decodeMoveAction, NUM_PLACE_CAPTURE_ACTIONS as _NUM_PLACE_CAPTURE_ACTIONS } from "../utils/actions";

// SVG canvas is 560×560, 7×7 grid with 80px cells, 40px padding.
// Board positions indexed 0-23 matching morris_rl POSITION_LABELS.
// Notation: columns a-g (left→right), rows 1-7 (bottom→top), chess-style.
const SIZE = 560;
const CELL = 80;
const PAD = 40;

// [col, row] in the 7×7 grid (0-indexed from top-left)
const GRID: [number, number][] = [
  [0, 0], [3, 0], [6, 0],   // 0-2  outer top
  [6, 3],                    // 3    outer right-mid
  [6, 6], [3, 6], [0, 6],   // 4-6  outer bottom
  [0, 3],                    // 7    outer left-mid
  [1, 1], [3, 1], [5, 1],   // 8-10 middle top
  [5, 3],                    // 11   middle right-mid
  [5, 5], [3, 5], [1, 5],   // 12-14 middle bottom
  [1, 3],                    // 15   middle left-mid
  [2, 2], [3, 2], [4, 2],   // 16-18 inner top
  [4, 3],                    // 19   inner right-mid
  [4, 4], [3, 4], [2, 4],   // 20-22 inner bottom
  [2, 3],                    // 23   inner left-mid
];

function basePosition(pos: number): [number, number] {
  const [col, row] = GRID[pos];
  return [PAD + col * CELL, PAD + row * CELL];
}

function applyJitter(
  positions: [number, number][],
  jitter: readonly Jitter[] | undefined,
): [number, number][] {
  if (!jitter) return positions;
  return positions.map(([x, y], i) => {
    const j = jitter[i];
    if (!j) return [x, y];
    return [x + j[0], y + j[1]];
  });
}

// Board lines: pairs of position indices that share a straight segment
const LINES: [number, number][][] = [
  // Outer square
  [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0]],
  // Middle square
  [[8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 8]],
  // Inner square
  [[16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 16]],
  // Spokes connecting rings
  [[1, 9], [9, 17]],
  [[3, 11], [11, 19]],
  [[5, 13], [13, 21]],
  [[7, 15], [15, 23]],
];

// Must stay in sync with morris_rl/inference/play.py POSITION_LABELS
const POSITION_LABELS: string[] = [
  "a7", "d7", "g7",  // 0-2  outer top
  "g4",              // 3    outer right-mid
  "g1", "d1", "a1",  // 4-6  outer bottom
  "a4",              // 7    outer left-mid
  "b6", "d6", "f6",  // 8-10 middle top
  "f4",              // 11   middle right-mid
  "f2", "d2", "b2",  // 12-14 middle bottom
  "b4",              // 15   middle left-mid
  "c5", "d5", "e5",  // 16-18 inner top
  "e4",              // 19   inner right-mid
  "e3", "d3", "c3",  // 20-22 inner bottom
  "c4",              // 23   inner left-mid
];

// All 16 mills (3-in-a-row patterns), as triples of position indices.
// First triple is one endpoint, middle, other endpoint — used to render a
// banner along the full mill from endpoint to endpoint.
const MILLS: [number, number, number][] = [
  [0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 0],          // outer square
  [8, 9, 10], [10, 11, 12], [12, 13, 14], [14, 15, 8], // middle square
  [16, 17, 18], [18, 19, 20], [20, 21, 22], [22, 23, 16], // inner square
  [1, 9, 17], [3, 11, 19], [5, 13, 21], [7, 15, 23],   // spokes
];

const RADIUS = 18;
// Larger invisible hit target for touch screens. Apple HIG / Material both
// recommend ≥ 44 CSS-px tap targets. Our SVG is scaled to viewport so the
// effective hit area follows the SVG scale.
const HIT_RADIUS = 28;
// Player 1 moves first (engine invariant) → white, by chess/Morris convention.
// Mirror of the CSS palette in index.css. Keep the two in sync.
const COLORS = {
  surface: "#11141c",
  scaffold: "#2a2f3d",
  rail: "#8b94a3",
  positionDot: "#3b4252",
  positionLabel: "#d946ef",
  player1: "#f5f6f8",
  player1Edge: "#2a2f3d",
  // Black piece sits on a near-black surface, so it carries a bright neutral
  // ring + soft outer halo to read clearly without losing its "black" identity.
  player2: "#0a0c12",
  player2Edge: "#a4adb9",
  selected: "#a78bfa",
  legal: "#22d3ee",
  millWhite: "#22d3ee",
  millBlack: "#f472b6",
};

// Single canvas-style font shorthand reused for pretext measurement and SVG
// rendering. Keeping them in sync is required for the justification math.
const RAIL_FONT_SIZE = 11;
const RAIL_FAMILY = "ui-monospace, SFMono-Regular, Menlo, monospace";
const RAIL_FONT = `${RAIL_FONT_SIZE}px ${RAIL_FAMILY}`;
const RAIL_UNIT = "MORRIS · ";
const RAIL_END_PAD = 24; // keep text from sliding under the position discs

const RING_FONT_SIZE = 7;
const RING_FAMILY = "ui-monospace, SFMono-Regular, Menlo, monospace";
const RING_FONT = `${RING_FONT_SIZE}px ${RING_FAMILY}`;
const RING_RADIUS = RADIUS + 7;
const RING_UNIT_P1 = "ALPHAZERO ";
const RING_UNIT_P2 = "MILLS ";

const MILL_FONT_SIZE = 22;
const MILL_FAMILY =
  '"Iowan Old Style", "Palatino Linotype", Georgia, serif';
const MILL_FONT = `${MILL_FONT_SIZE}px ${MILL_FAMILY}`;
const MILL_TEXT = "M I L L";
const MILL_END_PAD = 32;

interface Props {
  board: number[];
  legalActions: number[];
  selectedPos: number | null;
  onPositionClick: (pos: number) => void;
  disabled: boolean;
  jitter?: readonly Jitter[];
  // The most recent square where a piece was added. Drives a brief flash so
  // dark pieces on a near-black surface don't visually disappear on placement.
  lastPlacedPos?: number | null;
  // Monotonic key (typically moveHistory.length) used to retrigger the flash
  // animation when the same square is re-placed across moves.
  lastMoveKey?: number;
}

interface RailSegment {
  ax: number;
  ay: number;
  bx: number;
  by: number;
  midX: number;
  midY: number;
  angleDeg: number;
  length: number;
}

function buildRailSegment(
  positions: [number, number][],
  a: number,
  b: number,
): RailSegment {
  const [ax, ay] = positions[a];
  const [bx, by] = positions[b];
  const dx = bx - ax;
  const dy = by - ay;
  const length = Math.hypot(dx, dy);
  let angleDeg = (Math.atan2(dy, dx) * 180) / Math.PI;
  // Keep text upright: never render upside-down on horizontal/vertical lines.
  if (angleDeg > 90) angleDeg -= 180;
  if (angleDeg < -90) angleDeg += 180;
  return {
    ax, ay, bx, by,
    midX: (ax + bx) / 2,
    midY: (ay + by) / 2,
    angleDeg,
    length,
  };
}

function activeMills(board: number[]): [number, number, number][] {
  const out: [number, number, number][] = [];
  for (const mill of MILLS) {
    const [a, b, c] = mill;
    const owner = board[a];
    if (owner !== 0 && board[b] === owner && board[c] === owner) {
      out.push(mill);
    }
  }
  return out;
}

export default function Board({
  board,
  legalActions,
  selectedPos,
  onPositionClick,
  disabled,
  jitter,
  lastPlacedPos = null,
  lastMoveKey = 0,
}: Props) {
  // Per-position coordinates (jittered if a shake is active). All renderers —
  // scaffold, rails, mill banners, pieces, ring paths — read from this single
  // array so the whole board moves coherently each frame.
  const positions: [number, number][] = applyJitter(
    board.map((_owner, pos) => basePosition(pos)),
    jitter,
  );

  // Decode legalActions into the set of board positions the human can tap.
  // Movement actions go through decodeMoveAction (utils/actions.ts) which
  // mirrors the backend's MOVE_EDGES table — using the legacy dense 24×24
  // layout here silently mislabels positions and the destination cell
  // becomes unclickable in the movement phase.
  const legalPositions = new Set<number>();
  for (const action of legalActions) {
    if (action < _NUM_PLACE_CAPTURE_ACTIONS) {
      legalPositions.add(action);
      continue;
    }
    const decoded = decodeMoveAction(action);
    if (!decoded) continue;
    const [src, dst] = decoded;
    if (selectedPos === src) {
      legalPositions.add(dst);
    } else if (selectedPos === null) {
      legalPositions.add(src);
    }
  }

  const mills = activeMills(board);
  const millKeySet = new Set(mills.map(([a, _b, c]) => `${a}-${c}`));

  return (
    <svg
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      width="100%"
      height="auto"
      preserveAspectRatio="xMidYMid meet"
      style={{
        background: COLORS.surface,
        borderRadius: 16,
        display: "block",
        maxWidth: SIZE,
        width: "100%",
        height: "auto",
        touchAction: "manipulation",
        boxShadow:
          "0 0 0 1px rgba(255,255,255,0.04), 0 30px 80px rgba(0,0,0,0.55), 0 0 60px rgba(167,139,250,0.06)",
      }}
    >
      <defs>
        {/* One circular path per board position so <textPath> can wrap a ring of
            text exactly around each piece. Path starts at the leftmost point and
            sweeps clockwise so reading order is "top of the disc". */}
        {board.map((_owner, pos) => {
          const [cx, cy] = positions[pos];
          const r = RING_RADIUS;
          const d =
            `M ${cx - r},${cy} ` +
            `a ${r},${r} 0 1,1 ${2 * r},0 ` +
            `a ${r},${r} 0 1,1 ${-2 * r},0`;
          return <path key={`ring-${pos}`} id={`ring-${pos}`} d={d} fill="none" />;
        })}
      </defs>

      {/* Faint geometric scaffold: keeps the board readable even before the text
          rails paint, and gives mill banners something subtle to sit on. */}
      {LINES.map((group, gi) =>
        group.map(([a, b], li) => {
          const [x1, y1] = positions[a];
          const [x2, y2] = positions[b];
          return (
            <line
              key={`scaffold-${gi}-${li}`}
              x1={x1} y1={y1} x2={x2} y2={y2}
              stroke={COLORS.scaffold}
              strokeWidth={1}
              opacity={0.55}
            />
          );
        })
      )}

      {/* Idea 1 — board lines are now strips of text that pretext justifies to
          fill the exact pixel length of each segment. Letter-spacing comes from
          (target - natural width) / gaps, measured by pretext's canvas pass. */}
      {LINES.flatMap((group, gi) =>
        group.map(([a, b], li) => {
          const seg = buildRailSegment(positions, a, b);
          const targetWidth = Math.max(20, seg.length - 2 * RAIL_END_PAD);
          const { text, letterSpacing } = justifyRepeatToWidth(
            RAIL_UNIT,
            RAIL_FONT,
            targetWidth,
          );
          if (text.length === 0) return null;
          return (
            <g
              key={`rail-${gi}-${li}`}
              transform={`translate(${seg.midX},${seg.midY}) rotate(${seg.angleDeg})`}
            >
              <text
                x={0}
                y={0}
                textAnchor="middle"
                dominantBaseline="middle"
                fontFamily={RAIL_FAMILY}
                fontSize={RAIL_FONT_SIZE}
                letterSpacing={letterSpacing}
                fill={COLORS.rail}
                opacity={0.55}
                style={{ pointerEvents: "none", userSelect: "none" }}
              >
                {text}
              </text>
            </g>
          );
        }),
      )}

      {/* Idea 3 — mill banners. We re-justify a fixed `MILL` token to span the
          full endpoint-to-endpoint length of any active mill. */}
      {mills.map(([a, _b, c]) => {
        const seg = buildRailSegment(positions, a, c);
        const targetWidth = Math.max(40, seg.length - 2 * MILL_END_PAD);
        const { text, letterSpacing } = justifyToWidth(
          MILL_TEXT,
          MILL_FONT,
          targetWidth,
        );
        const owner = board[a];
        const accent = owner === 1 ? COLORS.millWhite : COLORS.millBlack;
        return (
          <g
            key={`mill-${a}-${c}`}
            transform={`translate(${seg.midX},${seg.midY}) rotate(${seg.angleDeg})`}
            style={{
              pointerEvents: "none",
              filter: `drop-shadow(0 0 6px ${accent}88)`,
            }}
          >
            <text
              x={0}
              y={0}
              textAnchor="middle"
              dominantBaseline="middle"
              fontFamily={MILL_FAMILY}
              fontSize={MILL_FONT_SIZE}
              fontWeight={700}
              letterSpacing={letterSpacing}
              fill={accent}
              opacity={0.95}
            >
              {text}
            </text>
          </g>
        );
      })}

      {/* Positions */}
      {board.map((owner, pos) => {
        const [cx, cy] = positions[pos];
        const isSelected = selectedPos === pos;
        const isLegal = !disabled && legalPositions.has(pos);
        const fill =
          owner === 1 ? COLORS.player1 :
          owner === 2 ? COLORS.player2 :
          "transparent";
        const edge =
          owner === 1 ? COLORS.player1Edge :
          owner === 2 ? COLORS.player2Edge :
          COLORS.scaffold;

        const inMill =
          owner !== 0 &&
          MILLS.some(([x, _y, z]) =>
            millKeySet.has(`${x}-${z}`) &&
            (x === pos || _y === pos || z === pos),
          );

        return (
          <g
            key={pos}
            onPointerDown={(e) => {
              if (!disabled && isLegal) {
                e.preventDefault();
                onPositionClick(pos);
              }
            }}
            style={{
              cursor: isLegal && !disabled ? "pointer" : "default",
              touchAction: "manipulation",
            }}
          >
            {/* Invisible hit target — keeps the visual piece small while
                giving touch screens a comfortable tap zone (≥ 44 CSS-px). */}
            <circle
              cx={cx}
              cy={cy}
              r={HIT_RADIUS}
              fill="transparent"
              pointerEvents={isLegal && !disabled ? "all" : "none"}
            />
            {isSelected && (
              <circle
                cx={cx} cy={cy} r={RADIUS + 7}
                fill="none"
                stroke={COLORS.selected}
                strokeWidth={1.5}
                opacity={0.9}
                style={{ filter: `drop-shadow(0 0 8px ${COLORS.selected}cc)` }}
              />
            )}
            {isLegal && !isSelected && (
              <circle
                cx={cx} cy={cy} r={RADIUS + 5}
                fill="none"
                stroke={COLORS.legal}
                strokeWidth={1.25}
                opacity={0.85}
                style={{ filter: `drop-shadow(0 0 6px ${COLORS.legal}99)` }}
              />
            )}
            <circle
              cx={cx} cy={cy} r={RADIUS}
              fill={fill}
              stroke={isSelected ? COLORS.selected : edge}
              strokeWidth={isSelected ? 2 : owner === 2 ? 1.5 : 1}
              style={
                owner === 2
                  ? { filter: "drop-shadow(0 0 6px rgba(255,255,255,0.08))" }
                  : undefined
              }
            />
            {owner === 0 && (
              <circle cx={cx} cy={cy} r={3} fill={COLORS.positionDot} opacity={0.85} />
            )}

            {/* Idea 2 — circular text ring around each owned piece. Pretext
                computes how many copies of the unit fit along the circumference
                and the letter-spacing that closes the loop seamlessly. */}
            {owner !== 0 && (() => {
              const unit = owner === 1 ? RING_UNIT_P1 : RING_UNIT_P2;
              const { text, letterSpacing } = justifyRepeatToCircumference(
                unit,
                RING_FONT,
                RING_RADIUS,
              );
              if (text.length === 0) return null;
              const millAccent = owner === 1 ? COLORS.millWhite : COLORS.millBlack;
              const ringFill = inMill
                ? millAccent
                : owner === 1
                  ? COLORS.player2 // dark text on white piece
                  : COLORS.rail;   // muted light text on black piece
              return (
                <text
                  fontFamily={RING_FAMILY}
                  fontSize={RING_FONT_SIZE}
                  letterSpacing={letterSpacing}
                  fill={ringFill}
                  opacity={inMill ? 1 : 0.7}
                  style={{ pointerEvents: "none", userSelect: "none" }}
                >
                  <textPath href={`#ring-${pos}`} startOffset={0}>
                    {text}
                  </textPath>
                </text>
              );
            })()}

            <text
              x={cx}
              y={cy - RADIUS - 8}
              textAnchor="middle"
              fontSize={10}
              fontWeight={700}
              fontFamily="ui-monospace, JetBrains Mono, SF Mono, Menlo, monospace"
              fill={COLORS.positionLabel}
              opacity={0.95}
              style={{
                pointerEvents: "none",
                userSelect: "none",
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                filter: `drop-shadow(0 0 4px ${COLORS.positionLabel}88)`,
              }}
            >
              {POSITION_LABELS[pos]}
            </text>
          </g>
        );
      })}

      {/* Just-placed flash. Re-keyed on each move so the CSS animation re-fires. */}
      {lastPlacedPos !== null && (
        <g
          key={`flash-${lastMoveKey}`}
          className="just-placed-flash"
          style={{
            transformOrigin: `${positions[lastPlacedPos][0]}px ${positions[lastPlacedPos][1]}px`,
            pointerEvents: "none",
          }}
        >
          <circle
            cx={positions[lastPlacedPos][0]}
            cy={positions[lastPlacedPos][1]}
            r={RADIUS + 2}
            fill="none"
            stroke={COLORS.legal}
            strokeWidth={2.5}
          />
        </g>
      )}
    </svg>
  );
}
