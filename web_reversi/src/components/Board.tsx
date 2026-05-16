import { justifyRepeatToCircumference } from "../utils/pretextLayout";

// SVG canvas: 520×520, 8×8 grid, 60px cells, 20px padding.
// Positions are row-major: pos = row * 8 + col (row 0 = top, col 0 = left).
const SIZE = 520;
const CELL = 60;
const PAD = 20;

// Piece radius — generous enough to show the ring text clearly.
const RADIUS = 22;

const COLORS = {
  surface: "#11141c",
  grid: "#0d1117",
  gridBorder: "#2e4a6a",
  labelColor: "#e879f9",
  // Black piece (PLAYER_1 = 1): near-black fill, light-grey stroke
  p1Fill: "#0a0c12",
  p1Stroke: "#a4adb9",
  // White piece (PLAYER_2 = 2): near-white fill, dark stroke
  p2Fill: "#f5f6f8",
  p2Stroke: "#2a2f3d",
  legal: "#22d3ee",
  lastMove: "#ffd700",
  lastMoveBg: "#1a1500",
};

const RING_FONT_SIZE = 6.5;
const RING_FAMILY = "ui-monospace, SFMono-Regular, Menlo, monospace";
const RING_FONT = `${RING_FONT_SIZE}px ${RING_FAMILY}`;
const RING_RADIUS = RADIUS + 7;
const RING_UNIT_P1 = "REVERSI · ";   // Black pieces
const RING_UNIT_P2 = "OTHELLO · ";   // White pieces

const COL_LABELS = "abcdefgh";

interface Props {
  board: number[];          // 64 ints: 0=empty, 1=black, 2=white
  legalActions: number[];   // cell indices that are legal clicks
  onCellClick: (pos: number) => void;
  disabled: boolean;
  lastMove: number | null;  // last played position (for gold highlight)
  lastMoveKey: number;      // monotonic key to retrigger CSS flash animation
}

function cx(col: number): number {
  return PAD + col * CELL + CELL / 2;
}

function cy(row: number): number {
  return PAD + row * CELL + CELL / 2;
}

export default function Board({
  board,
  legalActions,
  onCellClick,
  disabled,
  lastMove,
  lastMoveKey,
}: Props) {
  const legalSet = new Set(legalActions);

  return (
    <svg
      width={SIZE}
      height={SIZE}
      style={{
        background: COLORS.surface,
        borderRadius: 16,
        display: "block",
        boxShadow:
          "0 0 0 1px rgba(255,255,255,0.04), 0 30px 80px rgba(0,0,0,0.55), 0 0 60px rgba(167,139,250,0.06)",
      }}
    >
      <defs>
        {/* Circular path per cell for ring text around owned pieces */}
        {board.map((_owner, pos) => {
          const row = Math.floor(pos / 8);
          const col = pos % 8;
          const px = cx(col);
          const py = cy(row);
          const r = RING_RADIUS;
          const d =
            `M ${px - r},${py} ` +
            `a ${r},${r} 0 1,1 ${2 * r},0 ` +
            `a ${r},${r} 0 1,1 ${-2 * r},0`;
          return (
            <path key={`ring-path-${pos}`} id={`rp-${pos}`} d={d} fill="none" />
          );
        })}

        {/* Glow filter for legal-move dots */}
        <filter id="glow-cyan" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Column labels (a-h) above the grid */}
      {Array.from({ length: 8 }, (_, col) => (
        <text
          key={`col-${col}`}
          x={cx(col)}
          y={12}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={9}
          fontWeight={700}
          fontFamily="ui-monospace, JetBrains Mono, SF Mono, Menlo, monospace"
          fill={COLORS.labelColor}
          opacity={0.9}
          style={{
            pointerEvents: "none",
            userSelect: "none",
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            filter: `drop-shadow(0 0 4px ${COLORS.labelColor}88)`,
          }}
        >
          {COL_LABELS[col]}
        </text>
      ))}

      {/* Row labels (1-8) left of the grid */}
      {Array.from({ length: 8 }, (_, row) => (
        <text
          key={`row-${row}`}
          x={10}
          y={cy(row)}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={9}
          fontWeight={700}
          fontFamily="ui-monospace, JetBrains Mono, SF Mono, Menlo, monospace"
          fill={COLORS.labelColor}
          opacity={0.9}
          style={{
            pointerEvents: "none",
            userSelect: "none",
            letterSpacing: "0.12em",
            filter: `drop-shadow(0 0 4px ${COLORS.labelColor}88)`,
          }}
        >
          {row + 1}
        </text>
      ))}

      {/* Grid cells */}
      {board.map((_owner, pos) => {
        const row = Math.floor(pos / 8);
        const col = pos % 8;
        const x = PAD + col * CELL;
        const y = PAD + row * CELL;
        const isLastMove = pos === lastMove;

        return (
          <rect
            key={`cell-${pos}`}
            x={x}
            y={y}
            width={CELL}
            height={CELL}
            fill={isLastMove ? COLORS.lastMoveBg : COLORS.grid}
            stroke={isLastMove ? COLORS.lastMove : COLORS.gridBorder}
            strokeWidth={isLastMove ? 2 : 1.5}
            rx={1}
          />
        );
      })}

      {/* Pieces and legal-move dots */}
      {board.map((owner, pos) => {
        const row = Math.floor(pos / 8);
        const col = pos % 8;
        const px = cx(col);
        const py = cy(row);
        const isLegal = !disabled && legalSet.has(pos);

        if (owner === 0) {
          // Empty cell: show legal move dot if applicable
          if (!isLegal) return null;
          return (
            <g
              key={`legal-${pos}`}
              onClick={() => onCellClick(pos)}
              style={{ cursor: "pointer" }}
            >
              {/* Hover zone — invisible, large target */}
              <rect
                x={PAD + col * CELL}
                y={PAD + row * CELL}
                width={CELL}
                height={CELL}
                fill="transparent"
              />
              <circle
                cx={px}
                cy={py}
                r={6}
                fill={COLORS.legal}
                opacity={0.85}
                filter="url(#glow-cyan)"
              />
            </g>
          );
        }

        // Occupied cell
        const fill = owner === 1 ? COLORS.p1Fill : COLORS.p2Fill;
        const stroke = owner === 1 ? COLORS.p1Stroke : COLORS.p2Stroke;
        const strokeW = owner === 1 ? 1.5 : 1;

        // Ring text
        const ringUnit = owner === 1 ? RING_UNIT_P1 : RING_UNIT_P2;
        const { text: ringText, letterSpacing: ringLS } = justifyRepeatToCircumference(
          ringUnit,
          RING_FONT,
          RING_RADIUS,
        );
        const ringFill = owner === 1 ? COLORS.p2Stroke : "#2a2f3d";

        return (
          <g key={`piece-${pos}`}>
            <circle
              cx={px}
              cy={py}
              r={RADIUS}
              fill={fill}
              stroke={stroke}
              strokeWidth={strokeW}
              style={
                owner === 2
                  ? { filter: "drop-shadow(0 0 6px rgba(255,255,255,0.08))" }
                  : undefined
              }
            />
            {ringText.length > 0 && (
              <text
                fontFamily={RING_FAMILY}
                fontSize={RING_FONT_SIZE}
                letterSpacing={ringLS}
                fill={ringFill}
                opacity={0.65}
                style={{ pointerEvents: "none", userSelect: "none" }}
              >
                <textPath href={`#rp-${pos}`} startOffset={0}>
                  {ringText}
                </textPath>
              </text>
            )}
          </g>
        );
      })}

      {/* Just-placed flash — cyan ring that expands and fades on each move */}
      {lastMove !== null && lastMove !== 64 && (
        <g
          key={`flash-${lastMoveKey}`}
          className="just-placed-flash"
          style={{
            transformOrigin: `${cx(lastMove % 8)}px ${cy(Math.floor(lastMove / 8))}px`,
            pointerEvents: "none",
          }}
        >
          <circle
            cx={cx(lastMove % 8)}
            cy={cy(Math.floor(lastMove / 8))}
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
