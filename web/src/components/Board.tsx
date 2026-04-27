
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

function toXY(pos: number): [number, number] {
  const [col, row] = GRID[pos];
  return [PAD + col * CELL, PAD + row * CELL];
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

const RADIUS = 18;
// Player 1 moves first (engine invariant) → white, by chess/Morris convention.
const COLORS = {
  empty: "#c8b87a",
  player1: "#e8e8e8",
  player2: "#1a1a2e",
  highlight: "#f0a500",
  legal: "#7bc67e",
  line: "#5a3e1b",
  board: "#d4a043",
};

interface Props {
  board: number[];
  legalActions: number[];
  selectedPos: number | null;
  onPositionClick: (pos: number) => void;
  disabled: boolean;
}

export default function Board({
  board,
  legalActions,
  selectedPos,
  onPositionClick,
  disabled,
}: Props) {
  // Determine which positions are valid click targets
  const NUM_PLACE_CAPTURE_ACTIONS = 24;

  const legalPositions = new Set<number>();
  for (const action of legalActions) {
    if (action < NUM_PLACE_CAPTURE_ACTIONS) {
      legalPositions.add(action);
    } else {
      const relative = action - NUM_PLACE_CAPTURE_ACTIONS;
      const src = Math.floor(relative / 24);
      const dst = relative % 24;
      if (selectedPos === src) {
        legalPositions.add(dst);
      } else if (selectedPos === null) {
        legalPositions.add(src);
      }
    }
  }

  return (
    <svg
      width={SIZE}
      height={SIZE}
      style={{ background: COLORS.board, borderRadius: 8, display: "block" }}
    >
      {/* Board lines */}
      {LINES.map((group, gi) =>
        group.map(([a, b], li) => {
          const [x1, y1] = toXY(a);
          const [x2, y2] = toXY(b);
          return (
            <line
              key={`${gi}-${li}`}
              x1={x1} y1={y1} x2={x2} y2={y2}
              stroke={COLORS.line}
              strokeWidth={3}
            />
          );
        })
      )}

      {/* Positions */}
      {board.map((owner, pos) => {
        const [cx, cy] = toXY(pos);
        const isSelected = selectedPos === pos;
        const isLegal = !disabled && legalPositions.has(pos);
        const fill =
          owner === 1 ? COLORS.player1 :
          owner === 2 ? COLORS.player2 :
          COLORS.empty;

        return (
          <g
            key={pos}
            onClick={() => !disabled && isLegal && onPositionClick(pos)}
            style={{ cursor: isLegal && !disabled ? "pointer" : "default" }}
          >
            {isSelected && (
              <circle cx={cx} cy={cy} r={RADIUS + 5} fill={COLORS.highlight} opacity={0.5} />
            )}
            {isLegal && !isSelected && (
              <circle cx={cx} cy={cy} r={RADIUS + 4} fill={COLORS.legal} opacity={0.6} />
            )}
            <circle
              cx={cx} cy={cy} r={RADIUS}
              fill={fill}
              stroke={isSelected ? COLORS.highlight : COLORS.line}
              strokeWidth={isSelected ? 3 : 1.5}
            />
            {owner === 0 && (
              <circle cx={cx} cy={cy} r={4} fill={COLORS.line} opacity={0.4} />
            )}
            <text
              x={cx}
              y={cy + RADIUS + 11}
              textAnchor="middle"
              fontSize={9}
              fill={COLORS.line}
              opacity={0.55}
              style={{ pointerEvents: "none", userSelect: "none" }}
            >
              {POSITION_LABELS[pos]}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
