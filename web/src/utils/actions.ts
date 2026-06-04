// Mirror of morris_rl/env/board.py — kept in TypeScript so the frontend
// and backend agree on the action-index layout without round-tripping
// through the API.  Two ways the encoding can go out of sync would be:
//   1. The dense 24×24 layout the frontend used before the move-action
//      space was packed (would yield action indices ≫ 80 and the server
//      would reject them as illegal).
//   2. Forgetting one of the two helper functions below in a new place
//      that needs to encode/decode moves.

export const NUM_POSITIONS = 24;
export const NUM_PLACE_CAPTURE_ACTIONS = NUM_POSITIONS;

// Same neighbour table as morris_rl/env/board.py ADJACENCY, sorted
// ascending within each source so the action ordering matches the
// backend exactly.
export const ADJACENCY: readonly (readonly number[])[] = [
  [1, 7],         // 0  a7
  [0, 2, 9],      // 1  d7
  [1, 3],         // 2  g7
  [2, 4, 11],     // 3  g4
  [3, 5],         // 4  g1
  [4, 6, 13],     // 5  d1
  [5, 7],         // 6  a1
  [0, 6, 15],     // 7  a4
  [9, 15],        // 8  b6
  [1, 8, 10, 17], // 9  d6
  [9, 11],        // 10 f6
  [3, 10, 12, 19],// 11 f4
  [11, 13],       // 12 f2
  [5, 12, 14, 21],// 13 d2
  [13, 15],       // 14 b2
  [7, 8, 14, 23], // 15 b4
  [17, 23],       // 16 c5
  [9, 16, 18],    // 17 d5
  [17, 19],       // 18 e5
  [11, 18, 20],   // 19 e4
  [19, 21],       // 20 e3
  [13, 20, 22],   // 21 d3
  [21, 23],       // 22 c3
  [15, 16, 22],   // 23 c4
];

// Packed (src,dst) → action index, parallel to MOVE_EDGES + EDGE_INDEX
// in the backend. Non-adjacent pairs stay -1 so calling code can guard
// against bogus moves.
const _EDGE_INDEX: number[][] = Array.from({ length: NUM_POSITIONS }, () =>
  Array(NUM_POSITIONS).fill(-1),
);
const _MOVE_EDGES: [number, number][] = [];
{
  let k = 0;
  for (let src = 0; src < NUM_POSITIONS; src++) {
    for (const dst of ADJACENCY[src]) {
      _EDGE_INDEX[src][dst] = NUM_PLACE_CAPTURE_ACTIONS + k;
      _MOVE_EDGES.push([src, dst]);
      k += 1;
    }
  }
}

export const EDGE_INDEX: readonly (readonly number[])[] = _EDGE_INDEX;
export const MOVE_EDGES: readonly (readonly [number, number])[] = _MOVE_EDGES;
export const NUM_MOVE_ACTIONS = _MOVE_EDGES.length;
export const ACTION_SPACE_SIZE = NUM_PLACE_CAPTURE_ACTIONS + NUM_MOVE_ACTIONS;

// Fly action range — mirrors morris_rl/env/board.py FLY_ACTION_BASE.
// Only used in the Flying variant when a player is down to 3 pieces and may
// jump from any own piece to any empty cell. Encoded ABOVE ACTION_SPACE_SIZE
// so the network's policy head (no-flying-only) is unaffected.
export const FLY_ACTION_BASE = ACTION_SPACE_SIZE;
export const NUM_FLY_ACTIONS = NUM_POSITIONS * NUM_POSITIONS;
export const EXTENDED_ACTION_SPACE_SIZE = ACTION_SPACE_SIZE + NUM_FLY_ACTIONS;

/** Decode a movement action index back to (src, dst). Returns null for
 *  non-movement actions (placement / capture range). Handles both the
 *  packed adjacency range and the extended fly range. */
export function decodeMoveAction(action: number): [number, number] | null {
  if (action < NUM_PLACE_CAPTURE_ACTIONS) return null;
  if (action >= FLY_ACTION_BASE) {
    const rel = action - FLY_ACTION_BASE;
    const src = Math.floor(rel / NUM_POSITIONS);
    const dst = rel % NUM_POSITIONS;
    if (src >= NUM_POSITIONS || dst >= NUM_POSITIONS) return null;
    return [src, dst];
  }
  const idx = action - NUM_PLACE_CAPTURE_ACTIONS;
  if (idx < 0 || idx >= _MOVE_EDGES.length) return null;
  return _MOVE_EDGES[idx] as [number, number];
}

/** Encode (src, dst). Returns -1 if the pair is not a legal Morris
 *  adjacency (i.e. EDGE_INDEX[src][dst] === -1). */
export function encodeMoveAction(src: number, dst: number): number {
  if (src < 0 || src >= NUM_POSITIONS || dst < 0 || dst >= NUM_POSITIONS) {
    return -1;
  }
  return _EDGE_INDEX[src][dst];
}

/** Encode a flying move (any (src, dst), src != dst). The Python rules
 *  engine accepts this in FLYING variant when the mover has 3 pieces. */
export function encodeFlyAction(src: number, dst: number): number {
  if (src < 0 || src >= NUM_POSITIONS || dst < 0 || dst >= NUM_POSITIONS) {
    return -1;
  }
  return FLY_ACTION_BASE + src * NUM_POSITIONS + dst;
}
