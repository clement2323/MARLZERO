import { useState, useCallback, useEffect, useRef } from "react";
import { fetchAgents, fetchNewGame, fetchPlay, fetchState } from "../api/client";
import type { AgentOption, BoardState, MoveInfo, PlayResponse } from "../types/game";

const NUM_PLACE_CAPTURE_ACTIONS = 24;

// Must stay in sync with morris_rl/inference/play.py POSITION_LABELS
const POSITION_LABELS: string[] = [
  "a7", "d7", "g7",
  "g4",
  "g1", "d1", "a1",
  "a4",
  "b6", "d6", "f6",
  "f4",
  "f2", "d2", "b2",
  "b4",
  "c5", "d5", "e5",
  "e4",
  "e3", "d3", "c3",
  "c4",
];

// Mirror of morris_rl/env/board.py ADJACENCY. Each entry lists the positions
// that share a board line with the indexed position (sorted ascending).
// Must stay byte-for-byte identical to the backend or movement actions will
// mismatch and the server will reject the replay with IllegalActionError.
const ADJACENCY: number[][] = [
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

// Compact (src, dst) → action_index lookup, built from ADJACENCY in the same
// order the backend uses (src ascending, then dst ascending within each src).
// Reproduces morris_rl/env/board.py MOVE_EDGES + EDGE_INDEX so this stays
// consistent without any backend round-trip.
const EDGE_INDEX: number[][] = (() => {
  const idx: number[][] = Array.from({ length: 24 }, () => Array(24).fill(-1));
  let k = 0;
  for (let src = 0; src < 24; src++) {
    for (const dst of ADJACENCY[src]) {
      idx[src][dst] = NUM_PLACE_CAPTURE_ACTIONS + k;
      k += 1;
    }
  }
  return idx;
})();

function describeAction(action: number, mustCapture: boolean): string {
  if (action < NUM_PLACE_CAPTURE_ACTIONS) {
    return mustCapture
      ? `Capture ${POSITION_LABELS[action]}`
      : `Place ${POSITION_LABELS[action]}`;
  }
  // Find (src, dst) by linear scan over EDGE_INDEX — fine for ≤ 56 edges.
  for (let src = 0; src < 24; src++) {
    for (const dst of ADJACENCY[src]) {
      if (EDGE_INDEX[src][dst] === action) {
        return `${POSITION_LABELS[src]} → ${POSITION_LABELS[dst]}`;
      }
    }
  }
  return `move ${action}`;
}

export interface MoveEntry {
  player: 1 | 2;
  desc: string;
}

export interface GameState {
  board: number[];
  currentPlayer: number;
  piecesInHand: [number, number];
  mustCapture: boolean;
  gameOver: boolean;
  winner: number | null;
  actions: number[];
  moveHistory: MoveEntry[];
  selectedPos: number | null;
  topMoves: MoveInfo[];
  valueEstimate: number;
  // Signed (own - opp) from the human's POV. Updated after every agent move.
  // Drives the auto-shake / auto-loser anim triggers in App.tsx.
  piecesDiff: number;
  millDiff: number;
  // Monotonic id incremented on every received PlayResponse so App.tsx can
  // run side effects exactly once per server reply (instead of debouncing
  // on valueEstimate/piecesDiff which may repeat across moves).
  responseTick: number;
  usingNetwork: boolean;
  agentDescription: string;
  agentName: string;
  humanPlayer: 1 | 2;   // 1 = white (moves first), 2 = black
  status: "idle" | "thinking" | "waiting_human" | "game_over" | "error";
  errorMsg: string;
  serverReady: boolean;
  // Authoritative legal actions from the server. Used during the capture phase
  // because mill-protection rules are tricky to reimplement client-side.
  serverLegalActions: number[];
  // Last position where a piece was newly added (placement or movement
  // destination). null on captures or before the first move. Drives the
  // "just placed" flash so dark pieces on a dark surface stay visible.
  lastPlacedPos: number | null;
  // Adversary catalog and current selection. Populated on mount via /agents.
  availableAgents: AgentOption[];
  selectedAgent: string | null;
}

function findNewlyOccupied(prev: number[], next: number[]): number | null {
  for (let i = 0; i < next.length; i++) {
    if (prev[i] === 0 && next[i] !== 0) return i;
  }
  return null;
}

const INITIAL_BOARD = Array<number>(24).fill(0);

function initialState(humanPlayer: 1 | 2 = 1): GameState {
  return {
    board: INITIAL_BOARD,
    currentPlayer: 1,
    piecesInHand: [9, 9],
    mustCapture: false,
    gameOver: false,
    winner: null,
    actions: [],
    moveHistory: [],
    selectedPos: null,
    topMoves: [],
    valueEstimate: 0,
    piecesDiff: 0,
    millDiff: 0,
    responseTick: 0,
    usingNetwork: false,
    agentDescription: "",
    agentName: "",
    humanPlayer,
    status: "idle",
    errorMsg: "",
    serverReady: false,
    serverLegalActions: [],
    lastPlacedPos: null,
    availableAgents: [],
    selectedAgent: null,
  };
}

function applyBoardState(gs: GameState, bs: BoardState): GameState {
  return {
    ...gs,
    board: bs.board,
    currentPlayer: bs.current_player,
    piecesInHand: [bs.pieces_in_hand[0], bs.pieces_in_hand[1]],
    mustCapture: bs.must_capture,
    gameOver: bs.game_over,
    winner: bs.winner ?? null,
    serverLegalActions: bs.legal_actions,
  };
}

export function useGame() {
  const [gs, setGs] = useState<GameState>(() => initialState(1));
  // Latest selected agent for in-flight network calls. Mirrors gs.selectedAgent
  // so the long-lived useCallbacks below can read the freshest value without
  // having to be re-created on every state change.
  const selectedAgentRef = useRef<string | null>(null);

  // The agent runs on the post-action state; it must continue while it's still
  // its turn (e.g. it just formed a mill and now owes a capture).
  const callAgent = useCallback((actions: number[], humanPlayer: 1 | 2) => {
    setGs((prev) => ({ ...prev, status: "thinking" }));
    const minDelay = new Promise<void>((resolve) => setTimeout(resolve, 700));
    Promise.all([fetchPlay(actions, selectedAgentRef.current), minDelay])
      .then(([resp]: [PlayResponse, void]) => {
        const newActions = [...actions, resp.action];
        const agentPlayer = humanPlayer === 1 ? 2 : 1;
        const agentMustContinue =
          !resp.board_after.game_over && resp.board_after.current_player === agentPlayer;
        setGs((prev) => ({
          ...applyBoardState(prev, resp.board_after),
          actions: newActions,
          moveHistory: [...prev.moveHistory, { player: agentPlayer, desc: resp.description }],
          topMoves: resp.top_moves,
          valueEstimate: resp.value_estimate,
          // Server returns pieces_diff/mill_diff from the POV of whoever moves
          // next on board_after (typically the human) — feed that straight in.
          piecesDiff: resp.pieces_diff,
          millDiff: resp.mill_diff,
          responseTick: prev.responseTick + 1,
          usingNetwork: resp.using_network,
          agentDescription: resp.description,
          agentName: resp.agent_name,
          selectedPos: null,
          status: resp.board_after.game_over
            ? "game_over"
            : agentMustContinue
              ? "thinking"
              : "waiting_human",
          errorMsg: "",
          // Diff against the optimistic prev board: a non-null result means the
          // agent just added a piece (place or move dst); null means a capture.
          lastPlacedPos: findNewlyOccupied(prev.board, resp.board_after.board),
        }));
        if (agentMustContinue) {
          setTimeout(() => callAgent(newActions, humanPlayer), 400);
        }
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : String(err);
        setGs((prev) => ({ ...prev, status: "error", errorMsg: msg }));
      });
  }, []);

  // After a human action: ask the server for the authoritative post-action
  // state. If the human's turn continues (e.g. a mill was just formed and a
  // capture is owed), we must NOT call /play — that would let the agent pick
  // the human's capture target. Only call the agent when control has passed.
  const proceedAfterHuman = useCallback(
    (actions: number[], humanPlayer: 1 | 2) => {
      fetchState(actions)
        .then((bs) => {
          if (bs.game_over) {
            setGs((prev) => ({
              ...applyBoardState(prev, bs),
              actions,
              status: "game_over",
            }));
            return;
          }
          if (bs.current_player === humanPlayer) {
            // Human's turn continues (capture pending, or simple turn passing
            // back to them — only happens during capture). Sync state and wait.
            setGs((prev) => ({
              ...applyBoardState(prev, bs),
              actions,
              selectedPos: null,
              status: "waiting_human",
            }));
          } else {
            callAgent(actions, humanPlayer);
          }
        })
        .catch((err: unknown) => {
          const msg = err instanceof Error ? err.message : String(err);
          setGs((prev) => ({ ...prev, status: "error", errorMsg: msg }));
        });
    },
    [callAgent]
  );

  const startGame = useCallback(
    (humanPlayer: 1 | 2) => {
      fetchNewGame()
        .then((bs) => {
          const fresh = initialState(humanPlayer);
          setGs({
            ...fresh,
            ...applyBoardState(fresh, bs),
            humanPlayer,
            status: humanPlayer === 1 ? "waiting_human" : "thinking",
            serverReady: true,
          });
          // Player 1 always moves first. If the human chose black, the agent
          // (player 1) plays the opening move.
          if (humanPlayer === 2) {
            callAgent([], humanPlayer);
          }
        })
        .catch(() => {
          setGs((prev) => ({
            ...prev,
            status: "error",
            errorMsg:
              "Cannot connect to backend. Start the server with: uvicorn morris_rl.inference.server:app",
            serverReady: false,
          }));
        });
    },
    [callAgent]
  );

  useEffect(() => {
    startGame(1);
  }, [startGame]);

  // Fetch the agent catalog once and seed the default selection.
  useEffect(() => {
    fetchAgents()
      .then((resp) => {
        selectedAgentRef.current = resp.default;
        setGs((prev) => ({
          ...prev,
          availableAgents: resp.options,
          selectedAgent: resp.default,
        }));
      })
      .catch(() => {
        // Non-fatal: /agents may briefly be unavailable on startup. fetchPlay
        // will fall back to the server's own default in that case.
      });
  }, []);

  const setSelectedAgent = useCallback((agent: string) => {
    selectedAgentRef.current = agent;
    setGs((prev) => ({ ...prev, selectedAgent: agent }));
  }, []);

  const resetGame = useCallback(
    (humanPlayer: 1 | 2 = gs.humanPlayer) => {
      startGame(humanPlayer);
    },
    [startGame, gs.humanPlayer]
  );

  const handlePositionClick = useCallback(
    (pos: number) => {
      // Side effects (setTimeout / network) MUST live outside the setGs updater:
      // React StrictMode invokes updaters twice in dev to surface impurity, which
      // would otherwise schedule two backend calls and duplicate the move history.
      if (gs.status !== "waiting_human" || gs.gameOver) return;

      const {
        mustCapture,
        piecesInHand,
        currentPlayer,
        selectedPos,
        actions,
        humanPlayer,
        board,
      } = gs;

      // --- Capture phase ---
      if (mustCapture) {
        const newActions = [...actions, pos];
        const newBoard = [...board];
        newBoard[pos] = 0;
        setGs((prev) => ({
          ...prev,
          board: newBoard,
          actions: newActions,
          moveHistory: [...prev.moveHistory, { player: humanPlayer, desc: describeAction(pos, true) }],
          status: "thinking",
          lastPlacedPos: null,
        }));
        proceedAfterHuman(newActions, humanPlayer);
        return;
      }

      // --- Placement phase ---
      if (piecesInHand[currentPlayer - 1] > 0) {
        const newActions = [...actions, pos];
        const newBoard = [...board];
        newBoard[pos] = currentPlayer;
        const newHand: [number, number] = [piecesInHand[0], piecesInHand[1]];
        newHand[currentPlayer - 1]--;
        setGs((prev) => ({
          ...prev,
          board: newBoard,
          piecesInHand: newHand,
          actions: newActions,
          moveHistory: [...prev.moveHistory, { player: humanPlayer, desc: describeAction(pos, false) }],
          status: "thinking",
          lastPlacedPos: pos,
        }));
        proceedAfterHuman(newActions, humanPlayer);
        return;
      }

      // --- Movement phase: select source ---
      if (selectedPos === null) {
        if (board[pos] === currentPlayer) {
          setGs((prev) => ({ ...prev, selectedPos: pos }));
        }
        return;
      }

      // Reselect if clicking own piece again
      if (board[pos] === currentPlayer) {
        setGs((prev) => ({ ...prev, selectedPos: pos }));
        return;
      }

      // Commit movement — use the packed edge encoding so it matches the
      // backend's MOVE_EDGES table (not the legacy 24×24 dense layout).
      const action = EDGE_INDEX[selectedPos][pos];
      if (action < 0) {
        // Not an adjacent position — bail out instead of sending a bogus
        // action the server will reject.
        return;
      }
      const newActions = [...actions, action];
      const newBoard = [...board];
      newBoard[selectedPos] = 0;
      newBoard[pos] = currentPlayer;
      setGs((prev) => ({
        ...prev,
        board: newBoard,
        actions: newActions,
        moveHistory: [...prev.moveHistory, { player: humanPlayer, desc: describeAction(action, false) }],
        selectedPos: null,
        status: "thinking",
        lastPlacedPos: pos,
      }));
      proceedAfterHuman(newActions, humanPlayer);
    },
    [gs, proceedAfterHuman]
  );

  const legalActions: number[] = gs.status === "waiting_human"
    ? (() => {
        const { mustCapture, piecesInHand, currentPlayer, selectedPos, board, serverLegalActions } = gs;
        // Capture is rule-heavy (mill protection); always trust the server.
        if (mustCapture) {
          return serverLegalActions;
        }
        if (piecesInHand[currentPlayer - 1] > 0) {
          return board.map((owner, i) => (owner === 0 ? i : -1)).filter((i) => i >= 0);
        }
        if (selectedPos !== null) {
          // Only adjacent EMPTY squares are legal destinations. EDGE_INDEX
          // gives -1 for non-adjacent pairs so we filter both conditions in
          // one pass — keeps the legal set aligned with the backend.
          return ADJACENCY[selectedPos]
            .filter((dst) => board[dst] === 0)
            .map((dst) => EDGE_INDEX[selectedPos][dst]);
        }
        return board.map((owner, i) => (owner === currentPlayer ? i : -1)).filter((i) => i >= 0);
      })()
    : [];

  return { gs, legalActions, handlePositionClick, resetGame, setSelectedAgent };
}
