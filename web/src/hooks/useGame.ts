import { useState, useCallback, useEffect, useRef } from "react";
import { fetchAgents, fetchNewGame, fetchPlay, fetchState } from "../api/client";
import type { Variant } from "../api/client";
import type { AgentOption, BoardState, MoveInfo, PlayResponse } from "../types/game";
import {
  ADJACENCY,
  EDGE_INDEX,
  FLY_ACTION_BASE,
  NUM_PLACE_CAPTURE_ACTIONS,
  decodeMoveAction,
  encodeFlyAction,
} from "../utils/actions";

const VARIANT_STORAGE_KEY = "morris.variant";

function loadVariant(): Variant {
  if (typeof window === "undefined") return "flying";
  const v = window.localStorage.getItem(VARIANT_STORAGE_KEY);
  return v === "no-flying" ? "no-flying" : "flying";
}

function saveVariant(v: Variant): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(VARIANT_STORAGE_KEY, v);
}

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

function describeAction(action: number, mustCapture: boolean): string {
  if (action < NUM_PLACE_CAPTURE_ACTIONS) {
    return mustCapture
      ? `Capture ${POSITION_LABELS[action]}`
      : `Place ${POSITION_LABELS[action]}`;
  }
  const decoded = decodeMoveAction(action);
  if (decoded) {
    const [src, dst] = decoded;
    const arrow = action >= FLY_ACTION_BASE ? "✈" : "→";
    return `${POSITION_LABELS[src]} ${arrow} ${POSITION_LABELS[dst]}`;
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
  variant: Variant;     // "flying" allows fly-at-3-pieces; "no-flying" is adjacency-only
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

function initialState(humanPlayer: 1 | 2 = 1, variant: Variant = "flying"): GameState {
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
    variant,
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
  const [gs, setGs] = useState<GameState>(() => initialState(1, loadVariant()));
  // Latest selected agent for in-flight network calls. Mirrors gs.selectedAgent
  // so the long-lived useCallbacks below can read the freshest value without
  // having to be re-created on every state change.
  const selectedAgentRef = useRef<string | null>(null);
  // Latest variant — same pattern as selectedAgentRef so the network callbacks
  // can read it without being recreated when it changes.
  const variantRef = useRef<Variant>(gs.variant);

  // The agent runs on the post-action state; it must continue while it's still
  // its turn (e.g. it just formed a mill and now owes a capture).
  const callAgent = useCallback((actions: number[], humanPlayer: 1 | 2) => {
    setGs((prev) => ({ ...prev, status: "thinking" }));
    // Minimum "thinking" delay so the agent doesn't feel instant. Kept short
    // so the loser overlay fires close to the game-ending move instead of
    // lagging by most of a second.
    const minDelay = new Promise<void>((resolve) => setTimeout(resolve, 280));
    Promise.all([fetchPlay(actions, selectedAgentRef.current, variantRef.current), minDelay])
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
      fetchState(actions, variantRef.current)
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
    (humanPlayer: 1 | 2, variant: Variant = variantRef.current) => {
      variantRef.current = variant;
      fetchNewGame(variant)
        .then((bs) => {
          const fresh = initialState(humanPlayer, variant);
          setGs({
            ...fresh,
            ...applyBoardState(fresh, bs),
            humanPlayer,
            variant,
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
    startGame(1, loadVariant());
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
    (humanPlayer: 1 | 2 = gs.humanPlayer, variant: Variant = variantRef.current) => {
      saveVariant(variant);
      startGame(humanPlayer, variant);
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

      // Commit movement. Adjacent moves use the packed edge encoding
      // (matches the backend's MOVE_EDGES table). When flying is active
      // (variant=flying, mover has 3 pieces) we fall back to the extended
      // fly encoding above ACTION_SPACE_SIZE so non-adjacent moves are also
      // representable.
      const ownPieces = board.filter((x) => x === currentPlayer).length;
      const isFlying = gs.variant === "flying" && ownPieces === 3;
      let action = EDGE_INDEX[selectedPos][pos];
      if (action < 0) {
        if (isFlying) {
          action = encodeFlyAction(selectedPos, pos);
        } else {
          // Not an adjacent position and we're not flying — bail out.
          return;
        }
      }
      if (action < 0) return;
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
        const { mustCapture, piecesInHand, currentPlayer, selectedPos, board, serverLegalActions, variant } = gs;
        // Capture is rule-heavy (mill protection); always trust the server.
        if (mustCapture) {
          return serverLegalActions;
        }
        if (piecesInHand[currentPlayer - 1] > 0) {
          return board.map((owner, i) => (owner === 0 ? i : -1)).filter((i) => i >= 0);
        }
        const ownPieces = board.filter((x) => x === currentPlayer).length;
        const isFlying = variant === "flying" && ownPieces === 3;
        if (selectedPos !== null) {
          if (isFlying) {
            // Fly mode: any empty cell is a legal destination from any own piece.
            return board
              .map((owner, dst) => (owner === 0 ? encodeFlyAction(selectedPos, dst) : -1))
              .filter((a) => a >= 0);
          }
          return ADJACENCY[selectedPos]
            .filter((dst) => board[dst] === 0)
            .map((dst) => EDGE_INDEX[selectedPos][dst]);
        }
        return board.map((owner, i) => (owner === currentPlayer ? i : -1)).filter((i) => i >= 0);
      })()
    : [];

  return { gs, legalActions, handlePositionClick, resetGame, setSelectedAgent };
}
