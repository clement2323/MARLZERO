import { useState, useCallback, useEffect } from "react";
import { fetchNewGame, fetchPlay, fetchState } from "../api/client";
import type { BoardState, MoveInfo, PlayResponse } from "../types/game";

const NUM_PLACE_CAPTURE_ACTIONS = 24;
const NUM_POSITIONS = 24;

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
  const src = Math.floor((action - NUM_PLACE_CAPTURE_ACTIONS) / NUM_POSITIONS);
  const dst = (action - NUM_PLACE_CAPTURE_ACTIONS) % NUM_POSITIONS;
  return `${POSITION_LABELS[src]} → ${POSITION_LABELS[dst]}`;
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
    usingNetwork: false,
    agentDescription: "",
    agentName: "",
    humanPlayer,
    status: "idle",
    errorMsg: "",
    serverReady: false,
    serverLegalActions: [],
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

  // The agent runs on the post-action state; it must continue while it's still
  // its turn (e.g. it just formed a mill and now owes a capture).
  const callAgent = useCallback((actions: number[], humanPlayer: 1 | 2) => {
    setGs((prev) => ({ ...prev, status: "thinking" }));
    const minDelay = new Promise<void>((resolve) => setTimeout(resolve, 700));
    Promise.all([fetchPlay(actions), minDelay])
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

  const resetGame = useCallback(
    (humanPlayer: 1 | 2 = gs.humanPlayer) => {
      startGame(humanPlayer);
    },
    [startGame, gs.humanPlayer]
  );

  const handlePositionClick = useCallback(
    (pos: number) => {
      setGs((prev) => {
        if (prev.status !== "waiting_human" || prev.gameOver) return prev;

        const {
          mustCapture,
          piecesInHand,
          currentPlayer,
          selectedPos,
          actions,
          moveHistory,
          humanPlayer,
        } = prev;

        // --- Capture phase ---
        if (mustCapture) {
          const newActions = [...actions, pos];
          const newBoard = [...prev.board];
          newBoard[pos] = 0;
          setTimeout(() => proceedAfterHuman(newActions, humanPlayer), 0);
          return {
            ...prev,
            board: newBoard,
            actions: newActions,
            moveHistory: [...moveHistory, { player: humanPlayer, desc: describeAction(pos, true) }],
            status: "thinking",
          };
        }

        // --- Placement phase ---
        if (piecesInHand[currentPlayer - 1] > 0) {
          const newActions = [...actions, pos];
          const newBoard = [...prev.board];
          newBoard[pos] = currentPlayer;
          const newHand: [number, number] = [prev.piecesInHand[0], prev.piecesInHand[1]];
          newHand[currentPlayer - 1]--;
          setTimeout(() => proceedAfterHuman(newActions, humanPlayer), 0);
          return {
            ...prev,
            board: newBoard,
            piecesInHand: newHand,
            actions: newActions,
            moveHistory: [...moveHistory, { player: humanPlayer, desc: describeAction(pos, false) }],
            status: "thinking",
          };
        }

        // --- Movement phase: select source ---
        if (selectedPos === null) {
          if (prev.board[pos] === currentPlayer) {
            return { ...prev, selectedPos: pos };
          }
          return prev;
        }

        // Reselect if clicking own piece again
        if (prev.board[pos] === currentPlayer) {
          return { ...prev, selectedPos: pos };
        }

        // Commit movement
        const action = NUM_PLACE_CAPTURE_ACTIONS + selectedPos * 24 + pos;
        const newActions = [...actions, action];
        const newBoard = [...prev.board];
        newBoard[selectedPos] = 0;
        newBoard[pos] = currentPlayer;
        setTimeout(() => proceedAfterHuman(newActions, humanPlayer), 0);
        return {
          ...prev,
          board: newBoard,
          actions: newActions,
          moveHistory: [...moveHistory, { player: humanPlayer, desc: describeAction(action, false) }],
          selectedPos: null,
          status: "thinking",
        };
      });
    },
    [proceedAfterHuman]
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
          return board.map((owner, i) => (owner === 0 ? NUM_PLACE_CAPTURE_ACTIONS + selectedPos * 24 + i : -1)).filter((i) => i >= 0);
        }
        return board.map((owner, i) => (owner === currentPlayer ? i : -1)).filter((i) => i >= 0);
      })()
    : [];

  return { gs, legalActions, handlePositionClick, resetGame };
}
