import { useState, useCallback, useEffect, useRef } from "react";
import { fetchAgents, fetchNewGame, fetchPlay, fetchState } from "../api/client";
import type { AgentOption, BoardState, MoveInfo, PlayResponse } from "../types/game";

const PASS_ACTION = 64;

const COL_LABELS = "abcdefgh";

function posLabel(pos: number): string {
  const row = Math.floor(pos / 8);
  const col = pos % 8;
  return `${COL_LABELS[col]}${row + 1}`;
}

function describeAction(action: number): string {
  if (action === PASS_ACTION) return "pass";
  return posLabel(action);
}

export interface MoveEntry {
  player: 1 | 2;
  desc: string;
}

export interface GameState {
  board: number[];          // 64 ints
  currentPlayer: number;
  gameOver: boolean;
  winner: number | null;
  actions: number[];
  moveHistory: MoveEntry[];
  topMoves: MoveInfo[];
  valueEstimate: number;    // Black's POV [-1, 1]
  usingNetwork: boolean;
  agentName: string;
  humanPlayer: 1 | 2;
  status: "idle" | "thinking" | "waiting_human" | "game_over" | "error";
  errorMsg: string;
  legalActions: number[];   // from server
  lastMove: number | null;  // last position played (for highlighting); null = pass
  availableAgents: AgentOption[];
  selectedAgent: string | null;
  passCount: number;
}

const INITIAL_BOARD = Array<number>(64).fill(0);

function initialState(humanPlayer: 1 | 2 = 1): GameState {
  return {
    board: INITIAL_BOARD,
    currentPlayer: 1,
    gameOver: false,
    winner: null,
    actions: [],
    moveHistory: [],
    topMoves: [],
    valueEstimate: 0,
    usingNetwork: false,
    agentName: "",
    humanPlayer,
    status: "idle",
    errorMsg: "",
    legalActions: [],
    lastMove: null,
    availableAgents: [],
    selectedAgent: null,
    passCount: 0,
  };
}

function applyBoardState(gs: GameState, bs: BoardState): GameState {
  return {
    ...gs,
    board: bs.board,
    currentPlayer: bs.current_player,
    gameOver: bs.game_over,
    winner: bs.winner ?? null,
    legalActions: bs.legal_actions,
    passCount: bs.pass_count,
  };
}

export function useGame() {
  const [gs, setGs] = useState<GameState>(() => initialState(1));
  const selectedAgentRef = useRef<string | null>(null);

  const callAgent = useCallback((actions: number[], humanPlayer: 1 | 2) => {
    setGs((prev) => ({ ...prev, status: "thinking" }));
    const minDelay = new Promise<void>((resolve) => setTimeout(resolve, 600));
    Promise.all([fetchPlay(actions, selectedAgentRef.current), minDelay])
      .then(([resp]: [PlayResponse, void]) => {
        const newActions = [...actions, resp.action];
        const agentPlayer: 1 | 2 = humanPlayer === 1 ? 2 : 1;
        // Agent's turn continues only if it's still the agent's move and not game over.
        // In Reversi this doesn't happen after a single move (no captures-after-move),
        // so we simply check current_player after the action.
        const agentMustContinue =
          !resp.board_after.game_over &&
          resp.board_after.current_player === agentPlayer;

        setGs((prev) => ({
          ...applyBoardState(prev, resp.board_after),
          actions: newActions,
          moveHistory: [
            ...prev.moveHistory,
            { player: agentPlayer, desc: resp.description },
          ],
          topMoves: resp.top_moves,
          valueEstimate: resp.value_estimate,
          usingNetwork: resp.using_network,
          agentName: resp.agent_name,
          status: resp.board_after.game_over
            ? "game_over"
            : agentMustContinue
              ? "thinking"
              : "waiting_human",
          errorMsg: "",
          lastMove: resp.action === PASS_ACTION ? prev.lastMove : resp.action,
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

  // After a human action: fetch authoritative state. If human has no legal
  // moves (pass-only), auto-pass; otherwise wait for next click.
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
            // Still human's turn — in Reversi this means human must pass (no legal flips).
            // Auto-pass if the only legal action is PASS_ACTION.
            if (bs.legal_actions.length === 1 && bs.legal_actions[0] === PASS_ACTION) {
              const passActions = [...actions, PASS_ACTION];
              setGs((prev) => ({
                ...applyBoardState(prev, bs),
                actions,
                moveHistory: [...prev.moveHistory, { player: humanPlayer, desc: "pass" }],
                status: "thinking",
                lastMove: prev.lastMove,
              }));
              // Give feedback to the UI then hand control to agent
              setTimeout(() => callAgent(passActions, humanPlayer), 400);
            } else {
              setGs((prev) => ({
                ...applyBoardState(prev, bs),
                actions,
                status: "waiting_human",
              }));
            }
          } else {
            callAgent(actions, humanPlayer);
          }
        })
        .catch((err: unknown) => {
          const msg = err instanceof Error ? err.message : String(err);
          setGs((prev) => ({ ...prev, status: "error", errorMsg: msg }));
        });
    },
    [callAgent],
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
          });
          // Black (P1) always moves first. If human chose White, agent goes first.
          if (humanPlayer === 2) {
            callAgent([], humanPlayer);
          }
        })
        .catch(() => {
          setGs((prev) => ({
            ...prev,
            status: "error",
            errorMsg:
              "Cannot connect to backend. Start the server with: python scripts/serve_reversi.py",
          }));
        });
    },
    [callAgent],
  );

  useEffect(() => {
    startGame(1);
  }, [startGame]);

  // Fetch available agents once on mount.
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
        // Non-fatal: server may not be fully ready yet.
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
    [startGame, gs.humanPlayer],
  );

  const handleCellClick = useCallback(
    (pos: number) => {
      if (gs.status !== "waiting_human" || gs.gameOver) return;

      const { actions, humanPlayer, legalActions } = gs;

      // Only allow legal actions
      if (!legalActions.includes(pos)) return;

      const newActions = [...actions, pos];
      const newBoard = [...gs.board];
      newBoard[pos] = gs.currentPlayer;

      setGs((prev) => ({
        ...prev,
        board: newBoard,
        actions: newActions,
        moveHistory: [
          ...prev.moveHistory,
          { player: humanPlayer, desc: describeAction(pos) },
        ],
        status: "thinking",
        lastMove: pos,
      }));

      proceedAfterHuman(newActions, humanPlayer);
    },
    [gs, proceedAfterHuman],
  );

  return { gs, handleCellClick, resetGame, setSelectedAgent };
}
